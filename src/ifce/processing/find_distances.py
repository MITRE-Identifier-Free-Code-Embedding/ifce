# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
import logging

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from pandas import DataFrame
from torch.utils.data import DataLoader
from tqdm import tqdm

from ifce.config import yaml_safe_load
from ifce.device import get_device
from ifce.modeling.curvature import validate_curvature
from ifce.paths import DISTANCES_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)


def evaluation_settings(config, distance_type=None, curvature=None):
    """Resolve CLI/library overrides over YAML evaluation settings."""
    settings = config.get("evaluation", {})
    distance_type = (
        distance_type
        if distance_type is not None
        else settings.get("distance_type", "cosine")
    )
    curvature = curvature if curvature is not None else settings.get("curvature", 0.1)
    if distance_type not in ("cosine", "hyperbolic"):
        raise ValueError(f"Unknown distance_type: {distance_type}")
    validate_curvature(curvature)
    return distance_type, curvature


def distance_tag(distance_type, curvature=0.1):
    evaluation_settings({}, distance_type, curvature)
    return (
        "cosine" if distance_type == "cosine" else f"hyperbolic_c{float(curvature)!r}"
    )


def evaluation_results_dir(distance_type, curvature=0.1):
    """Separate evaluation products by metric and hyperbolic curvature."""
    return RESULTS_DIR / distance_tag(distance_type, curvature)


@torch.no_grad()
def similarity_matrix_calculation(
    full_embeddings,
    device: str,
    split_size=1024,
    distance_type="cosine",
    curvature=0.1,
):
    """Return FP32 scores, with larger values ranking first for both metrics.

    Hyperbolic scores are negative Lorentz distances using the training map.
    Compute in blocks while keeping the full score matrix on the CPU.
    """
    evaluation_settings({}, distance_type, curvature)
    if split_size <= 0:
        raise ValueError("split_size must be positive")
    full_embeddings = full_embeddings.to(device=device, dtype=torch.float32)
    if distance_type == "cosine":
        full_embeddings = F.normalize(full_embeddings, p=2, dim=1)
    else:
        from ifce.modeling.losses import negative_lorentz_distance

    count = full_embeddings.shape[0]
    scores = np.empty((count, count), dtype=np.float32)
    with torch.autocast(device_type=full_embeddings.device.type, enabled=False):
        for start in tqdm(
            range(0, count, split_size), desc=f"Calculating {distance_type} scores"
        ):
            anchors = full_embeddings[start : start + split_size]
            if distance_type == "cosine":
                scores[start : start + split_size] = (
                    (anchors @ full_embeddings.T).cpu().numpy()
                )
            else:
                for column in range(0, count, split_size):
                    candidates = full_embeddings[column : column + split_size]
                    scores[start : start + split_size, column : column + split_size] = (
                        negative_lorentz_distance(anchors, candidates, curvature)
                        .cpu()
                        .numpy()
                    )
    return scores


def cosine_matrix_calculation(full_embeddings, device: str, split_size=1024):
    """Backward-compatible cosine scoring entry point."""
    return similarity_matrix_calculation(full_embeddings, device, split_size)


def find_distances(
    data_input_dir: str,
    model_config_path: str,
    caching: bool = True,
    distance_type: str | None = None,
    curvature: float | None = None,
):
    ds = load_from_disk(data_input_dir)
    ds_length = len(ds)

    yaml_config_data = yaml_safe_load(model_config_path)
    distance_type, curvature = evaluation_settings(
        yaml_config_data, distance_type, curvature
    )
    model_list = yaml_config_data["models"]
    device = get_device()
    similarity_files = []
    cache_dir = DISTANCES_DIR / distance_tag(distance_type, curvature)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for model in model_list:
        new_file_path = cache_dir / f"{model}.parquet"
        if new_file_path.is_file() and caching:
            similarity_files.append(new_file_path)
            continue
        # Figure out embedding dimension from the first row
        ex0 = ds[0]
        d_source = len(ex0[f"{model}-embedding-source"])
        d_decomp = len(ex0[f"{model}-embedding-decompilation"])
        assert d_source == d_decomp, (
            "Source and decompilation embeddings must have same dim"
        )
        dim = d_source

        num_ids = ds_length * 2
        # full_embeddings[id] will contain the embedding for that id

        def _collate_fn(batch, model=model):
            src_ids = torch.tensor([ex["source_id"] for ex in batch])
            dec_ids = torch.tensor([ex["decompilation_id"] for ex in batch])
            src_embs = torch.stack(
                [torch.tensor(ex[f"{model}-embedding-source"]) for ex in batch]
            )
            dec_embs = torch.stack(
                [torch.tensor(ex[f"{model}-embedding-decompilation"]) for ex in batch]
            )
            return src_ids, dec_ids, src_embs, dec_embs

        loader = DataLoader(
            ds, batch_size=256, num_workers=8, collate_fn=_collate_fn, pin_memory=False
        )

        full_embeddings = torch.empty((num_ids, dim), dtype=torch.float32)

        for src_ids, dec_ids, src_embs, dec_embs in tqdm(
            loader, desc="Loading embeddings"
        ):
            full_embeddings[src_ids] = src_embs
            full_embeddings[dec_ids] = dec_embs

        score_matrix = similarity_matrix_calculation(
            full_embeddings=full_embeddings,
            device=device,
            distance_type=distance_type,
            curvature=curvature,
        )

        all_ids = list(range(num_ids))
        df = DataFrame(score_matrix, columns=all_ids, index=all_ids)
        df.attrs["model"] = model
        df.attrs["distance_type"] = distance_type
        if distance_type == "hyperbolic":
            df.attrs["curvature"] = curvature
        df.attrs["score_convention"] = "higher_is_better"
        df.attrs["embedding_size"] = full_embeddings.shape[1]
        df.to_parquet(new_file_path)
        similarity_files.append(new_file_path)

    logger.info(
        f"Processed {distance_type} scores at {dict(zip(model_list, similarity_files, strict=False))}"
    )
    return model_list, similarity_files
