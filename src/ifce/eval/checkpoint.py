# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Evaluate a SentenceTransformer checkpoint with the rebuttal retrieval protocol.

Pairs remain in dataset row order. Candidates contain all sources followed by all
decompilations. Each query's self-match is excluded; its target is the opposite
representation from the same row. Ties receive the optimistic rank used by the
original eval_ret.py: 1 + the number of strictly better candidates.
"""

from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk


def load_eval_dataset(data_dir: str | Path, split: str | None = None):
    """Load a saved Dataset, or select the test/only/explicit DatasetDict split."""
    dataset = load_from_disk(str(data_dir))
    selected_split = None
    if isinstance(dataset, DatasetDict):
        selected_split = split
        if selected_split is None:
            if "test" in dataset:
                selected_split = "test"
            elif len(dataset) == 1:
                selected_split = next(iter(dataset))
            else:
                raise ValueError(
                    f"Choose --split from the available splits: {list(dataset)}"
                )
        if selected_split not in dataset:
            raise ValueError(
                f"Unknown split {selected_split!r}; available: {list(dataset)}"
            )
        dataset = dataset[selected_split]
    elif split is not None:
        raise ValueError("--split applies only to a saved DatasetDict.")
    missing = {"source", "decompilation"} - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if len(dataset) == 0:
        raise ValueError("The evaluation dataset is empty.")
    for column in ("source", "decompilation"):
        if any(not isinstance(value, str) for value in dataset[column]):
            raise ValueError(
                f"Column {column!r} must contain code strings, without nulls."
            )
    return dataset, selected_split


@torch.inference_mode()
def combined_pool_metrics(
    source: torch.Tensor, decompiled: torch.Tensor, rank_batch_size: int = 512
) -> dict[str, dict[str, float]]:
    """Score normalized embeddings in query chunks, without a full 2N×2N matrix."""
    if rank_batch_size < 1:
        raise ValueError("rank_batch_size must be positive.")
    if (
        source.ndim != 2
        or source.shape != decompiled.shape
        or source.shape[0] == 0
        or source.shape[1] == 0
    ):
        raise ValueError(
            "Expected nonempty source/decompiled embeddings of equal shape."
        )
    if not torch.isfinite(source).all() or not torch.isfinite(decompiled).all():
        raise ValueError("Embeddings contain NaN or infinite values.")
    n = len(source)
    pool = torch.cat([source, decompiled], dim=0)
    metrics = {}
    for name, queries, self_offset, target_offset in (
        ("decompiled_to_source", decompiled, n, 0),
        ("source_to_decompiled", source, 0, n),
    ):
        ranks = np.empty(n, dtype=np.int64)
        for start in range(0, n, rank_batch_size):
            scores = queries[start : start + rank_batch_size] @ pool.T
            rows = torch.arange(len(scores), device=pool.device)
            indices = rows + start
            target = scores[rows, target_offset + indices].clone()
            scores[rows, self_offset + indices] = -torch.inf
            ranks[start : start + len(scores)] = (
                ((scores > target[:, None]).sum(dim=1) + 1).cpu().numpy()
            )
        metrics[name] = {
            "mrr": float(np.mean(1.0 / ranks)),
            "recall_at_1": float(np.mean(ranks <= 1)),
            "recall_at_10": float(np.mean(ranks <= 10)),
        }
    return metrics


def evaluate_checkpoint(
    checkpoint: str,
    data_dir: str | Path,
    *,
    device: str | None = None,
    max_seq_length: int = 2048,
    batch_size: int = 128,
    rank_batch_size: int = 512,
    split: str | None = None,
    trust_remote_code: bool = False,
) -> dict:
    """Load any compatible local/Hub checkpoint and return metrics plus settings."""
    from sentence_transformers import SentenceTransformer

    if min(max_seq_length, batch_size, rank_batch_size) < 1:
        raise ValueError("Sequence length and batch sizes must be positive.")
    start = perf_counter()
    dataset, selected_split = load_eval_dataset(data_dir, split)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(
        checkpoint,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    model.max_seq_length = max_seq_length
    encode_args = {
        "batch_size": batch_size,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": True,
    }
    source = torch.as_tensor(
        model.encode(list(dataset["source"]), **encode_args), device=device
    )
    decompiled = torch.as_tensor(
        model.encode(list(dataset["decompilation"]), **encode_args), device=device
    )
    metrics = combined_pool_metrics(source, decompiled, rank_batch_size)
    return {
        "checkpoint": checkpoint,
        "data_dir": str(data_dir),
        "split": selected_split,
        "num_pairs": len(dataset),
        "pool_size": 2 * len(dataset),
        "candidate_pool": "combined",
        "self_matches": "excluded",
        "tie_policy": "optimistic: 1 + count(scores > target_score)",
        "device": device,
        "max_seq_length": max_seq_length,
        "batch_size": batch_size,
        "rank_batch_size": rank_batch_size,
        "metrics": metrics,
        "elapsed_seconds": round(perf_counter() - start, 3),
    }
