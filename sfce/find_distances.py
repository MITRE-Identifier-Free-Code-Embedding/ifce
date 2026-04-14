from ast import Break
from datasets import load_from_disk
from add_embeddings import yaml_safe_load
from pandas import DataFrame
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pathlib
from sklearn.metrics import pairwise_distances
import os
from torch.utils.data import DataLoader


def cosine_matrix_calculation(full_embeddings, device: str, split_size=1024):

    full_embeddings = full_embeddings.to(device)

    # Normalize for cosine similarity: cos(A,B)= (A·B)/(||A|| ||B||)
    full_embeddings = F.normalize(full_embeddings, p=2, dim=1)

    # Compute full pairwise cosine similarity:
    num_data_points = full_embeddings.shape[0]
    template = torch.Tensor(np.zeros((num_data_points, num_data_points)))
    split_already = 0
    pbar = tqdm(
        total=(
            int(num_data_points / split_size) + 1
            if num_data_points % split_size != 0
            else int(num_data_points / split_size)
        ),
        desc="Calculating cosine distances",
    )
    while split_already < num_data_points:

        partial_embeddings = full_embeddings[split_already : split_already + split_size]

        partial_cosine_matrix = partial_embeddings @ full_embeddings.T  # [2N,2N]
        template[split_already : split_already + split_size] = partial_cosine_matrix
        split_already += split_size
        pbar.update(1)

    cosine_matrix = template.cpu().numpy()
    return cosine_matrix


def find_distances(
    data_input_dir: str,
    model_config_path: str,
    save_euclidean: bool,
    caching: bool = True,
):
    ds = load_from_disk(data_input_dir)
    ds_length = len(ds)

    yaml_config_data = yaml_safe_load(model_config_path)
    model_list = yaml_config_data["models"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    similarity_files = []

    for model in model_list:

        new_file_path = (
            pathlib.Path(__file__).parent.parent.resolve()
            / "data"
            / f"cosine_{model}.parquet"
        )
        if (
            new_file_path.is_file() and caching == True and save_euclidean == False
        ):  # caching
            similarity_files.append(new_file_path)
            continue
        # Figure out embedding dimension from the first row
        ex0 = ds[0]
        d_source = len(ex0[f"{model}-embedding-source"])
        d_decomp = len(ex0[f"{model}-embedding-decompilation"])
        assert (
            d_source == d_decomp
        ), "Source and decompilation embeddings must have same dim"
        dim = d_source

        num_ids = ds_length * 2
        # full_embeddings[id] will contain the embedding for that id

        def _collate_fn(batch):
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

        euclidean_matrix = 1 / pairwise_distances(
            full_embeddings, n_jobs=-1
        )  # This will contain infs

        cosine_matrix = cosine_matrix_calculation(
            full_embeddings=full_embeddings, device=device
        )

        all_ids = list(range(num_ids))

        distances = [(cosine_matrix, "cosine")]
        if save_euclidean:
            distances.append((euclidean_matrix, "euclidean"))

        for matrix, distance_type in distances:

            df = DataFrame(
                matrix,
                columns=all_ids,
                index=all_ids,
            )
            df.attrs["model"] = model
            df.attrs["distance_type"] = distance_type
            df.attrs["embedding_size"] = full_embeddings.shape[
                1
            ]  # This should be an embedding size

            # Save / use df as needed
            tqdm.pandas()
            df.to_parquet(new_file_path)

            if (
                distance_type == "cosine"
            ):  # TODO modify other to support non cosine distances
                similarity_files.append(new_file_path)

    print(f"Processed cosine distances at {dict(zip(model_list, similarity_files))}")
    return model_list, similarity_files
