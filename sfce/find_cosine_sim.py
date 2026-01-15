from datasets import load_from_disk
from add_embeddings import yaml_safe_load
from pandas import DataFrame
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pathlib

def find_cosine_sim(data_input_dir:str, model_config_path:str):
    ds = load_from_disk(data_input_dir)
    ds_length = len(ds)
    
    yaml_config_data = yaml_safe_load(model_config_path)
    model_list = yaml_config_data["models"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    similarity_files = []

    for model in model_list:
        # Figure out embedding dimension from the first row
        ex0 = ds[0]
        d_source = len(ex0[f"{model}-embedding-source"])
        d_decomp = len(ex0[f"{model}-embedding-decompilation"])
        assert d_source == d_decomp, "Source and decompilation embeddings must have same dim"
        dim = d_source

        num_ids = ds_length * 2
        # full_embeddings[id] will contain the embedding for that id
        full_embeddings = torch.empty((num_ids, dim), dtype=torch.float32)

        # Fill the embedding matrix once
        for ex in tqdm(ds, desc="Loading embeddings"):
            src_id = ex["source_id"]
            dec_id = ex["decompilation_id"]
            full_embeddings[src_id] = torch.tensor(ex[f"{model}-embedding-source"])
            full_embeddings[dec_id] = torch.tensor(ex[f"{model}-embedding-decompilation"])

        full_embeddings = full_embeddings.to(device)

        # Normalize for cosine similarity: cos(A,B)= (A·B)/(||A|| ||B||)
        full_embeddings = F.normalize(full_embeddings, p=2, dim=1)

        # Compute full pairwise cosine similarity:
        # shape: (2N, 2N)
        # sim[i,j] = cosine(full_embeddings[i], full_embeddings[j])
        cosine_matrix = full_embeddings @ full_embeddings.T  # [2N,2N]

        # Move to CPU and convert to numpy
        cosine_matrix = cosine_matrix.cpu().numpy()

        all_ids = list(range(num_ids))
        cosine_similarity_df = DataFrame(
            cosine_matrix,
            columns=all_ids,
            index=all_ids,
        )
        cosine_similarity_df.attrs["model"] = model
        cosine_similarity_df.attrs["embedding_size"] = full_embeddings.shape[1] # This should be an embedding size

        # Save / use df as needed
        new_file_path = pathlib.Path(__file__).parent.parent.resolve() / "data" / f"cosine_{model}.parquet"
        tqdm.pandas()
        cosine_similarity_df.to_parquet(new_file_path)
        similarity_files.append(new_file_path)
    
    return model_list, similarity_files
    
