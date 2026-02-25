from csv import Error
from models import MODEL_CLASS_MAP
from datasets import load_from_disk
from tqdm import tqdm
import yaml
from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
import os
from pathlib import Path
"""
Note "id'd" data refers to data that has been id'd by 
1. function 
2. source function (starting from 0 to ds length) and 
3. decompilation function (starting from ds length to ds length * 2)
"""

def yaml_safe_load(filename:str):
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)
        return data


def _stack_embeddings(emb_list):
    """
    emb_list: list where each element is a vector-like (list/np.ndarray) or None
    returns: (X, keep_mask) with X as float32 ndarray
    """
    keep = [e is not None for e in emb_list]
    if not any(keep):
        return np.empty((0, 0), dtype=np.float32), np.array(keep, dtype=bool)

    X = np.asarray([emb_list[i] for i, k in enumerate(keep) if k], dtype=np.float32)
    return X, np.array(keep, dtype=bool)


def _sample_indices(n, per_rep_limit, random_state=42):
    if per_rep_limit is None or per_rep_limit >= n:
        return np.arange(n)
    rng = np.random.default_rng(random_state)
    return rng.choice(n, size=per_rep_limit, replace=False)


def save_tsne_figures_from_embeddings(
    ds,
    model_name: str,
    out_path: str,
    per_rep_limit: int = 5000,
    random_state: int = 42,
    tsne_kwargs_2d=None,
    tsne_kwargs_3d=None,
):
    """
    Uses *existing* embedding columns in `ds` created by `create_embeddings`.
    Produces and saves:
      - 2D t-SNE plot
      - 3D t-SNE plot
    Filenames and titles include `model_name`. No caching.
    """
    os.makedirs(out_path, exist_ok=True)

    decomp_col = f"{model_name}-embedding-decompilation"
    source_col = f"{model_name}-embedding-source"
    if decomp_col not in ds.column_names or source_col not in ds.column_names:
        raise KeyError(
            f"Missing embedding columns for model '{model_name}'. "
            f"Expected '{decomp_col}' and '{source_col}'."
        )

    n = len(ds)
    idx_decomp = _sample_indices(n, per_rep_limit, random_state=random_state)
    idx_source = _sample_indices(n, per_rep_limit, random_state=random_state + 1)

    decomp_emb = ds.select(idx_decomp.tolist())[decomp_col]
    source_emb = ds.select(idx_source.tolist())[source_col]

    X_decomp, _ = _stack_embeddings(decomp_emb)
    X_source, _ = _stack_embeddings(source_emb)

    if X_decomp.size == 0 or X_source.size == 0:
        raise ValueError(f"No embeddings found to visualize for model '{model_name}'.")

    X = np.concatenate([X_decomp, X_source], axis=0)
    y = np.array(
        (["decompilation"] * len(X_decomp)) + (["source"] * len(X_source)), dtype=object
    )

    uniques = ["decompilation", "source"]
    cmap = cm.get_cmap("tab10", len(uniques))
    color_map = {lab: cmap(i) for i, lab in enumerate(uniques)}

    # --- 2D t-SNE
    tsne_kwargs_2d = tsne_kwargs_2d or {}
    tsne_2d = TSNE(
        n_components=2,
        init=tsne_kwargs_2d.pop("init", "pca"),
        random_state=tsne_kwargs_2d.pop("random_state", random_state),
        verbose=tsne_kwargs_2d.pop("verbose", 1),
        **tsne_kwargs_2d,
    )
    X2 = tsne_2d.fit_transform(X)

    fig2 = plt.figure(figsize=(12, 9))
    ax2 = fig2.add_subplot(111)
    for lab in uniques:
        m = y == lab
        ax2.scatter(
            X2[m, 0],
            X2[m, 1],
            s=6,
            alpha=0.35,
            c=[color_map[lab]],
            label=lab,
        )
    ax2.set_xlabel("Component 1")
    ax2.set_ylabel("Component 2")
    ax2.set_title(f"t-SNE (2D) — {model_name} — source vs decompilation")
    ax2.legend(loc="best", frameon=False)

    out_2d = os.path.join(out_path, f"{model_name}_tsne_2d.png")
    plt.tight_layout()
    plt.savefig(out_2d, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # --- 3D t-SNE
    tsne_kwargs_3d = tsne_kwargs_3d or {}
    tsne_3d = TSNE(
        n_components=3,
        init=tsne_kwargs_3d.pop("init", "pca"),
        random_state=tsne_kwargs_3d.pop("random_state", random_state),
        verbose=tsne_kwargs_3d.pop("verbose", 1),
        **tsne_kwargs_3d,
    )
    X3 = tsne_3d.fit_transform(X)

    fig3 = plt.figure(figsize=(12, 9))
    ax3 = fig3.add_subplot(111, projection="3d")
    for lab in uniques:
        m = y == lab
        ax3.scatter(
            X3[m, 0],
            X3[m, 1],
            X3[m, 2],
            s=6,
            alpha=0.75,
            color=color_map[lab],
            label=lab,
        )

    ax3.set_xlabel("Component 1")
    ax3.set_ylabel("Component 2")
    ax3.set_zlabel("Component 3")
    ax3.set_title(f"t-SNE (3D) — {model_name} — source vs decompilation")
    ax3.legend(loc="upper right", fontsize="small", frameon=False)

    out_3d = os.path.join(out_path, f"{model_name}_tsne_3d.png")
    plt.tight_layout()
    plt.savefig(out_3d, dpi=300, bbox_inches="tight")
    plt.close(fig3)

    return {"tsne_2d_path": out_2d, "tsne_3d_path": out_3d}
    
def create_embeddings(data_directory: str, 
                      model_config_path: str, 
                      batch_size: int,
                      out_path : str = str(Path(__file__).resolve().parent / Path("experiment_results")),
                      tsne_per_rep_limit: int = 5000):
    
    device = "cuda"
    yaml_config = yaml_safe_load(model_config_path)

    ds = load_from_disk(data_directory)
    batched_ds = ds.batch(batch_size = batch_size)
    max_attempts = 5
    for model in yaml_config["models"]:
        local_model = None
        for i in range(max_attempts):
            try:
                local_model = MODEL_CLASS_MAP[model](device=device, name=model)
                break
            except ValueError:
                sleep(5)
                print(f"retrying... attempt {i}")
                if i == max_attempts - 1:
                    raise(ValueError(f"Model {model} cannot be loaded"))
        # batch and process
        embedding_list = [[None] * len(ds) for _ in range(2)]
        for _, batch in enumerate(tqdm(batched_ds, desc="Embedding batches...")):
            for key_index, key in enumerate(['decompilation','source']):
                embeddings = local_model.embed_batch(batch[key])

                for index in range(len(batch[key])):
                    embedding_list[key_index][batch["function_id"][index]] = embeddings[index]
                

        for key_index, key in enumerate(['decompilation','source']):
            ds = ds.add_column(f"{model}-embedding-{key}", embedding_list[key_index])

        print(f"Composing dimensionality reduction plots for {model}")
        ssave_tsne_figures_from_embeddings(
            ds=ds,
            model_name=model,
            out_path=out_path,
            per_rep_limit=tsne_per_rep_limit,
        )

    new_ds_directory = f"{data_directory}_embedded"
    ds.save_to_disk(new_ds_directory)
    return new_ds_directory

if __name__ == "__main__":
    # Retained for test case
    DATA_DIR = "/home/ewolos/semantic-free-code-embedding/data/decompiled_summaries_TEST_copy_idd"
    CONFIG_PATH = "/home/ewolos/semantic-free-code-embedding/config/experiment_config.yaml"
    BATCH_SIZE = 64
    create_embeddings(data_directory = DATA_DIR, 
                      model_config_path = CONFIG_PATH, 
                      batch_size = BATCH_SIZE)
