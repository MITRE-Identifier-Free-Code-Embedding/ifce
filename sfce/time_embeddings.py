from modeling.models import MODEL_CLASS_MAP
from datasets import load_from_disk
from tqdm import tqdm
from processing.add_embeddings import yaml_safe_load
from time import sleep
import matplotlib.pyplot as plt
from pathlib import Path
from time import sleep, perf_counter
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import re
import pandas as pd
import gc

from pathlib import Path

TARGET_EMBED_COUNT = 10_000

VLLM_MEM_PATTERNS = {
    "weights_gb": re.compile(r"Model loading took\s+([\d.]+)\s*GiB"),
    "kv_cache_gb": re.compile(r"Available KV cache memory:\s+([\d.]+)\s*GiB"),
}


def get_memory(log_file, all_mem_stats, model_name: str):  # TODO fix this stuff

    # After health check confirms server is up
    mem_stats = {k: None for k in VLLM_MEM_PATTERNS}
    with open(log_file, "r") as f:
        lines = "".join(f.readlines())
        for key, pattern in VLLM_MEM_PATTERNS.items():
            if "cpu" in model_name:
                mem_stats[key] = 0
                continue
            if mem_stats[key] is None:
                m = pattern.search(lines)
                if m:
                    mem_stats[key] = float(m.group(1))

    all_mem_stats[model_name] = mem_stats
    return all_mem_stats


def write_mem_stats(all_mem_stats, out_path):

    df = pd.DataFrame(all_mem_stats).T.rename(
        columns={
            "weights_gb": "Weights (GiB)",
            "kv_cache_gb": "KV Cache (GiB)",
        }
    )
    df.index.name = "Model"

    # Save CSV with both columns
    df.to_csv(out_path / "vram_usage.csv")

    # ---- Plot *only* weights by model ----
    sns.set_theme(style="whitegrid", palette="pastel")  # [web:6][web:10]

    df_plot = df.reset_index()  # keep all cols, but only use Weights (GiB) in the plot

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=df_plot,
        x="Model",
        y="Weights (GiB)",  # single variable barplot [web:4][web:16][web:24]
        edgecolor="black",
    )

    ax.set_title("Model Weights VRAM Usage")
    ax.set_xlabel("Model")
    ax.set_ylabel("Weights (GiB)")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plt.savefig(out_path / "vram_weights_usage.png", dpi=200)
    plt.close()


def plot_embed_times(timing_results: dict[str, dict], out_path: Path):
    """
    timing_results: {
        model_name: {
            "decompilation": float (seconds per 10k),
            "source":        float (seconds per 10k)
        }
    }
    """
    models = list(timing_results.keys())
    keys = ["decompilation", "source"]
    palette = {"decompilation": "#4C72B0", "source": "#DD8452"}

    rows = [
        {"model": model, "key": key, "time_s": timing_results[model][key]}
        for model in models
        for key in keys
    ]

    import pandas as pd

    df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.8), 6))

    sns.barplot(
        data=df,
        x="model",
        y="time_s",
        hue="key",
        palette=palette,
        edgecolor="white",
        linewidth=0.8,
        ax=ax,
    )

    # Annotate bars with value labels
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.1fs",
            label_type="edge",
            padding=3,
            fontsize=9,
            color="#333333",
        )

    ax.set_title(
        f"Time to Embed {TARGET_EMBED_COUNT:,} Samples by Model & Key",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f s"))
    ax.legend(title="Key", title_fontsize=10, framealpha=0.85)
    sns.despine(left=True)

    fig.tight_layout()
    plot_file = out_path / "embed_timing.png"
    fig.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved timing plot → {plot_file}")


def main(
    model_config_path: str,
    data_directory: str,
    batch_size: int = 64,
    out_path_name: str = "experiment_results",
):
    device = "cuda"
    yaml_config = yaml_safe_load(model_config_path)
    out_path = Path(__file__).resolve().parent / Path(out_path_name)
    out_path.mkdir(parents=True, exist_ok=True)

    all_mem_stats = {}

    ds = load_from_disk(data_directory).select(list(range(TARGET_EMBED_COUNT)))
    batched_ds = ds.batch(batch_size=batch_size)

    # timing_results[model_name][key] = extrapolated seconds for TARGET_EMBED_COUNT
    timing_results: dict[str, dict[str, float]] = {}

    local_model = None
    for model in yaml_config["models"]:
        if "cpu" in model:
            device = "cpu"
        else:
            device = "cuda"
        sleep(5)
        if "quant" in model:
            local_model = MODEL_CLASS_MAP[model](
                device=device, name=model, quantization="fp8"
            )
        else:
            local_model = MODEL_CLASS_MAP[model](device=device, name=model)
        sleep(5)

        timing_results[model] = {}
        total_embedded: dict[str, int] = {"decompilation": 0, "source": 0}
        elapsed: dict[str, float] = {"decompilation": 0.0, "source": 0.0}

        all_mem_stats = get_memory(local_model._log_file_path, all_mem_stats, model)

        for _, batch in enumerate(
            tqdm(batched_ds, desc=f"[{model}] Embedding batches...")
        ):
            for key in ["decompilation", "source"]:
                t0 = perf_counter()
                embeddings = local_model.embed_batch(batch[key])
                elapsed[key] += perf_counter() - t0
                total_embedded[key] += len(batch[key])

        del local_model
        gc.collect()

        for key in ["decompilation", "source"]:
            timing_results[model][key] = elapsed[key]
            print(
                f"  {model} | {key}: "
                f"{elapsed[key]:.2f}s for {total_embedded[key]} samples "
                f" ~{timing_results[model][key]:.1f}s per {TARGET_EMBED_COUNT:,}"
            )

    plot_embed_times(timing_results, out_path)
    write_mem_stats(all_mem_stats, out_path)


if __name__ == "__main__":
    main(
        model_config_path="../config/time_experiment_config.yaml",
        data_directory="../data/decompiled_summaries_TEST_copy_idd/",
    )
