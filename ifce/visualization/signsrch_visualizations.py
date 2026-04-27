# Copyright 2026 The MITRE corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the “Software”), to deal
# in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_group_map_bars(
    group_aps_exp2: dict[str, dict[str, list[float]]],
    fig_path: str | Path,
    figsize: tuple = (20, 8),
):
    """
    Bar chart: x=algorithm group, y=MAP, hue=model.
    group_aps_exp2: model_name -> {group_name -> [per-constant APs]}
    """
    records = []
    for model_name, group_aps in group_aps_exp2.items():
        for group_name, aps in group_aps.items():
            records.append(
                {
                    "model": model_name,
                    "group": group_name,
                    "MAP": float(np.mean(aps)),
                    "n_constants": len(aps),
                }
            )
    plot_df = pd.DataFrame(records)

    group_variance = plot_df.groupby("group")["MAP"].var().sort_values(ascending=False)
    top5_groups = group_variance.head(5).index.tolist()
    plot_df = plot_df[plot_df["group"].isin(top5_groups)]
    groups = plot_df["group"].unique()
    models = plot_df["model"].unique()
    x = np.arange(len(groups))
    width = 0.8 / len(models)

    # tab20 palette — one stable color per model
    tab20 = plt.get_cmap("tab20")
    colors = [tab20(i / max(len(models) - 1, 1)) for i in range(len(models))]

    fig, ax = plt.subplots(figsize=figsize)
    for i, (model, color) in enumerate(zip(models, colors)):
        vals = [
            plot_df.loc[
                (plot_df["model"] == model) & (plot_df["group"] == g), "MAP"
            ].values[0]
            for g in groups
        ]
        ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAP (mean over constants in group)")
    ax.set_title("Per-Group MAP by Model — Experiment 2 (Top 5 Most Variant Groups)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_wilcoxon_pvalue_heatmap(
    paired_results,
    exp_name: str,
    alpha=0.05,
    log10_transform=True,
    cmap="viridis",
    file_path=None,
    annotate=True,
    figsize=(12, 10),
):
    """
    Create a heatmap of pairwise Wilcoxon p-values from paired_results.

    paired_results: dict[(model_i, model_j)] -> {
        "wilcoxon": {"p_value": float, ...}
    }
    alpha: significance threshold; optionally used to draw lines or interpret.
    log10_transform: if True, plot -log10(p) so small p-values are dark.
    cmap: matplotlib colormap name.
    file_path: if not None, save figure to this path.
    annotate: if True, write p-values (or -log10(p)) in cells.
    figsize: figure size tuple.
    """
    # Collect model names
    models = set()
    for m1, m2 in paired_results.keys():
        models.add(m1)
        models.add(m2)
    models = sorted(models)

    n = len(models)
    # Initialize matrix with NaN
    p_mat = np.full((n, n), np.nan, dtype=float)

    # Fill symmetric matrix
    index = {m: i for i, m in enumerate(models)}
    for (m1, m2), tests in paired_results.items():
        p = tests["wilcoxon"]["p_value"]
        i, j = index[m1], index[m2]
        p_mat[i, j] = p

    # Optionally transform to -log10(p) for plotting
    data = p_mat.copy()
    if log10_transform:
        with np.errstate(divide="ignore"):
            data = -np.log10(data)
        # Replace infinities from p=0 with finite max
        inf_mask = np.isinf(data)
        if np.any(inf_mask):
            max_finite = np.nanmax(data[~inf_mask])
            data[inf_mask] = max_finite

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    # Use masked array to hide NaNs (diagonal)
    masked_data = np.ma.masked_invalid(data)
    im = ax.imshow(masked_data, cmap=cmap, origin="upper")

    # Ticks and labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels([m.replace("_", "\\_") for m in models], rotation=45, ha="right")
    ax.set_yticklabels([m.replace("_", "\\_") for m in models])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    if log10_transform:
        cbar.set_label(r"$-\log_{10}(p)$")
    else:
        cbar.set_label("p-value")

    # Optional annotations
    if annotate:
        for i in range(n):
            for j in range(n):
                if np.isnan(p_mat[i, j]) or i == j:
                    continue
                if log10_transform:
                    text_val = f"{p_mat[i, j]:.3f}"
                else:
                    text_val = f"{p_mat[i, j]:.3f}"
                ax.text(
                    j,
                    i,
                    text_val,
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if masked_data[i, j] > np.nanmax(masked_data) / 2
                        else "black"
                    ),
                    fontsize=8,
                )

    title = f"Pairwise Wilcoxon p-values, {exp_name}"
    if log10_transform:
        title += " (darker = more significant)"
    ax.set_title(title)

    fig.tight_layout()

    if file_path is not None:
        fig.savefig(file_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_overlaid_histograms(ap_dict, bins=20, density=False, alpha=0.4):
    """
    Plot overlaid histograms for each model on the same axes.
    """
    plt.figure(figsize=(16, 10))
    for model, ap_list in ap_dict.items():
        plt.hist(
            ap_list,
            bins=bins,
            density=density,
            alpha=alpha,
            label=model,
            edgecolor="black",
            linewidth=0.5,
        )
    plt.xlabel("Average Precision")
    plt.ylabel("Density" if density else "Count")
    plt.title("Overlaid AP Distributions by Model")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("overlaid_histograms.png")
    plt.close()


def plot_overlaid_kde(ap_dict, title: str, fig_path: str | Path, bw_method="scott"):
    """
    Optional: plot overlaid kernel density estimates for smoother distribution comparison.
    """
    from scipy.stats import gaussian_kde

    plt.figure(figsize=(10, 7))
    xs = np.linspace(0.0, 1.0, 200)
    for model, ap_list in ap_dict.items():
        arr = np.asarray(ap_list)
        kde = gaussian_kde(arr, bw_method=bw_method)
        plt.plot(xs, kde(xs), label=model)
    plt.xlabel("Average Precision")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(fig_path)
    plt.close()


def plot_map_histograms(
    out_path, AP_dictionary, suptitle: str, extension: str, bins=50
):
    plt.figure(figsize=(16, 10))

    # Choose a global range over all models to align bins
    all_values = np.concatenate([np.array(v) for v in AP_dictionary.values()])
    min_val, max_val = np.min(all_values), np.max(all_values)

    # To make sure bins cover the range properly
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    for model_name, aps in AP_dictionary.items():
        aps = np.array(aps)
        plt.hist(
            aps,
            bins=bin_edges,
            alpha=0.4,  # transparency so overlaps are visible
            label=model_name,
            density=True,  # compare shapes as distributions
            edgecolor="black",
            linewidth=0.5,
        )

        # Optional: vertical line at mean
        mean_val = aps.mean()
        plt.axvline(
            mean_val,
            linestyle="--",
            linewidth=1,
            label=f"{model_name} mean: {mean_val:.3f}",
        )

    plt.xlabel("MAP")
    plt.ylabel("Density")
    plt.title("MAP distributions by model")
    plt.suptitle(suptitle)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{out_path}/sig_hist_{extension}.png")


def plot_model_comparison_bars(
    descriptives: dict[str, dict[str, float]],
    exp_name: str,
    extension_name: str,
    map_or_ap: str,
    fig_dir: str | Path,
    figsize: tuple = (12, 6),
):
    """
    Bar chart comparing models by MAP or AP with error bars (standard error).

    descriptives: model_name -> {"mean": float, "se": float}
    exp_name:        experiment name (used in title + filename)
    extension_name:  extension/dataset name (used in title + filename)
    map_or_ap:       "MAP" or "AP" — used for y-axis label and title
    fig_dir:         directory to save the figure
    figsize:         figure dimensions
    """
    models = list(descriptives.keys())
    means = [descriptives[m]["mean"] for m in models]
    ses = [descriptives[m]["se"] for m in models]

    tab20 = plt.get_cmap("tab20")
    colors = [tab20(i / max(len(models) - 1, 1)) for i in range(len(models))]

    x = np.arange(len(models))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        x,
        means,
        width=0.6,
        color=colors,
        alpha=0.85,
        yerr=ses,
        capsize=5,
        error_kw={"elinewidth": 1.2, "ecolor": "black"},
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(f"{map_or_ap}")
    ax.set_title(f"Mean {map_or_ap} by Model — {exp_name} / {extension_name}")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()

    slug = f"bar_plot_{exp_name}_{extension_name}_{map_or_ap}".replace(" ", "_").lower()
    fig_path = Path(fig_dir) / f"{slug}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return fig_path
