# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_from_disk
from joblib import Parallel, delayed

from ifce.modeling.models import MODEL_SIZE_MAP
from ifce.paths import RESULTS_DIR
from ifce.visualization.assemblage_plots import (
    log_histogram,
    scatter_with_fit,
    trimmed_quantile_boxplot,
)

K_FOR_RECALLS = [1, 3, 5, 10]


logger = logging.getLogger(__name__)


def recall_at_k(rank_series: np.ndarray, k: int):
    # Recall@k: fraction of items whose rank is <= k
    return np.sum(rank_series <= k) / rank_series.size


def MRR(rank_series: np.ndarray):
    # Standard MRR over all items: mean of 1/rank
    reciprocal_rank_series = 1.0 / rank_series.astype(float)
    return np.mean(reciprocal_rank_series)


def MRR_at_k(rank_series: np.ndarray, k: int):
    # Conditional MRR@k: mean reciprocal rank over items with rank <= k
    mask = rank_series <= k
    if not np.any(mask):
        return np.nan
    template = np.zeros(rank_series.shape)
    template[mask] = 1.0 / rank_series[mask].astype(float)
    return np.mean(template)


def get_function_lengths(dataset):

    src_ids = np.array(dataset["source_id"], dtype=np.int64)
    dec_ids = np.array(dataset["decompilation_id"], dtype=np.int64)
    src_lens = np.array([len(s) for s in dataset["source"]])
    dec_lens = np.array([len(s) for s in dataset["decompilation"]])

    template = np.zeros(len(dataset) * 2, dtype=np.int64)
    template[src_ids] = src_lens
    template[dec_ids] = dec_lens
    return template


def get_function_signature_count(dataset):
    src_ids = np.array(dataset["source_id"], dtype=np.int64)
    dec_ids = np.array(dataset["decompilation_id"], dtype=np.int64)

    def sig_arg_count(code):
        inner = code[code.find("(") + 1 : code.find(")")]
        return (
            inner.count(",") + 1 if (inner.strip() and inner.strip() != "void") else 0
        )

    src_counts = np.array([sig_arg_count(s) for s in dataset["source"]])
    dec_counts = np.array([sig_arg_count(s) for s in dataset["decompilation"]])

    template = np.zeros(len(dataset) * 2, dtype=np.int64)
    template[src_ids] = src_counts
    template[dec_ids] = dec_counts
    return template


def _get_safe_group(candidate_group):
    return candidate_group.replace(" ", "_").replace("/", "_").replace("\\", "_")


def make_correlation_scatter(metrics_df, out_path, candidate_groups):

    loc_figsize = (9, 6)
    for candidate_group in candidate_groups:
        df_group = metrics_df[metrics_df["candidate_group"] == candidate_group].copy()
        safe_group = _get_safe_group(candidate_group=candidate_group)

        fig, ax = plt.subplots(figsize=loc_figsize, constrained_layout=True)

        palette = sns.color_palette("tab20", n_colors=len(df_group))
        for i, (_, row) in enumerate(df_group.iterrows()):
            ax.scatter(
                row["Spearman Correlation - Function Length"],
                row["Spearman Correlation - Function Argument Count"],
                color=palette[i],
                s=80,
                label=row["model"],
                zorder=3,
            )

        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Spearman r  (Rank vs. Function Length)")
        ax.set_ylabel("Spearman r  (Rank vs. Argument Count)")
        ax.legend(title="Model")
        plt.suptitle("Rank Correlation by Model", fontsize=12)
        plt.title(candidate_group, fontsize=10)

        plt.savefig(
            f"{out_path}/rank_corr_scatter_{safe_group}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        fig, ax = plt.subplots(figsize=loc_figsize, constrained_layout=True)

        for i, (_, row) in enumerate(df_group.iterrows()):
            ax.scatter(
                row["Spearman Correlation - Function Length"],
                row["MRR"],
                color=palette[i],
                s=80,
                label=row["model"],
                zorder=3,
            )

        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Spearman r  (Rank vs. Function Length)")
        ax.set_ylabel("MRR")
        ax.legend(title="Model")
        plt.suptitle("Function Length Rank Correlation and MRR by Model", fontsize=12)
        plt.title(candidate_group, fontsize=10)

        plt.savefig(
            f"{out_path}/length_corr_scatter_{safe_group}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        fig, ax = plt.subplots(figsize=loc_figsize, constrained_layout=True)

        for i, (_, row) in enumerate(df_group.iterrows()):
            ax.scatter(
                row["Spearman Correlation - Function Argument Count"],
                row["MRR"],
                color=palette[i],
                s=80,
                label=row["model"],
                zorder=3,
            )

        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Spearman r  (Rank vs. Function Argument Count)")
        ax.set_ylabel("MRR")
        ax.legend(title="Model")
        plt.suptitle(
            "Function Arguments Rank Correlation and MRR by Model", fontsize=12
        )
        plt.title(candidate_group, fontsize=10)

        plt.savefig(
            f"{out_path}/args_corr_scatter_{safe_group}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        fig, ax = plt.subplots(figsize=loc_figsize, constrained_layout=True)
        for i, (_, row) in enumerate(df_group.iterrows()):
            ax.scatter(
                row["size (B of params)"],
                row["MRR"],
                color=palette[i],
                s=80,
                label=row["model"],
                zorder=3,
            )

        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Model Size (B of params)")
        ax.set_ylabel("MRR")
        ax.legend(title="Model")
        plt.suptitle("Model Size and MRR", fontsize=12)
        plt.title(candidate_group, fontsize=10)

        plt.savefig(
            f"{out_path}/size_corr_scatter_{safe_group}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def compute_query_metrics_from_ranks(ranks, k_for_recalls):
    """
    ranks: 1D iterable of integer ranks (1 = perfect, >k means 'miss' for Recall@k)
    Returns:
        metrics_per_query: dict with keys:
          'MRR', 'MRR@k', 'Recall@k' -> 1D numpy arrays of length n_queries
    """
    ranks = np.asarray(ranks)

    # Per-query reciprocal rank (for full MRR)
    rr = 1.0 / ranks

    metrics = {}
    metrics["MRR"] = rr

    # MRR@k: reciprocal rank truncated at k (0 if rank > k)
    for k in k_for_recalls:
        rr_k = np.where(ranks <= k, 1.0 / ranks, 0.0)
        metrics[f"MRR@{k}"] = rr_k

    # Recall@k: 1 if rank <= k, else 0
    for k in k_for_recalls:
        rec_k = (ranks <= k).astype(float)
        metrics[f"Recall@{k}"] = rec_k

    return metrics


def _get_metric_sem_from_ranks(ranks, k_for_recalls):
    """
    Compute per-metric standard errors from per-query metric values induced by ranks.

    Returns:
        dict mapping metric name -> standard error of the mean for that metric
    """
    metrics_per_query = compute_query_metrics_from_ranks(ranks, k_for_recalls)
    sems = {}

    for metric_name, values in metrics_per_query.items():
        values = np.asarray(values, dtype=float)
        n = values.size
        if n == 0:
            sem = np.nan
        elif n == 1:
            sem = 0.0
        else:
            sem = values.std(ddof=1) / np.sqrt(n)
        sems[metric_name] = sem

    return sems


# unused
def _attach_metric_se_columns(df_group, candidate_group, all_ranks, k_for_recalls):
    """
    Add <metric>_SE columns to a candidate-group dataframe, one row per model.

    Expected input columns:
        - 'model' or 'Model'
        - metric columns like 'MRR', 'MRR@10', 'Recall@10', ...

    Returns:
        Copy of df_group with added SE columns.
    """
    df_group = df_group.copy()

    model_col = "model" if "model" in df_group.columns else "Model"
    metric_cols = [c for c in df_group.columns if ("MRR" in c or "Recall" in c)]

    se_by_model = {}
    for model_name in df_group[model_col]:
        ranks = all_ranks[(model_name, candidate_group)]
        se_by_model[model_name] = _get_metric_sem_from_ranks(ranks, k_for_recalls)

    for metric in metric_cols:
        se_col = f"{metric}_SE"
        df_group[se_col] = df_group[model_col].map(
            lambda m, metric=metric: se_by_model[m].get(metric, np.nan)
        )

    return df_group


def _plot_metric_with_errorbars(
    df_plot, metric, se_col, out_path, candidate_group, safe_group, ranks_size
):
    x_labels = df_plot["Model"].tolist()
    hue_levels = sorted(df_plot["size (B of params)"].unique())
    n_hue = len(hue_levels)
    n_x = len(x_labels)

    group_width = 0.8
    bar_width = group_width
    x_base = np.arange(n_x)
    offsets = np.linspace(
        -group_width / 2 + bar_width / 2, group_width / 2 - bar_width / 2, n_hue
    )

    metric_safe = metric.replace("@", "at")

    # ── Plot 1: colored by model size (original) ─────────────────────────────
    palette_size = sns.color_palette("tab20", n_hue)

    fig, ax = plt.subplots(figsize=(13, 10), constrained_layout=True)
    for i, (hue_val, _offset) in enumerate(zip(hue_levels, offsets, strict=False)):
        sub = df_plot[df_plot["size (B of params)"] == hue_val]
        means = np.array(
            [
                (
                    sub.loc[sub["Model"] == m, metric].values[0]
                    if m in sub["Model"].values
                    else 0.0
                )
                for m in x_labels
            ]
        )
        ses = np.array(
            [
                (
                    sub.loc[sub["Model"] == m, se_col].values[0]
                    if m in sub["Model"].values
                    else 0.0
                )
                for m in x_labels
            ]
        )
        ax.bar(
            x_base, means, width=bar_width, color=palette_size[i], label=str(hue_val)
        )
        ax.errorbar(
            x=x_base,
            y=means,
            yerr=ses,
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=4,
            capthick=1,
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend(title="size (B of params)")
    ax.set_ylabel(metric)
    fig.suptitle(
        f"Model {metric}, colored by size, {int(ranks_size)} functions", fontsize=18
    )
    ax.set_title(f"{candidate_group} (error bars = ±1 SE)", fontsize=10)
    plt.savefig(
        f"{out_path}/{metric_safe}_by_model_and_size_{safe_group}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ── Plot 2: one color per model, tab20 by x-position ────────────────────
    palette_model = sns.color_palette("tab20", n_x)  # <-- n_x, not n_hue
    palette_model = plt.get_cmap("tab20")
    # tab20 = plt.get_cmap("tab20")
    palette_model = [
        palette_model(i / max(len(x_labels) - 1, 1)) for i in range(len(x_labels))
    ]
    # colors = [tab20(i / max(len(models) - 1, 1)) for i in range(len(models))]

    fig, ax = plt.subplots(figsize=(16, 12), constrained_layout=True)
    for i, (hue_val, _offset) in enumerate(zip(hue_levels, offsets, strict=False)):
        sub = df_plot[df_plot["size (B of params)"] == hue_val]
        means = np.array(
            [
                (
                    sub.loc[sub["Model"] == m, metric].values[0]
                    if m in sub["Model"].values
                    else 0.0
                )
                for m in x_labels
            ]
        )
        ses = np.array(
            [
                (
                    sub.loc[sub["Model"] == m, se_col].values[0]
                    if m in sub["Model"].values
                    else 0.0
                )
                for m in x_labels
            ]
        )
        # Assign color per x-position, not per hue group
        bar_colors = [palette_model[j] for j in range(n_x)]
        for j, (x_pos, mean, se, color) in enumerate(
            zip(x_base, means, ses, bar_colors, strict=False)
        ):
            ax.bar(
                x_pos,
                mean,
                width=bar_width,
                color=color,
                label=x_labels[j] if i == 0 else None,
            )  # label once
            ax.errorbar(
                x=x_pos,
                y=mean,
                yerr=se,
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=4,
                capthick=1,
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.legend(title="Model", ncol=2)
    ax.set_ylabel(metric)
    fig.suptitle(f"Model {metric}, {int(ranks_size)} functions", fontsize=18)
    ax.set_title(f"{candidate_group} (error bars = ±1 SE)", fontsize=10)
    plt.savefig(
        f"{out_path}/{metric_safe}_by_model_{safe_group}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def gen_products(metrics_df, out_path, all_ranks, k_for_recalls, model_names):

    candidate_groups = metrics_df["candidate_group"].unique()
    make_correlation_scatter(metrics_df, out_path, candidate_groups)

    # Create TWO LaTeX tables: one per candidate group
    for candidate_group in candidate_groups:
        df_group = metrics_df[metrics_df["candidate_group"] == candidate_group].copy()
        df_group.fillna(0, inplace=True)
        columns_to_drop = [
            col
            for col in df_group.columns
            if "Spearman" in col or "size" in col or "@3" in col
        ]
        columns_to_drop.append("MRR@5")
        smaller_df_group = df_group.drop(columns_to_drop, axis=1)

        df_group = _attach_metric_se_columns(
            df_group=df_group,
            candidate_group=candidate_group,
            all_ranks=all_ranks,
            k_for_recalls=k_for_recalls,
        )

        ranks_size = all_ranks[(model_names[0], candidate_group)].size
        if " and " in candidate_group.lower():
            ranks_size *= 2

        # Clean up column header name so it renders nicely in LaTeX
        df_group.rename(
            columns={
                "candidate_group": "Candidate Group",
                "model": "Model",
            },
            inplace=True,
        )

        smaller_df_group.rename(
            columns={
                "candidate_group": "Candidate Group",
                "model": "Model",
            },
            inplace=True,
        )

        # Use a safe version of the group name for filenames/labels
        safe_group = _get_safe_group(candidate_group=candidate_group)
        smaller_df_group = smaller_df_group.drop("Candidate Group", axis=1)
        df_group = df_group.drop("Candidate Group", axis=1)

        col_format = "ll" + "r" * (1 + 2 * len(k_for_recalls))
        caption = f"MRR, MRR@ k and Recall@ k for {candidate_group}."
        df_label = f"tab:metrics_{safe_group}"
        # Full metrics table
        df_group.to_latex(
            f"{out_path}/metrics_table_{safe_group}.tex",
            index=False,
            float_format="%.4f",
            column_format=col_format,
            caption=caption,
            label=df_label,
        )

        # Reduced ("smaller") metrics table
        smaller_df_group.to_latex(
            f"{out_path}/metrics_table_{safe_group}_smaller.tex",
            index=False,
            float_format="%.4f",
            column_format=col_format,
            caption=caption + " (reduced)",
            label=df_label + "_smaller",
        )

        metrics_cols = [
            col
            for col in df_group.columns
            if ("MRR" in col or "Recall" in col) and ("SE" not in col)
        ]
        for metric in metrics_cols:
            se_col = f"{metric}_SE"

            _plot_metric_with_errorbars(
                df_plot=df_group,
                metric=metric,
                se_col=se_col,
                out_path=out_path,
                candidate_group=candidate_group,
                safe_group=safe_group,
                ranks_size=ranks_size,
            )

    # Create two distinct plots (one per candidate group) of the recall@k curve,
    # each model plotted as a different color on each plot.

    for candidate_group in candidate_groups:
        plt.figure(figsize=(10, 7), constrained_layout=True)
        palette = sns.color_palette("tab20", n_colors=len(model_names))
        for color, model_name in zip(palette, model_names, strict=False):
            ranks = all_ranks[(model_name, candidate_group)]
            # ECDF gives Recall@k curve: P(rank <= n) vs n
            sns.ecdfplot(data=ranks, stat="proportion", label=model_name, color=color)

        plt.xlabel("k")
        plt.ylabel("Fraction of Dataset Pairs Recalled ≤ k")
        plt.title(f"Recall @ k for all k\n{candidate_group}")
        plt.legend(title="Model")

        safe_group = (
            candidate_group.replace(" ", "_").replace("/", "_").replace("\\", "_")
        )
        out_name = f"{out_path}/recall_curve_{safe_group}.png"
        plt.savefig(out_name, dpi=300, bbox_inches="tight")
        plt.close()


def analyze_sims(
    model_names: list[str], cosine_parquet_paths: list[str], data_input_dir: str
):

    out_path = RESULTS_DIR
    os.makedirs(out_path, exist_ok=True)

    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count == 0:
        raise ValueError("No CPUs detected")
    procs = int(cpu_count * 7 / 8)

    # k values for which we will report Recall@k and MRR@k
    k_for_recalls = K_FOR_RECALLS

    # Collect per-(model, candidate_group) metrics
    metrics_rows = []
    # Store per-(model, candidate_group) rank arrays for plotting later
    all_ranks = {}

    ds = load_from_disk(data_input_dir)
    function_lengths = get_function_lengths(dataset=ds)
    function_signature_counts = get_function_signature_count(dataset=ds)
    log_histogram(
        data=function_lengths,
        title="Function Lengths (characters) by representation",
        suptitle="",
        xlabel="Function Lengths (characters)",
        label_a="source",
        label_b="decompilation",
        filename=f"{out_path}/function_lengths_histogram.png",
    )
    log_histogram(
        data=function_signature_counts,
        title="Function Argument Counts by representation",
        suptitle="",
        xlabel="Function Arguments",
        label_a="source",
        label_b="decompilation",
        filename=f"{out_path}/function_sigs_histogram.png",
    )
    trimmed_quantile_boxplot(
        data=function_lengths,
        title="Function Lengths (characters) by representation",
        suptitle="",
        xlabel="Function Lengths (characters)",
        label_a="source",
        label_b="decompilation",
        filename=f"{out_path}/function_lengths_boxplot.png",
    )
    trimmed_quantile_boxplot(
        data=function_signature_counts,
        title="Function Argument Counts by representation",
        suptitle="",
        xlabel="Function Arguments",
        label_a="source",
        label_b="decompilation",
        filename=f"{out_path}/function_sigs_boxplot.png",
    )

    for model_name, parquet_path in zip(
        model_names, cosine_parquet_paths, strict=False
    ):
        df = pd.read_parquet(parquet_path, engine="pyarrow")

        half_length = round(len(df) / 2)

        # Rank of the matching function among all other functions
        # Searching Source -> Decompiled
        def s2d_reciprocal_rank_of_all_other_functions(
            x, df=df, half_length=half_length
        ):
            return (
                df.loc[x]
                .drop(labels=x)
                .sort_values(ascending=False)
                .index.get_loc(x + half_length)
                + 1
            )  # 1-based rank

        # Rank of the matching function among decompilation-only candidates
        # Searching Source -> Decompiled
        def s2d_reciprocal_rank_of_just_decomp(x, df=df, half_length=half_length):
            return (
                df.loc[x][half_length:]
                .sort_values(ascending=False)
                .index.get_loc(x + half_length)
                + 1  # 1-based rank
            )

        # Rank of the matching function among all other functions
        # Searching Decompiled -> Source
        def d2s_reciprocal_rank_of_all_other_functions(
            x, df=df, half_length=half_length
        ):
            return (
                df.loc[x]
                .drop(labels=x)
                .sort_values(ascending=False)
                .index.get_loc(x - half_length)
                + 1
            )  # 1-based rank

        # Rank of the matching function among source-only candidates
        # Searching Decompiled -> Source
        def d2s_reciprocal_rank_of_just_source(x, df=df, half_length=half_length):
            return (
                df.loc[x][:half_length]
                .sort_values(ascending=False)
                .index.get_loc(x - half_length)
                + 1  # 1-based rank
            )

        s2d_iters = list(range(half_length))
        d2s_iters = list(range(half_length, half_length * 2))

        s2d_ranked_all, s2d_ranked_decomp = (
            Parallel(n_jobs=procs, verbose=2)(delayed(func)(i) for i in s2d_iters)
            for func in [
                s2d_reciprocal_rank_of_all_other_functions,
                s2d_reciprocal_rank_of_just_decomp,
            ]
        )
        s2d_ranked_all, s2d_ranked_decomp = (
            np.array(s2d_ranked_all),
            np.array(s2d_ranked_decomp),
        )

        d2s_ranked_all, d2s_ranked_source = (
            Parallel(n_jobs=procs, verbose=2)(delayed(func)(i) for i in d2s_iters)
            for func in [
                d2s_reciprocal_rank_of_all_other_functions,
                d2s_reciprocal_rank_of_just_source,
            ]
        )
        d2s_ranked_all, d2s_ranked_source = (
            np.array(d2s_ranked_all),
            np.array(d2s_ranked_source),
        )

        candidates = {
            "Searching Source to Decompiled, Decompilation and Source Candidates": s2d_ranked_all,
            "Searching Source to Decompiled, Decompilation Candidates Only": s2d_ranked_decomp,
            "Searching Decompiled to Source, Decompilation and Source Candidates": d2s_ranked_all,
            "Searching Decompiled to Source, Source Candidates Only": d2s_ranked_source,
        }

        for exp_name, data_arr in candidates.items():
            # data_arr already contains 1-based ranks
            ranks = data_arr.astype(int)

            # Store ranks for plotting later
            all_ranks[(model_name, exp_name)] = ranks

            safe_group = _get_safe_group(candidate_group=exp_name)
            lengths_mask = None
            rank_lengths = len(ranks)
            if "Searching Decompiled to Source" in exp_name:
                lengths_mask = function_lengths[:rank_lengths]
                signatures_mask = function_signature_counts[:rank_lengths]
            elif "Searching Source to Decompiled" in exp_name:
                lengths_mask = function_lengths[rank_lengths:]
                signatures_mask = function_signature_counts[rank_lengths:]
            lengths_r, lengths_p = scatter_with_fit(
                ranks,
                lengths_mask,
                title=f"Ranks v. Function Length, {model_name}",
                suptitle=exp_name,
                xlabel="Function Rank (1 = Perfect)",
                ylabel="Function length (characters)",
                filename=f"{out_path}/ranks_v_length_{safe_group}_{model_name}.png",
            )
            sigs_r, sigs_p = scatter_with_fit(
                ranks,
                signatures_mask,
                title=f"Ranks v. Function Argument Count, {model_name}",
                suptitle=exp_name,
                xlabel="Function Rank (1 = Perfect)",
                ylabel="Function Argument Count",
                filename=f"{out_path}/ranks_v_sigs_{safe_group}_{model_name}.png",
            )

            # Global MRR over all ranks
            mrr_exp = MRR(ranks)

            # Recall@k and MRR@k for each requested k
            recall_vals = {k: recall_at_k(ranks, k) for k in k_for_recalls}
            mrr_at_n_vals = {
                k: MRR_at_k(ranks, k) for k in [x for x in k_for_recalls if x != 1]
            }  # We don't need to study MRR@1, it's just Recall@1

            # Accumulate metrics into a row
            row = {
                "model": model_name,
                "size (B of params)": MODEL_SIZE_MAP[model_name],
                "candidate_group": exp_name,
                "MRR": mrr_exp,
                "Spearman Correlation - Function Length": lengths_r,
                "Spearman p-value - Function Length": lengths_p,
                "Spearman Correlation - Function Argument Count": sigs_r,
                "Spearman p-value - Function Argument Count": sigs_p,
            }
            for k in k_for_recalls:
                row[f"Recall@{k}"] = recall_vals[k]
                if k != 1:  # see previous comment about MRR@1
                    row[f"MRR@{k}"] = mrr_at_n_vals[k]

            metrics_rows.append(row)

    # Build metrics DataFrame for all models and candidate groups
    metrics_df = pd.DataFrame(metrics_rows)

    # save everything
    try:
        metrics_df.to_pickle("metrics_df.pkl")
        with open("all_ranks.pkl", "wb") as file:
            pickle.dump(all_ranks, file)
    except Exception as e:
        logger.info(e)
    gen_products(metrics_df, out_path, all_ranks, k_for_recalls, model_names)
