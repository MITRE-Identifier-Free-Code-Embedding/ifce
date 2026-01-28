import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed
import os
import seaborn as sns
import matplotlib.pyplot as plt
from models import MODEL_CLASS_MAP
from pathlib import Path


def recall_at_n(rank_series: np.ndarray, n: int):
    # Recall@n: fraction of items whose rank is <= n
    return np.sum(rank_series <= n) / rank_series.size


def MRR(rank_series: np.ndarray):
    # Standard MRR over all items: mean of 1/rank
    reciprocal_rank_series = 1.0 / rank_series.astype(float)
    return np.mean(reciprocal_rank_series)


def MRR_at_n(rank_series: np.ndarray, n: int):
    # Conditional MRR@n: mean reciprocal rank over items with rank <= n
    mask = rank_series <= n
    if not np.any(mask):
        return np.nan
    return np.mean(1.0 / rank_series[mask].astype(float))


def analyze_sims(model_names, cosine_parquet_paths):

    sns.set(style="whitegrid")

    out_path = Path(__file__).resolve() / Path("experiment_results")

    cpu_count = os.cpu_count()
    if cpu_count is None or cpu_count == 0:
        raise ValueError("No CPUs detected")
    procs = int(cpu_count * 7 / 8)

    # n values for which we will report Recall@n and MRR@n
    n_for_recalls = [1, 3, 5, 10]

    # Collect per-(model, candidate_group) metrics
    metrics_rows = []
    # Store per-(model, candidate_group) rank arrays for plotting later
    all_ranks = {}

    model_size_dictionary = {}
    for model_name in model_names:
        # model = MODEL_CLASS_MAP[model_name](device="cpu", name=model_name)
        # model_size_dictionary[model_name] = model.parameter_count()
        import random
        model_size_dictionary[model_name] = random.randint(100,1000)
    

    for model_name, parquet_path in zip(model_names, cosine_parquet_paths):

        df = pd.read_parquet(parquet_path, engine="pyarrow")
        half_length = round(len(df) / 2)

        # Rank of the matching function among all other functions
        # Searching Source -> Decompiled
        s2d_reciprocal_rank_of_all_other_functions = (
            lambda x: df.loc[x]
            .drop(labels=x)
            .sort_values(ascending=False)
            .index.get_loc(x + half_length)
            + 1  # 1-based rank
        )

        # Rank of the matching function among decompilation-only candidates
        # Searching Source -> Decompiled
        s2d_reciprocal_rank_of_just_decomp = (
            lambda x: df.loc[x][half_length:]
            .sort_values(ascending=False)
            .index.get_loc(x + half_length)
            + 1  # 1-based rank
        )

        # Rank of the matching function among all other functions
        # Searching Decompiled -> Source
        d2s_reciprocal_rank_of_all_other_functions = (
            lambda x: df.loc[x]
            .drop(labels=x)
            .sort_values(ascending=False)
            .index.get_loc(x - half_length)
            + 1  # 1-based rank
        )

        # Rank of the matching function among source-only candidates
        # Searching Decompiled -> Source
        d2s_reciprocal_rank_of_just_source = (
            lambda x: df.loc[x][:half_length]
            .sort_values(ascending=False)
            .index.get_loc(x - half_length)
            + 1  # 1-based rank
        )

        s2d_iters = list(range(half_length))
        d2s_iters = list(range(half_length, half_length*2))

        s2d_ranked_all, s2d_ranked_decomp = (
            Parallel(n_jobs=procs, verbose=2)(delayed(func)(i) for i in s2d_iters)
            for func in [
                s2d_reciprocal_rank_of_all_other_functions,
                s2d_reciprocal_rank_of_just_decomp,
            ]
        )
        s2d_ranked_all, s2d_ranked_decomp = np.array(s2d_ranked_all), np.array(s2d_ranked_decomp)

        d2s_ranked_all, d2s_ranked_source = (
            Parallel(n_jobs=procs, verbose=2)(delayed(func)(i) for i in d2s_iters)
            for func in [
                d2s_reciprocal_rank_of_all_other_functions,
                d2s_reciprocal_rank_of_just_source,
            ]
        )
        d2s_ranked_all, d2s_ranked_source = np.array(d2s_ranked_all), np.array(d2s_ranked_source)

        candidates = {
            "Searching Source to Decompiled, Decompilation and Source Candidates": s2d_ranked_all,
            "Searching Source to Decompiled, Decompilation Candidates Only": s2d_ranked_decomp,
            "Searching Decompiled to Source, Decompilation and Source Candidates": d2s_ranked_all,
            "Searching Decompiled to Source, Source Candidates Only": d2s_ranked_source
        }

        for exp_name, data_arr in candidates.items():
            # data_arr already contains 1-based ranks
            ranks = data_arr.astype(int)

            # Store ranks for plotting later
            all_ranks[(model_name, exp_name)] = ranks

            # Global MRR over all ranks
            mrr_exp = MRR(ranks)

            # Recall@n and MRR@n for each requested n
            recall_vals = {n: recall_at_n(ranks, n) for n in n_for_recalls}
            mrr_at_n_vals = {n: MRR_at_n(ranks, n) for n in [x for x in n_for_recalls if x != 1]} # We don't need to study MRR@1, it's always 1.

            # Accumulate metrics into a row
            row = {
                "model": model_name,
                "size (B of params)": model_size_dictionary[model_name],
                "candidate_group": exp_name,
                "MRR": mrr_exp,
            }
            for n in n_for_recalls:
                row[f"Recall@{n}"] = recall_vals[n]
                if n != 1: # see previous comment about MRR@1
                    row[f"MRR@{n}"] = mrr_at_n_vals[n]

            metrics_rows.append(row)

    # Build metrics DataFrame for all models and candidate groups
    metrics_df = pd.DataFrame(metrics_rows)

    # Create TWO LaTeX tables: one per candidate group
    candidate_groups = metrics_df["candidate_group"].unique()
    for candidate_group in candidate_groups:
        df_group = metrics_df[metrics_df["candidate_group"] == candidate_group].copy()

        # Clean up column header name so it renders nicely in LaTeX
        df_group.rename(
            columns={
                "candidate_group": "Candidate Group",
                "model": "Model",
            },
            inplace=True,
        )

        # Use a safe version of the group name for filenames/labels
        safe_group = (
            candidate_group.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        df_group = df_group.drop("Candidate Group", axis=1) # For aesthetics

        # LaTeX table with caption mentioning the candidate group
        df_group.to_latex(
            f"{out_path}/metrics_table_{safe_group}.tex",
            index=False,
            float_format="%.4f",
            column_format="ll" + "r" * (1 + 2 * len(n_for_recalls)),
            caption=f"MRR, MRR@ n and Recall@ n for {candidate_group}.",
            label=f"tab:metrics_{safe_group}",
        )

        metrics_rows = [col for col in df_group.columns if "MRR" in col or "Recall" in col]
        for metric in metrics_rows:
            plt.figure()
            sns.barplot(df_group, x="Model", y=metric, hue="size (B of params)", legend=True)
            plt.suptitle(f"Model {metric}, colored by size", fontsize=18)
            plt.title(candidate_group, fontsize=10)
            plt.savefig(f"{out_path}/{metric.replace("@", "at")}_by_model_and_size_{safe_group}")
            plt.close()
        

    # Create two distinct plots (one per candidate group) of the recall@n curve,
    # each model plotted as a different color on each plot.
    

    for candidate_group in candidate_groups:
        plt.figure()
        for model_name in model_names:
            ranks = all_ranks[(model_name, candidate_group)]
            # ECDF gives Recall@n curve: P(rank <= n) vs n
            sns.ecdfplot(
                data=ranks,
                stat="proportion",
                label=model_name,
            )

        plt.xlabel("n")
        plt.ylabel("Fraction of Dataset Pairs Recalled ≤ n")
        plt.title(f"Recall @ n for all n\n{candidate_group}")
        plt.legend(title="Model")

        safe_group = (
            candidate_group.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        out_name = f"{out_path}/recall_curve_{safe_group}.png"
        plt.savefig(out_name, dpi=300, bbox_inches="tight")
        plt.close()