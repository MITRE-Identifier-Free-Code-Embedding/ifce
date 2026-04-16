from time import sleep
import re
import sys

from processing.add_embeddings import yaml_safe_load
from modeling.models import MODEL_CLASS_MAP
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import pickle
from processing.find_distances import cosine_matrix_calculation
from joblib import Parallel, delayed
import os
import itertools
import joblib
from scipy import stats
from pathlib import Path
from visualization.signsrch_visualizations import (
    plot_group_map_bars,
    plot_wilcoxon_pvalue_heatmap,
    plot_overlaid_kde,
    plot_map_histograms,
    plot_model_comparison_bars,
)

memory = joblib.Memory(
    location=str(Path(__file__).resolve().parent / Path("joblib_cache"))
)


def parse_signsrch(filepath: str) -> dict[str, str]:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    blocks = re.split(r"#{5,}", text)
    result: dict[str, str] = {}

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        title_m = re.search(r"TITLE\s*\n(.*?)(?=\nTYPE\b)", block, re.DOTALL)
        if not title_m:
            continue
        title = " ".join(title_m.group(1).split())

        data_m = re.search(r"DATA\s*\n(.*)", block, re.DOTALL)
        if not data_m:
            continue

        raw = _clean_data(data_m.group(1))
        if raw:
            result[title] = raw

    return result


def _clean_data(data_raw: str) -> str:
    """Strip comments and suffixes, return comma-separated values."""
    # Remove block comments
    data = re.sub(r"/\*.*?\*/", " ", data_raw, flags=re.DOTALL)
    # Remove line comments
    lines = []
    for line in data.splitlines():
        lines.append(_strip_line_comment(line))
    data = "\n".join(lines)

    # Find all tokens (strings, chars, numbers)
    tokens = []
    for tok in _TOKEN_RE.finditer(data):
        raw = tok.group(0)
        if raw.startswith('"') or raw.startswith("'"):
            tokens.append(raw)
        else:
            clean = _clean_suffix(raw)
            # Normalize: parse and re-emit as uppercase hex
            try:
                if "." in clean or ("e" in clean.lower() and "0x" not in clean.lower()):
                    tokens.append(str(float(clean)))
                else:
                    val = int(clean, 0)
                    tokens.append(f"0x{val:X}")
            except ValueError:
                tokens.append(clean)

    return ", ".join(tokens)


def _strip_line_comment(line: str) -> str:
    in_str = False
    esc = False
    for i, ch in enumerate(line):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if not in_str:
            if ch in ("#", ";"):
                return line[:i]
            if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
                return line[:i]
    return line


_TOKEN_RE = re.compile(
    r"""
      "(?:[^"\\]|\\.)*"
    | '(?:[^'\\]|\\.)*'
    | [+-]?0[xX][0-9A-Fa-f]+(?:UL|ul|U|u|L|l)*
    | [+-]?\d+\.\d*(?:[eE][+-]?\d+)?[fFdD]?
    | [+-]?\d+[eE][+-]?\d+[fFdD]?
    | [+-]?\d+(?:UL|ul|U|u|L|l)*
    """,
    re.VERBOSE,
)


def _clean_suffix(raw: str) -> str:
    if "0x" in raw.lower():
        return re.sub(r"(?:UL|ul|Ul|uL|U|u|L|l)+$", "", raw)
    else:
        return re.sub(r"[UuLlFfDd]+$", "", raw)


def check_lengths_and_align(ap_dict):
    """Ensure all models have the same number of APs."""
    lengths = {k: len(v) for k, v in ap_dict.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"All models must have same length for paired tests, got lengths: {lengths}"
        )
    n = next(iter(lengths.values()))
    return n


def summarize_descriptives(ap_dict, smaller=False):
    """Compute basic descriptive stats for each model."""
    summary = {}
    for model, ap_list in ap_dict.items():
        arr = np.asarray(ap_list)
        stdev = float(np.std(arr, ddof=1))
        se = stdev / np.sqrt(arr.size)
        if not smaller:
            summary[model] = {
                "mean": float(np.mean(arr)),
                "std": stdev,
                "se": se,
                "median": float(np.median(arr)),
                "q1": float(np.percentile(arr, 25)),
                "q3": float(np.percentile(arr, 75)),
            }
        else:
            summary[model] = {
                "mean": float(np.mean(arr)),
                "se": se,
                "median": float(np.median(arr)),
            }
    return summary


def latex_from_summary(
    summary, filepath, caption="Descriptive statistics", label="tab:descriptives"
):
    """
    Generate a LaTeX table from summarize_descriptives output and save to file.

    Parameters
    ----------
    summary : dict
        Output of summarize_descriptives.
    filepath : str
        Path where the .tex file will be written.
    caption : str, optional
        Table caption.
    label : str, optional
        LaTeX label for referencing the table.
    """

    cols = list(list(summary.values())[0].keys())
    col_header = " & ".join(["Model"] + [c.capitalize() for c in cols]) + r" \\"

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{lccccc}")
    lines.append(r"    \hline")
    lines.append("    " + col_header)
    lines.append(r"    \hline")

    for model, _stats in summary.items():
        row_vals = [f"{_stats[k]:.6f}" for k in cols]
        row = " & ".join([str(model)] + row_vals) + r" \\"
        lines.append("    " + row)

    lines.append(r"    \hline")
    lines.append(r"  \end{tabular}")
    lines.append(f"  \caption{{{caption}}}")
    lines.append(f"  \label{{{label}}}")
    lines.append(r"\end{table}")

    content = "\n".join(lines) + "\n"
    with open(filepath, "w") as f:
        f.write(content)

    return content


def test_normality(ap_dict, alpha=0.05):
    """
    Test normality of each model's AP distribution using Shapiro–Wilk.
    Returns dict: model -> (statistic, p_value, is_normal_at_alpha).
    """
    results = {}
    for model, ap_list in ap_dict.items():
        arr = np.asarray(ap_list)
        # Shapiro–Wilk has an upper limit on sample size, but usually fine for AP vectors.
        stat, p = stats.shapiro(arr)
        results[model] = {
            "statistic": float(stat),
            "p_value": float(p),
            "normal_at_alpha": bool(p > alpha),
        }
    return results


def latex_table_normality(
    normality_results,
    caption="AP Normality tests (Shapiro–Wilk)",
    label="tab:normality",
    float_format="{:.4g}",
    file_path=None,
):
    """
    normality_results: dict[model] -> {"statistic": float, "p_value": float, "normal_at_alpha": bool}
    Returns LaTeX string. If file_path is not None, also writes it to that file.
    """
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\hline")
    lines.append("Model & $W$ & $p$ & Normal?\\\\")
    lines.append("\\hline")

    for model, res in normality_results.items():
        stat = float_format.format(res["statistic"])
        pval = float_format.format(res["p_value"])
        normal_flag = "Yes" if res["normal_at_alpha"] else "No"
        # Escape underscores in model names for LaTeX
        model_tex = model.replace("_", "\\_")
        lines.append(f"{model_tex} & {stat} & {pval} & {normal_flag}\\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    table_str = "\n".join(lines)

    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(table_str)

    return table_str


def latex_table_paired_wilcoxon(
    paired_results,
    caption="Pairwise Wilcoxon signed-rank tests",
    label="tab:paired_wilcoxon",
    float_format="{:.4g}",
    file_path=None,
):
    """
    paired_results: dict[(model_i, model_j)] -> {
        "wilcoxon": {
            "statistic": float,
            "p_value": float,
            "significant_at_alpha": bool
        }
    }
    Returns LaTeX table string; optionally writes to file.
    """
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lllccc}")
    lines.append("\\hline")
    lines.append("Model 1 & Model 2 & Test & Statistic & $p$ & Sig.?\\\\")
    lines.append("\\hline")

    for (m1, m2), tests in paired_results.items():
        w = tests["wilcoxon"]
        stat = float_format.format(w["statistic"])
        pval = float_format.format(w["p_value"])
        sig = "Yes" if w["significant_at_alpha"] else "No"

        m1_tex = m1.replace("_", "\\_")
        m2_tex = m2.replace("_", "\\_")

        lines.append(f"{m1_tex} & {m2_tex} & Wilcoxon & {stat} & {pval} & {sig}\\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    table_str = "\n".join(lines)

    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            f.write(table_str)

    return table_str


def paired_tests(ap_dict, alpha=0.05):
    """
    Perform pairwise paired tests between models, assuming APs are aligned per sample.
    Uses both parametric (paired t-test) and nonparametric (Wilcoxon) tests.
    Returns nested dict: (model_i, model_j) -> {test_name: {...}}.
    """
    check_lengths_and_align(ap_dict)
    models = list(ap_dict.keys())
    results = {}

    for m1, m2 in itertools.permutations(models, 2):
        x = np.asarray(ap_dict[m1])
        y = np.asarray(ap_dict[m2])

        # Wilcoxon signed-rank test (nonparametric, paired)
        # Note: requires no zero differences or handles them specially depending on SciPy version.
        w_stat, w_p = stats.wilcoxon(
            x, y, zero_method="wilcox", correction=False, alternative="greater"
        )

        results[(m1, m2)] = {
            "wilcoxon": {
                "statistic": float(w_stat),
                "p_value": float(w_p),
                "significant_at_alpha": bool(w_p < alpha),
            },
        }
    return results


def get_embeddings_from_model(keys, local_model, batch_size, model_name):
    embedding_dictionary = {}
    for start in tqdm(
        range(0, len(keys), batch_size), desc=f"Embedding with: {model_name}"
    ):
        end = start + batch_size
        batch_keys = keys[start:end]
        if isinstance(batch_keys, str):
            batch_keys = [batch_keys]  # edge case?
        batch_embeddings = local_model.embed_batch(batch_keys)

        for k, emb in zip(batch_keys, batch_embeddings):
            embedding_dictionary[k] = emb
    return embedding_dictionary


def _get_embeddings_cached(
    keys: list, model_name: str, batch_size: int, device: str
) -> dict:
    """
    Cached wrapper around get_embeddings_from_model.
    Loads the model internally so joblib only needs to serialize
    primitive arguments (keys, model_name, batch_size, device).
    """
    sleep(5)
    max_attempts = 5
    local_model = None
    for i in range(max_attempts):
        try:
            if "quant" in model_name:
                local_model = MODEL_CLASS_MAP[model_name](
                    device=device, name=model_name, quantization="fp8"
                )
            else:
                local_model = MODEL_CLASS_MAP[model_name](
                    device=device, name=model_name
                )
            break
        except Exception as e:
            print(f"retrying... attempt {i}")
            if i == max_attempts - 1:
                raise (
                    ValueError(
                        f"Model {model_name} cannot be loaded with exception: {e}"
                    )
                )
        sleep(5)
    sleep(5)

    result = get_embeddings_from_model(
        keys=keys,
        local_model=local_model,
        batch_size=batch_size,
        model_name=model_name,
    )
    del local_model
    return result


_get_embeddings_cached = memory.cache(_get_embeddings_cached)


def get_AP(relevant, ranked):

    hits = np.fromiter((c in relevant for c in ranked), dtype=np.int32)
    R = hits.sum()  # should equal len(relevant) if all relevant are in candidates

    if R == 0:
        AP = 0.0
    else:
        cum_hits = np.cumsum(hits)
        ranks = np.arange(1, len(hits) + 1)
        precision_at_k = cum_hits / ranks
        AP = (precision_at_k * hits).sum() / len(relevant)
    return AP


def get_group_MAPs(
    df_no_algo_classes: pd.DataFrame,
    sigs: dict[str, set],
    reversed_sigs: dict[str, str],
    procs: int,
) -> dict[str, list[float]]:
    """
    For each algorithm group, compute the MAP across all its constants.
    Returns a dict: group_name -> list[AP-per-constant], one entry per group.
    The caller averages to get MAP, but the list is kept for per-group stats.
    """

    def find_ranked(x, df_no_algo_classes=df_no_algo_classes):
        return (
            df_no_algo_classes.loc[x].drop(index=x).sort_values(ascending=False).index
        )

    all_constants = list(df_no_algo_classes.index)
    all_ranked = Parallel(n_jobs=procs, verbose=2)(
        delayed(find_ranked)(c) for c in all_constants
    )

    # Group APs by algorithm name
    group_aps: dict[str, list[float]] = {name: [] for name in sigs}
    for _, (constant, ranked) in enumerate(
        tqdm(zip(all_constants, all_ranked), desc="Finding group APs")
    ):
        algo_names = reversed_sigs[constant]
        for algo_name in algo_names:
            relevant = sigs[algo_name]
            relevant.remove(constant)
            ap = get_AP(relevant=relevant, ranked=list(ranked))
            group_aps[algo_name].append(ap)

    return group_aps  # group -> [AP_c1, AP_c2, ...]


def get_AP_from_indices(relevant_idx: np.ndarray, ranked_idx: np.ndarray) -> float:
    if relevant_idx.size == 0:
        return 0.0

    is_rel = np.isin(ranked_idx, relevant_idx, assume_unique=False).astype(np.int8)
    R = is_rel.sum()
    if R == 0:
        return 0.0

    cum_hits = np.cumsum(is_rel)
    ranks = np.arange(1, len(is_rel) + 1, dtype=np.float32)
    precision_at_k = cum_hits / ranks
    return float((precision_at_k * is_rel).sum() / relevant_idx.size)


def get_group_MAPs_precomputed(
    row_rank_ids: np.ndarray,
    all_constants: list[str],
    sigs: dict[str, set[str]],
    reversed_sigs: dict[str, list[str]],
) -> dict[str, list[float]]:
    const_to_id = {c: i for i, c in enumerate(all_constants)}

    # Precompute full relevant indices per algo (including the algo's own constant)
    algo_to_rel_idx_full: dict[str, np.ndarray] = {}
    for algo_name, consts in sigs.items():
        algo_to_rel_idx_full[algo_name] = np.fromiter(
            (const_to_id[c] for c in consts if c in const_to_id),
            dtype=np.int64,
        )

    group_aps: dict[str, list[float]] = {name: [] for name in sigs}

    for i, constant in enumerate(all_constants):
        ranked_ids = row_rank_ids[i]
        const_id = i  # same as const_to_id[constant]

        for algo_name in reversed_sigs[constant]:
            full_rel = algo_to_rel_idx_full[algo_name]
            # Exclude the query constant from relevant set
            rel_idx = full_rel[full_rel != const_id]

            ap = get_AP_from_indices(rel_idx, ranked_ids)
            group_aps[algo_name].append(ap)

    return group_aps


def group_aps_to_MAPs(group_aps: dict[str, list[float]]) -> list[float]:
    """
    Average each group's per-constant APs into one MAP value per group.
    Returns a flat list of MAPs, one per group — equal-weighted across groups.
    """
    return [float(np.mean(aps)) for aps in group_aps.values()]


def gen_signsrch_products(AP_dict_exp1, AP_dict_exp2, group_aps_exp2, out_path):

    for AP_dict, exp_name, extension_name, map_or_ap in (
        [AP_dict_exp1, "Given Group name, finding relevant constants", "g_2_c", "AP"],
        [
            AP_dict_exp2,
            "Given Constant, Finding Constants in Same Group",
            "c_2_c",
            "MAP",
        ],
    ):
        plot_map_histograms(
            out_path, AP_dict, suptitle=exp_name, extension=extension_name
        )
        alpha = 0.05
        descriptives = summarize_descriptives(AP_dict)
        # pprint(descriptives)
        latex_from_summary(
            summary=descriptives,
            filepath=out_path / Path(f"distributions_summary_{extension_name}.tex"),
            caption=f"Summary Statistics for {map_or_ap} by model, {exp_name}",
            label=f"tab:{map_or_ap.lower()}_statistics",
        )

        descriptives_smaller = summarize_descriptives(AP_dict, smaller=True)

        latex_from_summary(
            summary=descriptives_smaller,
            filepath=out_path
            / Path(f"distributions_summary_{extension_name}_smaller.tex"),
            caption=f"Summary Statistics for {map_or_ap} by model, {exp_name}",
            label=f"tab:{map_or_ap.lower()}_statistics",
        )

        plot_model_comparison_bars(
            descriptives,
            exp_name,
            extension_name,
            map_or_ap,
            out_path,
        )

        normality = test_normality(AP_dict, alpha=alpha)
        # pprint(normality)
        latex_table_normality(
            normality_results=normality,
            caption=f"Shapiro–Wilk normality tests for {map_or_ap} distributions, {exp_name}",
            label=f"tab:{map_or_ap.lower()}_normality",
            file_path=out_path / Path(f"sig_normalit_{extension_name}.tex"),
        )
        paired_results = paired_tests(AP_dict, alpha=alpha)  # if APs are aligned
        # pprint(paired_results)

        tex_paired = latex_table_paired_wilcoxon(
            paired_results,
            caption=f"Pairwise Wilcoxon signed-rank tests for {map_or_ap} by model, {exp_name}",
            label=f"tab:{map_or_ap.lower()}_paired_wilcoxon",
            file_path=out_path / Path(f"ap_paired_wilcoxon_{extension_name}.tex"),
        )
        # Plain p-value heatmap
        plot_wilcoxon_pvalue_heatmap(
            paired_results,
            exp_name=exp_name,
            alpha=0.05,
            log10_transform=False,
            cmap="crest",
            file_path=out_path / Path(f"ap_wilcoxon_pvalues_{extension_name}.png"),
        )

        plot_overlaid_kde(
            AP_dict,
            title=f"KDE of {map_or_ap} Distributions by Model, {exp_name}",
            fig_path=out_path / Path(f"overlaid_kde_{extension_name}.png"),
        )

    plot_group_map_bars(
        group_aps_exp2,
        fig_path=out_path / "group_map_bars_c_2_c.png",
    )


def run_signsrch_exps(
    model_config_path: str, signsrch_path: str, batch_size: int = 512
):

    out_path = Path(__file__).resolve().parent / Path("experiment_results")

    sigs = parse_signsrch(signsrch_path)

    for name, data in sigs.items():
        split_set = set(data.split(", "))
        sigs[name] = split_set

    reversed_sigs = {}
    for key, value in sigs.items():
        for ind_value in value:
            if ind_value in reversed_sigs:
                reversed_sigs[ind_value].append(key)
            else:
                reversed_sigs[ind_value] = [key]

    AP_dict_exp1, AP_dict_exp2 = {}, {}
    group_aps_exp2 = {}

    device = "cuda"
    yaml_config = yaml_safe_load(model_config_path)
    cpu_count = os.cpu_count()
    embedding_dictionary_keys = set(sigs.keys())
    for _, data in sigs.items():
        for d in data:
            embedding_dictionary_keys.add(d)

    for model_name in yaml_config["models"]:
        keys = list(embedding_dictionary_keys)
        embedding_dictionary = _get_embeddings_cached(
            keys=keys,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
        )

        embeddings = np.array(list(embedding_dictionary.values()))
        cosine_matrix = cosine_matrix_calculation(
            full_embeddings=torch.Tensor(embeddings), device=device, split_size=4096
        )
        df = pd.DataFrame(
            cosine_matrix,
            index=embedding_dictionary.keys(),
            columns=embedding_dictionary.keys(),
        )
        df.attrs["model"] = model_name
        df.attrs["embedding_size"] = embeddings.shape[1]

        # Experiment 1: for each algorithm, rank constants

        procs = int(
            cpu_count * 7 / 8
        )  # we arbitrarily select 7/8s of the available pool
        APs = []

        sorted_keys = sorted(sigs.keys())

        def find_constants_index(x, df=df, sorted_keys=sorted_keys):
            return df.loc[x].drop(index=sorted_keys).sort_values(ascending=False).index

        constants_ranked_by_algorithm = Parallel(n_jobs=procs, verbose=2)(
            delayed(find_constants_index)(i) for i in sorted_keys
        )

        for i, sig in enumerate(
            tqdm(
                sorted_keys,
                desc=f"Processing signatures during Experiment 1 for {model_name}",
            )
        ):
            relevant = sigs[sig]
            ranked = list(
                constants_ranked_by_algorithm[i]
            )  # ranked constant ids/strings

            AP = get_AP(relevant=relevant, ranked=ranked)
            APs.append(AP)

        AP_dict_exp1[model_name] = APs

        # Experiment 2: Rank constants in same "algorithm class"
        APs = []
        df_no_algo_classes = df.drop(columns=list(sigs.keys()), index=list(sigs.keys()))
        values = df_no_algo_classes.copy().to_numpy().copy()

        # Mask out diagonal so it never appears in rankings
        np.fill_diagonal(values, -np.inf)

        # argsort descending per row
        # argsort is ascending, so negate
        row_rank_ids = np.argsort(-values, axis=1, stable=True)  # shape (N, N)

        # Map row index -> constant, and constant -> row index
        all_constants = df_no_algo_classes.index.to_list()

        group_aps_exp2[model_name] = get_group_MAPs_precomputed(
            row_rank_ids, all_constants, sigs, reversed_sigs
        )

        # Flat list of per-group MAPs — one value per group, equal weight
        AP_dict_exp2[model_name] = group_aps_to_MAPs(group_aps_exp2[model_name])

    with open("group_aps_exp2.pkl", "wb") as file:
        pickle.dump(group_aps_exp2, file)
    with open("AP_dict_exps.pkl", "wb") as file:
        pickle.dump([AP_dict_exp1, AP_dict_exp2], file)

    gen_signsrch_products(AP_dict_exp1, AP_dict_exp2, group_aps_exp2, out_path)

    print("Completed Signsrch Experiments")
