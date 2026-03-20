from time import sleep
import re
import sys

from transformers import MODEL_NAMES_MAPPING
from add_embeddings import yaml_safe_load
from models import MODEL_CLASS_MAP
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import pathlib
from find_distances import cosine_matrix_calculation
from joblib import Parallel, delayed
import os
import itertools
import joblib
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint

memory = joblib.Memory(location = str(Path(__file__).resolve().parent / Path("joblib_cache")))

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
        raise ValueError(f"All models must have same length for paired tests, got lengths: {lengths}")
    n = next(iter(lengths.values()))
    return n

def summarize_descriptives(ap_dict):
    """Compute basic descriptive stats for each model."""
    summary = {}
    for model, ap_list in ap_dict.items():
        arr = np.asarray(ap_list)
        summary[model] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)),
            "median": float(np.median(arr)),
            "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)),
        }
    return summary


def latex_from_summary(summary,
                       filepath,
                       caption="Descriptive statistics",
                       label="tab:descriptives"):
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
    # Column order
    cols = ["mean", "std", "median", "q1", "q3"]
    col_header = " & ".join(["Model"] + [c.capitalize() for c in cols]) + r" \\"

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"  \centering")
    lines.append(r"  \begin{tabular}{lccccc}")
    lines.append(r"    \hline")
    lines.append("    " + col_header)
    lines.append(r"    \hline")

    for model, stats in summary.items():
        row_vals = [f"{stats[k]:.6f}" for k in cols]
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



def latex_table_normality(normality_results, caption="AP Normality tests (Shapiro–Wilk)",
                          label="tab:normality", float_format="{:.4g}",
                          file_path=None):
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

def plot_group_map_bars(
    group_aps_exp2: dict[str, dict[str, list[float]]],
    fig_path: str | Path,
    figsize: tuple = (14, 6),
):
    """
    Bar chart: x=algorithm group, y=MAP, hue=model.
    group_aps_exp2: model_name -> {group_name -> [per-constant APs]}
    """
    records = []
    for model_name, group_aps in group_aps_exp2.items():
        for group_name, aps in group_aps.items():
            records.append({
                "model": model_name,
                "group": group_name,
                "MAP": float(np.mean(aps)),
                "n_constants": len(aps),
            })

    plot_df = pd.DataFrame(records)
    groups = plot_df["group"].unique()
    models = plot_df["model"].unique()
    x = np.arange(len(groups))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=figsize)
    for i, model in enumerate(models):
        vals = [
            plot_df.loc[
                (plot_df["model"] == model) & (plot_df["group"] == g), "MAP"
            ].values[0]
            for g in groups
        ]
        ax.bar(x + i * width, vals, width, label=model, alpha=0.85)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MAP (mean over constants in group)")
    ax.set_title("Per-Group MAP by Model — Experiment 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_wilcoxon_pvalue_heatmap(paired_results,
                                 exp_name:str,
                                 alpha=0.05,
                                 log10_transform=True,
                                 cmap="viridis",
                                 file_path=None,
                                 annotate=True,
                                 figsize=(6, 5)):
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
    for (m1, m2) in paired_results.keys():
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
        p_mat[j, i] = p

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
                ax.text(j, i, text_val,
                        ha="center", va="center",
                        color="white" if masked_data[i, j] > np.nanmax(masked_data) / 2 else "black",
                        fontsize=8)

    title = "Pairwise Wilcoxon p-values, {exp_name}"
    if log10_transform:
        title += " (darker = more significant)"
    ax.set_title(title)

    fig.tight_layout()

    if file_path is not None:
        fig.savefig(file_path, dpi=300, bbox_inches="tight")

    return fig, ax

def latex_table_paired_wilcoxon(paired_results, caption="Pairwise Wilcoxon signed-rank tests",
                                label="tab:paired_wilcoxon", float_format="{:.4g}",
                                file_path=None):
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

        # Commented: data doesn't follow normality
        # Paired t-test (assumes normality of differences)
        # t_stat, t_p = stats.ttest_rel(x, y, nan_policy="omit")

        # Wilcoxon signed-rank test (nonparametric, paired)
        # Note: requires no zero differences or handles them specially depending on SciPy version.
        w_stat, w_p = stats.wilcoxon(x, y, zero_method="wilcox", correction=False, alternative="greater")

        results[(m1, m2)] = {
            #"paired_t_test": {
            #    "statistic": float(t_stat),
            #    "p_value": float(t_p),
            #    "significant_at_alpha": bool(t_p < alpha),
            #},
            "wilcoxon": {
                "statistic": float(w_stat),
                "p_value": float(w_p),
                "significant_at_alpha": bool(w_p < alpha),
            },
        }
    return results

def plot_overlaid_histograms(ap_dict, bins=20, density=False, alpha=0.4):
    """
    Plot overlaid histograms for each model on the same axes.
    """
    plt.figure(figsize=(8, 5))
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

def plot_overlaid_kde(ap_dict, title:str, fig_path:str | Path, bw_method="scott"):
    """
    Optional: plot overlaid kernel density estimates for smoother distribution comparison.
    """
    from scipy.stats import gaussian_kde

    plt.figure(figsize=(8, 5))
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

def plot_map_histograms(out_path, AP_dictionary, suptitle:str, extension:str,bins=50):
    plt.figure(figsize=(8, 5))

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
            alpha=0.4,          # transparency so overlaps are visible
            label=model_name,
            density=True,       # compare shapes as distributions
            edgecolor="black",
            linewidth=0.5
        )

        # Optional: vertical line at mean
        mean_val = aps.mean()
        plt.axvline(
            mean_val,
            linestyle="--",
            linewidth=1,
            label=f"{model_name} mean: {mean_val:.3f}"
        )

    plt.xlabel("MAP")
    plt.ylabel("Density")
    plt.title("MAP distributions by model")
    plt.suptitle(suptitle)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{out_path}/sig_hist_{extension}.png")

def get_embeddings_from_model(keys, local_model, batch_size, model_name):
    embedding_dictionary = {}
    for start in tqdm(range(0, len(keys), batch_size),
                    desc=f"Embedding with: {model_name}"):
        end = start + batch_size
        batch_keys = keys[start:end]
        if type(batch_keys) == str:
            batch_keys = [batch_keys] # edge case?
        batch_embeddings = local_model.embed_batch(batch_keys)

        for k, emb in zip(batch_keys, batch_embeddings):
            embedding_dictionary[k] = emb
    return embedding_dictionary

def _get_embeddings_cached(keys: list, model_name: str, batch_size: int, device: str) -> dict:
    """
    Cached wrapper around get_embeddings_from_model.
    Loads the model internally so joblib only needs to serialize
    primitive arguments (keys, model_name, batch_size, device).
    """
    local_model = None
    max_attempts = 5
    for i in range(max_attempts):
        try:
            local_model = MODEL_CLASS_MAP[model_name](device=device, name=model_name)
            break
        except Exception as e:
            sleep(5)
            print(f"retrying... attempt {i}")
            if i == max_attempts - 1:
                raise ValueError(f"Model {model_name} cannot be loaded with exception: {e}")

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
    find_ranked = lambda x: (
        df_no_algo_classes.loc[x]
        .drop(index=x)
        .sort_values(ascending=False)
        .index
    )

    all_constants = list(df_no_algo_classes.index)
    all_ranked = Parallel(n_jobs=procs, verbose=2)(
        delayed(find_ranked)(c) for c in all_constants
    )

    # Group APs by algorithm name
    group_aps: dict[str, list[float]] = {name: [] for name in sigs}
    for constant, ranked in zip(all_constants, all_ranked):
        algo_name = reversed_sigs[constant]
        relevant = sigs[algo_name]
        ap = get_AP(relevant=relevant, ranked=list(ranked))
        group_aps[algo_name].append(ap)

    return group_aps  # group -> [AP_c1, AP_c2, ...]

def group_aps_to_MAPs(group_aps: dict[str, list[float]]) -> list[float]:
    """
    Average each group's per-constant APs into one MAP value per group.
    Returns a flat list of MAPs, one per group — equal-weighted across groups.
    """
    return [float(np.mean(aps)) for aps in group_aps.values()]



def run_signsrch_exps(model_config_path: str, 
                      signsrch_path: str | None = None):

    # Takes one input arg: path to signsrch.sig file
    if len(sys.argv) < 2 and signsrch_path == None:
        print(f"Usage: {sys.argv[0]} <signsrch.sig>")
        sys.exit(1)

    out_path = Path(__file__).resolve().parent / Path("experiment_results")

    sigs = parse_signsrch(signsrch_path if signsrch_path != None else sys.argv[1])
    for name, data in sigs.items():
        split_set = set(data.split(', '))
        sigs[name] = split_set

    reversed_sigs = {}
    for key, value in sigs.items():
        for ind_value in value:
            reversed_sigs[ind_value] = key

    AP_dict_exp1, AP_dict_exp2 = {}, {}
    group_aps_exp2 = {}

    device = "cuda"
    yaml_config = yaml_safe_load(model_config_path)
    max_attempts = 5
    batch_size = 512 
    cpu_count = os.cpu_count()
    embedding_dictionary_keys = set(sigs.keys())
    for _, data in sigs.items():
        for d in data:
            embedding_dictionary_keys.add(d)

    for model_name in yaml_config["models"]:
        
        local_model = None
        for i in range(max_attempts):
            try:
                local_model = MODEL_CLASS_MAP[model_name](device=device, name=model_name)
                break
            except Exception as e:
                sleep(5)
                print(f"retrying... attempt {i}")
                if i == max_attempts - 1:
                    raise(ValueError(f"Model {model_name} cannot be loaded with exception: {e}"))

        keys = list(embedding_dictionary_keys)
        embedding_dictionary = _get_embeddings_cached(
            keys=keys,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
        )
        del(local_model)   
        
        embeddings = np.array(list(embedding_dictionary.values()))
        cosine_matrix = cosine_matrix_calculation(full_embeddings=torch.Tensor(embeddings), 
                                                  device=device, 
                                                  split_size=4096)
        df = pd.DataFrame(cosine_matrix, 
                          index=embedding_dictionary.keys(), 
                          columns=embedding_dictionary.keys())
        df.attrs["model"] = model_name
        df.attrs["embedding_size"] = embeddings.shape[1] 
        new_file_path = pathlib.Path(__file__).parent.parent.resolve() / "data" / f"signsrch_cosine_{model_name}.parquet"
        # df.to_parquet(new_file_path)

        # Experiment 1: for each algorithm, rank constants

        procs = int(cpu_count * 7 / 8)
        APs = []
        
        sorted_keys = sorted(sigs.keys())
        
        find_constants_index = lambda x: (df.loc[x]
            .drop(index=sorted_keys)
            .sort_values(ascending=False)
            .index)
        
        constants_ranked_by_algorithm = Parallel(n_jobs=procs, verbose=2)(delayed(find_constants_index)(i) for i in sorted_keys)
        
        
        for i, sig in enumerate(tqdm(sorted_keys, 
                                     desc=f"Processing signatures during Experiment 1 for {model_name}")):
            
            relevant = sigs[sig]
            ranked = list(constants_ranked_by_algorithm[i])  # ranked constant ids/strings

            AP = get_AP(relevant=relevant, ranked=ranked)
            APs.append(AP)
            
        AP_dict_exp1[model_name] = APs

        # Experiment 2: Rank constants in same "algorithm class"
        APs = []
        df_no_algo_classes = df.drop(
            columns=list(sigs.keys()), index=list(sigs.keys())
        )

        group_aps_exp2[model_name] = get_group_MAPs(
            df_no_algo_classes=df_no_algo_classes,
            sigs=sigs,
            reversed_sigs=reversed_sigs,
            procs=procs,
        )
        # Flat list of per-group MAPs — one value per group, equal weight
        AP_dict_exp2[model_name] = group_aps_to_MAPs(group_aps_exp2[model_name])


    for AP_dict, exp_name, extension_name in ([AP_dict_exp1, "Given Group name, finding relevant constants", "g_2_c"], 
                              [AP_dict_exp2, "Given Constant, Finding Constants in Same Group", "c_2_c"]):
        plot_map_histograms(out_path,AP_dict, suptitle=exp_name, extension=extension_name)
        alpha = 0.05
        descriptives = summarize_descriptives(AP_dict) 
        pprint(descriptives)
        latex_from_summary(summary=descriptives, 
                           filepath = out_path / Path(f"distributions_summary_{extension_name}.tex"),
                           caption=f"Summary Statistics for AP by model, {exp_name}",
                           label="tab:ap_statistics")


        normality = test_normality(AP_dict, alpha=alpha)
        pprint(normality)
        latex_table_normality(normality_results=normality, 
                            caption=f"Shapiro–Wilk normality tests for AP distributions, {exp_name}",
                            label="tab:ap_normality",
                            file_path = out_path / Path(f"sig_normalit_{extension_name}.tex"))
        paired_results = paired_tests(AP_dict, alpha=alpha)   # if APs are aligned
        pprint(paired_results)

        tex_paired = latex_table_paired_wilcoxon(
        paired_results,
        caption=f"Pairwise Wilcoxon signed-rank tests for AP by model, {exp_name}",
        label="tab:ap_paired_wilcoxon",
        file_path=out_path / Path(f"ap_paired_wilcoxon_{extension_name}.tex"),
        )
        # Plain p-value heatmap
        plot_wilcoxon_pvalue_heatmap(
            paired_results,
            exp_name=exp_name,
            alpha=0.05,
            log10_transform=False,
            cmap="viridis",
            file_path=out_path / Path(f"ap_wilcoxon_pvalues_{extension_name}.png"),
        )

        # -log10(p) heatmap (recommended)
        plot_wilcoxon_pvalue_heatmap(
            paired_results,
            exp_name=exp_name,
            alpha=0.05,
            log10_transform=True,
            cmap="magma",
            file_path=out_path / Path(f"ap_wilcoxon_pvalues_logp_{extension_name}.png"),
        )

        # plot_overlaid_histograms(AP_dict, bins=50, density=True) # This hasn't been useful and is ugly

        plot_overlaid_kde(AP_dict, 
                          title=f"KDE of AP Distributions by Model, {exp_name}",
                          fig_path=out_path / Path(f"overlaid_kde_{extension_name}.png"))

    plot_group_map_bars(
        group_aps_exp2,
        fig_path=out_path / "group_map_bars_c_2_c.png",
    )
    print(f"Completed Signsrch Experiments")

if __name__ == "__main__":
    run_signsrch_exps(model_config_path = "../config/experiment_config.yaml")
