from pathlib import Path

from find_distances import find_distances
import numpy as np
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score,
)


def partner_index(i: int, num_ids: int) -> int:
    """
    Given id i in [0, num_ids), return the id of its opposite representation,
    using the rule:
        if there are 100 total pairs -> 200 ids,
        pairs are (0,100), (1,101), (2,102), ...
    """
    half = num_ids // 2
    if i < half:
        return i + half
    else:
        return i - half


def _compute_knn_nonconformity(
    cosine_matrix: np.ndarray,
    idx_points: np.ndarray,
    idx_reference: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Nonconformity score for each point i in idx_points:
      alpha_i = average distance to its k nearest neighbors among idx_reference.
    Distance is 1 - cosine similarity.
    """
    if len(idx_reference) < k:
        raise ValueError(
            f"k={k} is larger than number of reference points={len(idx_reference)}"
        )

    # Submatrix of cosine sims between points and reference set
    sims = cosine_matrix[np.ix_(idx_points, idx_reference)]
    dists = 1.0 - sims  # convert cosine similarity to distance

    # kth-order statistic via partition: first k entries are the k smallest distances
    knn_dists = np.partition(dists, kth=k - 1, axis=1)[:, :k]
    scores = knn_dists.mean(axis=1)
    return scores


def build_pair_based_splits_for_f1(
    num_ids: int,
    train_frac: float = 0.5,
    val_frac_id: float = 0.5,
    val_frac_ood: float = 0.5,
    random_state: int = 0,
):
    """
    Construct splits based on pair presence/absence in the train set, for F1 optimization.

    Steps:
      1. Randomly choose train_ids (used as kNN reference set).
      2. Remaining ids are candidate_ids.
      3. Candidate ids are labeled:
           - ID candidate if their pair is in train_ids.
           - OOD candidate if their pair is NOT in train_ids.
      4. Split ID candidates into ID_val and ID_test.
      5. Split OOD candidates into OOD_val and OOD_test.

    ID_val + OOD_val are used to choose k and threshold to maximize F1.
    ID_test + OOD_test are used to evaluate the chosen (k, threshold).
    """
    if num_ids % 2 != 0:
        raise ValueError("Number of ids must be even to form pairs")
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0,1)")
    if not (0.0 < val_frac_id < 1.0):
        raise ValueError("val_frac_id must be in (0,1)")
    if not (0.0 < val_frac_ood < 1.0):
        raise ValueError("val_frac_ood must be in (0,1)")

    rng = np.random.default_rng(seed=random_state)
    all_ids = np.arange(num_ids)
    rng.shuffle(all_ids)

    n_train = int(round(train_frac * num_ids))
    if n_train <= 0 or n_train >= num_ids:
        raise ValueError("train_frac yields empty train or candidate set")

    train_ids = np.sort(all_ids[:n_train])
    candidate_ids = np.sort(all_ids[n_train:])
    train_set = set(train_ids.tolist())

    id_candidates = []
    ood_candidates = []
    for i in candidate_ids:
        j = partner_index(i, num_ids)
        if j in train_set:
            id_candidates.append(i)
        else:
            ood_candidates.append(i)

    id_candidates = np.array(id_candidates, dtype=int)
    ood_candidates = np.array(ood_candidates, dtype=int)

    if len(id_candidates) == 0:
        raise ValueError(
            "No ID candidates: no candidate id has its pair in the train set."
        )
    if len(ood_candidates) == 0:
        raise ValueError(
            "No OOD candidates: all candidate ids have their pair in the train set."
        )

    # Split ID candidates
    rng.shuffle(id_candidates)
    n_val_id = int(round(val_frac_id * len(id_candidates)))
    n_val_id = max(
        1, min(n_val_id, len(id_candidates) - 1)
    )  # keep both splits non-empty
    ID_val = np.sort(id_candidates[:n_val_id])
    ID_test = np.sort(id_candidates[n_val_id:])

    # Split OOD candidates
    rng.shuffle(ood_candidates)
    n_val_ood = int(round(val_frac_ood * len(ood_candidates)))
    n_val_ood = max(1, min(n_val_ood, len(ood_candidates) - 1))
    OOD_val = np.sort(ood_candidates[:n_val_ood])
    OOD_test = np.sort(ood_candidates[n_val_ood:])

    return {
        "ID_train": train_ids,
        "ID_val": ID_val,
        "ID_test": ID_test,
        "OOD_val": OOD_val,
        "OOD_test": OOD_test,
    }


def _precision_recall_f1_from_scores(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    threshold: float,
):
    """
    Treat OOD as positive class (y=1), ID as negative (y=0).
    Predict positive if score > threshold.
    """
    # ID: negatives
    y_id = np.zeros_like(id_scores, dtype=int)
    pred_id = (id_scores > threshold).astype(int)

    # OOD: positives
    y_ood = np.ones_like(ood_scores, dtype=int)
    pred_ood = (ood_scores > threshold).astype(int)

    y_true = np.concatenate([y_id, y_ood])
    y_pred = np.concatenate([pred_id, pred_ood])

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, tp, fp, fn


def _bootstrap_f1_ci(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    threshold: float,
    num_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 0,
):
    """
    Bootstrap CI for F1 by resampling test points with replacement.
    """
    rng = np.random.default_rng(seed=random_state)
    n_id = len(id_scores)
    n_ood = len(ood_scores)
    if n_id + n_ood == 0:
        return (0.0, 1.0)

    f1_samples = []
    for _ in range(num_bootstrap):
        if n_id > 0:
            idx_id = rng.integers(0, n_id, size=n_id)
            bs_id_scores = id_scores[idx_id]
        else:
            bs_id_scores = np.array([], dtype=float)
        if n_ood > 0:
            idx_ood = rng.integers(0, n_ood, size=n_ood)
            bs_ood_scores = ood_scores[idx_ood]
        else:
            bs_ood_scores = np.array([], dtype=float)

        _, _, f1_bs, _, _, _ = _precision_recall_f1_from_scores(
            bs_id_scores, bs_ood_scores, threshold
        )
        f1_samples.append(f1_bs)

    f1_samples = np.sort(np.array(f1_samples))
    lower = f1_samples[int((alpha / 2) * num_bootstrap)]
    upper = f1_samples[int((1 - alpha / 2) * num_bootstrap)]
    return float(lower), float(upper)


def knn_ood_f1_with_pairs(
    cosine_parquet_path: str,
    k_list,
    train_frac: float = 0.5,
    val_frac_id: float = 0.5,
    val_frac_ood: float = 0.5,
    random_state: int = 0,
    num_bootstrap: int = 1000,
):
    """
    Run kNN-based OOD detection using pair structure, optimizing F1 on validation.

    Positive class: OOD (samples whose opposite representation is NOT in train set).
    Negative class: ID (samples whose opposite representation IS in train set).

    Splitting:
      - ID_train: reference set for kNN.
      - ID_val, OOD_val: used to optimize k and threshold for F1.
      - ID_test, OOD_test: used to evaluate F1 of selected (k, threshold).

    Returns
    -------
    dict with keys:
      - "best_k"
      - "best_threshold"
      - "split_sizes"
      - "val_results": per k, best F1 and threshold
      - "test_results": metrics, bootstrap CI for F1, ROC/PR data
    """
    cosine_df = pd.read_parquet(cosine_parquet_path)
    cosine_matrix = cosine_df.values
    num_ids = cosine_matrix.shape[0]
    cosine_matrix = np.clip(
        cosine_matrix, a_max=np.max(np.nan_to_num(cosine_matrix, posinf=0)), a_min=0
    )  # values should never exceed 1.

    splits = build_pair_based_splits_for_f1(
        num_ids=num_ids,
        train_frac=train_frac,
        val_frac_id=val_frac_id,
        val_frac_ood=val_frac_ood,
        random_state=random_state,
    )

    ID_train = splits["ID_train"]
    ID_val = splits["ID_val"]
    ID_test = splits["ID_test"]
    OOD_val = splits["OOD_val"]
    OOD_test = splits["OOD_test"]

    max_k = max(k_list)
    if len(ID_train) <= max_k:
        raise ValueError(
            f"Largest k={max_k} is >= number of ID_train points={len(ID_train)}; "
            "increase train_frac or reduce k_list."
        )

    val_results = {}
    best_global_f1 = -1.0
    best_k = None
    best_threshold = None

    # For each k, compute scores on validation splits and optimize threshold
    for k in sorted(k_list):
        id_val_scores = _compute_knn_nonconformity(
            cosine_matrix=cosine_matrix,
            idx_points=ID_val,
            idx_reference=ID_train,
            k=k,
        )
        ood_val_scores = _compute_knn_nonconformity(
            cosine_matrix=cosine_matrix,
            idx_points=OOD_val,
            idx_reference=ID_train,
            k=k,
        )

        # Candidate thresholds: all unique scores from val sets
        all_scores = np.concatenate([id_val_scores, ood_val_scores])
        unique_scores = np.unique(all_scores)
        if len(unique_scores) == 0:
            continue

        best_f1_k = -1.0
        best_tau_k = None
        best_prec_k = 0.0
        best_rec_k = 0.0

        for tau in unique_scores:
            prec, rec, f1, _, _, _ = _precision_recall_f1_from_scores(
                id_val_scores, ood_val_scores, tau
            )
            if f1 > best_f1_k or (
                f1 == best_f1_k and (best_tau_k is None or tau < best_tau_k)
            ):
                best_f1_k = f1
                best_tau_k = tau
                best_prec_k = prec
                best_rec_k = rec

        val_results[k] = {
            "best_F1_val": float(best_f1_k),
            "best_precision_val": float(best_prec_k),
            "best_recall_val": float(best_rec_k),
            "best_threshold_val": float(best_tau_k),
            "num_ID_val": int(len(ID_val)),
            "num_OOD_val": int(len(OOD_val)),
        }

        # Track global best across k (break ties with smaller k)
        if best_f1_k > best_global_f1 or (
            math.isclose(best_f1_k, best_global_f1) and (best_k is None or k < best_k)
        ):
            best_global_f1 = best_f1_k
            best_k = k
            best_threshold = best_tau_k

    # Evaluate best (k, threshold) on test
    if len(ID_test) > 0:
        id_test_scores = _compute_knn_nonconformity(
            cosine_matrix=cosine_matrix,
            idx_points=ID_test,
            idx_reference=ID_train,
            k=best_k,
        )
    else:
        id_test_scores = np.array([], dtype=float)

    if len(OOD_test) > 0:
        ood_test_scores = _compute_knn_nonconformity(
            cosine_matrix=cosine_matrix,
            idx_points=OOD_test,
            idx_reference=ID_train,
            k=best_k,
        )
    else:
        ood_test_scores = np.array([], dtype=float)

    precision_test, recall_test, f1_test, tp, fp, fn = _precision_recall_f1_from_scores(
        id_test_scores, ood_test_scores, best_threshold
    )

    f1_ci_low, f1_ci_high = _bootstrap_f1_ci(
        id_scores=id_test_scores,
        ood_scores=ood_test_scores,
        threshold=best_threshold,
        num_bootstrap=num_bootstrap,
        alpha=0.05,
        random_state=random_state,
    )

    # Build ROC / PR data on the test split (using scores and true labels)
    y_true_test = np.concatenate(
        [
            np.zeros(len(id_test_scores), dtype=int),
            np.ones(len(ood_test_scores), dtype=int),
        ]
    )
    scores_test = np.concatenate([id_test_scores, ood_test_scores])

    if len(np.unique(y_true_test)) < 2:
        # Degenerate case: only one class present
        fpr, tpr, roc_thresh = np.array([0.0]), np.array([0.0]), np.array([np.inf])
        precision_curve, recall_curve, pr_thresh = (
            np.array([1.0]),
            np.array([0.0]),
            np.array([np.inf]),
        )
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        fpr, tpr, roc_thresh = roc_curve(y_true_test, scores_test, pos_label=1)
        precision_curve, recall_curve, pr_thresh = precision_recall_curve(
            y_true_test, scores_test, pos_label=1
        )
        roc_auc = auc(fpr, tpr)
        pr_auc = average_precision_score(y_true_test, scores_test)

    test_results = {
        "precision_test": float(precision_test),
        "recall_test": float(recall_test),
        "F1_test": float(f1_test),
        "F1_ci_95": (float(f1_ci_low), float(f1_ci_high)),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "num_ID_test": int(len(ID_test)),
        "num_OOD_test": int(len(OOD_test)),
        "y_true_test": y_true_test,
        "scores_test": scores_test,
        "roc_curve": {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": roc_thresh,
            "auc": float(roc_auc),
        },
        "pr_curve": {
            "precision": precision_curve,
            "recall": recall_curve,
            "thresholds": pr_thresh,
            "auc": float(pr_auc),
        },
    }

    results = {
        "best_k": int(best_k),
        "best_threshold": float(best_threshold),
        "split_sizes": {
            "num_ids": int(num_ids),
            "num_ID_train": int(len(ID_train)),
            "num_ID_val": int(len(ID_val)),
            "num_ID_test": int(len(ID_test)),
            "num_OOD_val": int(len(OOD_val)),
            "num_OOD_test": int(len(OOD_test)),
        },
        "val_results": val_results,
        "test_results": test_results,
    }
    return results


def plot_roc_pr_curves(
    results_by_model: dict, title_suffix: str = "", save_path: str = None
):
    """
    results_by_model: dict mapping model_name -> results dict from knn_ood_f1_with_pairs.

    Produces a single figure with:
      - Left: ROC curves (FPR vs TPR) for each model.
      - Right: PR curves (Recall vs Precision) for each model.
    Each model is a separate colored line, labeled by model name.
    """
    plt.figure(figsize=(12, 5))
    ax_roc = plt.subplot(1, 2, 1)
    ax_pr = plt.subplot(1, 2, 2)

    for model_name, res in results_by_model.items():
        tr = res["test_results"]
        roc = tr["roc_curve"]
        pr = tr["pr_curve"]

        fpr = roc["fpr"]
        tpr = roc["tpr"]
        roc_auc = roc["auc"]

        precision = pr["precision"]
        recall = pr["recall"]
        pr_auc = pr["auc"]

        ax_roc.plot(
            fpr,
            tpr,
            label=f"{model_name} (AUC={roc_auc:.3f})",
        )
        ax_pr.plot(
            recall,
            precision,
            label=f"{model_name} (AP={pr_auc:.3f})",
        )

    # ROC plot configuration
    ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curves{title_suffix}")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, linestyle=":", alpha=0.6)

    # PR plot configuration
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision-Recall Curves{title_suffix}")
    ax_pr.legend(loc="lower left")
    ax_pr.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return ax_roc, ax_pr


def make_latex_summary_table(
    results_by_model: dict,
    caption: str = "OOD Detection Summary",
    label: str = "tab:ood_summary",
):
    """
    Build a LaTeX table summarizing key metrics per model.

    Columns:
      Model, k*, tau*, F1_test (with 95% CI), Precision_test, Recall_test, ROC AUC, PR AUC.
    """
    lines = []
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lrrrrrrr}")
    lines.append("\\hline")
    lines.append(
        "Model & $k^*$ & $\\tau^*$ & F1$_{\\text{test}}$ & Prec$_{\\text{test}}$ & Rec$_{\\text{test}}$ & AUC$_{\\text{ROC}}$ & AUC$_{\\text{PR}}$\\\\"
    )
    lines.append("\\hline")

    for model_name, res in results_by_model.items():
        best_k = res["best_k"]
        tau = res["best_threshold"]
        tr = res["test_results"]

        f1 = tr["F1_test"]
        f1_ci_low, f1_ci_high = tr["F1_ci_95"]
        prec = tr["precision_test"]
        rec = tr["recall_test"]
        roc_auc = tr["roc_curve"]["auc"]
        pr_auc = tr["pr_curve"]["auc"]

        # Format: F1 [CI_low, CI_high]
        f1_with_ci = f"{f1:.3f} [${f1_ci_low:.3f}$, ${f1_ci_high:.3f}$]"

        line = (
            f"{model_name} & "
            f"{best_k:d} & "
            f"{tau:.8f} & "
            f"{f1_with_ci} & "
            f"{prec:.3f} & "
            f"{rec:.3f} & "
            f"{roc_auc:.3f} & "
            f"{pr_auc:.3f}\\\\"
        )
        lines.append(line)

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def get_stats_for_f1(
    model_list=["Qwen3-Embedding-0.6B", "bge-small-en-v1.5"],
    similarity_files=[
        "../data/cosine_Qwen3-Embedding-0.6B.parquet",
        "../data/cosine_bge-small-en-v1.5.parquet",
    ],
):

    out_path = Path(__file__).resolve().parent / Path("experiment_results")
    os.makedirs(out_path, exist_ok=True)

    if not model_list:
        # 1) Build cosine similarity parquet files (per model)
        model_list, similarity_files = find_distances(
            data_input_dir="path/to/hf_dataset",
            model_config_path="path/to/models.yaml",
        )

    # 2) Run F1-optimized kNN OOD detection for each model
    k_list = [1, 3, 5, 10, 20, 50]

    for distance_type in ["Cosine", "Euclidean"]:
        results_by_model = {}

        for model, distance_file_path in zip(model_list, similarity_files):
            if distance_type.lower() not in distance_file_path:
                continue

            res = knn_ood_f1_with_pairs(
                cosine_parquet_path=str(distance_file_path),
                k_list=k_list,
                train_frac=0.5,
                val_frac_id=0.5,
                val_frac_ood=0.5,
                random_state=42,
                num_bootstrap=1000,
            )
            results_by_model[model] = res
            tr = res["test_results"]
            print(
                f"{model}: k*={res['best_k']}, tau*={res['best_threshold']:.8f}, "
                f"F1_test={tr['F1_test']:.3f}, Prec={tr['precision_test']:.3f}, "
                f"Rec={tr['recall_test']:.3f}"
            )

        # 3) Plot ROC / PR curves across models
        plot_roc_pr_curves(
            results_by_model,
            title_suffix=f" (Test Split) - {distance_type} Distance",
            save_path=f"{out_path}/roc_pr_models_{distance_type.lower()}.png",
        )

        # 4) Generate LaTeX table summary
        latex_table = make_latex_summary_table(results_by_model)
        latex_file_path = f"{out_path}/f1_stats_models_{distance_type.lower()}.tex"
        with open(latex_file_path, "w") as f:
            f.write(latex_table)

    print("Calibrated OOD detection")


if __name__ == "__main__":
    get_stats_for_f1()
