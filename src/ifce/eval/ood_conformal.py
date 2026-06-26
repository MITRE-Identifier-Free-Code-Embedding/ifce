# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
import math

import numpy as np
import pandas as pd

from ifce.eval.ood_common import compute_knn_nonconformity, partner_index
from ifce.processing.find_distances import find_distances


def _conformal_threshold(
    calib_scores: np.ndarray,
    epsilon: float,
) -> float:
    """
    Compute inductive conformal threshold at significance level epsilon.

    Let calib_scores = {alpha_1, ..., alpha_n} from ID calibration points.
    Sort them: s_(1) <= ... <= s_(n).
    Use index j = ceil((n + 1) * (1 - epsilon)) (1-based), threshold = s_(j).
    """
    if not (0 < epsilon < 1):
        raise ValueError("epsilon must be in (0,1)")
    sorted_scores = np.sort(calib_scores)
    n = len(sorted_scores)
    if n == 0:
        raise ValueError("Calibration scores array is empty")
    j = int(math.ceil((n + 1) * (1.0 - epsilon))) - 1  # convert to 0-based
    j = max(0, min(j, n - 1))
    return sorted_scores[j]


def _wilson_interval(
    num_success: int,
    num_total: int,
    alpha: float = 0.05,
) -> (float, float):
    """
    Wilson score confidence interval for a binomial proportion p.
    """
    if num_total == 0:
        return (0.0, 1.0)
    # 1.96 ≈ z_{0.975} for 95% CI
    z = 1.96 if abs(alpha - 0.05) < 1e-9 else 1.96
    phat = num_success / num_total
    denom = 1.0 + z**2 / num_total
    center = (phat + z**2 / (2.0 * num_total)) / denom
    half_width = (
        z * math.sqrt(phat * (1.0 - phat) / num_total + z**2 / (4.0 * num_total**2))
    ) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def build_pair_based_splits(
    num_ids: int,
    train_frac: float = 0.5,
    ood_val_frac: float = 0.5,
    random_state: int = 0,
):
    """
    Construct splits based on pair presence/absence in the training set.

    Steps:
      1. Randomly choose train_ids (used as kNN reference set).
      2. All remaining ids are candidate_ids (for calibration and testing).
      3. For each candidate id i, check if its pair partner_index(i) is in train_ids:
         - If yes -> ID candidate (its opposite representation is present in the dataset).
         - If no  -> OOD candidate (its opposite representation is not present).
      4. Split ID candidates into:
         - ID_calib: for conformal calibration
         - ID_test: for estimating empirical ID FPR
      5. Split OOD candidates into:
         - OOD_val: for choosing k (hyperparameter selection)
         - OOD_test: for final evaluation of P(detect | OOD)

    Returns a dict with numpy arrays of indices.
    """
    if not (0.0 < train_frac < 1.0):
        raise ValueError("train_frac must be in (0,1)")
    rng = np.random.default_rng(seed=random_state)

    all_ids = np.arange(num_ids)
    rng.shuffle(all_ids)

    n_train = int(round(train_frac * num_ids))
    if n_train <= 0 or n_train >= num_ids:
        raise ValueError("train_frac yields empty train or candidate set")

    train_ids = np.sort(all_ids[:n_train])
    candidate_ids = np.sort(all_ids[n_train:])

    train_set = set(train_ids)

    # Label candidate ids based on whether their opposite representation is in the train set
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

    # Split ID candidates into calibration and ID test (50/50 by default)
    rng.shuffle(id_candidates)
    n_calib = max(1, len(id_candidates) // 2)
    ID_calib = np.sort(id_candidates[:n_calib])
    ID_test = (
        np.sort(id_candidates[n_calib:])
        if len(id_candidates) > n_calib
        else np.array([], dtype=int)
    )

    # Split OOD candidates into OOD_val and OOD_test
    rng.shuffle(ood_candidates)
    if not (0.0 < ood_val_frac < 1.0):
        raise ValueError("ood_val_frac must be in (0,1)")

    n_val_ood = int(round(ood_val_frac * len(ood_candidates)))
    n_val_ood = max(
        1, min(n_val_ood, len(ood_candidates) - 1)
    )  # ensure both splits non-empty
    OOD_val = np.sort(ood_candidates[:n_val_ood])
    OOD_test = np.sort(ood_candidates[n_val_ood:])

    return {
        "ID_train": train_ids,
        "ID_calib": ID_calib,
        "ID_test": ID_test,
        "OOD_val": OOD_val,
        "OOD_test": OOD_test,
    }


def knn_conformal_ood_with_pairs(
    cosine_parquet_path: str,
    k_list,
    epsilon: float = 0.05,
    train_frac: float = 0.5,
    ood_val_frac: float = 0.5,
    random_state: int = 0,
):
    """
    Run inductive conformal kNN anomaly detection on one cosine matrix,
    using the pairing structure to define ID vs OOD.

    An id i is:
      - ID if its opposite representation partner_index(i) is in the train set.
      - OOD if its opposite representation is not in the train set.

    The procedure:
      1. Build splits via build_pair_based_splits.
      2. For each k in k_list:
         a. Compute kNN nonconformity scores for ID_calib against ID_train.
         b. Compute conformal threshold tau(k) at level epsilon.
         c. Compute TPR on OOD_val (fraction of OOD_val flagged as anomalous).
      3. Choose best_k that maximizes OOD_val TPR (ties broken in favor of smaller k).
      4. With best_k:
         a. Recompute tau(best_k) from ID_calib.
         b. Evaluate on OOD_test to estimate P(detect | OOD) with Wilson CI.
         c. Optionally, compute empirical FPR on ID_test.

    Returns
    -------
    dict with keys:
      - "best_k"
      - "epsilon"
      - "split_sizes"
      - "val_results" (per k)
      - "test_results"
    """
    cosine_df = pd.read_parquet(cosine_parquet_path)
    cosine_matrix = cosine_df.values
    num_ids = cosine_matrix.shape[0]

    if num_ids % 2 != 0:
        raise ValueError("Number of ids must be even to form source/decomp pairs")

    # Build splits based on pair presence
    splits = build_pair_based_splits(
        num_ids=num_ids,
        train_frac=train_frac,
        ood_val_frac=ood_val_frac,
        random_state=random_state,
    )

    ID_train = splits["ID_train"]
    ID_calib = splits["ID_calib"]
    ID_test = splits["ID_test"]
    OOD_val = splits["OOD_val"]
    OOD_test = splits["OOD_test"]

    # Ensure k is feasible
    max_k = max(k_list)
    if len(ID_train) <= max_k:
        raise ValueError(
            f"Largest k={max_k} is >= number of ID_train points={len(ID_train)}; "
            "increase train_frac or reduce k_list."
        )

    val_results = {}

    # Hyperparameter search over k
    for k in sorted(k_list):
        # Nonconformity scores for calibration ID points
        calib_scores = compute_knn_nonconformity(
            cosine_matrix=cosine_matrix,
            idx_points=ID_calib,
            idx_reference=ID_train,
            k=k,
        )
        tau = _conformal_threshold(calib_scores, epsilon=epsilon)

        # Nonconformity scores for OOD validation points
        ood_val_scores = compute_knn_nonconformity(
            cosine_matrix=cosine_matrix,
            idx_points=OOD_val,
            idx_reference=ID_train,
            k=k,
        )
        detected_val = ood_val_scores > tau
        tpr_ood_val = detected_val.mean() if len(detected_val) > 0 else np.nan

        val_results[k] = {
            "tpr_ood_val": float(tpr_ood_val),
            "num_ood_val": int(len(OOD_val)),
            "threshold": float(tau),
        }

    # Pick k with highest OOD validation TPR; break ties by smaller k
    k_sorted = sorted(k_list)
    best_k = max(k_sorted, key=lambda kk: val_results[kk]["tpr_ood_val"])

    # With best_k, recompute calibration scores and threshold
    best_calib_scores = compute_knn_nonconformity(
        cosine_matrix=cosine_matrix,
        idx_points=ID_calib,
        idx_reference=ID_train,
        k=best_k,
    )
    best_tau = _conformal_threshold(best_calib_scores, epsilon=epsilon)

    # Evaluate on OOD test
    ood_test_scores = compute_knn_nonconformity(
        cosine_matrix=cosine_matrix,
        idx_points=OOD_test,
        idx_reference=ID_train,
        k=best_k,
    )
    detected_ood_test = ood_test_scores > best_tau
    num_ood_test = len(OOD_test)
    num_detected_ood_test = int(detected_ood_test.sum())
    tpr_ood_test = (
        num_detected_ood_test / num_ood_test if num_ood_test > 0 else float("nan")
    )
    ci_low, ci_high = _wilson_interval(
        num_success=num_detected_ood_test,
        num_total=num_ood_test,
        alpha=0.05,
    )

    # Empirical FPR on held-out ID test set (optional)
    if len(ID_test) > 0:
        id_test_scores = compute_knn_nonconformity(
            cosine_matrix=cosine_matrix,
            idx_points=ID_test,
            idx_reference=ID_train,
            k=best_k,
        )
        id_test_flags = id_test_scores > best_tau
        empirical_fpr_id_test = id_test_flags.mean()
    else:
        empirical_fpr_id_test = float("nan")

    test_results = {
        "tpr_ood_test": float(tpr_ood_test),
        "tpr_ci_95": (float(ci_low), float(ci_high)),
        "num_ood_test": int(num_ood_test),
        "num_detected_ood_test": int(num_detected_ood_test),
        "empirical_fpr_id_test": float(empirical_fpr_id_test),
        "num_id_test": int(len(ID_test)),
        "threshold": float(best_tau),
    }

    results = {
        "best_k": int(best_k),
        "epsilon": float(epsilon),
        "split_sizes": {
            "num_ids": int(num_ids),
            "num_ID_train": int(len(ID_train)),
            "num_ID_calib": int(len(ID_calib)),
            "num_ID_test": int(len(ID_test)),
            "num_OOD_val": int(len(OOD_val)),
            "num_OOD_test": int(len(OOD_test)),
        },
        "val_results": val_results,
        "test_results": test_results,
    }
    return results


def main(
    model_list=None,
    similarity_files=None,
):
    if model_list is None:
        model_list = ["Qwen3-Embedding-0.6B"]
    if similarity_files is None:
        similarity_files = ["../data/cosine_Qwen3-Embedding-0.6B.parquet"]

    if not model_list:
        # 1) Build cosine similarity parquet files (per model)
        model_list, similarity_files = find_distances(
            data_input_dir="path/to/hf_dataset",
            model_config_path="path/to/models.yaml",
            save_euclidean=False,
        )

    # 2) Inductive conformal kNN OOD detection per model, using pair structure
    k_list = [1, 3, 5, 10, 20, 50]
    epsilon = 0.05  # target ID false positive rate
    train_frac = 0.5  # fraction of ids used as kNN reference set
    ood_val_frac = 0.5  # fraction of OOD candidates used for validation vs test

    for model, cosine_path in zip(model_list, similarity_files, strict=False):
        results = knn_conformal_ood_with_pairs(
            cosine_parquet_path=str(cosine_path),
            k_list=k_list,
            epsilon=epsilon,
            train_frac=train_frac,
            ood_val_frac=ood_val_frac,
            random_state=42,
        )
        print(f"Model: {model}")
        print("Split sizes:", results["split_sizes"])
        print("Best k:", results["best_k"])
        print("Validation TPR per k:")
        for k in sorted(k_list):
            vr = results["val_results"][k]
            print(
                f"  k={k}: TPR_OOD_val={vr['tpr_ood_val']:.3f}, num_OOD_val={vr['num_ood_val']}"
            )
        tr = results["test_results"]
        print(f"Test OOD detection probability P(detect|OOD)={tr['tpr_ood_test']:.3f}")
        print(f"95% CI (Wilson): {tr['tpr_ci_95']}")
        print(
            f"Empirical ID FPR on ID_test={tr['empirical_fpr_id_test']:.3f} over {tr['num_id_test']} points"
        )
        print(f"Chosen threshold tau={tr['threshold']:.4f}")


if __name__ == "__main__":
    main()
