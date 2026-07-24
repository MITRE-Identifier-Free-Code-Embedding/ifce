# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Shared helpers for the pair-based out-of-distribution (OOD) evaluations.

These were previously duplicated verbatim across ``ood_conformal`` and
``ood_f1_eval``.
"""

from __future__ import annotations

import numpy as np


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


def compute_knn_nonconformity(
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
