# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Unit tests for the ranking metrics."""

import numpy as np

from ifce.eval.assemblage_retrieval import MRR, MRR_at_k, recall_at_k


def test_recall_at_k_hand_computed():
    ranks = np.array([1, 2, 5, 11])
    # rank <= 1 -> 1/4; <= 5 -> 3/4; <= 10 -> 3/4
    assert recall_at_k(ranks, 1) == 0.25
    assert recall_at_k(ranks, 5) == 0.75
    assert recall_at_k(ranks, 10) == 0.75
    assert recall_at_k(ranks, 11) == 1.0


def test_recall_at_k_is_monotonic_in_k():
    rng = np.random.default_rng(0)
    ranks = rng.integers(1, 50, size=100)
    values = [recall_at_k(ranks, k) for k in range(1, 50)]
    assert all(b >= a for a, b in zip(values, values[1:], strict=False))


def test_mrr_hand_computed():
    ranks = np.array([1, 2, 4])
    assert np.isclose(MRR(ranks), (1.0 + 0.5 + 0.25) / 3)


def test_mrr_perfect_ranking_is_one():
    assert MRR(np.ones(10)) == 1.0


def test_mrr_at_k_truncates_misses():
    ranks = np.array([1, 2, 20])
    # only ranks <= 5 contribute: (1 + 0.5 + 0) / 3
    assert np.isclose(MRR_at_k(ranks, 5), (1.0 + 0.5 + 0.0) / 3)


def test_mrr_at_k_all_misses_is_nan():
    assert np.isnan(MRR_at_k(np.array([10, 20]), 5))
