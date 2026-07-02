# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Unit tests for the shared OOD helpers."""

import numpy as np
import pytest

from ifce.eval.ood_common import compute_knn_nonconformity, partner_index


@pytest.mark.parametrize("num_ids", [2, 4, 10, 200])
def test_partner_index_is_its_own_inverse(num_ids):
    for i in range(num_ids):
        j = partner_index(i, num_ids)
        assert 0 <= j < num_ids
        assert partner_index(j, num_ids) == i
        assert j != i  # a point is never its own partner


def test_partner_index_pairs_first_and_second_half():
    # ids 0..3 -> pairs (0,2),(1,3) for num_ids=4
    assert partner_index(0, 4) == 2
    assert partner_index(1, 4) == 3
    assert partner_index(2, 4) == 0


def test_knn_nonconformity_identical_points_have_zero_distance():
    # Identical reference points => cosine sim 1 => distance 0.
    cosine = np.ones((4, 4))
    scores = compute_knn_nonconformity(
        cosine_matrix=cosine,
        idx_points=np.array([0, 1]),
        idx_reference=np.array([2, 3]),
        k=2,
    )
    assert np.allclose(scores, 0.0)


def test_knn_nonconformity_orders_by_similarity():
    # Point 0 is more similar to the reference set than point 1,
    # so it should have a lower (less anomalous) nonconformity score.
    cosine = np.array(
        [
            [1.0, 0.0, 0.9, 0.8],
            [0.0, 1.0, 0.1, 0.2],
            [0.9, 0.1, 1.0, 0.5],
            [0.8, 0.2, 0.5, 1.0],
        ]
    )
    scores = compute_knn_nonconformity(
        cosine_matrix=cosine,
        idx_points=np.array([0, 1]),
        idx_reference=np.array([2, 3]),
        k=2,
    )
    assert scores[0] < scores[1]


def test_knn_nonconformity_raises_when_k_too_large():
    with pytest.raises(ValueError):
        compute_knn_nonconformity(
            cosine_matrix=np.ones((3, 3)),
            idx_points=np.array([0]),
            idx_reference=np.array([1, 2]),
            k=5,
        )


def test_wilson_interval_bounds_and_default_z():
    from ifce.eval.ood_conformal import _wilson_interval

    # Empty sample => maximally uncertain interval.
    assert _wilson_interval(0, 0) == (0.0, 1.0)

    lo, hi = _wilson_interval(5, 10)
    assert 0.0 <= lo < hi <= 1.0
    # The 95% interval should be wider than the 80% interval.
    lo80, hi80 = _wilson_interval(5, 10, alpha=0.20)
    assert (hi - lo) > (hi80 - lo80)
