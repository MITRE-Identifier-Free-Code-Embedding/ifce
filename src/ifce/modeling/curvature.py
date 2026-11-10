# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Curvature limits shared by training and retrieval scoring."""

import math

# The Lorentz map and pairwise distances run in FP32. These bounds keep its
# coordinates representable and leave headroom when squared distances are formed.
MIN_CURVATURE = 1e-30
MAX_CURVATURE = 100.0


def validate_curvature(curvature: float) -> None:
    if (
        not math.isfinite(curvature)
        or curvature < MIN_CURVATURE
        or curvature > MAX_CURVATURE
    ):
        raise ValueError(
            f"curvature must be finite and between {MIN_CURVATURE:g} "
            f"and {MAX_CURVATURE:g}, got {curvature}"
        )
