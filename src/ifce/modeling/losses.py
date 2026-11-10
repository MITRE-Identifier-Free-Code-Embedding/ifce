# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Configurable cached InfoNCE objectives for paired code embeddings."""

import math
from functools import partial, update_wrapper

import torch
from sentence_transformers import util
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from torch.nn import functional as F

from ifce.modeling.curvature import validate_curvature


def _positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value}")


def lorentz_expmap(embeddings: torch.Tensor, curvature: float = 0.1) -> torch.Tensor:
    """Normalize pooled embeddings and apply Exp_o at fixed curvature -c.

    Returns time-first hyperboloid coordinates in FP32, including under AMP.
    Zero embeddings map to the origin.
    """
    validate_curvature(curvature)
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        tangent = F.normalize(embeddings.float(), p=2, dim=-1)
        radius = tangent.norm(dim=-1, keepdim=True)
        scaled_radius = math.sqrt(curvature) * radius
        time = scaled_radius.cosh() / math.sqrt(curvature)
        # sinh(t)/t has limit 1 at zero; avoid undefined gradients there.
        denominator = scaled_radius.clamp_min(torch.finfo(torch.float32).tiny)
        factor = torch.where(scaled_radius > 0, scaled_radius.sinh() / denominator, 1.0)
        return torch.cat((time, factor * tangent), dim=-1)


def negative_lorentz_distance(
    anchors: torch.Tensor, candidates: torch.Tensor, curvature: float = 0.1
) -> torch.Tensor:
    """Pairwise -d_L after normalized exponential mapping (higher is better)."""
    with torch.autocast(device_type=anchors.device.type, enabled=False):
        x = lorentz_expmap(anchors, curvature)
        y = lorentz_expmap(candidates, curvature)
        # On the hyperboloid, -c<x,y>_L = 1 + c/2 <x-y,x-y>_L.
        # Direct differences avoid cancellation for identical/nearby vectors.
        spatial_sq = torch.cdist(
            x[:, 1:], y[:, 1:], compute_mode="donot_use_mm_for_euclid_dist"
        ).square()
        time_sq = (x[:, :1] - y[:, :1].T).square()
        lorentz_sq = (spatial_sq - time_sq).clamp_min(0)
        # acosh(1 + c*d²/2) = 2*asinh(sqrt(c)*d/2). Avoid adding a tiny
        # quantity to 1, which erases all distances when curvature is small.
        # Clamp inside sqrt to keep its backward pass finite at exact matches.
        radius = torch.sqrt(lorentz_sq.clamp_min(torch.finfo(torch.float32).tiny))
        distance = 2 * torch.asinh(0.5 * math.sqrt(curvature) * radius)
        distance = distance / math.sqrt(curvature)
        return -torch.where(lorentz_sq > 0, distance, torch.zeros_like(distance))


def build_loss(
    model,
    loss_name: str = "cosine_infonce",
    temperature: float = 0.05,
    curvature: float = 0.1,
    mini_batch_size: int = 1,
) -> CachedMultipleNegativesRankingLoss:
    """Keep cached in-batch negatives and change only the similarity geometry."""
    _positive_finite(temperature, "temperature")
    validate_curvature(curvature)
    if loss_name == "cosine_infonce":
        similarity = util.cos_sim
    elif loss_name == "hyperbolic_infonce":
        # SentenceTransformers records similarity_fct.__name__ in model cards.
        similarity = update_wrapper(
            partial(negative_lorentz_distance, curvature=curvature),
            negative_lorentz_distance,
        )
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
    return CachedMultipleNegativesRankingLoss(
        model,
        scale=1 / temperature,
        similarity_fct=similarity,
        mini_batch_size=mini_batch_size,
    )
