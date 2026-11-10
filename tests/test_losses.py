# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Geometry, objective, and cached-gradient checks without model downloads."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from ifce.modeling.losses import (
    build_loss,
    lorentz_expmap,
    negative_lorentz_distance,
)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 3)

    def __getitem__(self, index):
        return self.projection

    def forward(self, features):
        return {"sentence_embedding": self.projection(features["input_ids"])}


@pytest.mark.parametrize("curvature", [0.01, 0.1, 1.0])
def test_lorentz_mapping_and_distance_match_plan(curvature):
    torch.manual_seed(12)
    embeddings = torch.randn(5, 4)
    mapped = lorentz_expmap(embeddings, curvature)
    norm = -mapped[:, 0].square() + mapped[:, 1:].square().sum(-1)
    torch.testing.assert_close(norm, torch.full_like(norm, -1 / curvature))
    assert (mapped[:, 0] > 0).all()
    lorentz_product = -mapped[:, :1] @ mapped[:, :1].T
    lorentz_product += mapped[:, 1:] @ mapped[:, 1:].T
    expected = -torch.acosh((-curvature * lorentz_product).clamp_min(1))
    expected /= curvature**0.5
    actual = negative_lorentz_distance(embeddings, embeddings, curvature)
    mask = ~torch.eye(5, dtype=torch.bool)
    torch.testing.assert_close(actual[mask], expected[mask], atol=1e-4, rtol=1e-4)
    assert actual.argmax(dim=1).tolist() == list(range(5))
    torch.testing.assert_close(actual, actual.T)


def test_duplicate_zero_and_small_embeddings_have_finite_gradients_under_amp():
    embeddings = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1e-9, 1e-9]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        scores = negative_lorentz_distance(embeddings, embeddings)
        loss = F.cross_entropy(scores / 0.05, torch.arange(4))
    assert scores.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(embeddings.grad).all()


def test_tiny_curvature_preserves_distinct_scores_and_gradients():
    anchors = torch.eye(2, requires_grad=True)
    scores = negative_lorentz_distance(anchors, torch.eye(2), curvature=1e-9)

    assert torch.isfinite(scores).all()
    assert scores[0, 0] > scores[0, 1]
    F.cross_entropy(scores, torch.arange(2)).backward()
    assert torch.count_nonzero(anchors.grad)


def test_smallest_supported_curvature_preserves_euclidean_limit():
    scores = negative_lorentz_distance(torch.eye(2), torch.eye(2), curvature=1e-30)
    torch.testing.assert_close(
        scores[0, 1], torch.tensor(-(2**0.5)), atol=1e-5, rtol=1e-5
    )


def test_curvature_that_overflows_fp32_is_rejected():
    with pytest.raises(ValueError, match="curvature"):
        negative_lorentz_distance(torch.eye(2), torch.eye(2), curvature=10000)
    with pytest.raises(ValueError, match="curvature"):
        build_loss(TinyEncoder(), "hyperbolic_infonce", curvature=10000)


@pytest.mark.parametrize("loss_name", ["cosine_infonce", "hyperbolic_infonce"])
def test_cached_loss_matches_full_batch_objective_and_gradients(loss_name):
    torch.manual_seed(42)
    model = TinyEncoder()
    features = [{"input_ids": torch.randn(4, 4)} for _ in range(2)]
    loss = build_loss(model, loss_name, temperature=0.1, mini_batch_size=2)
    assert loss.get_config_dict()["similarity_fct"]
    embeddings = [model(feature)["sentence_embedding"] for feature in features]
    scores = loss.similarity_fct(*embeddings) / 0.1
    expected = F.cross_entropy(scores, torch.arange(4))
    expected.backward()
    expected_gradients = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    actual = loss(features, labels=None)
    actual.backward()
    torch.testing.assert_close(actual, expected)
    for parameter, gradient in zip(model.parameters(), expected_gradients, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("parameter", ["temperature", "curvature"])
def test_invalid_parameters_rejected(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        build_loss(TinyEncoder(), **{parameter: value})


def test_unknown_loss_rejected():
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss(TinyEncoder(), "unknown")


def test_default_loss_preserves_cosine_baseline():
    loss = build_loss(TinyEncoder())
    assert loss.scale == 20.0
    a, b = torch.randn(3, 4), torch.randn(5, 4)
    torch.testing.assert_close(
        loss.similarity_fct(a, b), F.normalize(a) @ F.normalize(b).T
    )


def test_training_cli_options(monkeypatch):
    from ifce.modeling.train import arg_parse

    monkeypatch.setattr(
        "sys.argv",
        [
            "ifce-train",
            "--data_dir_path",
            "unused",
            "--loss",
            "hyperbolic_infonce",
            "--temperature",
            "0.2",
            "--curvature",
            "0.5",
        ],
    )
    args = arg_parse()
    assert (args.loss, args.temperature, args.curvature) == (
        "hyperbolic_infonce",
        0.2,
        0.5,
    )
