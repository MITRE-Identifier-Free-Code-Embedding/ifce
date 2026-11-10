# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Evaluation geometry, cache separation, and experiment configuration checks."""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from ifce.modeling.losses import negative_lorentz_distance
from ifce.processing import find_distances as distances


@pytest.mark.parametrize("metric", ["cosine", "hyperbolic"])
@pytest.mark.parametrize("split_size", [1, 3, 20])
def test_block_scores_match_training_and_rank_nearest_first(metric, split_size):
    embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.2], [-1.0, 0.0], [0.0, 0.0]])
    original = embeddings.clone()
    scores = distances.similarity_matrix_calculation(
        embeddings, "cpu", split_size, metric, curvature=0.3
    )
    expected = (
        F.normalize(embeddings) @ F.normalize(embeddings).T
        if metric == "cosine"
        else negative_lorentz_distance(embeddings, embeddings, curvature=0.3)
    )
    np.testing.assert_allclose(scores, expected.numpy(), atol=1e-6)
    assert scores.dtype == np.float32
    assert np.isfinite(scores).all()
    assert scores[0, 1] > scores[0, 2]
    torch.testing.assert_close(embeddings, original)


def test_legacy_cosine_entry_point():
    np.testing.assert_allclose(
        distances.cosine_matrix_calculation(torch.eye(3), "cpu"), np.eye(3)
    )


def test_yaml_defaults_overrides_and_cache_names():
    config = {"evaluation": {"distance_type": "hyperbolic", "curvature": 0.5}}
    assert distances.evaluation_settings({}) == ("cosine", 0.1)
    assert distances.evaluation_settings(config) == ("hyperbolic", 0.5)
    assert distances.evaluation_settings(config, "cosine", 0.2) == ("cosine", 0.2)
    tags = [
        distances.distance_tag("cosine"),
        distances.distance_tag("hyperbolic", 0.1),
        distances.distance_tag("hyperbolic", 0.2),
    ]
    assert len(set(tags)) == 3
    assert (
        distances.evaluation_results_dir("cosine") == distances.RESULTS_DIR / "cosine"
    )
    assert distances.evaluation_results_dir("hyperbolic").name == tags[1]


@pytest.mark.parametrize("curvature", [0, -1, float("nan"), float("inf")])
def test_bad_curvature_rejected(curvature):
    with pytest.raises(ValueError, match="curvature"):
        distances.evaluation_settings({}, "hyperbolic", curvature)


def test_overflow_curvature_rejected_before_paper_benchmarks():
    with pytest.raises(ValueError, match="curvature"):
        distances.evaluation_settings({}, "hyperbolic", 10000)


def test_bad_metric_rejected():
    with pytest.raises(ValueError, match="distance_type"):
        distances.similarity_matrix_calculation(
            torch.eye(2), "cpu", distance_type="bad"
        )


def test_find_distances_writes_separate_metric_caches(monkeypatch, tmp_path):
    from datasets import Dataset
    from torch.utils.data import DataLoader

    dataset = Dataset.from_dict(
        {
            "source_id": [0],
            "decompilation_id": [1],
            "tiny-embedding-source": [[1.0, 0.0]],
            "tiny-embedding-decompilation": [[0.8, 0.2]],
        }
    )
    monkeypatch.setattr(distances, "load_from_disk", lambda _: dataset)
    monkeypatch.setattr(distances, "yaml_safe_load", lambda _: {"models": ["tiny"]})
    monkeypatch.setattr(distances, "DISTANCES_DIR", tmp_path / "distances")
    monkeypatch.setattr(distances, "get_device", lambda: "cpu")
    monkeypatch.setattr(
        distances,
        "DataLoader",
        lambda *a, **kw: DataLoader(*a, **{**kw, "num_workers": 0}),
    )
    paths = []
    for metric, curvature in [
        ("cosine", 0.1),
        ("hyperbolic", 0.1),
        ("hyperbolic", 0.2),
    ]:
        _, files = distances.find_distances(
            "unused", "unused", distance_type=metric, curvature=curvature
        )
        path = files[0]
        paths.append(path)
        frame = pd.read_parquet(path)
        assert frame.attrs["distance_type"] == metric
        assert frame.attrs["score_convention"] == "higher_is_better"
        if metric == "hyperbolic":
            assert frame.attrs["curvature"] == curvature
        assert frame.shape == (2, 2)
    assert len(set(paths)) == 3
    assert paths[0] == tmp_path / "distances" / "cosine" / "tiny.parquet"
    assert paths[1] == tmp_path / "distances" / "hyperbolic_c0.1" / "tiny.parquet"
    assert paths[2] == tmp_path / "distances" / "hyperbolic_c0.2" / "tiny.parquet"
    scorer = Mock(side_effect=AssertionError("Should use cache"))
    monkeypatch.setattr(distances, "similarity_matrix_calculation", scorer)
    distances.find_distances(
        "unused", "unused", distance_type="hyperbolic", curvature=0.1
    )
    scorer.assert_not_called()


@pytest.mark.parametrize("metric", ["cosine", "hyperbolic"])
def test_experiment_passes_metric_to_both_benchmarks(monkeypatch, tmp_path, metric):
    from ifce.cli import conduct_experiment as cli

    monkeypatch.setattr(
        "sys.argv",
        [
            "ifce-experiment",
            "--data_dir_path",
            "unused",
            "--distance_type",
            metric,
            "--curvature",
            "0.3",
        ],
    )
    monkeypatch.setattr(cli, "yaml_safe_load", lambda _: {})
    monkeypatch.setattr(cli, "evaluation_results_dir", lambda *args: tmp_path)
    monkeypatch.setattr(cli, "create_embeddings", Mock(return_value="embedded"))
    find = Mock(return_value=(["tiny"], ["scores.parquet"]))
    signsrch = Mock()
    monkeypatch.setattr(cli, "find_distances", find)
    monkeypatch.setattr(cli, "analyze_sims", Mock())
    monkeypatch.setattr(cli, "run_signsrch_exps", signsrch)
    cli.conduct_experiment()
    for call in [find.call_args, signsrch.call_args]:
        assert call.kwargs["distance_type"] == metric
        assert call.kwargs["curvature"] == 0.3


def test_experiment_checkpoint_overrides_models_for_all_stages(monkeypatch, tmp_path):
    from ifce.cli import conduct_experiment as cli

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "models: [unused]\nevaluation: {distance_type: hyperbolic, curvature: 0.3}\n"
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "ifce-experiment",
            str(checkpoint),
            "--data_dir_path",
            "unused",
            "--yaml_file_path",
            str(config),
        ],
    )
    monkeypatch.setattr(cli, "evaluation_results_dir", lambda *args: tmp_path)
    model_lists = []

    def record_config(**kwargs):
        model_lists.append(cli.yaml_safe_load(kwargs["model_config_path"])["models"])
        return "embedded"

    def find(**kwargs):
        record_config(**kwargs)
        assert kwargs["distance_type"] == "hyperbolic"
        assert kwargs["curvature"] == 0.3
        return model_lists[-1], ["scores.parquet"]

    monkeypatch.setattr(cli, "create_embeddings", record_config)
    monkeypatch.setattr(cli, "find_distances", find)
    monkeypatch.setattr(cli, "analyze_sims", Mock())
    monkeypatch.setattr(cli, "run_signsrch_exps", record_config)
    cli.conduct_experiment()
    assert len(model_lists) == 3
    assert model_lists[0] == model_lists[1] == model_lists[2]
    assert len(model_lists[0]) == 1
    assert model_lists[0] != ["unused"]
    assert cli.yaml_safe_load(config)["models"] == ["unused"]
