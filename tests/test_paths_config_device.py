# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Unit tests for the lightweight path / config / device helpers."""

import pytest


def test_project_paths_are_under_project_root():
    from sfce import paths

    assert paths.CONFIG_DIR == paths.PROJECT_ROOT / "config"
    assert paths.CHECKPOINTS_DIR == paths.OUTPUTS_DIR / "checkpoints"
    assert paths.RESULTS_DIR == paths.OUTPUTS_DIR / "results"
    assert paths.DEFAULT_EXPERIMENT_CONFIG.name == "experiment_config.yaml"


def test_path_overrides_via_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SFCE_OUTPUTS_DIR", str(tmp_path / "out"))
    import importlib

    from sfce import paths

    importlib.reload(paths)
    try:
        assert paths.OUTPUTS_DIR == tmp_path / "out"
        assert paths.CHECKPOINTS_DIR == tmp_path / "out" / "checkpoints"
    finally:
        monkeypatch.delenv("SFCE_OUTPUTS_DIR", raising=False)
        importlib.reload(paths)


def test_ensure_output_dirs_creates_dirs(monkeypatch, tmp_path):
    import importlib

    monkeypatch.setenv("SFCE_OUTPUTS_DIR", str(tmp_path / "out"))
    from sfce import paths

    importlib.reload(paths)
    try:
        paths.ensure_output_dirs()
        assert (tmp_path / "out" / "checkpoints").is_dir()
        assert (tmp_path / "out" / "results").is_dir()
        assert (tmp_path / "out" / "distances").is_dir()
    finally:
        monkeypatch.delenv("SFCE_OUTPUTS_DIR", raising=False)
        importlib.reload(paths)


def test_yaml_safe_load(tmp_path):
    from sfce.config import yaml_safe_load

    cfg = tmp_path / "c.yaml"
    cfg.write_text("models: [a, b]\n")
    assert yaml_safe_load(cfg) == {"models": ["a", "b"]}
    assert yaml_safe_load(str(cfg)) == {"models": ["a", "b"]}


def test_get_device_returns_valid_string():
    from sfce.device import get_device

    assert get_device() in {"cuda", "mps", "cpu"}
    assert get_device(prefer="cpu") == "cpu"


@pytest.mark.parametrize("config_name", ["experiment_config.yaml", "train_config.yaml"])
def test_shipped_configs_parse(config_name):
    from sfce import paths
    from sfce.config import yaml_safe_load

    config_file = paths.CONFIG_DIR / config_name
    if not config_file.exists():
        pytest.skip(f"{config_name} not present")
    assert isinstance(yaml_safe_load(config_file), (dict, list))
