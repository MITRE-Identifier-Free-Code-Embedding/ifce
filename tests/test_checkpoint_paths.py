"""Explicit checkpoint validation, selection, and cache separation."""

from unittest.mock import Mock

import pytest

from ifce.modeling.models import registry


def test_checkpoint_paths_are_distinct_and_forwarded(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "MODEL_CLASS_MAP", {})
    monkeypatch.setattr(registry, "MODEL_SIZE_MAP", {})
    backend = Mock()
    monkeypatch.setattr(registry, "OnlineCustomEmbeddingModel", backend)
    names = []
    for parent in ["run-a", "run-b"]:
        checkpoint = tmp_path / parent / "checkpoint-100"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}")
        name = registry.register_checkpoint(str(checkpoint))
        assert name == registry.register_checkpoint(str(checkpoint))
        assert "/" not in name
        registry.build_model(name, device="cuda")
        backend.assert_called_with(
            name=name, device="cuda", model_path=checkpoint.resolve()
        )
        names.append(name)
    assert names[0] != names[1]


def test_checkpoint_requires_config(tmp_path):
    with pytest.raises(ValueError, match="must contain config.json"):
        registry.register_checkpoint(str(tmp_path))


def test_explicit_checkpoint_path_does_not_request_fp8(monkeypatch, tmp_path):
    checkpoint = tmp_path / "unquantized-run" / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}")
    monkeypatch.setattr(registry, "OnlineCustomEmbeddingModel", lambda **kwargs: kwargs)

    name = registry.register_checkpoint(str(checkpoint))
    arguments = registry.build_model(name)

    assert arguments["model_path"] == checkpoint.resolve()
    assert "quantization" not in arguments
