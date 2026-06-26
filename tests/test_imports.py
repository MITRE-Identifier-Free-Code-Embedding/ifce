# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Import smoke tests.

Every module must import cleanly on a core (CPU/macOS) install, with no GPU-only
optional dependencies present. Heavy backends (vllm, FlagEmbedding, openai) are
imported lazily inside the methods that use them, so importing the modules that
reference them must still succeed. This guards against the import-time crashes
that previously made the package unusable on a fresh clone.
"""

import importlib

import pytest

MODULES = [
    "ifce",
    "ifce.paths",
    "ifce.config",
    "ifce.device",
    "ifce.modeling",
    "ifce.modeling.models",
    "ifce.modeling.train",
    "ifce.processing",
    "ifce.processing.add_embeddings",
    "ifce.processing.find_distances",
    "ifce.eval",
    "ifce.eval.assemblage_retrieval",
    "ifce.eval.signsrch_benchmark",
    "ifce.eval.ood_conformal",
    "ifce.eval.ood_f1_eval",
    "ifce.visualization",
    "ifce.visualization.assemblage_plots",
    "ifce.visualization.signsrch_plots",
    "ifce.cli",
    "ifce.cli.conduct_experiment",
    "ifce.cli.time_embeddings",
    "ifce.cli.regenerate_figures",
    "ifce.cli.train",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)


def test_version():
    import ifce

    assert isinstance(ifce.__version__, str)


def test_model_registry_is_consistent():
    from ifce.modeling.models import MODEL_CLASS_MAP, MODEL_SIZE_MAP

    assert set(MODEL_CLASS_MAP) == set(MODEL_SIZE_MAP)


def test_build_model_unknown_name_raises():
    from ifce.modeling.models import build_model

    with pytest.raises(KeyError):
        build_model("definitely-not-a-real-model", device="cpu")
