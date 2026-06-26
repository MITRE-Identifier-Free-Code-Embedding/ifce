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
    "sfce",
    "sfce.paths",
    "sfce.config",
    "sfce.device",
    "sfce.modeling",
    "sfce.modeling.models",
    "sfce.modeling.train",
    "sfce.processing",
    "sfce.processing.add_embeddings",
    "sfce.processing.find_distances",
    "sfce.eval",
    "sfce.eval.assemblage_retrieval",
    "sfce.eval.signsrch_benchmark",
    "sfce.eval.ood_conformal",
    "sfce.eval.ood_f1_eval",
    "sfce.visualization",
    "sfce.visualization.assemblage_plots",
    "sfce.visualization.signsrch_plots",
    "sfce.cli",
    "sfce.cli.conduct_experiment",
    "sfce.cli.time_embeddings",
    "sfce.cli.regenerate_figures",
    "sfce.cli.train",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    importlib.import_module(module)


def test_version():
    import sfce

    assert isinstance(sfce.__version__, str)


def test_model_registry_is_consistent():
    from sfce.modeling.models import MODEL_CLASS_MAP, MODEL_SIZE_MAP

    assert set(MODEL_CLASS_MAP) == set(MODEL_SIZE_MAP)
