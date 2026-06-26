# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Canonical filesystem locations for IFCE.

A single source of truth for project directories, replacing the per-script
``~/identifier-free-code-embedding/...`` defaults and ad-hoc ``../data`` relative
paths that were scattered across the codebase. Every location can be overridden
with an environment variable so the package works from an installed wheel or a
relocated checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root: src/ifce/paths.py -> src/ifce -> src -> <repo root>
PROJECT_ROOT = Path(
    os.environ.get(
        "IFCE_PROJECT_ROOT",
        Path(__file__).resolve().parents[2],
    )
)

CONFIG_DIR = Path(os.environ.get("IFCE_CONFIG_DIR", PROJECT_ROOT / "config"))
DATA_DIR = Path(os.environ.get("IFCE_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUTS_DIR = Path(os.environ.get("IFCE_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))

CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
RESULTS_DIR = OUTPUTS_DIR / "results"
DISTANCES_DIR = OUTPUTS_DIR / "distances"

DEFAULT_EXPERIMENT_CONFIG = CONFIG_DIR / "experiment_config.yaml"
DEFAULT_TIME_CONFIG = CONFIG_DIR / "time_experiment_config.yaml"
DEFAULT_SIGNSRCH_PATH = DATA_DIR / "signsrch.sig"


def ensure_output_dirs() -> None:
    """Create the generated-output directories if they do not yet exist."""
    for directory in (CHECKPOINTS_DIR, RESULTS_DIR, DISTANCES_DIR):
        directory.mkdir(parents=True, exist_ok=True)
