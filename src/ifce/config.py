# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Configuration loading helpers.

Lives in its own lightweight module so that reading a YAML config does not drag
in the heavy ML stack (this helper previously lived inside the torch-heavy
``processing.add_embeddings`` module).
"""

from __future__ import annotations

from pathlib import Path

import yaml


def yaml_safe_load(file_path: str | Path) -> dict:
    """Load a YAML file and return its parsed contents."""
    with open(file_path) as f:
        return yaml.safe_load(f)
