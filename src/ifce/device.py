# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Device selection.

Replaces the hardcoded ``device='cuda'`` strings that were scattered across the
embedding modules and that crashed on CPU-only / macOS machines.
"""

from __future__ import annotations


def get_device(prefer: str | None = None) -> str:
    """Return the best available torch device string.

    Order of preference: an explicit ``prefer`` value if given, then CUDA, then
    Apple MPS, then CPU.
    """
    import torch

    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"
