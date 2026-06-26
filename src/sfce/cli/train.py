# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""CLI entry point for training (``sfce-train``).

Thin shim over the library training routine in :mod:`sfce.modeling.train`.
Distributed training is launched via ``accelerate launch`` (see the README).
"""

from sfce.modeling.train import main

if __name__ == "__main__":
    main()
