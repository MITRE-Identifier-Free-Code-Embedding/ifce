# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""CLI entry point for training (``ifce-train``).

Thin shim over the library training routine in :mod:`ifce.modeling.train`.
Distributed training is launched via ``accelerate launch`` (see the README).
"""

from ifce.modeling.train import main

if __name__ == "__main__":
    main()
