# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Identifier-Free Code Embedding (IFCE).

Train and evaluate code-embedding models that locate functions in embedding
space by what they do rather than by their identifiers or surface naming.

Heavy machine-learning dependencies are imported lazily inside the submodules
that need them, so ``import ifce`` stays lightweight.
"""

__version__ = "0.1.0"
