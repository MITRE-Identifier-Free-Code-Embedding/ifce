# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Embedding model wrappers.

This package was split out of a single ``models.py`` module; the public names
are re-exported here so ``from ifce.modeling.models import ...`` keeps working.
"""

from .base import EmbeddingModel
from .hf_models import (
    CustomHFEmbeddingModel,
    HFBGE3EmbeddingModel,
    HFBGEEmbeddingModel,
    HFEmbeddingModel,
    Qwen3HFEmbeddingModel,
)
from .registry import (
    MODEL_CLASS_MAP,
    MODEL_SIZE_MAP,
    build_model,
    discover_custom_models,
)
from .sentence_transformer_models import (
    BaseSentenceTransformerModel,
    BinSeekEmbeddingModel,
    CodeSageEmbeddingModel,
    GemmaEmbeddingModel,
    JinaAIEmbeddingModel,
    NomicAIEmbeddingModel,
    SalesforceEmbeddingModel,
    SnowflakeEmbeddingModel,
)
from .vllm_models import (
    CustomEmbeddingModelVLLM,
    OnlineCustomEmbeddingModel,
    OnlineQwen3EmbeddingModel,
    Qwen3EmbeddingModel,
)

__all__ = [
    "EmbeddingModel",
    "HFEmbeddingModel",
    "HFBGEEmbeddingModel",
    "Qwen3HFEmbeddingModel",
    "CustomHFEmbeddingModel",
    "HFBGE3EmbeddingModel",
    "Qwen3EmbeddingModel",
    "OnlineQwen3EmbeddingModel",
    "OnlineCustomEmbeddingModel",
    "CustomEmbeddingModelVLLM",
    "GemmaEmbeddingModel",
    "BaseSentenceTransformerModel",
    "SnowflakeEmbeddingModel",
    "SalesforceEmbeddingModel",
    "NomicAIEmbeddingModel",
    "JinaAIEmbeddingModel",
    "CodeSageEmbeddingModel",
    "BinSeekEmbeddingModel",
    "MODEL_CLASS_MAP",
    "MODEL_SIZE_MAP",
    "build_model",
    "discover_custom_models",
]
