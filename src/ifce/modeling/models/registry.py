# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Model registry: name -> class / size maps, discovery, and a build factory."""

from functools import partial
from hashlib import sha256
from pathlib import Path

from ifce.paths import CHECKPOINTS_DIR

from .base import EmbeddingModel
from .hf_models import HFBGE3EmbeddingModel, HFBGEEmbeddingModel
from .sentence_transformer_models import (
    BinSeekEmbeddingModel,
    CodeSageEmbeddingModel,
    GemmaEmbeddingModel,
    JinaAIEmbeddingModel,
    NomicAIEmbeddingModel,
    SalesforceEmbeddingModel,
    SnowflakeEmbeddingModel,
)
from .vllm_models import OnlineCustomEmbeddingModel, OnlineQwen3EmbeddingModel

MODEL_CLASS_MAP = {
    "bge-small-en-v1.5": HFBGEEmbeddingModel,
    "bge-base-en-v1.5": HFBGEEmbeddingModel,
    "bge-large-en-v1.5": HFBGEEmbeddingModel,
    "bge-m3": HFBGE3EmbeddingModel,
    "Qwen3-Embedding-0.6B": OnlineQwen3EmbeddingModel,
    "Qwen3-Embedding-8B": OnlineQwen3EmbeddingModel,
    "embeddinggemma-300m": GemmaEmbeddingModel,
    "snowflake-arctic-embed-l-v2.0": SnowflakeEmbeddingModel,
    "SFR-Embedding-Mistral": SalesforceEmbeddingModel,
    "nomic-embed-code": NomicAIEmbeddingModel,
    "jina-embeddings-v2-base-code": JinaAIEmbeddingModel,
    "codesage-large-v2": CodeSageEmbeddingModel,
    "BinSeek-Embedding": BinSeekEmbeddingModel,
}


MODEL_SIZE_MAP = {
    "bge-small-en-v1.5": 0.033,
    "bge-base-en-v1.5": 0.1,
    "bge-large-en-v1.5": 0.3,
    "bge-m3": 0.56,
    "Qwen3-Embedding-0.6B": 0.6,
    "Qwen3-Embedding-8B": 8,
    "embeddinggemma-300m": 0.3,
    "snowflake-arctic-embed-l-v2.0": 0.3,
    "SFR-Embedding-Mistral": 7,
    "nomic-embed-code": 7,
    "jina-embeddings-v2-base-code": 0.16,
    "codesage-large-v2": 1.3,
    "BinSeek-Embedding": 0.3,
}


def discover_custom_models() -> None:
    """Register any fine-tuned checkpoints found under the checkpoints directory.

    Safe to call on a fresh clone: if the checkpoints directory does not exist
    (or contains no model subdirectories) this is a no-op. This was previously
    executed unconditionally at import time and crashed with FileNotFoundError
    on a clean checkout.
    """
    if not CHECKPOINTS_DIR.exists():
        return
    for entry in CHECKPOINTS_DIR.iterdir():
        if not entry.is_dir():
            continue
        if "Qwen" in entry.name:
            MODEL_CLASS_MAP[entry.name] = OnlineCustomEmbeddingModel
            # Hardcoded for now; adjust if a non-0.6B custom model is introduced.
            MODEL_SIZE_MAP[entry.name] = 0.6
        else:
            raise ValueError(
                f"Unrecognized custom checkpoint '{entry.name}': update "
                "MODEL_CLASS_MAP to handle model types other than Qwen3-Embedding."
            )
    assert set(MODEL_CLASS_MAP.keys()) == set(MODEL_SIZE_MAP.keys())


def register_checkpoint(checkpoint_path: str) -> str:
    """Register an explicit Qwen3-0.6B checkpoint with a path-specific cache key."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not (path / "config.json").is_file():
        raise ValueError(
            f"Checkpoint directory must contain config.json: {path}. "
            "Point to the saved checkpoint-* directory, not its parent."
        )
    digest = sha256(str(path).encode()).hexdigest()[:12]
    # The build factory treats names containing "quant" or "fp8" as a request
    # to quantize. Keep arbitrary path components out of this cache key.
    name = f"checkpoint-{digest}"
    MODEL_CLASS_MAP[name] = partial(OnlineCustomEmbeddingModel, model_path=path)
    MODEL_SIZE_MAP[name] = 0.6
    return name


def build_model(name: str, device: str = "cpu", **kwargs) -> EmbeddingModel:
    """Instantiate the embedding model registered under ``name``.

    Quantized checkpoints (those whose name contains "quant" or "fp8") default
    to fp8 quantization unless overridden, centralizing the quant handling that
    was previously copy-pasted across the embedding/timing/benchmark entry points.
    """
    cls = MODEL_CLASS_MAP[name]
    if "quant" in name or "fp8" in name.lower():
        kwargs.setdefault("quantization", "fp8")
    return cls(name=name, device=device, **kwargs)


discover_custom_models()
