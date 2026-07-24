# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""sentence-transformers backed embedding models."""

import itertools

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import EmbeddingModel


class GemmaEmbeddingModel(EmbeddingModel):
    def __init__(self, name: str, device: str = "cpu"):
        super().__init__(name, device=device)
        self.model = SentenceTransformer(
            model_name_or_path=f"google/{self.name}", device=self.device
        )

    def embed(self, str_to_embed: str):
        return self.model.encode_document([str_to_embed])

    def embed_batch(self, strs_to_embed: list[str]):
        return self.model.encode_document(strs_to_embed)


class BaseSentenceTransformerModel(EmbeddingModel):
    def __init__(self, name: str, base_name: str, device: str = "cpu", **kwargs):
        super().__init__(name=name, device=device)
        self.base_name = base_name
        self.model = SentenceTransformer(
            model_name_or_path=f"{base_name}/{self.name}", device=self.device, **kwargs
        )

    def embed(self, str_to_embed: str) -> np.ndarray:
        return self.model.encode([str_to_embed])

    def embed_batch(self, strs_to_embed: list[str]) -> np.ndarray:
        return self.model.encode(strs_to_embed)

    def embed_batch_as_smaller_pieces(self, strs_to_embed: list[str]) -> np.ndarray:
        mini_bs = 4
        # When the batch is large, manage batching in smaller chunks than usual.
        if len(strs_to_embed) > mini_bs:
            all_embeddings = []
            for batch in itertools.batched(strs_to_embed, mini_bs):
                all_embeddings.extend(self.model.encode(list(batch)))
            return np.array(all_embeddings)
        else:
            return self.model.encode(strs_to_embed)


class SnowflakeEmbeddingModel(BaseSentenceTransformerModel):
    def __init__(self, name: str, base_name: str = "Snowflake", device: str = "cpu"):
        super().__init__(name=name, base_name=base_name, device=device)


class SalesforceEmbeddingModel(BaseSentenceTransformerModel):
    def __init__(self, name: str, base_name: str = "Salesforce", device: str = "cpu"):
        super().__init__(name=name, base_name=base_name, device=device)


class NomicAIEmbeddingModel(BaseSentenceTransformerModel):
    def __init__(
        self,
        name: str,
        base_name: str = "nomic-ai",
        device: str = "cpu",
        model_kwargs=None,
    ):
        if model_kwargs is None:
            model_kwargs = {"dtype": "bfloat16"}
        super().__init__(
            name=name, base_name=base_name, device=device, model_kwargs=model_kwargs
        )

    def embed_batch(self, strs_to_embed: list[str]) -> np.ndarray:
        if isinstance(strs_to_embed, str):
            raise TypeError("embed_batch must take a list of strs")
        return self.embed_batch_as_smaller_pieces(strs_to_embed=strs_to_embed)


class JinaAIEmbeddingModel(BaseSentenceTransformerModel):
    def __init__(self, name: str, base_name: str = "jinaai", device: str = "cpu"):
        super().__init__(name=name, base_name=base_name, device=device)


class CodeSageEmbeddingModel(BaseSentenceTransformerModel):
    def __init__(self, name: str, base_name: str = "codesage", device: str = "cpu"):
        super().__init__(name=name, base_name=base_name, device=device)


class BinSeekEmbeddingModel(BaseSentenceTransformerModel):
    def __init__(self, name: str, base_name: str = "XingTuLab", device: str = "cpu"):
        super().__init__(name=name, base_name=base_name, device=device)
