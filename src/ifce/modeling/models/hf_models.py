# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Hugging Face Transformers / FlagEmbedding backed embedding models."""

import torch
from transformers import AutoModel, AutoTokenizer

from ifce.paths import CHECKPOINTS_DIR

from .base import EmbeddingModel


class HFEmbeddingModel(EmbeddingModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _infer(self, encoded_input):
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            # Perform pooling. In this case, cls pooling.
            sentence_embeddings = model_output[0][:, 0]
            sentence_embeddings = torch.nn.functional.normalize(
                sentence_embeddings, p=2, dim=1
            )
        return sentence_embeddings.to("cpu").tolist()

    def embed(self, str_to_embed: str) -> list[float]:
        encoded_input = self.tokenizer(
            [str_to_embed], padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        return self._infer(encoded_input)

    def embed_batch(self, strs_to_embed: list[str]) -> list[list[float]]:
        encoded_input = self.tokenizer(
            strs_to_embed, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        return self._infer(encoded_input)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.model.model.parameters())


class HFBGEEmbeddingModel(HFEmbeddingModel):
    def __init__(
        self, model_tokenizer_hf_name: str | None = None, device: str = "cpu", **kwargs
    ):
        super().__init__(**kwargs)
        self.device = device
        self.model_hf_name = f"BAAI/{self.name}"
        self.model = AutoModel.from_pretrained(
            self.model_hf_name, device_map=device
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_tokenizer_hf_name
            if model_tokenizer_hf_name is not None
            else self.model_hf_name
        )


class Qwen3HFEmbeddingModel(HFEmbeddingModel):
    def __init__(
        self, model_tokenizer_hf_name: str | None = None, device: str = "cpu", **kwargs
    ):
        super().__init__(**kwargs)
        self.device = device
        self.model_hf_name = f"Qwen/{self.name}"
        self.model = AutoModel.from_pretrained(
            self.model_hf_name, device_map=device
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_tokenizer_hf_name
            if model_tokenizer_hf_name is not None
            else self.model_hf_name
        )


class CustomHFEmbeddingModel(HFEmbeddingModel):
    def __init__(self, name: str, device: str = "cpu"):
        super().__init__(name=name, device=device)
        models_path = CHECKPOINTS_DIR / name
        self.model = AutoModel.from_pretrained(models_path, device_map=device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(models_path)


class HFBGE3EmbeddingModel(EmbeddingModel):
    def __init__(self, name: str = "bge-m3", device: str = "cpu", **kwargs):
        super().__init__(name, device, **kwargs)
        from FlagEmbedding import BGEM3FlagModel

        self.model = BGEM3FlagModel(f"BAAI/{name}", use_fp16=True)

    def embed(self, str_to_embed: str) -> list[float]:
        result = self.model.encode(
            [str_to_embed],
            batch_size=1,
            max_length=8192,
        )["dense_vecs"]
        return result.tolist()

    def embed_batch(self, strs_to_embed: list[str]) -> list[list[float]]:
        result = self.model.encode(
            strs_to_embed,
            batch_size=len(strs_to_embed),
            max_length=8192,
        )["dense_vecs"]
        return result.tolist()
