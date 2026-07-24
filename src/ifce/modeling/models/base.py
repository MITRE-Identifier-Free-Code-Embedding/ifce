# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The MITRE Corporation
"""Base embedding-model interface."""


class EmbeddingModel:
    def __init__(self, name: str, device: str = "cpu"):
        self.name = name
        self.device = device
        self.model = None

    def embed(self, str_to_embed: str) -> list[float]:
        raise NotImplementedError

    def embed_batch(self, strs_to_embed: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def parameter_count(self) -> int:
        if self.model is None:
            raise ValueError("Model must be instantiated")
        return sum(p.numel() for p in self.model.model.parameters())
