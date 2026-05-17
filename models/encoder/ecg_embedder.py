# ecg_embedder.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
from torch import nn



class ECGEmbedder(nn.Module, ABC):
    """
    Abstract base for models that map a 12-lead ECG tensor to a fixed-dimensional embedding.

    Expected input shape: (B, C, T) where:
      - B = batch size
      - C = num leads (default 12)
      - T = time samples

    Functions:
      - forward(x): return embeddings of shape (B, embedding_dim)
      - embedding_dim: int property describing the output dimension

    """

    def __init__(
        self,
        num_channels: int = 12,
    ) -> None:
        super().__init__()
        self._num_channels = int(num_channels)

    @property
    def num_channels(self) -> int:
        return self._num_channels

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            embeddings: Tensor of shape (B, embedding_dim)
        """
        ...

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
