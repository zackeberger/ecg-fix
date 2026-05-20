#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Architecture hyperparameters
# -----------------------------
c1 = 12   # 12 ECG leads as input channels
c2 = 4
c3 = 16
c4 = 32
k = 7
s = 3


class cnn_network_contrastive(nn.Module):
    """
    Contrastive CNN for 12-lead ECG with two temporal segments as views.

    Input:  x = (B, T, C, V)  e.g. (B, 2500, 12, 2)
    Output: z = (B, E, V)
    """

    def __init__(
        self,
        dropout_type="drop1d",
        p1=0.1,
        p2=0.1,
        p3=0.1,
        embedding_dim=256,
        **kwargs
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Dropout
        if dropout_type == "drop2d":
            self.dropout1 = nn.Dropout2d(p=p1)
            self.dropout2 = nn.Dropout2d(p=p2)
            self.dropout3 = nn.Dropout2d(p=p3)
        else:
            self.dropout1 = nn.Dropout(p=p1)
            self.dropout2 = nn.Dropout(p=p2)
            self.dropout3 = nn.Dropout(p=p3)

        # Encoder blocks (paper-style)
        self.encoder = nn.Sequential(
            nn.Conv1d(c1, c2, k, s),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            self.dropout1,

            nn.Conv1d(c2, c3, k, s),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            self.dropout2,

            nn.Conv1d(c3, c4, k, s),
            nn.BatchNorm1d(c4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            self.dropout3,
        )

        # Paper uses Linear(320 -> E) for 2500 samples after conv/pool stack
        # (32 channels * 10 length = 320)
        self.fc = nn.Sequential(
            nn.Linear(c4 * 10, embedding_dim),
            nn.ReLU()
        )

    def _encode_view_batch(self, x_bv):
        """
        Encode a batch of views.

        x_bv: (B*V, T, C)
        returns: (B*V, E)
        """
        # Conv1d wants (B*V, C, T)
        x_bv = x_bv.permute(0, 2, 1).contiguous()   # (B*V, C, T)
     #   print(x_bv.shape)
        h = self.encoder(x_bv)                      # (B*V, c4, L)
      #  print(h.shape)
        h = h.reshape(h.size(0), -1)                # (B*V, 320)
        z = self.fc(h)                              # (B*V, E)
        return z

    def forward(self, x):
        """
        x: (B, T, C, V)
        returns: (B, E, V)
        """
        B, T, C, V = x.shape

        # Vectorize across views:
        # (B,T,C,V) -> (B,V,T,C) -> (B*V,T,C)
        x_bv = x.permute(0, 3, 1, 2).contiguous().reshape(B * V, T, C)
        #print(x_bv.shape)

        z_bv = self._encode_view_batch(x_bv)        # (B*V, E)
       # print(z_bv.shape)

        # back to (B, E, V)
        z = z_bv.reshape(B, V, self.embedding_dim).permute(0, 2, 1).contiguous()
        return z

    @torch.no_grad()
    def get_embedding(self, x, aggregate=None):
        """
        Returns embeddings from the encoder head (same as forward output, but aggregated).

        x: (B, T, C, V)
        aggregate:
          - "mean"  -> (B, E)
          - "first" -> (B, E) (view 0)
          - None    -> (B, E, V)
        """
        z = self.forward(x)  # (B, E, V)

        if aggregate == "mean":
            return z.mean(dim=2)
        elif aggregate == "first":
            return z[:, :, 0]
        elif aggregate is None:
            return z
        else:
            raise ValueError("aggregate must be 'mean', 'first', or None")


class second_cnn_network(nn.Module):
    """
    Linear downstream head (paper Layer 5) : E -> C
    """

    def __init__(self, first_model: cnn_network_contrastive, noutputs: int):
        super().__init__()
        self.first_model = first_model
        self.linear = nn.Linear(first_model.embedding_dim, noutputs)

    def forward(self, x):
        h = self.first_model.get_embedding(x, aggregate="mean")  # (B, E)
        return self.linear(h)  # (B, noutputs)
 