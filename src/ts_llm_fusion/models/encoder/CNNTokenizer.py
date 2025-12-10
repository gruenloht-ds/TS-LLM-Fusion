#
# This source file is part of the OpenTSLM open-source project
#
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn
from einops import rearrange

from src.ts_llm_fusion.utils.model_config import TRANSFORMER_INPUT_DIM, ENCODER_OUTPUT_DIM, PATCH_SIZE
from src.ts_llm_fusion.models.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase
from src.ts_llm_fusion.models.encoder.ResNetBlock import WT_series_decomp
class WaveletTokenizer(TimeSeriesEncoderBase):
    def __init__(self,
                 output_dim: int = ENCODER_OUTPUT_DIM,
                 dropout: float = 0.0,
                 transformer_input_dim: int = TRANSFORMER_INPUT_DIM,
                 ):
        """

        Args:
            output_dim:
            dropout:
            transformer_input_dim:

        """
        super().__init__(output_dim, dropout)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3)

        # 1) Conv1d patch embedding: (B, 1, L) -> (B, embed_dim, L)
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=transformer_input_dim,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        # 2) Wavelet Series Decomposition
        self.decomposition = WT_series_decomp()
        self.conv2 = nn.Conv1d(int(transformer_input_dim/4), 128, kernel_size=3, stride=1,padding=1,bias=False)

        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)
        # 3) Input norm + dropout
        self.input_norm = nn.LayerNorm(transformer_input_dim)
        self.input_dropout = nn.Dropout(0.3)
    def forward(self, x):

        x = x.unsqueeze(1)
        residual = x

        x = self.gelu(self.conv1(x))
        x = self.dropout(x)
        splits = torch.split(x, int(x.shape[1]/4), dim=1)
        out = [splits[0]]
        #out = out.unsqueeze(1)

        x_season, x_trend = [], []
        for out_i in out:
            season, trend = self.decomposition(out_i)
            x_season.append(season)
            x_trend.append(trend)
        for i in range(1, len(x_season)):
            x_season[i] = x_season[i] + x_season[i - 1]
        for i in range(1, len(x_trend)):
            x_trend[i] = x_trend[i] + x_trend[i - 1]
        for i in range(len(out)):
            out[i] = x_season[i] + x_trend[i]
        out = torch.cat(out,dim=1)
        out = self.conv2(out)
        out += residual


        out = out.squeeze(2)
        out = self.pool(out)


        # GELU + dropout
        out = self.gelu(out)
        out = self.dropout(out)
        out = torch.transpose(out,1,2)

        return out

class CNNTokenizer(TimeSeriesEncoderBase):
    def __init__(
        self,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.0,
        transformer_input_dim: int = TRANSFORMER_INPUT_DIM,
        patch_size: int = PATCH_SIZE,
        max_patches: int = 2600,
    ):
        """
        Args:
            embed_dim: dimension of patch embeddings
            num_heads: number of attention heads
            num_layers: number of TransformerEncoder layers
            patch_size: length of each patch
            ff_dim: hidden size of the feed‐forward network inside each encoder layer
            dropout: dropout probability
            max_patches: maximum number of patches expected per sequence (for pos emb)
        """
        super().__init__(output_dim, dropout)
        self.patch_size = patch_size

        # 1) Conv1d patch embedding: (B, 1, L) -> (B, embed_dim, L/patch_size)
        self.patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=transformer_input_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # 2) Learnable positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_patches, transformer_input_dim)
        )

        # 3) Input norm + dropout
        self.input_norm = nn.LayerNorm(transformer_input_dim)
        self.input_dropout = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape [B, L], a batch of raw time series.
        Returns:
            FloatTensor of shape [B, N, embed_dim], where N = L // patch_size.
        """

        B, L = x.shape
        if L % self.patch_size != 0:
            raise ValueError(
                f"Sequence length {L} not divisible by patch_size {self.patch_size}"
            )

        # reshape to (B, 1, L)
        x = x.unsqueeze(1)

        # conv patch embedding -> (B, embed_dim, N)
        x = self.patch_embed(x)

        # transpose to (B, N, embed_dim)
        x = x.transpose(1, 2)

        # add positional embeddings (truncate or expand as needed)
        N = x.size(1)
        if N > self.pos_embed.size(1):
            raise ValueError(
                f"Time series of length {N*4} is too long; max supported is {self.pos_embed.size(1)*4}. Change max_patches parameter in {__file__}"
            )
        pos = self.pos_embed[:, :N, :]
        x = x + pos

        # norm + dropout
        x = self.input_norm(x)
        x = self.input_dropout(x)

        return x
