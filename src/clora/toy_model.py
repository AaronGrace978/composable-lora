"""A tiny, randomly-initialized stand-in for a UNet/transformer noise
predictor, used so the whole test suite and examples run on CPU with no
network access or pretrained weights.

This is **not** a real diffusion backbone. For real SDXL/Flux models, use
:mod:`clora.hf_models`, which is only imported lazily (behind a flag/config)
so it never affects the offline, dependency-light toy test path.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class ToyBlock(nn.Module):
    """A minimal cross-attention + feed-forward block with diffusers-like
    submodule names (``to_q``, ``to_k``, ``to_v``, ``to_out``) so that LoRA
    target-module matching code paths look like the real thing.
    """

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(cond_dim, channels, bias=False)
        self.to_v = nn.Linear(cond_dim, channels, bias=False)
        self.to_out = nn.Linear(channels, channels)
        self.ff_in = nn.Linear(channels, channels * 2)
        self.ff_out = nn.Linear(channels * 2, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        q = self.to_q(x)
        k = self.to_k(cond)
        v = self.to_v(cond)
        scale = q.shape[-1] ** 0.5
        attn = torch.softmax(q @ k.transpose(-1, -2) / scale, dim=-1)
        x = x + self.to_out(attn @ v)
        x = x + self.ff_out(torch.relu(self.ff_in(x)))
        return x


class ToyUNet(nn.Module):
    """Tiny multi-block cross-attention stand-in for a UNet/transformer."""

    def __init__(
        self, channels: int = 16, cond_dim: int = 16, depth: int = 2, seed: int = 0, init_std: float = 0.3
    ) -> None:
        super().__init__()
        self.init_std = init_std
        self.channels = channels
        self.cond_dim = cond_dim
        self.in_proj = nn.Linear(channels, channels)
        self.blocks = nn.ModuleList([ToyBlock(channels, cond_dim) for _ in range(depth)])
        self.out_proj = nn.Linear(channels, channels)
        self._init_weights(seed)

    def _init_weights(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        for p in self.parameters():
            if p.dim() > 1:
                values = torch.empty_like(p).normal_(mean=0.0, std=self.init_std, generator=generator)
                p.data.copy_(values)
            else:
                p.data.zero_()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(h)


def build_toy_model(
    channels: int = 16, cond_dim: int = 16, depth: int = 2, seed: int = 0
) -> ToyUNet:
    return ToyUNet(channels=channels, cond_dim=cond_dim, depth=depth, seed=seed)


def toy_target_modules() -> List[str]:
    """Submodule-name substrings suitable for LoRA injection into :class:`ToyUNet`."""
    return ["to_q", "to_k", "to_v", "to_out", "ff_in", "ff_out"]
