"""Toy-CPU LoRA training with an optional "composability" penalty.

Trains LoRA adapters injected into :class:`~clora.toy_model.ToyUNet` on a
synthetic denoising-style regression task (random latents, random text-like
conditioning, no dataset or network access needed), in the spirit of a
DreamBooth-style LoRA fine-tune.

The composability penalty has two independent terms, both optional:

- ``group_sparsity_weight``: group-lasso penalty over each adapter's rank
  slots (``A[r, :]`` and ``B[:, r]`` for each rank index ``r``), encouraging
  each adapter to use only a few of its allocated rank slots.
- ``orth_weight`` with ``compose_against``: an orthogonality penalty (mean
  squared cosine similarity, see :mod:`clora.overlap`) between the adapter
  currently being trained and a list of frozen, previously-trained LoRAs
  loaded from ``.safetensors`` files.

For real SDXL/Flux training via diffusers + peft, see
:mod:`clora.hf_models`; this module intentionally never imports diffusers or
peft so the toy path stays dependency-light and fully offline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
import yaml

from .lora_io import LoraWeights, load_lora, save_lora
from .lora_nn import LoRALinear, extract_lora_weights, inject_lora
from .overlap import frobenius_sq_raw, inner_product_raw
from .toy_model import build_toy_model, toy_target_modules

logger = logging.getLogger(__name__)


@dataclass
class ComposabilityConfig:
    group_sparsity_weight: float = 0.0
    orth_weight: float = 0.0
    compose_against: List[str] = field(default_factory=list)


@dataclass
class TrainConfig:
    rank: int = 4
    alpha: float = 4.0
    lr: float = 1e-3
    steps: int = 50
    batch_size: int = 4
    seq_len: int = 4
    cond_len: int = 4
    channels: int = 16
    cond_dim: int = 16
    depth: int = 2
    seed: int = 0
    log_every: int = 10
    output: Optional[str] = None
    compose: ComposabilityConfig = field(default_factory=ComposabilityConfig)


def load_train_config(path: "str | Path") -> TrainConfig:
    """Load a :class:`TrainConfig` from a YAML file."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    compose_data = data.pop("compose", {}) or {}
    return TrainConfig(compose=ComposabilityConfig(**compose_data), **data)


def group_sparsity_penalty(replaced: dict) -> torch.Tensor:
    """Group-lasso penalty over each adapter's rank slots."""
    total = None
    for module in replaced.values():
        A, B = module.lora_A, module.lora_B  # (rank, in), (out, rank)
        group_norms = torch.sqrt(A.pow(2).sum(dim=1) + B.pow(2).sum(dim=0) + 1e-12)
        term = group_norms.sum()
        total = term if total is None else total + term
    return total if total is not None else torch.tensor(0.0)


def orthogonality_penalty(replaced: dict, frozen_loras: List[LoraWeights]) -> torch.Tensor:
    """Mean squared cosine similarity between trainable adapters and frozen ones."""
    if not frozen_loras:
        return torch.tensor(0.0)
    total = None
    count = 0
    for name, module in replaced.items():
        A, B, scale = module.lora_A, module.lora_B, module.scale
        cur_norm = torch.sqrt(frobenius_sq_raw(A, B, scale).clamp_min(1e-12))
        for frozen in frozen_loras:
            layer = frozen.layers.get(name)
            if layer is None:
                continue
            inner = inner_product_raw(A, B, scale, layer.A, layer.B, layer.scale)
            frozen_norm = torch.sqrt(
                frobenius_sq_raw(layer.A, layer.B, layer.scale).clamp_min(1e-12)
            )
            cos = inner / (cur_norm * frozen_norm)
            term = cos.pow(2)
            total = term if total is None else total + term
            count += 1
    if total is None:
        return torch.tensor(0.0)
    return total / count


def train(config: TrainConfig) -> LoraWeights:
    """Train toy LoRA adapters and return the resulting :class:`LoraWeights`."""
    torch.manual_seed(config.seed)
    model = build_toy_model(config.channels, config.cond_dim, config.depth, seed=config.seed)
    for p in model.parameters():
        p.requires_grad_(False)

    replaced: dict[str, LoRALinear] = inject_lora(
        model, toy_target_modules(), config.rank, config.alpha
    )
    params = [p for module in replaced.values() for p in (module.lora_A, module.lora_B)]
    optimizer = torch.optim.Adam(params, lr=config.lr)

    frozen_loras = [load_lora(p) for p in config.compose.compose_against]

    data_generator = torch.Generator().manual_seed(config.seed + 1)
    for step in range(config.steps):
        x0 = torch.randn(
            config.batch_size, config.seq_len, config.channels, generator=data_generator
        )
        cond = torch.randn(
            config.batch_size, config.cond_len, config.cond_dim, generator=data_generator
        )
        t = torch.rand(config.batch_size, 1, 1, generator=data_generator)
        noise = torch.randn(x0.shape, generator=data_generator)
        xt = (1 - t) * x0 + t * noise

        pred = model(xt, cond)
        target = noise - x0
        loss = F.mse_loss(pred, target)

        if config.compose.group_sparsity_weight:
            loss = loss + config.compose.group_sparsity_weight * group_sparsity_penalty(replaced)
        if config.compose.orth_weight and frozen_loras:
            loss = loss + config.compose.orth_weight * orthogonality_penalty(
                replaced, frozen_loras
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if config.log_every and step % config.log_every == 0:
            logger.info("step %d/%d loss=%.6f", step, config.steps, loss.item())

    lora = extract_lora_weights(replaced)
    if config.output:
        save_lora(lora, config.output)
    return lora


__all__ = [
    "ComposabilityConfig",
    "TrainConfig",
    "load_train_config",
    "group_sparsity_penalty",
    "orthogonality_penalty",
    "train",
]
