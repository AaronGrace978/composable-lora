"""Trainable LoRA ``nn.Module`` injection, used by :mod:`clora.train` to add
low-rank adapters to a (toy or real) model, and to extract the resulting
factors back into a :class:`~clora.lora_io.LoraWeights`.

For applying one or more *already-trained* :class:`LoraWeights` onto a frozen
model for generation/evaluation (no gradients needed), see
:func:`apply_loras` which uses lightweight forward hooks instead of replacing
modules.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Sequence

import torch
from torch import nn

from .lora_io import LoraLayer, LoraWeights


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable low-rank adapter.

    ``lora_A`` is initialized with Kaiming-uniform noise and ``lora_B`` is
    zero-initialized, so the wrapped module computes exactly the original
    linear layer's output before any training happens (standard LoRA init).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    @property
    def scale(self) -> float:
        return self.alpha / self.rank if self.rank else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scale * lora_out

    def to_layer(self, name: str) -> LoraLayer:
        """Snapshot current (detached) factors as a :class:`LoraLayer`."""
        return LoraLayer(
            name=name,
            A=self.lora_A.detach().clone(),
            B=self.lora_B.detach().clone(),
            alpha=float(self.alpha),
            orig_key=name,
        )


def inject_lora(
    model: nn.Module, target_substrings: Sequence[str], rank: int, alpha: float
) -> Dict[str, LoRALinear]:
    """Replace every ``nn.Linear`` whose dotted name contains one of
    ``target_substrings`` with a trainable :class:`LoRALinear`, in place.

    Returns a mapping from the (pre-replacement) dotted module name to the
    new :class:`LoRALinear`, which also serves as the canonical layer name
    used by :func:`~clora.lora_io.LoraWeights` for these adapters.
    """
    replaced: Dict[str, LoRALinear] = {}
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(s in name for s in target_substrings)
    ]
    for name in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        base = getattr(parent, attr)
        wrapped = LoRALinear(base, rank=rank, alpha=alpha)
        setattr(parent, attr, wrapped)
        replaced[name] = wrapped
    return replaced


def extract_lora_weights(replaced: Dict[str, LoRALinear], format: str = "diffusers") -> LoraWeights:
    """Snapshot all injected adapters into a single :class:`LoraWeights`."""
    layers = {name: module.to_layer(name) for name, module in replaced.items()}
    return LoraWeights(layers=layers, format=format)


@contextmanager
def apply_loras(model: nn.Module, loras: Sequence[LoraWeights]):
    """Temporarily add the (summed) deltas of ``loras`` onto matching
    ``nn.Linear`` submodules of ``model`` via forward hooks, without
    mutating any weights or requiring gradients. Layer matching is by dotted
    module name (as returned by ``model.named_modules()``).
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []
    try:
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            deltas = [lw.layers[name].delta() for lw in loras if name in lw]
            if not deltas:
                continue
            total_delta = sum(deltas)

            def _hook(_mod, inputs, output, _delta=total_delta):
                return output + inputs[0] @ _delta.T

            handles.append(module.register_forward_hook(_hook))
        yield model
    finally:
        for h in handles:
            h.remove()
