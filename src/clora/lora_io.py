"""Load and save LoRA adapter weights in kohya and diffusers/peft key formats,
normalized to a common per-layer ``(A, B, alpha)`` factor representation.

Both formats store, for each adapted linear layer, a low-rank down-projection
``A`` (shape ``rank x in_features``) and up-projection ``B``
(shape ``out_features x rank``), plus a scalar ``alpha`` such that the
effective weight delta is ``(alpha / rank) * B @ A``.

Key format reference
---------------------
kohya (sd-scripts style)::

    <base>.lora_down.weight   # A, (rank, in_features)
    <base>.lora_up.weight     # B, (out_features, rank)
    <base>.alpha              # scalar tensor (optional, defaults to rank)

diffusers / peft style::

    <base>.lora_A[.<adapter>].weight   # A, (rank, in_features)
    <base>.lora_B[.<adapter>].weight   # B, (out_features, rank)
    <base>.alpha                       # scalar tensor (optional, defaults to rank)

Canonical layer names
----------------------
For diffusers-format keys, ``<base>`` is already a dotted module path and is
used directly as the canonical layer name.

For kohya-format keys, ``<base>`` uses underscores in place of dots (e.g.
``lora_unet_down_blocks_0_attn_to_q``). Reconstructing the exact original
dotted module path from this is ambiguous in general (module names like
``to_q`` already contain underscores), so this module uses a best-effort
heuristic (:func:`kohya_key_to_canonical`) that strips the known
``lora_unet_`` / ``lora_te_`` / ``lora_te1_`` / ``lora_te2_`` /
``lora_transformer_`` prefixes and replaces remaining underscores with dots.
This is good enough to match layers *within* a kohya-format file and across
multiple kohya-format files that were produced the same way, but is not
guaranteed to line up with the true diffusers module path. Save-back to the
*same* format a LoRA was loaded from is always exact, since the original
format-specific key is preserved on each :class:`LoraLayer` as ``orig_key``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch

KOHYA_DOWN_SUFFIX = ".lora_down.weight"
KOHYA_UP_SUFFIX = ".lora_up.weight"
KOHYA_ALPHA_SUFFIX = ".alpha"

DIFFUSERS_A_RE = re.compile(r"\.lora_A(?:\.(?P<adapter>[^.]+))?\.weight$")
DIFFUSERS_B_RE = re.compile(r"\.lora_B(?:\.(?P<adapter>[^.]+))?\.weight$")

KOHYA_PREFIXES = {
    "lora_unet_": "unet.",
    "lora_te1_": "text_encoder.",
    "lora_te2_": "text_encoder_2.",
    "lora_te_": "text_encoder.",
    "lora_transformer_": "transformer.",
}
# Longest canonical-prefix first, so the reverse mapping picks the most
# specific match (e.g. "text_encoder_2." before "text_encoder.").
_REV_KOHYA_PREFIXES = sorted(
    ((repl, prefix) for prefix, repl in KOHYA_PREFIXES.items()),
    key=lambda item: -len(item[0]),
)


def kohya_key_to_canonical(base: str) -> str:
    """Best-effort conversion of a kohya base key to a dotted canonical name."""
    for prefix, repl in KOHYA_PREFIXES.items():
        if base.startswith(prefix):
            rest = base[len(prefix):]
            return repl + rest.replace("_", ".")
    return base.replace("_", ".")


def canonical_to_kohya_key(name: str) -> str:
    """Best-effort inverse of :func:`kohya_key_to_canonical`."""
    for repl, prefix in _REV_KOHYA_PREFIXES:
        if name.startswith(repl):
            rest = name[len(repl):]
            return prefix + rest.replace(".", "_")
    return "lora_unet_" + name.replace(".", "_")


@dataclass
class LoraLayer:
    """Normalized per-layer LoRA factors.

    ``A`` is the down-projection (rank, in_features), ``B`` is the
    up-projection (out_features, rank). The effective weight delta is
    ``scale * B @ A`` where ``scale = alpha / rank``.
    """

    name: str
    A: torch.Tensor
    B: torch.Tensor
    alpha: float
    orig_key: str = ""
    adapter: Optional[str] = None

    def __post_init__(self) -> None:
        if self.A.dim() != 2 or self.B.dim() != 2:
            raise ValueError(f"LoRA factors for {self.name!r} must be 2D tensors")
        if self.A.shape[0] != self.B.shape[1]:
            raise ValueError(
                f"Rank mismatch for {self.name!r}: A is {tuple(self.A.shape)}, "
                f"B is {tuple(self.B.shape)}"
            )
        if not self.orig_key:
            self.orig_key = self.name

    @property
    def rank(self) -> int:
        return self.A.shape[0]

    @property
    def in_features(self) -> int:
        return self.A.shape[1]

    @property
    def out_features(self) -> int:
        return self.B.shape[0]

    @property
    def scale(self) -> float:
        return self.alpha / self.rank if self.rank else 0.0

    def delta(self) -> torch.Tensor:
        """Materialize the full ``(out_features, in_features)`` weight delta."""
        return self.scale * (self.B @ self.A)


@dataclass
class LoraWeights:
    """A collection of :class:`LoraLayer`\\ s keyed by canonical layer name."""

    layers: Dict[str, LoraLayer]
    format: str = "diffusers"
    metadata: Dict[str, str] = field(default_factory=dict)

    def __iter__(self):
        return iter(self.layers.values())

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, name: str) -> LoraLayer:
        return self.layers[name]

    def __contains__(self, name: str) -> bool:
        return name in self.layers

    def names(self):
        return set(self.layers)


def detect_format(keys: Iterable[str]) -> str:
    """Detect whether `keys` look like kohya or diffusers/peft LoRA keys."""
    keys = list(keys)
    if any(k.endswith(KOHYA_DOWN_SUFFIX) or k.endswith(KOHYA_UP_SUFFIX) for k in keys):
        return "kohya"
    if any(DIFFUSERS_A_RE.search(k) or DIFFUSERS_B_RE.search(k) for k in keys):
        return "diffusers"
    raise ValueError(
        "Could not detect LoRA key format: expected kohya-style "
        "'*.lora_down.weight' / '*.lora_up.weight' keys or diffusers/peft-style "
        "'*.lora_A[.adapter].weight' / '*.lora_B[.adapter].weight' keys."
    )


def _parse_kohya(state_dict: Dict[str, torch.Tensor]) -> Dict[str, LoraLayer]:
    bases = sorted(
        k[: -len(KOHYA_DOWN_SUFFIX)] for k in state_dict if k.endswith(KOHYA_DOWN_SUFFIX)
    )
    layers: Dict[str, LoraLayer] = {}
    for base in bases:
        up_key = base + KOHYA_UP_SUFFIX
        if up_key not in state_dict:
            raise ValueError(f"Missing {up_key!r} for kohya LoRA base {base!r}")
        A = state_dict[base + KOHYA_DOWN_SUFFIX]
        B = state_dict[up_key]
        alpha_key = base + KOHYA_ALPHA_SUFFIX
        alpha = float(state_dict[alpha_key]) if alpha_key in state_dict else float(A.shape[0])
        name = kohya_key_to_canonical(base)
        layers[name] = LoraLayer(name=name, A=A, B=B, alpha=alpha, orig_key=base)
    return layers


def _parse_diffusers(state_dict: Dict[str, torch.Tensor]) -> Dict[str, LoraLayer]:
    bases: Dict[str, Dict[str, object]] = {}
    for key in state_dict:
        m = DIFFUSERS_A_RE.search(key)
        if m:
            base = key[: m.start()]
            entry = bases.setdefault(base, {})
            entry["A_key"] = key
            entry["adapter"] = m.group("adapter")
            continue
        m = DIFFUSERS_B_RE.search(key)
        if m:
            base = key[: m.start()]
            entry = bases.setdefault(base, {})
            entry["B_key"] = key
            entry["adapter"] = m.group("adapter")

    layers: Dict[str, LoraLayer] = {}
    for base in sorted(bases):
        entry = bases[base]
        if "A_key" not in entry or "B_key" not in entry:
            continue  # incomplete pair (shouldn't normally happen)
        A = state_dict[entry["A_key"]]
        B = state_dict[entry["B_key"]]
        alpha_key = base + ".alpha"
        alpha = float(state_dict[alpha_key]) if alpha_key in state_dict else float(A.shape[0])
        layers[base] = LoraLayer(
            name=base, A=A, B=B, alpha=alpha, orig_key=base, adapter=entry.get("adapter")
        )
    return layers


def load_lora_state_dict(
    state_dict: Dict[str, torch.Tensor],
    format: Optional[str] = None,
    name: Optional[str] = None,
) -> LoraWeights:
    """Parse a raw safetensors-style state dict into a :class:`LoraWeights`."""
    fmt = format or detect_format(state_dict.keys())
    if fmt == "kohya":
        layers = _parse_kohya(state_dict)
    elif fmt == "diffusers":
        layers = _parse_diffusers(state_dict)
    else:
        raise ValueError(f"Unknown LoRA format {fmt!r}, expected 'kohya' or 'diffusers'")
    metadata = {"name": name} if name else {}
    return LoraWeights(layers=layers, format=fmt, metadata=metadata)


def load_lora(path: "str | Path", format: Optional[str] = None) -> LoraWeights:
    """Load a LoRA from a ``.safetensors`` file, auto-detecting its key format."""
    from safetensors.torch import load_file

    path = Path(path)
    state_dict = load_file(str(path))
    return load_lora_state_dict(state_dict, format=format, name=path.stem)


def _to_kohya_state_dict(lora: LoraWeights) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}
    for layer in lora.layers.values():
        base = layer.orig_key if lora.format == "kohya" else canonical_to_kohya_key(layer.name)
        state_dict[base + KOHYA_DOWN_SUFFIX] = layer.A.contiguous()
        state_dict[base + KOHYA_UP_SUFFIX] = layer.B.contiguous()
        state_dict[base + KOHYA_ALPHA_SUFFIX] = torch.tensor(float(layer.alpha))
    return state_dict


def _to_diffusers_state_dict(lora: LoraWeights) -> Dict[str, torch.Tensor]:
    state_dict: Dict[str, torch.Tensor] = {}
    for layer in lora.layers.values():
        base = layer.orig_key if lora.format == "diffusers" else layer.name
        adapter_part = f".{layer.adapter}" if layer.adapter else ""
        state_dict[f"{base}.lora_A{adapter_part}.weight"] = layer.A.contiguous()
        state_dict[f"{base}.lora_B{adapter_part}.weight"] = layer.B.contiguous()
        state_dict[f"{base}.alpha"] = torch.tensor(float(layer.alpha))
    return state_dict


def to_state_dict(lora: LoraWeights, format: Optional[str] = None) -> Dict[str, torch.Tensor]:
    """Serialize a :class:`LoraWeights` back to a raw safetensors-style state dict."""
    fmt = format or lora.format
    if fmt == "kohya":
        return _to_kohya_state_dict(lora)
    if fmt == "diffusers":
        return _to_diffusers_state_dict(lora)
    raise ValueError(f"Unknown LoRA format {fmt!r}, expected 'kohya' or 'diffusers'")


def save_lora(lora: LoraWeights, path: "str | Path", format: Optional[str] = None) -> None:
    """Save a :class:`LoraWeights` to a ``.safetensors`` file."""
    from safetensors.torch import save_file

    state_dict = to_state_dict(lora, format=format)
    save_file(state_dict, str(path))
