"""Stacked-vs-standalone LoRA evaluation.

For a fixed prompt/seed grid, generates each LoRA alone and all of them
stacked together, then scores:

- **identity preservation**: embedding similarity between a stacked
  generation and that LoRA's own standalone generation (stand-in for
  DINOv2/ArcFace face/object similarity).
- **style fidelity**: likewise, with a separate style embedder (stand-in for
  CLIP/CSD style similarity).
- **prompt adherence**: embedding similarity between the stacked generation
  and the prompt text (stand-in for CLIP score).
- **interference**: the drop in a LoRA's own identity/style score when
  stacked vs. alone (alone is 1.0 by construction).

All scores are reported as means with 95%-bootstrap confidence intervals
(:func:`bootstrap_ci`).

The default embedders and image generator are tiny, deterministic,
hash-seeded random projections over :class:`~clora.toy_model.ToyUNet`
outputs -- enough to validate the full pipeline and the overlap/interference
correlation experiment entirely offline. Real CLIP/DINOv2/ArcFace scorers
are available via :mod:`clora.hf_models` (lazy imports, only used when
requested), since they require network access to download pretrained
checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml
from torch import nn

from .lora_io import LoraWeights
from .lora_nn import apply_loras

Embedder = Callable[[torch.Tensor], torch.Tensor]
TextEmbedder = Callable[[str], torch.Tensor]
PromptConditioner = Callable[[str, int], torch.Tensor]


@dataclass
class PromptSpec:
    prompt: str
    seed: int


def load_prompts(path: "str | Path") -> List[PromptSpec]:
    """Load a prompt/seed grid from a YAML file: ``{"prompts": [{"prompt":
    str, "seed": int}, ...]}``.
    """
    data = yaml.safe_load(Path(path).read_text()) or {}
    entries = data.get("prompts", [])
    return [PromptSpec(prompt=e["prompt"], seed=int(e.get("seed", 0))) for e in entries]


@dataclass
class ScoreResult:
    mean: float
    ci_low: float
    ci_high: float

    def as_dict(self) -> Dict[str, float]:
        return {"mean": self.mean, "ci_low": self.ci_low, "ci_high": self.ci_high}


def bootstrap_ci(
    values: Sequence[float], n_resamples: int = 1000, ci: float = 0.95, seed: int = 0
) -> ScoreResult:
    """Mean with a percentile bootstrap confidence interval."""
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return ScoreResult(mean=float("nan"), ci_low=float("nan"), ci_high=float("nan"))
    rng = np.random.default_rng(seed)
    n = values.size
    resample_means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        resample_means[i] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(resample_means, [alpha, 1.0 - alpha])
    return ScoreResult(mean=float(values.mean()), ci_low=float(lo), ci_high=float(hi))


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = (a.norm() * b.norm()).clamp_min(1e-8)
    return float((a @ b) / denom)


def _hash_seed(*parts) -> int:
    return abs(hash(parts)) % (2**31)


def make_toy_embedder(in_dim: int, out_dim: int = 16, seed: int = 0) -> Embedder:
    """A fixed random linear projection used as a stand-in image embedder."""
    generator = torch.Generator().manual_seed(seed)
    proj = torch.randn(out_dim, in_dim, generator=generator)

    def embed(image: torch.Tensor) -> torch.Tensor:
        flat = image.flatten()
        if flat.numel() < in_dim:
            flat = torch.nn.functional.pad(flat, (0, in_dim - flat.numel()))
        else:
            flat = flat[:in_dim]
        v = proj @ flat
        return v / v.norm().clamp_min(1e-8)

    return embed


def make_toy_text_embedder(out_dim: int = 16, seed: int = 0) -> TextEmbedder:
    """A deterministic hash-seeded random embedding for prompt strings."""

    def embed(prompt: str) -> torch.Tensor:
        generator = torch.Generator().manual_seed(_hash_seed("text", prompt, seed))
        v = torch.randn(out_dim, generator=generator)
        return v / v.norm().clamp_min(1e-8)

    return embed


def make_toy_prompt_conditioner(cond_len: int, cond_dim: int, seed: int = 0) -> PromptConditioner:
    """A deterministic hash-seeded random conditioning tensor for a prompt+seed."""

    def cond_fn(prompt: str, prompt_seed: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(_hash_seed("cond", prompt, prompt_seed, seed))
        return torch.randn(1, cond_len, cond_dim, generator=generator)

    return cond_fn


def default_toy_embedders(
    channels: int = 16, seq_len: int = 4, cond_len: int = 4, cond_dim: int = 16, seed: int = 0
) -> Dict[str, Callable]:
    image_dim = channels * seq_len
    return {
        "identity": make_toy_embedder(image_dim, out_dim=16, seed=seed + 1),
        "style": make_toy_embedder(image_dim, out_dim=16, seed=seed + 2),
        "text": make_toy_text_embedder(out_dim=16, seed=seed + 3),
        "prompt_cond": make_toy_prompt_conditioner(cond_len, cond_dim, seed=seed),
    }


def generate(
    model: nn.Module,
    loras: Sequence[LoraWeights],
    prompt_cond: PromptConditioner,
    prompt: str,
    seed: int,
    seq_len: int = 4,
) -> torch.Tensor:
    """Generate one deterministic toy "image" tensor for a prompt+seed,
    with the given LoRAs (none, one, or several stacked) applied.
    """
    cond = prompt_cond(prompt, seed)
    generator = torch.Generator().manual_seed(_hash_seed("latent", prompt, seed))
    x = torch.randn(1, seq_len, model.channels, generator=generator)
    with apply_loras(model, loras):
        with torch.no_grad():
            out = model(x, cond)
    return out


def identity_score(img_a: torch.Tensor, img_b: torch.Tensor, embedder: Embedder) -> float:
    return cosine_similarity(embedder(img_a), embedder(img_b))


@dataclass
class EvalResult:
    identity_interference: Dict[str, ScoreResult]
    style_interference: Dict[str, ScoreResult]
    prompt_adherence: ScoreResult

    def as_dict(self) -> Dict[str, object]:
        return {
            "identity_interference": {k: v.as_dict() for k, v in self.identity_interference.items()},
            "style_interference": {k: v.as_dict() for k, v in self.style_interference.items()},
            "prompt_adherence": self.prompt_adherence.as_dict(),
        }


def evaluate(
    model: nn.Module,
    loras: Dict[str, LoraWeights],
    prompts: Sequence[PromptSpec],
    embedders: Optional[Dict[str, Callable]] = None,
    bootstrap_n: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> EvalResult:
    """Evaluate stacked-vs-standalone quality for a set of named LoRAs."""
    embedders = embedders or default_toy_embedders(
        channels=model.channels, cond_dim=model.cond_dim, seed=seed
    )
    identity_emb = embedders["identity"]
    style_emb = embedders["style"]
    text_emb = embedders["text"]
    prompt_cond = embedders["prompt_cond"]

    names = list(loras)
    identity_drop: Dict[str, List[float]] = {n: [] for n in names}
    style_drop: Dict[str, List[float]] = {n: [] for n in names}
    adherence: List[float] = []

    for ps in prompts:
        alone_imgs = {
            n: generate(model, [loras[n]], prompt_cond, ps.prompt, ps.seed) for n in names
        }
        stacked_img = generate(
            model, [loras[n] for n in names], prompt_cond, ps.prompt, ps.seed
        )
        for n in names:
            id_stacked = identity_score(stacked_img, alone_imgs[n], identity_emb)
            identity_drop[n].append(1.0 - id_stacked)
            style_stacked = identity_score(stacked_img, alone_imgs[n], style_emb)
            style_drop[n].append(1.0 - style_stacked)
        adherence.append(cosine_similarity(text_emb(ps.prompt), identity_emb(stacked_img)))

    return EvalResult(
        identity_interference={
            n: bootstrap_ci(v, bootstrap_n, ci, seed) for n, v in identity_drop.items()
        },
        style_interference={
            n: bootstrap_ci(v, bootstrap_n, ci, seed) for n, v in style_drop.items()
        },
        prompt_adherence=bootstrap_ci(adherence, bootstrap_n, ci, seed),
    )
