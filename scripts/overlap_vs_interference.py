#!/usr/bin/env python3
"""Correlate measured subspace overlap with measured stacking interference.

Trains (or loads) a set of LoRAs, computes pairwise overlap scores via
`clora.overlap`, measures pairwise interference by generating stacked vs
standalone outputs via `clora.eval`, and reports/plots overlap vs
interference with a Pearson correlation across all pairs.

Runs entirely on the offline CPU toy model by default (no downloads):

    python scripts/overlap_vs_interference.py --n 6 --out overlap_vs_interference.png

Pass --lora-dir to use a directory of real trained LoRA `.safetensors` files
instead of the toy generator (overlap measurement works the same way either
way; interference measurement still uses the toy model/embedders unless you
adapt `measure_interference` to call a real generation pipeline from
`clora.hf_models`).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from clora.eval import PromptSpec, default_toy_embedders, generate, identity_score
from clora.lora_io import LoraWeights, load_lora
from clora.overlap import aggregate_lora_overlap
from clora.toy_model import build_toy_model
from clora.train import TrainConfig, train


def build_toy_lora_set(
    n: int, rank: int, steps: int, channels: int, cond_dim: int, depth: int, seed: int,
    lr: float = 1e-2, alpha_mult: float = 4.0,
) -> dict:
    loras = {}
    for i in range(n):
        cfg = TrainConfig(
            rank=rank,
            alpha=float(rank) * alpha_mult,
            lr=lr,
            steps=steps,
            channels=channels,
            cond_dim=cond_dim,
            depth=depth,
            seed=seed + i,
        )
        loras[f"lora{i}"] = train(cfg)
    return loras


def measure_interference(model, loras: dict, prompts: list) -> dict:
    names = list(loras)
    embedders = default_toy_embedders(channels=model.channels, cond_dim=model.cond_dim)
    prompt_cond = embedders["prompt_cond"]
    identity_emb = embedders["identity"]

    alone_imgs = {
        name: {p.prompt: generate(model, [loras[name]], prompt_cond, p.prompt, p.seed) for p in prompts}
        for name in names
    }

    results = {}
    for a, b in itertools.combinations(names, 2):
        drops = []
        for p in prompts:
            stacked = generate(model, [loras[a], loras[b]], prompt_cond, p.prompt, p.seed)
            drop_a = 1.0 - identity_score(stacked, alone_imgs[a][p.prompt], identity_emb)
            drop_b = 1.0 - identity_score(stacked, alone_imgs[b][p.prompt], identity_emb)
            drops.append((drop_a + drop_b) / 2.0)
        results[(a, b)] = float(np.mean(drops))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lora-dir", type=Path, default=None, help="dir of .safetensors to load instead of training toy LoRAs")
    parser.add_argument("--n", type=int, default=6, help="number of toy LoRAs to train")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--alpha-mult", type=float, default=4.0)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--cond-dim", type=int, default=8)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-prompts", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("overlap_vs_interference.png"))
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.lora_dir:
        loras: dict[str, LoraWeights] = {
            p.stem: load_lora(p) for p in sorted(args.lora_dir.glob("*.safetensors"))
        }
    else:
        loras = build_toy_lora_set(
            args.n, args.rank, args.steps, args.channels, args.cond_dim, args.depth, args.seed,
            lr=args.lr, alpha_mult=args.alpha_mult,
        )

    model = build_toy_model(args.channels, args.cond_dim, args.depth, seed=args.seed)
    prompts = [PromptSpec(prompt=f"a toy prompt {i}", seed=i) for i in range(args.n_prompts)]

    names = list(loras)
    overlaps = {
        (a, b): aggregate_lora_overlap(loras[a], loras[b])
        for a, b in itertools.combinations(names, 2)
    }
    interference = measure_interference(model, loras, prompts)

    pairs = list(overlaps)
    x = np.array([overlaps[p] for p in pairs])
    y = np.array([interference[p] for p in pairs])
    corr = float(np.corrcoef(x, y)[0, 1]) if len(pairs) > 1 else float("nan")

    print(f"Pearson correlation (overlap vs interference): {corr:.4f}")
    for a, b in pairs:
        print(f"{a:>10} vs {b:<10} overlap={overlaps[(a, b)]:.4f} interference={interference[(a, b)]:.4f}")

    if args.json_out:
        payload = {
            "pairs": [
                {"a": a, "b": b, "overlap": overlaps[(a, b)], "interference": interference[(a, b)]}
                for a, b in pairs
            ],
            "pearson_r": corr,
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.json_out}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(x, y)
        ax.set_xlabel("measured subspace overlap")
        ax.set_ylabel("measured interference (score drop)")
        ax.set_title(f"overlap vs interference (r={corr:.2f}, n={len(pairs)} pairs)")
        fig.tight_layout()
        fig.savefig(args.out, dpi=150)
        print(f"saved plot to {args.out}")
    except ImportError:
        print("matplotlib not installed; skipping plot (pip install 'composable-lora[plots]')")


if __name__ == "__main__":
    main()
