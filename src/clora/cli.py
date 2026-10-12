"""The ``clora`` command line tool.

    clora overlap a.safetensors b.safetensors [c.safetensors ...]
    clora merge --method ortho a.safetensors b.safetensors c.safetensors --out merged.safetensors
    clora eval --base sdxl --loras a.safetensors b.safetensors --prompts prompts.yaml
    clora train --config configs/train_toy.yaml --compose-against a.safetensors --output out.safetensors
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from . import eval as eval_mod
from .lora_io import load_lora, save_lora
from .merge import MERGE_METHODS, merge as merge_fn
from .overlap import all_pairs_overlap, pairwise_layer_overlap
from .toy_model import build_toy_model
from .train import ComposabilityConfig, TrainConfig, load_train_config, train as train_fn

app = typer.Typer(
    help="clora: measure and eliminate interference between stacked LoRA adapters.",
    no_args_is_help=True,
)


@app.command()
def overlap(
    lora_paths: List[Path] = typer.Argument(..., help="2 or more LoRA .safetensors files"),
    per_layer: bool = typer.Option(False, help="also print per-layer overlap scores"),
    json_out: Optional[Path] = typer.Option(None, "--json", help="write results as JSON"),
) -> None:
    """Measure pairwise subspace-overlap / interference scores between LoRAs."""
    if len(lora_paths) < 2:
        raise typer.BadParameter("Provide at least 2 LoRA files")
    loras = {p.stem: load_lora(p) for p in lora_paths}
    pair_scores = all_pairs_overlap(loras)

    results: dict = {"pairs": []}
    for (name_a, name_b), score in pair_scores.items():
        typer.echo(f"{name_a:>20} vs {name_b:<20} overlap={score:.4f}")
        entry = {"a": name_a, "b": name_b, "overlap": score}
        if per_layer:
            layer_scores = pairwise_layer_overlap(loras[name_a], loras[name_b])
            entry["per_layer"] = layer_scores
            for layer_name, layer_score in sorted(layer_scores.items()):
                typer.echo(f"    {layer_name}: {layer_score:.4f}")
        results["pairs"].append(entry)

    if json_out:
        json_out.write_text(json.dumps(results, indent=2))
        typer.echo(f"wrote {json_out}")


@app.command("merge")
def merge_cmd(
    lora_paths: List[Path] = typer.Argument(..., help="2 or more LoRA .safetensors files"),
    method: str = typer.Option("ortho", "--method", help=f"one of {sorted(MERGE_METHODS)}"),
    out: Path = typer.Option(..., "--out", help="output .safetensors path"),
    weight: Optional[List[float]] = typer.Option(
        None, "--weight", help="per-LoRA weight, repeatable (default: 1.0 each)"
    ),
    rank: Optional[int] = typer.Option(
        None, "--rank", help="target rank for naive_sum/weighted_sum/ties"
    ),
    trim_frac: float = typer.Option(0.2, "--trim-frac", help="trim fraction for ties"),
    grow_rank: bool = typer.Option(False, "--grow-rank", help="grow-rank mode for ortho"),
    out_format: Optional[str] = typer.Option(
        None, "--format", help="output key format: kohya|diffusers (default: first input's format)"
    ),
) -> None:
    """Merge 2+ LoRAs with a baseline method or our orthogonal reprojection."""
    if len(lora_paths) < 2:
        raise typer.BadParameter("Provide at least 2 LoRA files")
    loras = [load_lora(p) for p in lora_paths]

    kwargs: dict = {}
    if weight:
        kwargs["weights"] = weight
    if method in ("naive_sum", "weighted_sum", "ties") and rank is not None:
        kwargs["target_rank"] = rank
    if method == "ties":
        kwargs["trim_frac"] = trim_frac
    if method == "ortho":
        kwargs["grow_rank"] = grow_rank

    merged = merge_fn(method, loras, **kwargs)
    save_lora(merged, out, format=out_format)
    typer.echo(f"wrote merged LoRA ({method}) with {len(merged)} layers to {out}")


@app.command("eval")
def eval_cmd(
    base: str = typer.Option("toy", "--base", help="toy|sdxl|flux (sdxl/flux not yet wired up)"),
    lora_paths: List[Path] = typer.Option(..., "--loras", help="LoRA .safetensors files to stack"),
    prompts: Path = typer.Option(..., "--prompts", help="prompts.yaml with a prompts: [...] list"),
    out: Optional[Path] = typer.Option(None, "--out", help="write results as JSON"),
    channels: int = typer.Option(16, "--channels"),
    cond_dim: int = typer.Option(16, "--cond-dim"),
    depth: int = typer.Option(2, "--depth"),
    bootstrap_n: int = typer.Option(1000, "--bootstrap-n"),
) -> None:
    """Evaluate stacked-vs-standalone quality for a set of LoRAs."""
    if base != "toy":
        raise typer.BadParameter(
            f"base={base!r} is not implemented yet; use --base toy for the offline CPU model, "
            "or use clora.hf_models directly for real SDXL/Flux pipelines."
        )
    loras = {p.stem: load_lora(p) for p in lora_paths}
    model = build_toy_model(channels=channels, cond_dim=cond_dim, depth=depth)
    prompt_specs = eval_mod.load_prompts(prompts)

    result = eval_mod.evaluate(model, loras, prompt_specs, bootstrap_n=bootstrap_n)
    typer.echo(json.dumps(result.as_dict(), indent=2))
    if out:
        out.write_text(json.dumps(result.as_dict(), indent=2))
        typer.echo(f"wrote {out}")


@app.command("train")
def train_cmd(
    config: Optional[Path] = typer.Option(None, "--config", help="YAML TrainConfig"),
    output: Path = typer.Option(..., "--output", help="output .safetensors path"),
    compose_against: List[Path] = typer.Option(
        [], "--compose-against", help="frozen LoRA .safetensors files to stay orthogonal to"
    ),
    rank: int = typer.Option(4, "--rank"),
    alpha: float = typer.Option(4.0, "--alpha"),
    steps: int = typer.Option(50, "--steps"),
    lr: float = typer.Option(1e-3, "--lr"),
    group_sparsity_weight: float = typer.Option(0.0, "--group-sparsity-weight"),
    orth_weight: float = typer.Option(0.0, "--orth-weight"),
) -> None:
    """Train toy LoRA adapters, optionally with a composability penalty."""
    if config:
        cfg = load_train_config(config)
    else:
        cfg = TrainConfig(rank=rank, alpha=alpha, steps=steps, lr=lr)

    if compose_against:
        cfg.compose.compose_against = [str(p) for p in compose_against]
    if group_sparsity_weight:
        cfg.compose.group_sparsity_weight = group_sparsity_weight
    if orth_weight:
        cfg.compose.orth_weight = orth_weight
    cfg.output = str(output)

    lora = train_fn(cfg)
    typer.echo(f"trained {len(lora)} LoRA layers, wrote {output}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
