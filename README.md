# composable-lora

Tools and research code for **measuring and eliminating interference when stacking multiple LoRA adapters** on image diffusion models (SDXL and Flux, via Hugging Face `diffusers` + `peft`).

## The idea

A LoRA adds a low-rank update `ΔW = (alpha / r) · B @ A` to each adapted layer. When you stack several LoRAs, their updates add. They compose cleanly when they occupy **non-overlapping subspaces** of each layer, and interfere when they collide. This repo is built around that hypothesis:

1. **Measure** the overlap between LoRAs, per layer, directly from the factors (never materializing `ΔW`).
2. **Train** LoRAs that stay composable by design (group sparsity over rank slots + orthogonality against frozen LoRAs).
3. **Re-project** existing LoRAs into compatible subspaces without retraining.
4. **Evaluate** stacked quality rigorously, with bootstrap confidence intervals.

The hypothesis is not yet validated on real models; see [ROADMAP.md](ROADMAP.md).

## Install

```bash
pip install -e ".[dev]"
```

For CPU-only torch (avoids large CUDA downloads), install torch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

## Quickstart

Everything below runs on CPU with a tiny randomly initialized model and no downloads.

```bash
# Train two toy LoRAs
clora train --output a.safetensors --steps 100 --rank 4
clora train --output b.safetensors --steps 100 --rank 4 \
    --compose-against a.safetensors --orth-weight 5.0

# Measure overlap
clora overlap a.safetensors b.safetensors --per-layer

# Merge (methods: concat, naive_sum, weighted_sum, ties, ortho)
clora merge --method ortho a.safetensors b.safetensors --out merged.safetensors

# Evaluate stacked vs standalone on the toy model
clora eval --base toy --loras a.safetensors b.safetensors --prompts configs/prompts_example.yaml
```

Run the overlap-vs-interference experiment and plot:

```bash
python scripts/overlap_vs_interference.py --n 6 --out overlap_vs_interference.png
```

Run the tests:

```bash
pytest
```

## Modules

| Module | Purpose |
| --- | --- |
| `clora.lora_io` | Load/save LoRA safetensors in kohya and diffusers/peft key formats, normalized to per-layer `(A, B, alpha)`. |
| `clora.overlap` | Principal angles between column/row spaces, normalized Frobenius overlap, per-pair aggregate score. |
| `clora.merge` | Baselines (concat, naive/weighted sum, TIES-style) and orthogonal reprojection with optional rank growth. |
| `clora.train` | Toy-CPU LoRA training with group-sparsity and orthogonality penalties. |
| `clora.eval` | Stacked-vs-standalone scoring with 95% bootstrap intervals. |
| `clora.hf_models` | Real SDXL/Flux + peft + CLIP/DINOv2 helpers (lazy imports, opt-in). |
| `clora.cli` | The `clora` command. |

## Current limitations

- Training and evaluation are implemented against the toy model only. The evaluation scorers on the toy path are random projections that exercise the pipeline; they are not CLIP/DINOv2/ArcFace. Real-model helpers exist in `clora.hf_models` but are not yet wired into `clora train` / `clora eval` (`--base sdxl|flux` is rejected for now).
- kohya key names are mapped to dotted names heuristically (see `clora.lora_io`), so cross-format layer matching is best-effort. Saving back to the format a LoRA was loaded from is exact.
- `merge_orthogonal` is order-dependent: earlier LoRAs are left untouched and later ones are projected against them.

## Results

No experiments have been run on real models. Tables are intentionally empty until they are.

| Experiment | Setting | Result |
| --- | --- | --- |
| Overlap vs interference correlation | 20+ public LoRAs | _not run_ |
| Composable training, 5+ stacked LoRAs | SDXL / Flux | _not run_ |
| Reprojection vs naive / TIES merge | SDXL / Flux | _not run_ |