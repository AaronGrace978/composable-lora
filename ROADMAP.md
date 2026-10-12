# Roadmap

Milestones are unchecked until backed by a reproducible experiment with results recorded in the README.

- [ ] **1. Overlap predicts interference.** Across 20+ public LoRAs (SDXL and/or Flux), the overlap score from `clora.overlap` correlates with measured stacking interference (drop in a LoRA's own identity/style score when stacked vs alone). Requires wiring real generation and CLIP/DINOv2 scoring into `clora eval` and `scripts/overlap_vs_interference.py`.
- [ ] **2. Composable training works.** With the composability penalty, 5+ stacked LoRAs stay within a few percent of standalone quality. Requires a real-model DreamBooth-style training path (diffusers + peft) in `clora.train`.
- [ ] **3. Training-free reprojection beats baselines.** `merge --method ortho` beats naive sum and TIES-style merging on stacked quality, with 95% bootstrap intervals.
- [ ] **4. Plugin.** A ComfyUI / diffusers plugin that applies overlap checks and reprojection when stacking LoRAs.

## Known gaps before milestone 1

- [ ] Real SDXL/Flux generation in `clora eval` (currently `--base toy` only).
- [ ] Real embedders (DINOv2 / ArcFace / CLIP / CSD) selectable from `clora eval`.
- [ ] Reliable kohya-to-diffusers layer name mapping for cross-format comparisons.
- [ ] Toy experiment: determine whether adapter magnitude confounds overlap vs interference (an initial toy run showed no positive correlation).
