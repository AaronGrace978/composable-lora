"""composable-lora (clora): measure and eliminate interference between stacked
LoRA adapters on image diffusion models.

See the submodules:

- ``clora.lora_io``   -- load/save LoRA weights (kohya + diffusers/peft formats).
- ``clora.overlap``   -- per-layer subspace-overlap / interference metrics.
- ``clora.merge``     -- baseline and orthogonal-reprojection LoRA merging.
- ``clora.train``     -- toy-CPU LoRA training with a composability penalty.
- ``clora.eval``      -- stacked-vs-standalone evaluation with bootstrap CIs.
- ``clora.toy_model`` -- tiny randomly-initialized stand-in UNet/transformer.
- ``clora.lora_nn``   -- trainable LoRA nn.Module injection utilities.
- ``clora.hf_models`` -- real SDXL/Flux + peft integration (lazy imports).
- ``clora.cli``       -- the ``clora`` command line tool.
"""

__version__ = "0.1.0"
