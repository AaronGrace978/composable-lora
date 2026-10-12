"""Real SDXL/Flux model + peft integration (lazy imports).

Everything in this module imports ``diffusers``/``peft``/``transformers``
*inside* function bodies, never at module scope, so that importing
``clora`` (and running the toy/offline test suite) never requires these
heavy dependencies to actually succeed at import time even though they are
listed in ``pyproject.toml``. Functions here are only invoked when a
flag/config explicitly asks for real models (e.g. ``--real`` /
``base: sdxl`` or ``base: flux``), and typically require network access to
download pretrained checkpoints the first time they run.
"""

from __future__ import annotations

from typing import Sequence

from .lora_io import LoraWeights, load_lora_state_dict

DEFAULT_SDXL_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
DEFAULT_FLUX_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0", "proj_mlp", "proj_out"]


def load_sdxl_unet(
    model_id: str = "stabilityai/stable-diffusion-xl-base-1.0", subfolder: str = "unet", **kwargs
):
    """Load a real SDXL UNet via ``diffusers``. Requires network access
    unless the model is already cached locally.
    """
    from diffusers import UNet2DConditionModel

    return UNet2DConditionModel.from_pretrained(model_id, subfolder=subfolder, **kwargs)


def load_flux_transformer(
    model_id: str = "black-forest-labs/FLUX.1-dev", subfolder: str = "transformer", **kwargs
):
    """Load a real Flux transformer via ``diffusers``. Requires network
    access unless the model is already cached locally.
    """
    from diffusers import FluxTransformer2DModel

    return FluxTransformer2DModel.from_pretrained(model_id, subfolder=subfolder, **kwargs)


def load_sdxl_pipeline(model_id: str = "stabilityai/stable-diffusion-xl-base-1.0", **kwargs):
    from diffusers import StableDiffusionXLPipeline

    return StableDiffusionXLPipeline.from_pretrained(model_id, **kwargs)


def load_flux_pipeline(model_id: str = "black-forest-labs/FLUX.1-dev", **kwargs):
    from diffusers import FluxPipeline

    return FluxPipeline.from_pretrained(model_id, **kwargs)


def inject_peft_lora(
    model,
    rank: int = 16,
    alpha: int = 16,
    target_modules: Sequence[str] = DEFAULT_SDXL_TARGET_MODULES,
):
    """Wrap `model` with a peft :class:`~peft.LoraConfig` adapter."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=rank, lora_alpha=alpha, target_modules=list(target_modules), bias="none")
    return get_peft_model(model, config)


def extract_peft_lora_weights(peft_model, format: str = "diffusers") -> LoraWeights:
    """Snapshot a peft-wrapped model's adapter weights into a :class:`LoraWeights`."""
    from peft import get_peft_model_state_dict

    state_dict = get_peft_model_state_dict(peft_model)
    return load_lora_state_dict(state_dict, format=format)


def load_clip_score_fn(model_id: str = "openai/clip-vit-base-patch32"):
    """Return a ``(images, prompts) -> scores`` CLIP-score function. Requires
    network access to download the CLIP checkpoint the first time it runs.
    """
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(model_id)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()

    def score(images, prompts: Sequence[str]):
        import torch

        inputs = processor(text=list(prompts), images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        return (image_embeds * text_embeds).sum(dim=-1)

    return score


def load_dino_embedder(model_id: str = "facebook/dinov2-small"):
    """Return an ``image -> embedding`` function using DINOv2, for identity
    preservation scoring. Requires network access to download the checkpoint
    the first time it runs.
    """
    from transformers import AutoImageProcessor, AutoModel

    model = AutoModel.from_pretrained(model_id)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model.eval()

    def embed(image):
        import torch

        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        pooled = outputs.last_hidden_state[:, 0]
        return pooled / pooled.norm(dim=-1, keepdim=True)

    return embed
