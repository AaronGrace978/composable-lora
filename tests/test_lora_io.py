import pytest
import torch

from clora.lora_io import (
    LoraLayer,
    LoraWeights,
    detect_format,
    load_lora,
    load_lora_state_dict,
    save_lora,
)


def _kohya_state_dict(rank=2, in_f=6, out_f=4, alpha=3.0, base="lora_unet_block0_attnq", seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(rank, in_f, generator=g)
    B = torch.randn(out_f, rank, generator=g)
    sd = {
        f"{base}.lora_down.weight": A,
        f"{base}.lora_up.weight": B,
        f"{base}.alpha": torch.tensor(alpha),
    }
    return sd, A, B


def _diffusers_state_dict(rank=2, in_f=6, out_f=4, alpha=3.0, base="unet.block0.attnq", seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(rank, in_f, generator=g)
    B = torch.randn(out_f, rank, generator=g)
    sd = {
        f"{base}.lora_A.weight": A,
        f"{base}.lora_B.weight": B,
        f"{base}.alpha": torch.tensor(alpha),
    }
    return sd, A, B


def test_detect_format_kohya():
    sd, _, _ = _kohya_state_dict()
    assert detect_format(sd.keys()) == "kohya"


def test_detect_format_diffusers():
    sd, _, _ = _diffusers_state_dict()
    assert detect_format(sd.keys()) == "diffusers"


def test_detect_format_unknown_raises():
    with pytest.raises(ValueError):
        detect_format(["some.random.key"])


def test_kohya_parse_canonical_name_and_values():
    sd, A, B = _kohya_state_dict()
    lora = load_lora_state_dict(sd)
    assert lora.format == "kohya"
    assert "unet.block0.attnq" in lora.layers
    layer = lora.layers["unet.block0.attnq"]
    assert torch.allclose(layer.A, A)
    assert torch.allclose(layer.B, B)
    assert layer.alpha == pytest.approx(3.0)
    assert layer.rank == 2


def test_diffusers_parse_canonical_name_and_values():
    sd, A, B = _diffusers_state_dict()
    lora = load_lora_state_dict(sd)
    assert lora.format == "diffusers"
    assert "unet.block0.attnq" in lora.layers
    layer = lora.layers["unet.block0.attnq"]
    assert torch.allclose(layer.A, A)
    assert torch.allclose(layer.B, B)
    assert layer.alpha == pytest.approx(3.0)


def test_kohya_round_trip(tmp_path):
    sd, A, B = _kohya_state_dict()
    lora = load_lora_state_dict(sd)
    path = tmp_path / "a.safetensors"
    save_lora(lora, path)
    reloaded = load_lora(path)

    assert reloaded.format == "kohya"
    assert set(reloaded.layers) == set(lora.layers)
    for name, layer in lora.layers.items():
        other = reloaded.layers[name]
        assert torch.allclose(other.A, layer.A)
        assert torch.allclose(other.B, layer.B)
        assert other.alpha == pytest.approx(layer.alpha)
        assert other.orig_key == layer.orig_key


def test_diffusers_round_trip(tmp_path):
    sd, A, B = _diffusers_state_dict()
    lora = load_lora_state_dict(sd)
    path = tmp_path / "a.safetensors"
    save_lora(lora, path)
    reloaded = load_lora(path)

    assert reloaded.format == "diffusers"
    assert set(reloaded.layers) == set(lora.layers)
    for name, layer in lora.layers.items():
        other = reloaded.layers[name]
        assert torch.allclose(other.A, layer.A)
        assert torch.allclose(other.B, layer.B)
        assert other.alpha == pytest.approx(layer.alpha)


def test_diffusers_round_trip_with_adapter_name(tmp_path):
    from clora.lora_io import to_state_dict

    g = torch.Generator().manual_seed(1)
    A = torch.randn(2, 6, generator=g)
    B = torch.randn(4, 2, generator=g)
    sd = {
        "unet.block0.attnq.lora_A.default.weight": A,
        "unet.block0.attnq.lora_B.default.weight": B,
        "unet.block0.attnq.alpha": torch.tensor(2.0),
    }
    lora = load_lora_state_dict(sd)
    layer = lora.layers["unet.block0.attnq"]
    assert layer.adapter == "default"

    path = tmp_path / "a.safetensors"
    save_lora(lora, path)
    reloaded = load_lora(path)
    reloaded_sd = to_state_dict(reloaded)
    assert "unet.block0.attnq.lora_A.default.weight" in reloaded_sd
    assert reloaded.layers["unet.block0.attnq"].adapter == "default"


def test_alpha_defaults_to_rank_when_missing():
    g = torch.Generator().manual_seed(2)
    A = torch.randn(3, 5, generator=g)
    B = torch.randn(4, 3, generator=g)
    sd = {
        "lora_unet_foo.lora_down.weight": A,
        "lora_unet_foo.lora_up.weight": B,
    }
    lora = load_lora_state_dict(sd)
    layer = lora.layers["unet.foo"]
    assert layer.alpha == pytest.approx(3.0)
    assert layer.scale == pytest.approx(1.0)


def test_lora_layer_rejects_rank_mismatch():
    A = torch.randn(2, 5)
    B = torch.randn(4, 3)
    with pytest.raises(ValueError):
        LoraLayer(name="x", A=A, B=B, alpha=2.0, orig_key="x")


def test_lora_weights_container_helpers():
    A = torch.randn(2, 5)
    B = torch.randn(4, 2)
    layer = LoraLayer(name="x", A=A, B=B, alpha=2.0, orig_key="x")
    lw = LoraWeights(layers={"x": layer})
    assert len(lw) == 1
    assert "x" in lw
    assert lw["x"] is layer
    assert lw.names() == {"x"}
