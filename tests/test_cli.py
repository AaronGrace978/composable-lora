import json

from typer.testing import CliRunner

from clora.cli import app
from clora.lora_io import LoraLayer, LoraWeights, save_lora
import torch

runner = CliRunner()


def _write_lora(path, seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(2, 4, generator=g)
    B = torch.randn(3, 2, generator=g)
    layer = LoraLayer(name="unet.block0.attnq", A=A, B=B, alpha=2.0, orig_key="unet.block0.attnq")
    lora = LoraWeights(layers={"unet.block0.attnq": layer}, format="diffusers")
    save_lora(lora, path)


def test_cli_overlap(tmp_path):
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    _write_lora(a, 0)
    _write_lora(b, 1)
    json_out = tmp_path / "out.json"

    result = runner.invoke(app, ["overlap", str(a), str(b), "--json", str(json_out)])
    assert result.exit_code == 0, result.output
    data = json.loads(json_out.read_text())
    assert len(data["pairs"]) == 1


def test_cli_merge(tmp_path):
    a = tmp_path / "a.safetensors"
    b = tmp_path / "b.safetensors"
    _write_lora(a, 0)
    _write_lora(b, 1)
    out = tmp_path / "merged.safetensors"

    result = runner.invoke(app, ["merge", str(a), str(b), "--method", "ortho", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_cli_train(tmp_path):
    out = tmp_path / "trained.safetensors"
    result = runner.invoke(
        app, ["train", "--output", str(out), "--steps", "2", "--rank", "2"]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
