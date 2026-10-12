import torch

from clora.lora_io import save_lora
from clora.overlap import layer_overlap_score
from clora.train import ComposabilityConfig, TrainConfig, train


def test_train_returns_lora_with_expected_rank():
    cfg = TrainConfig(rank=3, alpha=3.0, steps=3, channels=8, cond_dim=8, depth=1, seed=0)
    lora = train(cfg)
    assert len(lora) > 0
    for layer in lora:
        assert layer.rank == 3


def test_train_writes_output_file(tmp_path):
    out = tmp_path / "trained.safetensors"
    cfg = TrainConfig(
        rank=2, alpha=2.0, steps=2, channels=8, cond_dim=8, depth=1, seed=0, output=str(out)
    )
    train(cfg)
    assert out.exists()


def test_composability_penalty_decreases_overlap_with_frozen_lora(tmp_path):
    frozen_cfg = TrainConfig(rank=2, alpha=2.0, steps=8, channels=8, cond_dim=8, depth=1, seed=0)
    frozen_lora = train(frozen_cfg)
    frozen_path = tmp_path / "frozen.safetensors"
    save_lora(frozen_lora, frozen_path)

    def mean_overlap(lora):
        scores = [
            layer_overlap_score(lora.layers[name], frozen_lora.layers[name])
            for name in lora.layers
            if name in frozen_lora.layers
        ]
        return sum(scores) / len(scores)

    no_penalty_cfg = TrainConfig(
        rank=2,
        alpha=2.0,
        steps=25,
        channels=8,
        cond_dim=8,
        depth=1,
        seed=1,
        compose=ComposabilityConfig(),
    )
    with_penalty_cfg = TrainConfig(
        rank=2,
        alpha=2.0,
        steps=25,
        channels=8,
        cond_dim=8,
        depth=1,
        seed=1,
        compose=ComposabilityConfig(orth_weight=8.0, compose_against=[str(frozen_path)]),
    )

    lora_no_penalty = train(no_penalty_cfg)
    lora_with_penalty = train(with_penalty_cfg)

    assert mean_overlap(lora_with_penalty) < mean_overlap(lora_no_penalty)


def test_group_sparsity_penalty_shrinks_total_group_norm():
    def total_group_norm(lora):
        total = 0.0
        for layer in lora:
            norms = torch.sqrt((layer.A**2).sum(dim=1) + (layer.B**2).sum(dim=0))
            total += norms.sum().item()
        return total

    plain_cfg = TrainConfig(
        rank=4, alpha=4.0, steps=20, channels=8, cond_dim=8, depth=1, seed=2,
        compose=ComposabilityConfig(),
    )
    sparse_cfg = TrainConfig(
        rank=4, alpha=4.0, steps=20, channels=8, cond_dim=8, depth=1, seed=2,
        compose=ComposabilityConfig(group_sparsity_weight=0.05),
    )

    lora_plain = train(plain_cfg)
    lora_sparse = train(sparse_cfg)

    assert total_group_norm(lora_sparse) < total_group_norm(lora_plain)
