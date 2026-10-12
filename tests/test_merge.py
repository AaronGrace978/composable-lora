import pytest
import torch

from clora.lora_io import LoraLayer, LoraWeights
from clora.merge import (
    merge_concat,
    merge_naive_sum,
    merge_orthogonal,
    merge_symmetric,
    merge_ties,
    merge_weighted_sum,
    sum_of_deltas,
)


def _layer(A, B, alpha, name="x"):
    return LoraLayer(name=name, A=A, B=B, alpha=alpha, orig_key=name)


def make_orthogonal_loras(out_f=8, in_f=6, ranks=(2, 2, 2), seed=0):
    """3 LoRAs on one shared layer whose column spaces (in output space) are
    pairwise orthogonal by construction."""
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(out_f, out_f, generator=g))
    loras = []
    idx = 0
    for i, r in enumerate(ranks):
        B = Q[:, idx : idx + r].clone()
        idx += r
        A = torch.randn(r, in_f, generator=g)
        loras.append(LoraWeights(layers={"layer0": _layer(A, B, float(r), "layer0")}))
    return loras


def test_merge_concat_is_exact_sum_of_deltas():
    loras = make_orthogonal_loras()
    merged = merge_concat(loras)
    expected = sum_of_deltas(loras)["layer0"]
    assert torch.allclose(merged.layers["layer0"].delta(), expected, atol=1e-5)


def test_merge_orthogonal_exact_when_inputs_already_orthogonal():
    loras = make_orthogonal_loras()
    merged = merge_orthogonal(loras)
    expected = sum_of_deltas(loras)["layer0"]
    actual = merged.layers["layer0"].delta()
    assert torch.allclose(actual, expected, atol=1e-4)


def test_merge_naive_sum_exact_when_rank_sufficient_and_orthogonal():
    loras = make_orthogonal_loras()
    merged = merge_naive_sum(loras, target_rank=6)
    expected = sum_of_deltas(loras)["layer0"]
    actual = merged.layers["layer0"].delta()
    assert torch.allclose(actual, expected, atol=1e-4)


def test_merge_weighted_sum_requires_matching_weight_count():
    loras = make_orthogonal_loras()
    with pytest.raises(ValueError):
        merge_weighted_sum(loras, weights=[1.0, 2.0])


def test_merge_weighted_sum_applies_weights():
    loras = make_orthogonal_loras(ranks=(2, 2))
    weights = [1.0, 0.0]
    merged = merge_weighted_sum(loras, weights=weights, target_rank=4)
    expected = sum_of_deltas(loras, weights=weights)["layer0"]
    actual = merged.layers["layer0"].delta()
    assert torch.allclose(actual, expected, atol=1e-4)


def test_merge_orthogonal_reduces_collision_energy_loss():
    """Two fully-colliding LoRAs (identical B): orthogonal merge should
    project the second one's contribution to near zero in the colliding
    direction, while concat/naive-sum keep both at full strength."""
    g = torch.Generator().manual_seed(0)
    B = torch.randn(8, 2, generator=g)
    A1 = torch.randn(2, 6, generator=g)
    A2 = torch.randn(2, 6, generator=g)
    l1 = LoraWeights(layers={"x": _layer(A1, B.clone(), 2.0, "x")})
    l2 = LoraWeights(layers={"x": _layer(A2, B.clone(), 2.0, "x")})

    merged_ortho = merge_orthogonal([l1, l2])
    B_merged = merged_ortho.layers["x"].B
    assert B_merged.shape[1] == 4  # rank still grows structurally (2 + 2)

    second_block = B_merged[:, 2:]
    orig_scale = l2.layers["x"].scale
    orig_norm = torch.linalg.norm(orig_scale * B)
    # second adapter's contribution should have been projected away since it
    # fully collides with the first adapter's column space
    assert torch.linalg.norm(second_block) < 0.1 * orig_norm


def test_merge_orthogonal_grow_rank_preserves_magnitude():
    g = torch.Generator().manual_seed(0)
    B = torch.randn(8, 2, generator=g)
    A1 = torch.randn(2, 6, generator=g)
    A2 = torch.randn(2, 6, generator=g)
    l1 = LoraWeights(layers={"x": _layer(A1, B.clone(), 2.0, "x")})
    l2 = LoraWeights(layers={"x": _layer(A2, B.clone(), 2.0, "x")})

    merged = merge_orthogonal([l1, l2], grow_rank=True, collision_threshold=0.1)
    second_block = merged.layers["x"].B[:, 2:]
    orig_scale = l2.layers["x"].scale
    orig_norm = torch.linalg.norm(orig_scale * B)
    # with grow_rank, magnitude should be preserved (not collapsed to ~0)
    assert torch.linalg.norm(second_block) > 0.5 * orig_norm


def test_merge_ties_runs_and_produces_target_rank():
    loras = make_orthogonal_loras(ranks=(2, 2, 2))
    merged = merge_ties(loras, target_rank=4)
    assert merged.layers["layer0"].rank == 4


def test_merge_requires_shape_match():
    l1 = LoraWeights(layers={"x": _layer(torch.randn(2, 4), torch.randn(3, 2), 2.0, "x")})
    l2 = LoraWeights(layers={"x": _layer(torch.randn(2, 5), torch.randn(3, 2), 2.0, "x")})
    with pytest.raises(ValueError):
        merge_concat([l1, l2])


def test_merge_only_uses_common_layers():
    l1 = LoraWeights(layers={"x": _layer(torch.randn(2, 4), torch.randn(3, 2), 2.0, "x")})
    l2 = LoraWeights(
        layers={
            "x": _layer(torch.randn(2, 4), torch.randn(3, 2), 2.0, "x"),
            "y": _layer(torch.randn(2, 4), torch.randn(3, 2), 2.0, "y"),
        }
    )
    merged = merge_concat([l1, l2])
    assert set(merged.layers) == {"x"}


def _random_loras(out_f=16, in_f=12, ranks=(3, 2, 4), seed=1):
    g = torch.Generator().manual_seed(seed)
    return [
        LoraWeights(layers={"layer0": _layer(torch.randn(r, in_f, generator=g), torch.randn(out_f, r, generator=g), float(r), "layer0")})
        for r in ranks
    ]


def test_merge_symmetric_exact_when_inputs_already_orthogonal():
    loras = make_orthogonal_loras()
    merged = merge_symmetric(loras)
    assert torch.allclose(merged.layers["layer0"].delta(), sum_of_deltas(loras)["layer0"], atol=1e-4)


def test_merge_symmetric_removes_cross_overlap_and_keeps_each_gram():
    loras = _random_loras()
    B = merge_symmetric(loras).layers["layer0"].B.double()
    G = B.T @ B
    start = 0
    for lw in loras:
        layer = lw.layers["layer0"]
        stop = start + layer.rank
        own = (layer.scale * layer.B).double()
        assert torch.allclose(G[start:stop, start:stop], own.T @ own, atol=1e-6, rtol=1e-5)
        assert torch.allclose(G[start:stop, :start], torch.zeros_like(G[start:stop, :start]), atol=1e-6)
        start = stop


def test_merge_symmetric_is_order_independent():
    loras = _random_loras()
    forward = merge_symmetric(loras).layers["layer0"].delta()
    backward = merge_symmetric(loras[::-1]).layers["layer0"].delta()
    assert torch.allclose(forward, backward, atol=1e-5)
    ortho_f = merge_orthogonal(loras).layers["layer0"].delta()
    ortho_b = merge_orthogonal(loras[::-1]).layers["layer0"].delta()
    assert not torch.allclose(ortho_f, ortho_b, atol=1e-3)
