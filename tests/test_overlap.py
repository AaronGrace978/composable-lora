import pytest
import torch

from clora.lora_io import LoraLayer, LoraWeights
from clora.overlap import (
    aggregate_lora_overlap,
    all_pairs_overlap,
    frobenius_cosine_similarity,
    layer_overlap_score,
    pairwise_layer_overlap,
    principal_angles,
    subspace_overlap_score,
)


def _layer(A, B, alpha, name="x"):
    return LoraLayer(name=name, A=A, B=B, alpha=alpha, orig_key=name)


def test_overlap_score_one_for_identical_layers():
    g = torch.Generator().manual_seed(1)
    A = torch.randn(4, 10, generator=g)
    B = torch.randn(8, 4, generator=g)
    l1 = _layer(A, B, 4.0)
    l2 = _layer(A.clone(), B.clone(), 4.0)

    assert layer_overlap_score(l1, l2) == pytest.approx(1.0, abs=1e-4)
    assert frobenius_cosine_similarity(l1, l2) == pytest.approx(1.0, abs=1e-4)
    assert subspace_overlap_score(l1, l2, space="column") == pytest.approx(1.0, abs=1e-4)
    assert subspace_overlap_score(l1, l2, space="row") == pytest.approx(1.0, abs=1e-4)


def test_overlap_score_zero_for_orthogonal_column_spaces():
    """Column spaces only orthogonal: the deltas don't collide in output
    space (frobenius overlap must be ~0), but the row spaces (of A) are
    independent random subspaces and need not be orthogonal themselves."""
    out_f = 8
    Q, _ = torch.linalg.qr(torch.randn(out_f, out_f))
    B1 = Q[:, :2]
    B2 = Q[:, 2:4]
    g = torch.Generator().manual_seed(2)
    A1 = torch.randn(2, 6, generator=g)
    A2 = torch.randn(2, 6, generator=g)
    l1 = _layer(A1, B1, 2.0)
    l2 = _layer(A2, B2, 2.0)

    assert subspace_overlap_score(l1, l2, space="column") == pytest.approx(0.0, abs=1e-4)
    assert abs(frobenius_cosine_similarity(l1, l2)) < 1e-4


def test_overlap_score_zero_for_fully_orthogonal_updates():
    """Column AND row spaces both orthogonal: a strictly non-overlapping
    pair of updates, so the aggregate per-layer score must be ~0 too."""
    out_f, in_f = 8, 8
    Q_out, _ = torch.linalg.qr(torch.randn(out_f, out_f))
    Q_in, _ = torch.linalg.qr(torch.randn(in_f, in_f))
    B1, B2 = Q_out[:, :2], Q_out[:, 2:4]
    A1, A2 = Q_in[:, :2].T, Q_in[:, 2:4].T
    l1 = _layer(A1, B1, 2.0)
    l2 = _layer(A2, B2, 2.0)

    assert layer_overlap_score(l1, l2) == pytest.approx(0.0, abs=1e-4)
    assert abs(frobenius_cosine_similarity(l1, l2)) < 1e-4


def test_principal_angles_orthogonal_are_pi_over_2():
    out_f = 6
    Q, _ = torch.linalg.qr(torch.randn(out_f, out_f))
    B1 = Q[:, :1]
    B2 = Q[:, 1:2]
    A1 = torch.randn(1, 4)
    A2 = torch.randn(1, 4)
    l1 = _layer(A1, B1, 1.0)
    l2 = _layer(A2, B2, 1.0)
    angles = principal_angles(l1, l2, space="column")
    assert angles.shape == (1,)
    assert angles[0] == pytest.approx(torch.pi / 2, abs=1e-4)


def test_overlap_invariant_to_rescaling():
    g = torch.Generator().manual_seed(3)
    A = torch.randn(3, 5, generator=g)
    B = torch.randn(7, 3, generator=g)
    A2 = torch.randn(3, 5, generator=torch.Generator().manual_seed(4))
    B2 = torch.randn(7, 3, generator=torch.Generator().manual_seed(5))

    l1 = _layer(A, B, 3.0)
    l2 = _layer(A2, B2, 3.0)

    c = 2.5
    l1_rescaled = _layer(A * c, B / c, 3.0)

    s1 = layer_overlap_score(l1, l2)
    s2 = layer_overlap_score(l1_rescaled, l2)
    assert s1 == pytest.approx(s2, abs=1e-4)

    f1 = frobenius_cosine_similarity(l1, l2)
    f2 = frobenius_cosine_similarity(l1_rescaled, l2)
    assert f1 == pytest.approx(f2, abs=1e-4)


def test_overlap_invariant_to_gauge_transform():
    """Invariant to any invertible rank x rank mixing B, A -> B@M^-1, M@A,
    not just scalar rescaling."""
    rank = 3
    g = torch.Generator().manual_seed(6)
    A = torch.randn(rank, 5, generator=g)
    B = torch.randn(7, rank, generator=g)
    M = torch.randn(rank, rank, generator=torch.Generator().manual_seed(7))
    while torch.linalg.matrix_rank(M) < rank:
        M = torch.randn(rank, rank)
    M_inv = torch.linalg.inv(M)

    A2 = torch.randn(rank, 5, generator=torch.Generator().manual_seed(8))
    B2 = torch.randn(7, rank, generator=torch.Generator().manual_seed(9))

    l1 = _layer(A, B, 3.0)
    l1_gauge = _layer(M @ A, B @ M_inv, 3.0)
    l2 = _layer(A2, B2, 3.0)

    f1 = frobenius_cosine_similarity(l1, l2)
    f2 = frobenius_cosine_similarity(l1_gauge, l2)
    assert f1 == pytest.approx(f2, abs=1e-3)


def test_pairwise_and_aggregate_overlap():
    A1 = torch.randn(2, 4)
    B1 = torch.randn(3, 2)
    A2 = torch.randn(2, 4)
    B2 = torch.randn(3, 2)
    lora1 = LoraWeights(layers={"x": _layer(A1, B1, 2.0, "x")})
    lora2 = LoraWeights(layers={"x": _layer(A2, B2, 2.0, "x")})

    per_layer = pairwise_layer_overlap(lora1, lora2)
    assert set(per_layer) == {"x"}
    agg = aggregate_lora_overlap(lora1, lora2)
    assert agg == pytest.approx(per_layer["x"])


def test_aggregate_overlap_empty_when_no_shared_layers():
    lora1 = LoraWeights(layers={"x": _layer(torch.randn(2, 4), torch.randn(3, 2), 2.0, "x")})
    lora2 = LoraWeights(layers={"y": _layer(torch.randn(2, 4), torch.randn(3, 2), 2.0, "y")})
    assert aggregate_lora_overlap(lora1, lora2) == 0.0


def test_all_pairs_overlap_three_loras():
    def make():
        return LoraWeights(layers={"x": _layer(torch.randn(2, 4), torch.randn(3, 2), 2.0, "x")})

    loras = {"a": make(), "b": make(), "c": make()}
    pairs = all_pairs_overlap(loras)
    assert set(pairs) == {("a", "b"), ("a", "c"), ("b", "c")}
    for score in pairs.values():
        assert 0.0 <= score <= 1.0


def test_shared_block_energy_separates_one_sided_from_two_sided_overlap():
    from clora.overlap import shared_block_energy, subspace_overlap_score

    eye = torch.eye(8)
    base = LoraLayer(name="x", A=eye[:2], B=eye[:, :2], alpha=2.0)
    same_block = LoraLayer(name="x", A=eye[:2] * 3, B=eye[:, :2] * 0.5, alpha=2.0)
    same_out_other_in = LoraLayer(name="x", A=eye[4:6], B=eye[:, :2], alpha=2.0)
    assert abs(shared_block_energy(base, same_block) - 1.0) < 1e-6
    assert subspace_overlap_score(base, same_out_other_in, space="column") > 0.99
    assert shared_block_energy(base, same_out_other_in) < 1e-6
