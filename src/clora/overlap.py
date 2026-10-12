"""Per-layer interference / subspace-overlap metrics between two or more LoRAs.

All metrics are computed directly from the low-rank ``(A, B, alpha)`` factors
without ever materializing the full ``out_features x in_features`` weight
delta, using the identity

    delta = scale * B @ A,   scale = alpha / rank

so that for two layers ``1`` and ``2``::

    <delta_1, delta_2>_F = s1 * s2 * trace((B1^T @ B2) @ (A2 @ A1^T))

Both matrices inside the trace are at most ``rank x rank`` in size, so this
is cheap even for huge ``in_features`` / ``out_features``.

Two complementary notions of overlap are provided:

- **Principal angles** between the column space of ``B`` (the directions a
  layer writes *into* in output space) or the row space of ``A`` (the
  directions it reads *from* in input space). A normalized score in
  ``[0, 1]`` is derived as the mean squared cosine of the principal angles
  (equivalently ``||Q1^T Q2||_F^2 / min(r1, r2)``): 0 for orthogonal
  subspaces, 1 for identical ones.
- **Normalized Frobenius overlap**: the cosine similarity between the two
  full weight deltas, computed via the trace identity above.

Both are invariant to the internal gauge freedom of a low-rank factorization
(``B, A -> B @ M^-1, M @ A`` for any invertible ``M``, which includes simple
rescaling ``B, A -> B / c, A * c``), since they only depend on the column
space of ``B``, the row space of ``A``, and the delta-level inner products.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import torch

from .lora_io import LoraLayer, LoraWeights

Space = str  # "column" | "row"


def frobenius_sq_raw(A: torch.Tensor, B: torch.Tensor, scale: float) -> torch.Tensor:
    """``||scale * B @ A||_F^2`` computed without materializing ``B @ A``."""
    gram = (B.T @ B) * (A @ A.T)
    return (scale ** 2) * gram.sum()


def inner_product_raw(
    A1: torch.Tensor, B1: torch.Tensor, s1: float, A2: torch.Tensor, B2: torch.Tensor, s2: float
) -> torch.Tensor:
    """``<s1*B1@A1, s2*B2@A2>_F`` computed without materializing either delta."""
    C = B1.T @ B2  # (r1, r2)
    D = A2 @ A1.T  # (r2, r1)
    return s1 * s2 * torch.trace(C @ D)


def layer_delta_frobenius_sq(layer: LoraLayer) -> float:
    return float(frobenius_sq_raw(layer.A, layer.B, layer.scale).item())


def layer_delta_inner_product(layer1: LoraLayer, layer2: LoraLayer) -> float:
    if layer1.out_features != layer2.out_features or layer1.in_features != layer2.in_features:
        raise ValueError(
            f"Layer shape mismatch: {layer1.name!r} is "
            f"({layer1.out_features}, {layer1.in_features}), {layer2.name!r} is "
            f"({layer2.out_features}, {layer2.in_features})"
        )
    return float(
        inner_product_raw(
            layer1.A, layer1.B, layer1.scale, layer2.A, layer2.B, layer2.scale
        ).item()
    )


def frobenius_cosine_similarity(layer1: LoraLayer, layer2: LoraLayer) -> float:
    """Cosine similarity between the two full weight deltas, in ``[-1, 1]``."""
    num = layer_delta_inner_product(layer1, layer2)
    denom = math.sqrt(
        max(layer_delta_frobenius_sq(layer1), 0.0) * max(layer_delta_frobenius_sq(layer2), 0.0)
    )
    if denom < 1e-12:
        return 0.0
    return num / denom


def _orthonormal_basis(mat: torch.Tensor) -> torch.Tensor:
    """Orthonormal basis for the column space of `mat` (out_features, rank)."""
    if mat.shape[1] == 0:
        return mat.new_zeros((mat.shape[0], 0))
    q, _ = torch.linalg.qr(mat, mode="reduced")
    return q[:, : min(mat.shape)]


def principal_angles(layer1: LoraLayer, layer2: LoraLayer, space: Space = "column") -> np.ndarray:
    """Principal angles (radians, ascending) between subspaces of two layers.

    ``space="column"`` compares the column spaces of ``B`` (output-space
    directions); ``space="row"`` compares the row spaces of ``A``
    (input-space directions).
    """
    if space == "column":
        Q1 = _orthonormal_basis(layer1.B)
        Q2 = _orthonormal_basis(layer2.B)
    elif space == "row":
        Q1 = _orthonormal_basis(layer1.A.T)
        Q2 = _orthonormal_basis(layer2.A.T)
    else:
        raise ValueError(f"space must be 'column' or 'row', got {space!r}")

    if Q1.shape[1] == 0 or Q2.shape[1] == 0:
        return np.array([])

    M = Q1.T @ Q2
    singular_values = torch.linalg.svdvals(M)
    singular_values = torch.clamp(singular_values, -1.0, 1.0)
    angles = torch.arccos(singular_values)
    return angles.detach().cpu().numpy()


def subspace_overlap_score(layer1: LoraLayer, layer2: LoraLayer, space: Space = "column") -> float:
    """Mean squared cosine of the principal angles, in ``[0, 1]``.

    0 for orthogonal subspaces, 1 for identical (or nested with equal rank)
    subspaces.
    """
    angles = principal_angles(layer1, layer2, space=space)
    if angles.size == 0:
        return 0.0
    return float(np.mean(np.cos(angles) ** 2))


def layer_overlap_score(
    layer1: LoraLayer,
    layer2: LoraLayer,
    column_weight: float = 0.4,
    row_weight: float = 0.4,
    frobenius_weight: float = 0.2,
) -> float:
    """Aggregate per-layer interference score in ``[0, 1]``.

    A weighted combination of the column-space overlap, row-space overlap,
    and absolute normalized Frobenius overlap. 0 means the two LoRAs' updates
    to this layer are fully composable (orthogonal); 1 means they are
    maximally colliding (identical direction).
    """
    total_weight = column_weight + row_weight + frobenius_weight
    if total_weight <= 0:
        raise ValueError("At least one weight must be positive")
    col = subspace_overlap_score(layer1, layer2, space="column")
    row = subspace_overlap_score(layer1, layer2, space="row")
    frob = abs(frobenius_cosine_similarity(layer1, layer2))
    score = (column_weight * col + row_weight * row + frobenius_weight * frob) / total_weight
    return float(min(max(score, 0.0), 1.0))


def pairwise_layer_overlap(lora1: LoraWeights, lora2: LoraWeights) -> Dict[str, float]:
    """Per-layer overlap score for every layer name common to both LoRAs."""
    common = sorted(set(lora1.layers) & set(lora2.layers))
    return {name: layer_overlap_score(lora1.layers[name], lora2.layers[name]) for name in common}


def aggregate_lora_overlap(lora1: LoraWeights, lora2: LoraWeights) -> float:
    """Mean per-layer overlap score across all shared layers (0 if none shared)."""
    scores = pairwise_layer_overlap(lora1, lora2)
    if not scores:
        return 0.0
    return float(np.mean(list(scores.values())))


def all_pairs_overlap(loras: Mapping[str, LoraWeights]) -> Dict[Tuple[str, str], float]:
    """Aggregate overlap score for every unordered pair of named LoRAs."""
    result: Dict[Tuple[str, str], float] = {}
    for name_a, name_b in itertools.combinations(sorted(loras), 2):
        result[(name_a, name_b)] = aggregate_lora_overlap(loras[name_a], loras[name_b])
    return result
