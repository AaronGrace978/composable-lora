"""Merge multiple LoRA adapters into one, as baselines and as our method.

All merge functions take a sequence of :class:`~clora.lora_io.LoraWeights`
that share at least one layer name (and matching shapes for those shared
layers) and return a single merged :class:`~clora.lora_io.LoraWeights` over
the layers common to all inputs.

Methods
-------
- :func:`merge_concat` -- stack each LoRA's factors side by side (rank grows
  to the sum of input ranks). Lossless: the merged delta is always exactly
  the sum of the individual (weighted) deltas.
- :func:`merge_naive_sum` / :func:`merge_weighted_sum` -- materialize the
  (weighted) sum of deltas and SVD-truncate back down to a fixed rank.
- :func:`merge_ties` -- TIES-style trim / elect-sign / disjoint-merge on the
  materialized deltas, then SVD-truncate.
- :func:`merge_orthogonal` -- **ours**: sequentially re-project each LoRA's
  output-space update to remove any overlap with the LoRAs already placed,
  before concatenating. Exactly recovers the naive sum when the inputs are
  already non-overlapping (nothing to project away); reduces destructive
  interference when they collide, optionally preserving each adapter's
  update magnitude by growing its effective contribution instead of letting
  the projection shrink it (``grow_rank=True``).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import torch

from .lora_io import LoraLayer, LoraWeights


def _common_names(loras: Sequence[LoraWeights]) -> List[str]:
    if not loras:
        return []
    names = set(loras[0].layers)
    for lw in loras[1:]:
        names &= set(lw.layers)
    return sorted(names)


def _check_shapes(layers: Sequence[LoraLayer]) -> None:
    out_features = layers[0].out_features
    in_features = layers[0].in_features
    for layer in layers[1:]:
        if layer.out_features != out_features or layer.in_features != in_features:
            raise ValueError(
                f"Shape mismatch for layer {layer.name!r}: expected "
                f"({out_features}, {in_features}), got "
                f"({layer.out_features}, {layer.in_features})"
            )


def _resolve_weights(weights: Optional[Sequence[float]], n: int) -> List[float]:
    if weights is None:
        return [1.0] * n
    weights = list(weights)
    if len(weights) != n:
        raise ValueError(f"Expected {n} weights, got {len(weights)}")
    return weights


def _orthogonal_complement_direction(
    Q_cum: torch.Tensor, out_features: int, seed: tuple
) -> torch.Tensor:
    """A deterministic unit vector orthogonal to the column space of `Q_cum`.

    Used by :func:`merge_orthogonal` with ``grow_rank=True`` when an
    adapter's update collides almost entirely with the already-placed
    subspace, so its projected residual is numerically degenerate (too small
    to normalize into a meaningful direction).
    """
    generator = torch.Generator().manual_seed(abs(hash(seed)) % (2**31))
    for _ in range(4):
        v = torch.randn(out_features, generator=generator)
        if Q_cum.shape[1] > 0:
            v = v - Q_cum @ (Q_cum.T @ v)
        norm = torch.linalg.norm(v)
        if norm > 1e-6:
            return v / norm
    raise RuntimeError("Could not find an orthogonal-complement direction; subspace may be full")


def sum_of_deltas(
    loras: Sequence[LoraWeights], weights: Optional[Sequence[float]] = None
) -> Dict[str, torch.Tensor]:
    """The exact (weighted) sum of materialized deltas per shared layer name.

    Useful as a reference/ground-truth in tests and scripts; not itself a
    merge method (it returns full dense deltas, not a low-rank LoRA).
    """
    weights = _resolve_weights(weights, len(loras))
    out: Dict[str, torch.Tensor] = {}
    for name in _common_names(loras):
        layers = [lw.layers[name] for lw in loras]
        _check_shapes(layers)
        out[name] = sum(w * layer.delta() for w, layer in zip(weights, layers))
    return out


def merge_concat(
    loras: Sequence[LoraWeights], weights: Optional[Sequence[float]] = None
) -> LoraWeights:
    """Concatenate factors; lossless, rank grows to the sum of input ranks."""
    weights = _resolve_weights(weights, len(loras))
    out_layers: Dict[str, LoraLayer] = {}
    for name in _common_names(loras):
        layers = [lw.layers[name] for lw in loras]
        _check_shapes(layers)
        B = torch.cat([w * layer.scale * layer.B for w, layer in zip(weights, layers)], dim=1)
        A = torch.cat([layer.A for layer in layers], dim=0)
        rank = A.shape[0]
        out_layers[name] = LoraLayer(name=name, A=A, B=B, alpha=float(rank), orig_key=name)
    fmt = loras[0].format if loras else "diffusers"
    return LoraWeights(layers=out_layers, format=fmt, metadata={"merge": "concat"})


def _svd_truncate(delta: torch.Tensor, rank: int):
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    r = max(min(rank, S.shape[0]), 0)
    sqrt_s = torch.sqrt(S[:r].clamp_min(0))
    B = U[:, :r] * sqrt_s.unsqueeze(0)
    A = sqrt_s.unsqueeze(1) * Vh[:r, :]
    return A, B, r


def merge_naive_sum(
    loras: Sequence[LoraWeights],
    weights: Optional[Sequence[float]] = None,
    target_rank: Optional[int] = None,
) -> LoraWeights:
    """Materialize the (weighted) sum of deltas, then SVD-truncate to a fixed rank."""
    weights = _resolve_weights(weights, len(loras))
    out_layers: Dict[str, LoraLayer] = {}
    for name in _common_names(loras):
        layers = [lw.layers[name] for lw in loras]
        _check_shapes(layers)
        delta = sum(w * layer.delta() for w, layer in zip(weights, layers))
        rank = target_rank if target_rank is not None else max(layer.rank for layer in layers)
        A, B, r = _svd_truncate(delta, rank)
        out_layers[name] = LoraLayer(name=name, A=A, B=B, alpha=float(max(r, 1)), orig_key=name)
    fmt = loras[0].format if loras else "diffusers"
    return LoraWeights(layers=out_layers, format=fmt, metadata={"merge": "naive_sum"})


def merge_weighted_sum(
    loras: Sequence[LoraWeights], weights: Sequence[float], target_rank: Optional[int] = None
) -> LoraWeights:
    """:func:`merge_naive_sum` with required per-LoRA weights."""
    if len(weights) != len(loras):
        raise ValueError(f"Expected {len(loras)} weights, got {len(weights)}")
    result = merge_naive_sum(loras, weights=weights, target_rank=target_rank)
    result.metadata["merge"] = "weighted_sum"
    return result


def merge_ties(
    loras: Sequence[LoraWeights],
    weights: Optional[Sequence[float]] = None,
    trim_frac: float = 0.2,
    target_rank: Optional[int] = None,
) -> LoraWeights:
    """TIES-style merge: trim small magnitudes, elect sign, average agreeing values.

    Applied per layer on the materialized (weighted) deltas, then
    SVD-truncated back down to a fixed rank.
    """
    if not (0.0 <= trim_frac < 1.0):
        raise ValueError("trim_frac must be in [0, 1)")
    weights = _resolve_weights(weights, len(loras))
    out_layers: Dict[str, LoraLayer] = {}
    for name in _common_names(loras):
        layers = [lw.layers[name] for lw in loras]
        _check_shapes(layers)
        deltas = [w * layer.delta() for w, layer in zip(weights, layers)]

        trimmed = []
        for d in deltas:
            flat = d.abs().flatten().float()
            if flat.numel() == 0 or trim_frac == 0.0:
                trimmed.append(d)
                continue
            thresh = torch.quantile(flat, trim_frac)
            trimmed.append(d * (d.abs() >= thresh))

        stacked = torch.stack(trimmed, dim=0)  # (n, out, in)
        elected_sign = torch.sign(stacked.sum(dim=0))
        agree_mask = torch.sign(stacked) == elected_sign.unsqueeze(0)
        agreeing_vals = stacked * agree_mask
        counts = agree_mask.sum(dim=0).clamp_min(1)
        merged_delta = agreeing_vals.sum(dim=0) / counts

        rank = target_rank if target_rank is not None else max(layer.rank for layer in layers)
        A, B, r = _svd_truncate(merged_delta, rank)
        out_layers[name] = LoraLayer(name=name, A=A, B=B, alpha=float(max(r, 1)), orig_key=name)
    fmt = loras[0].format if loras else "diffusers"
    return LoraWeights(layers=out_layers, format=fmt, metadata={"merge": "ties"})


def merge_orthogonal(
    loras: Sequence[LoraWeights],
    weights: Optional[Sequence[float]] = None,
    grow_rank: bool = False,
    collision_threshold: float = 0.3,
    tol: float = 1e-5,
) -> LoraWeights:
    """Our method: sequentially re-project each LoRA's output-space update to
    remove overlap with previously-placed LoRAs, then concatenate.

    Processing order follows the input order (earlier entries are left
    untouched; later entries are projected against the cumulative span of
    all earlier ones). When the inputs are already pairwise orthogonal in
    output space, every projection is a no-op and the result is exactly the
    (weighted) sum of individual deltas.

    If ``grow_rank`` is True, a later adapter whose collision with the
    already-placed subspace exceeds ``collision_threshold`` (fraction of its
    energy that would be projected away) has its post-projection direction
    rescaled back up to its original magnitude column-by-column, trading a
    larger directional change for not losing update strength to the
    collision.
    """
    weights = _resolve_weights(weights, len(loras))
    out_layers: Dict[str, LoraLayer] = {}
    for name in _common_names(loras):
        layers = [lw.layers[name] for lw in loras]
        _check_shapes(layers)
        out_features = layers[0].out_features

        Q_cum = layers[0].B.new_zeros((out_features, 0))
        B_blocks: List[torch.Tensor] = []
        A_blocks: List[torch.Tensor] = []

        for layer_idx, (layer, w) in enumerate(zip(layers, weights)):
            B_scaled = w * layer.scale * layer.B  # (out_features, r_i)

            if Q_cum.shape[1] > 0 and B_scaled.shape[1] > 0:
                proj = Q_cum @ (Q_cum.T @ B_scaled)
                residual = B_scaled - proj

                if grow_rank:
                    orig_norms = torch.linalg.norm(B_scaled, dim=0)
                    res_norms = torch.linalg.norm(residual, dim=0)
                    collision = 1.0 - (res_norms / orig_norms.clamp_min(1e-12))
                    columns = []
                    for j in range(B_scaled.shape[1]):
                        if collision[j].item() <= collision_threshold:
                            columns.append(residual[:, j])
                            continue
                        if res_norms[j] > tol:
                            direction = residual[:, j] / res_norms[j]
                        else:
                            # residual is numerically degenerate (near-total
                            # collision): fall back to a deterministic
                            # direction orthogonal to everything placed so
                            # far, so the adapter's magnitude isn't lost.
                            direction = _orthogonal_complement_direction(
                                Q_cum, out_features, seed=(layer_idx, j)
                            )
                        columns.append(direction * orig_norms[j])
                    B_final = torch.stack(columns, dim=1)
                else:
                    B_final = residual
            else:
                B_final = B_scaled

            B_blocks.append(B_final)
            A_blocks.append(layer.A)

            if B_final.shape[1] > 0 and torch.linalg.norm(B_final) > tol:
                combined = torch.cat([Q_cum, B_final], dim=1) if Q_cum.shape[1] > 0 else B_final
                Q_new, R = torch.linalg.qr(combined, mode="reduced")
                keep = torch.abs(torch.diagonal(R)) > tol
                Q_cum = Q_new[:, keep]

        B = torch.cat(B_blocks, dim=1)
        A = torch.cat(A_blocks, dim=0)
        rank = A.shape[0]
        out_layers[name] = LoraLayer(name=name, A=A, B=B, alpha=float(rank), orig_key=name)

    fmt = loras[0].format if loras else "diffusers"
    return LoraWeights(
        layers=out_layers, format=fmt, metadata={"merge": "ortho", "grow_rank": str(grow_rank)}
    )


MERGE_METHODS: Dict[str, Callable[..., LoraWeights]] = {
    "concat": merge_concat,
    "naive_sum": merge_naive_sum,
    "weighted_sum": merge_weighted_sum,
    "ties": merge_ties,
    "ortho": merge_orthogonal,
}


def merge(method: str, loras: Sequence[LoraWeights], **kwargs) -> LoraWeights:
    """Dispatch to one of :data:`MERGE_METHODS` by name."""
    if method not in MERGE_METHODS:
        raise ValueError(f"Unknown merge method {method!r}; options: {sorted(MERGE_METHODS)}")
    return MERGE_METHODS[method](loras, **kwargs)
