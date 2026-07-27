"""Entanglement-informed orbital bipartition search and parameter transfer.

This is a deliberately narrower extension than automatic TTN optimization.
It treats the root orbital split as a resource variable, inspired by local
TTN reconnection, while retaining the declared hard cutting budget.  The
search never uses CASCI energy to rank partitions: it uses a topology-neutral
pilot state, Schmidt diagnostics and Hamiltonian cross weight.  Candidate
energies are assessed only by the matched downstream control.
"""

from __future__ import annotations

from itertools import combinations
import math
from typing import Any, Iterable

import numpy as np
import torch

from . import overhead as oh


def _compact_bits(bits: int, orbitals: list[int]) -> int:
    out = 0
    for local, orbital in enumerate(orbitals):
        out |= ((int(bits) >> int(orbital)) & 1) << local
    return out


def schmidt_spectrum_spatial_partition(
    sector: Any,
    state: np.ndarray | torch.Tensor,
    left: Iterable[int],
) -> np.ndarray:
    """Return exact Schmidt probabilities across a spatial-orbital split.

    No dense full-register vector is allocated.  The coefficient matrix has
    only the determinant-sector entries.  A fermionic phase is included when
    the blocked global mode order ``A_alpha,B_alpha,A_beta,B_beta`` is changed
    to the tensor-product order ``A_alpha,A_beta | B_alpha,B_beta``.
    """

    left = sorted({int(x) for x in left})
    right = sorted(set(range(int(sector.norb))) - set(left))
    if not left or not right:
        raise ValueError("both sides of a Schmidt split must be nonempty")
    amplitudes = (
        state.detach().cpu().numpy() if torch.is_tensor(state) else np.asarray(state)
    )
    if amplitudes.shape != (sector.dimA, sector.dimB):
        raise ValueError("state shape does not match determinant sector")

    entries: list[tuple[tuple[int, int], tuple[int, int], complex]] = []
    left_keys: set[tuple[int, int]] = set()
    right_keys: set[tuple[int, int]] = set()
    for ia, sa_raw in enumerate(sector.alpha.strs):
        sa = int(sa_raw)
        la = _compact_bits(sa, left)
        ra = _compact_bits(sa, right)
        n_ra = int(ra).bit_count()
        for ib, sb_raw in enumerate(sector.beta.strs):
            amplitude = amplitudes[ia, ib]
            if abs(amplitude) < 1e-18:
                continue
            sb = int(sb_raw)
            lb = _compact_bits(sb, left)
            rb = _compact_bits(sb, right)
            # Move the A-beta occupied modes across the B-alpha occupied modes.
            phase = -1.0 if ((int(lb).bit_count() * n_ra) & 1) else 1.0
            lk, rk = (la, lb), (ra, rb)
            left_keys.add(lk)
            right_keys.add(rk)
            entries.append((lk, rk, phase * amplitude))

    left_order = {key: i for i, key in enumerate(sorted(left_keys))}
    right_order = {key: i for i, key in enumerate(sorted(right_keys))}
    matrix = np.zeros(
        (len(left_order), len(right_order)),
        dtype=np.result_type(amplitudes.dtype, np.float64),
    )
    for lk, rk, amplitude in entries:
        matrix[left_order[lk], right_order[rk]] += amplitude
    singular = np.linalg.svd(matrix, compute_uv=False)
    weights = np.square(np.abs(singular))
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("zero-norm state")
    weights = weights / total
    weights.sort()
    return weights[::-1]


def partition_diagnostics(
    prob: Any,
    sector: Any,
    state: np.ndarray | torch.Tensor,
    left: Iterable[int],
    *,
    chis: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
) -> dict[str, Any]:
    left = sorted({int(x) for x in left})
    right = sorted(set(range(int(prob.norb))) - set(left))
    weights = schmidt_spectrum_spatial_partition(sector, state, left)
    positive = weights[weights > 1e-16]
    entropy = float(-np.sum(positive * np.log(positive)))
    score = oh.pair_score(prob.h1e, prob.eri)
    cross_weight = float(score[np.ix_(left, right)].sum())
    total_weight = float(np.triu(score, 1).sum())
    tails = {
        str(int(chi)): float(weights[int(chi):].sum())
        for chi in chis
    }
    return {
        "left_block": left,
        "right_block": right,
        "schmidt_rank_at_1e-16": int(len(positive)),
        "schmidt_entropy_nats": entropy,
        "largest_schmidt_probabilities": [float(x) for x in weights[:16]],
        "discarded_weight_by_chi": tails,
        "hamiltonian_cross_weight": cross_weight,
        "hamiltonian_cross_fraction": (
            cross_weight / total_weight if total_weight > 0 else 0.0
        ),
    }


def canonical_left(left: Iterable[int], norb: int) -> tuple[int, ...]:
    left = tuple(sorted({int(x) for x in left}))
    right = tuple(sorted(set(range(int(norb))) - set(left)))
    # Remove mirror duplicates only for an even split; for odd widths the
    # smaller side is the unique left side by construction.
    if len(left) == len(right):
        return min(left, right)
    return left if len(left) < len(right) else right


def balanced_partitions(norb: int) -> list[tuple[int, ...]]:
    """Enumerate mirror-unique balanced spatial-orbital root splits."""

    size = int(norb) // 2
    if size <= 0:
        raise ValueError("at least two orbitals are required")
    partitions = {
        canonical_left(combo, norb) for combo in combinations(range(norb), size)
    }
    return sorted(partitions)


def local_swap_neighbours(left: Iterable[int], norb: int) -> list[tuple[int, ...]]:
    base = set(canonical_left(left, norb))
    right = set(range(int(norb))) - base
    out = {tuple(sorted(base))}
    for i in sorted(base):
        for j in sorted(right):
            candidate = (base - {i}) | {j}
            out.add(canonical_left(candidate, norb))
    return sorted(out)


def candidate_partitions(
    norb: int,
    incumbent_left: Iterable[int],
    *,
    exhaustive_max_norb: int = 8,
    local_rounds: int = 3,
    frontier_width: int = 8,
    evaluator=None,
) -> list[tuple[int, ...]]:
    """Enumerate small cuts or explore deterministic local reconnections.

    For larger spaces an evaluator may return preliminary diagnostics.  The
    Pareto-front candidates seed the next local-swap round, analogous to
    repeated root reconnections without claiming a full tree optimization.
    """

    if int(norb) <= int(exhaustive_max_norb):
        return balanced_partitions(int(norb))
    seen = {canonical_left(incumbent_left, norb)}
    seeds = sorted(seen)
    for _ in range(int(local_rounds)):
        expanded = set(seen)
        for seed in seeds:
            expanded.update(local_swap_neighbours(seed, norb))
        new = sorted(expanded - seen)
        seen = expanded
        if not new or evaluator is None:
            seeds = new[: int(frontier_width)]
            continue
        records = [evaluator(item) for item in sorted(seen)]
        front = pareto_front(records)
        seeds = [
            tuple(item["left_block"])
            for item in sorted(
                front,
                key=lambda x: (
                    x["schmidt_entropy_nats"],
                    x["hamiltonian_cross_weight"],
                    tuple(x["left_block"]),
                ),
            )[: int(frontier_width)]
        ]
    return sorted(seen)


def pareto_front(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Minimize entropy and Hamiltonian cross weight without hidden weights."""

    objectives = ("schmidt_entropy_nats", "hamiltonian_cross_weight")
    front = []
    for i, item in enumerate(records):
        dominated = False
        for j, other in enumerate(records):
            if i == j:
                continue
            no_worse = all(float(other[k]) <= float(item[k]) + 1e-15 for k in objectives)
            strict = any(float(other[k]) < float(item[k]) - 1e-15 for k in objectives)
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            front.append(item)
    return sorted(front, key=lambda x: tuple(x["left_block"]))


def pareto_knee(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Choose the normalized closest-to-utopia point on a Pareto front."""

    if not records:
        raise ValueError("empty Pareto front")
    keys = ("schmidt_entropy_nats", "hamiltonian_cross_weight")
    ranges = {}
    for key in keys:
        values = np.asarray([float(item[key]) for item in records])
        ranges[key] = (float(values.min()), float(values.max()))

    def distance(item):
        components = []
        for key in keys:
            lo, hi = ranges[key]
            components.append(0.0 if hi <= lo else (float(item[key]) - lo) / (hi - lo))
        return math.sqrt(sum(x * x for x in components))

    return min(records, key=lambda x: (distance(x), tuple(x["left_block"])))


def transfer_parameters(
    source_circuit: Any,
    target_circuit: Any,
    source_params: dict[str, torch.Tensor],
    *,
    seed: int,
    new_pair_scale: float = 0.0,
) -> tuple[dict[str, torch.nn.Parameter], dict[str, Any]]:
    """Map shared effective pair angles between widths/topologies.

    New pairs are identity-initialized by default.  The target's tanh cap is
    inverted so copied *effective* cross-cut angles are preserved exactly.
    """

    if source_circuit.acfg.layers != target_circuit.acfg.layers:
        raise ValueError("parameter transfer requires equal layer counts")
    target = target_circuit.init_params(int(seed))
    generator = torch.Generator().manual_seed(int(seed) + 991_337)
    with torch.no_grad():
        for tensor in target.values():
            if new_pair_scale == 0.0:
                tensor.zero_()
            else:
                tensor.copy_(
                    new_pair_scale
                    * torch.randn(*tensor.shape, generator=generator, dtype=tensor.dtype)
                )
    source_effective = {
        key: value.detach()
        for key, value in source_circuit._effective_angles(source_params).items()
    }
    source_pos = {tuple(pair): i for i, pair in enumerate(source_circuit.topo.pairs)}
    target_cross = set(int(i) for i in target_circuit.cross_idx)
    cap_double = (
        target_circuit.acfg.beta_cap_double
        if target_circuit.acfg.beta_cap_double is not None
        else target_circuit.acfg.beta_cap
    )
    copied = []
    clipped_values = 0
    with torch.no_grad():
        for target_index, pair in enumerate(target_circuit.topo.pairs):
            source_index = source_pos.get(tuple(pair))
            if source_index is None:
                continue
            copied.append(tuple(pair))
            for key, target_tensor in target.items():
                if key not in source_effective:
                    continue
                value = source_effective[key].select(1, source_index).clone()
                if target_index in target_cross:
                    cap = cap_double if key == "doubles" else target_circuit.acfg.beta_cap
                    ratio = value / cap
                    clipped_values += int(
                        torch.count_nonzero(torch.abs(ratio) >= 1.0 - 1e-9).item()
                    )
                    value = cap * torch.atanh(
                        torch.clamp(ratio, -1.0 + 1e-9, 1.0 - 1e-9)
                    )
                target_tensor.select(1, target_index).copy_(value)
    target = {
        key: torch.nn.Parameter(value.detach().clone()) for key, value in target.items()
    }
    return target, {
        "copied_pairs": [list(pair) for pair in copied],
        "copied_pair_count": len(copied),
        "new_pair_count": len(target_circuit.topo.pairs) - len(copied),
        "new_pair_scale": float(new_pair_scale),
        "exact_identity_for_new_pairs": bool(new_pair_scale == 0.0),
        "cross_cap_clipped_values": int(clipped_values),
        "transfer_exact_on_shared_values": bool(clipped_values == 0),
    }
