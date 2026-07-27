"""Budget-valid residual tokens for CUDA-QX Generative Quantum Eigensolver.

The CUDA-QX GQE implementation treats ``pool`` as an indexable Python
vocabulary and passes selected pool objects to the user cost function.  This
module defines chemically structured residual gate tokens for that vocabulary
and exact cutting-overhead accounting for every generated sequence.

Only cross-partition gates are included.  A caller must still mask tokens whose
cost exceeds the remaining sequence budget *before sampling*.  The companion
``scripts/qbraid_budgeted_gqe.py`` implements that mask by subclassing the
pinned CUDA-QX 0.6.0 Transformer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from . import overhead as oh


@dataclass(frozen=True)
class ResidualToken:
    """One discrete, particle-conserving residual action.

    ``kind`` is ``identity``, ``single`` or ``double``.  A single token acts
    on exactly one spin channel; a double token performs a paired excitation.
    Angles use the sector engine's matrix-angle convention.
    """

    kind: str
    p: int = -1
    q: int = -1
    angle: float = 0.0
    spin: str | None = None
    u_cost: float = 0.0
    pair_score: float = 0.0

    @property
    def label(self) -> str:
        if self.kind == "identity":
            return "I"
        spin = f"_{self.spin}" if self.spin is not None else ""
        return f"{self.kind}{spin}({self.p},{self.q};{self.angle:+.5f})"

    def record(self, index: int | None = None) -> dict:
        out = asdict(self)
        out["label"] = self.label
        if index is not None:
            out["pool_index"] = int(index)
        return out


def token_cost(kind: str, angle: float) -> float:
    """Return the exact log-overhead contribution of one cross-cut token."""
    if kind == "identity":
        return 0.0
    if kind == "single":
        return float(oh.u_single(angle))
    if kind == "double":
        return float(oh.u_pair_double(angle))
    raise ValueError(f"unknown residual-token kind: {kind}")


def sequence_u(tokens: Iterable[ResidualToken]) -> float:
    """Total additive log-overhead of a generated residual sequence."""
    return float(sum(token.u_cost for token in tokens))


def base_u_from_phi(phi: float) -> float:
    """Invert ``phi = 2 exp(2u) - 1`` for a frozen backbone."""
    if phi < 1.0:
        raise ValueError("cutting overhead phi must be at least one")
    return 0.5 * float(np.log((float(phi) + 1.0) / 2.0))


def remaining_u(phi_max: float, base_phi: float) -> float:
    """Exact residual budget after freezing a measured backbone circuit."""
    return max(0.0, oh.u_max_from_phi(phi_max) - base_u_from_phi(base_phi))


def blocked_to_interleaved_fock_state(
    blocked_state: np.ndarray, n_spatial: int
) -> np.ndarray:
    """Convert blocked-spin Fock amplitudes to interleaved-spin ordering.

    The sector engine orders modes as ``[alpha_0, ..., beta_0, ...]`` while
    CUDA-QX's chemistry bridge uses ``[alpha_0, beta_0, alpha_1, beta_1, ...]``.
    Reordering fermionic modes requires both a basis-index permutation and the
    parity phase of the occupied-mode permutation.
    """
    state = np.asarray(blocked_state)
    if n_spatial <= 0:
        raise ValueError("n_spatial must be positive")
    if state.ndim != 1 or state.size != 1 << (2 * int(n_spatial)):
        raise ValueError("state length must equal 2**(2*n_spatial)")

    interleaved = np.zeros_like(state)
    alpha_mask = (1 << n_spatial) - 1
    for blocked_index, amplitude in enumerate(state):
        alpha_bits = blocked_index & alpha_mask
        beta_bits = blocked_index >> n_spatial
        interleaved_index = 0
        fermionic_inversions = 0
        for orbital in range(n_spatial):
            interleaved_index |= ((alpha_bits >> orbital) & 1) << (2 * orbital)
            interleaved_index |= ((beta_bits >> orbital) & 1) << (2 * orbital + 1)
            if (beta_bits >> orbital) & 1:
                fermionic_inversions += (
                    alpha_bits >> (orbital + 1)
                ).bit_count()
        phase = -1 if fermionic_inversions & 1 else 1
        interleaved[interleaved_index] = phase * amplitude
    return interleaved


def build_residual_pool(
    prob,
    left_block: list[int],
    max_pairs: int,
    angle_bins: tuple[float, ...] = (0.0125, 0.025, 0.05),
    include_pair_doubles: bool = True,
) -> tuple[list[ResidualToken], list[tuple[int, int]]]:
    """Build a score-pruned cross-partition residual vocabulary.

    Every non-identity token is a genuine particle-number-conserving gate.
    The pool contains signed alpha/beta single excitations and, optionally,
    signed pair-double excitations for the highest-scoring cross pairs.
    """
    if max_pairs <= 0:
        raise ValueError("max_pairs must be positive")
    if not angle_bins or any(float(x) <= 0.0 for x in angle_bins):
        raise ValueError("angle_bins must contain positive values")

    score = oh.pair_score(prob.h1e, prob.eri)
    _, cross = oh.split_pairs(prob.norb, left_block)
    ranked = sorted(cross, key=lambda pq: (-score[pq[0], pq[1]], pq))
    chosen = ranked[: min(int(max_pairs), len(ranked))]

    pool = [ResidualToken("identity")]
    signed_angles = tuple(
        sign * float(angle)
        for angle in angle_bins
        for sign in (-1.0, 1.0)
    )
    for p, q in chosen:
        relevance = float(score[p, q])
        for angle in signed_angles:
            for spin in ("a", "b"):
                pool.append(
                    ResidualToken(
                        "single",
                        p=p,
                        q=q,
                        angle=angle,
                        spin=spin,
                        u_cost=token_cost("single", angle),
                        pair_score=relevance,
                    )
                )
            if include_pair_doubles:
                pool.append(
                    ResidualToken(
                        "double",
                        p=p,
                        q=q,
                        angle=angle,
                        spin=None,
                        u_cost=token_cost("double", angle),
                        pair_score=relevance,
                    )
                )
    return pool, chosen


def apply_token(state, sector, token: ResidualToken, convention: str = "qubit"):
    """Apply one token to a sector state and return the new state."""
    if token.kind == "identity":
        return state
    import torch

    angle = torch.tensor(token.angle, dtype=state.dtype, device=state.device)
    if token.kind == "single":
        if token.spin not in ("a", "b"):
            raise ValueError("single token requires spin 'a' or 'b'")
        return sector.apply_single(
            state, token.spin, token.p, token.q, angle, convention
        )
    if token.kind == "double":
        return sector.apply_pair_double(
            state, token.p, token.q, angle, convention
        )
    raise ValueError(f"unknown residual-token kind: {token.kind}")


def validate_sequence_budget(
    tokens: Iterable[ResidualToken],
    base_phi: float,
    phi_max: float,
    atol: float = 1e-12,
) -> dict:
    """Return exact budget accounting and raise on a violating sequence."""
    residual = sequence_u(tokens)
    base = base_u_from_phi(base_phi)
    total = base + residual
    budget = oh.u_max_from_phi(phi_max)
    phi = oh.phi_from_u(total)
    if total > budget + atol:
        raise ValueError(
            f"generated sequence violates hard budget: u={total:.12g} > "
            f"u_max={budget:.12g}"
        )
    return {
        "base_u": base,
        "residual_u": residual,
        "total_u": total,
        "u_max": budget,
        "remaining_u_before_sequence": max(0.0, budget - base),
        "phi": phi,
        "phi_max": float(phi_max),
        "within_budget": True,
    }
