"""Quantum-circuit-cutting overhead accounting and budgeted partitioned topologies.

Follows Nakamura & Sanji, arXiv:2509.08351 (their Eqs. 2-7, 9, 11), using the
optimal-cut convention of Harrow & Lowe. Angle conventions: our Givens matrix
angle beta corresponds to PennyLane SingleExcitation SE(theta) with
theta = 2*beta; our pair-double matrix angle delta to DoubleExcitation
DE(theta = 2*delta). Per-gate log-overheads for a gate crossing the cut:

    u_single(beta)  = 2 ln(|cos(beta/2)| + |sin(beta/2)|)      ~ |beta|
    u_pairdbl(delta) = 8 ln(|cos(delta/8)| + |sin(delta/8)|)   ~ |delta|

Circuit overhead phi = 2 exp(2 sum_u) - 1; shots scale as O(phi^2 / eps^2).
Budget: phi <= phi_max  <=>  sum_u <= u_max = 0.5 ln((phi_max + 1)/2).
"""
from __future__ import annotations

import numpy as np


def u_single(beta) -> np.ndarray:
    """Return the optimal-cut log-overhead for single-exchange angles."""

    b = np.abs(np.asarray(beta, dtype=np.float64))
    return 2.0 * np.log(np.abs(np.cos(b / 2)) + np.abs(np.sin(b / 2)))


def u_pair_double(delta) -> np.ndarray:
    """Return the optimal-cut log-overhead for pair-double angles."""

    d = np.abs(np.asarray(delta, dtype=np.float64))
    return 8.0 * np.log(np.abs(np.cos(d / 8)) + np.abs(np.sin(d / 8)))


def phi_from_u(total_u: float) -> float:
    """Convert cumulative log-overhead into the circuit factor phi."""

    return 2.0 * float(np.exp(2.0 * total_u)) - 1.0


def u_max_from_phi(phi_max: float) -> float:
    """Convert a hard circuit-factor limit into a log-overhead budget."""

    return 0.5 * float(np.log((phi_max + 1.0) / 2.0))


# ---------------- Hamiltonian-informed pair score & partitioning ----------------

def pair_score(h1e: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """A_pq on spatial orbitals: single- and pair-hop relevance of (p,q).

    Fermionic-level analogue of the manuscript's Eq. (24) restricted to
    excitation-relevant terms: |h_pq| + sum_r |(pq|rr)| (one-index hops)
    + |(pq|qp)| (exchange) + |(pq|pq)| (pair hop). Symmetric, zero diagonal.
    """
    no = h1e.shape[0]
    A = np.abs(h1e).astype(np.float64).copy()
    A += np.abs(np.einsum("pqrr->pq", eri))
    for p in range(no):
        for q in range(no):
            A[p, q] += abs(eri[p, q, q, p]) + abs(eri[p, q, p, q])
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    return A


def kl_bipartition(A: np.ndarray, seed: int = 0, sweeps: int = 60) -> tuple[list[int], list[int]]:
    """Balanced bipartition of spatial orbitals minimizing cross weight sum(A[p,q]).

    Simple Kernighan-Lin-style greedy pair swapping; exact enough at no <= 20.
    """
    no = A.shape[0]
    rng = np.random.default_rng(seed)
    left = list(range(no // 2))
    right = list(range(no // 2, no))

    def cross(l, r):
        return A[np.ix_(l, r)].sum()

    best = cross(left, right)
    for _ in range(sweeps):
        improved = False
        for i in range(len(left)):
            for j in range(len(right)):
                l2 = left.copy(); r2 = right.copy()
                l2[i], r2[j] = r2[j], l2[i]
                c = cross(l2, r2)
                if c < best - 1e-15:
                    left, right, best = l2, r2, c
                    improved = True
        if not improved:
            break
        rng.shuffle(left); rng.shuffle(right)
    return sorted(left), sorted(right)


def split_pairs(no: int, left: list[int]) -> tuple[list, list]:
    """All orbital pairs (p<q) split into intra-block and cross-block lists."""
    lset = set(left)
    intra, cross = [], []
    for p in range(no):
        for q in range(p + 1, no):
            (intra if ((p in lset) == (q in lset)) else cross).append((p, q))
    return intra, cross


def budget_cross_pairs(cross: list, A: np.ndarray, phi_max: float, beta_cap: float,
                       gate_kinds: tuple[str, ...] = ("single", "single", "double")):
    """Greedy Nakamura-style budget: admit cross pairs by descending A_pq.

    Each admitted pair costs its worst case at |beta| = beta_cap for every
    gate placed on it (per layer: alpha single + beta single + pair double,
    per `gate_kinds`). Returns (admitted_pairs, u_budget, u_committed).
    """
    u_budget = u_max_from_phi(phi_max)
    per_pair = 0.0
    for k in gate_kinds:
        per_pair += float(u_single(beta_cap)) if k == "single" else float(u_pair_double(beta_cap))
    order = sorted(cross, key=lambda pq: -A[pq[0], pq[1]])
    admitted, committed = [], 0.0
    for pq in order:
        if committed + per_pair <= u_budget + 1e-12:
            admitted.append(pq)
            committed += per_pair
    return admitted, u_budget, committed


def balanced_bipartitions(n: int):
    """All balanced qubit splits (qubit 0 pinned to side A to remove mirrors)."""
    import itertools

    for combo in itertools.combinations(range(1, n), n // 2 - 1):
        yield frozenset((0,) + combo)


def best_cut_phi(edges: list, betas: np.ndarray) -> tuple[float, frozenset]:
    """Min over balanced bipartitions of phi for Givens angles betas[layer, edge]."""
    best, best_p = np.inf, None
    for part in balanced_bipartitions(1 + max(max(e) for e in edges)):
        tot = 0.0
        for k, (i, j) in enumerate(edges):
            if (i in part) != (j in part):
                tot += float(np.sum(u_single(betas[:, k])))
        p = phi_from_u(tot)
        if p < best:
            best, best_p = p, part
    return best, best_p


def circuit_phi(single_betas_cross: np.ndarray, double_deltas_cross: np.ndarray) -> float:
    """Exact overhead phi of a circuit from its cross-cut angles."""
    total = 0.0
    if len(single_betas_cross):
        total += float(np.sum(u_single(single_betas_cross)))
    if len(double_deltas_cross):
        total += float(np.sum(u_pair_double(double_deltas_cross)))
    return phi_from_u(total)
