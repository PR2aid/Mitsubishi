"""Enhancement 4: exact quasi-probability (QPD) execution of cut circuits.

Validates the circuit-cutting claim end-to-end: a partitioned circuit's
energy is reconstructed EXACTLY (channel enumeration, no sampling) from
simulations of the two halves only — no joint 2^n statevector is ever
built during execution. Cross-block Givens gates are decomposed via the
Mitarai-Fujii quasi-probability identity applied to their two commuting
crossing Pauli rotations (Nakamura Eq. 1, generalized from Z(x)Z to
arbitrary P(x)Q):

  R_PQ(t) rho R_PQ(t)^dag = c^2 rho + s^2 (P(x)Q) rho (P(x)Q)
    - cs sum_{a1,a2 in {+-1}} a1 a2 [ M_P(a1) (x) R_Q(-a2 pi/2)
                                    + R_P(-a1 pi/2) (x) M_Q(a2) ] rho [..]^dag

with c = cos(t/2), s = sin(t/2), M_P(a) = (I + aP)/2 (single-Kraus CP map),
R_P(phi) = exp(-i phi/2 P). Every term is a product of LOCAL single-Kraus
maps, so each enumeration branch factorizes the pure-state simulation
across the cut; observables factorize because Pauli strings do.

The qubit-convention Givens G(beta) (manuscript Eq. 8) equals
SE(theta = 2 beta) and hence two commuting crossing rotations
R_XY(-beta) . R_YX(+beta) (orientation fixed by the unit test).

Scaling: enumeration costs 10^(2 k_cut) branches — a VALIDATOR for small
k_cut, which is exactly the regime the phi budget enforces. On hardware
one samples branches with probability |c_i|/gamma. The naive one-gate
Mitarai-Fujii overhead (1 + 8|cs|)... per-rotation sum |coeff| = 1 + 4|cs|
composes to more than the optimal Harrow-Lowe grouped value reported by
overhead.py; both numbers are returned so the distinction stays visible.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

I2 = np.eye(2, dtype=complex)
PAULI = {
    "I": I2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def _m_op(P: str, a: int) -> np.ndarray:
    return 0.5 * (I2 + a * PAULI[P])


def _r_op(P: str, phi: float) -> np.ndarray:
    return np.cos(phi / 2) * I2 - 1j * np.sin(phi / 2) * PAULI[P]


def rotation_qpd_terms(P: str, Q: str, t: float):
    """10 local-Kraus terms (K_first, K_second, coeff) for the R_PQ(t) channel."""
    c, s = np.cos(t / 2), np.sin(t / 2)
    terms = [(I2, I2, c * c), (PAULI[P], PAULI[Q], s * s)]
    for a1, a2 in itertools.product((1, -1), repeat=2):
        w = -c * s * a1 * a2
        terms.append((_m_op(P, a1), _r_op(Q, -a2 * np.pi / 2), w))
        terms.append((_r_op(P, -a1 * np.pi / 2), _m_op(Q, a2), w))
    return terms


def _givens_branches(beta: float):
    """G(beta) = R_YX(-beta) . R_XY(+beta), P acting on the first gate qubit.

    (Verified numerically: exp(-i(-b)/2 Y_i X_j) exp(-i(+b)/2 X_i Y_j)
    equals the manuscript Eq. 8 matrix in LSB-first gate indexing.)
    """
    t1 = rotation_qpd_terms("X", "Y", +beta)   # applied first
    t2 = rotation_qpd_terms("Y", "X", -beta)   # applied second
    out = []
    for (a1, b1, w1), (a2, b2, w2) in itertools.product(t1, t2):
        out.append((a2 @ a1, b2 @ b1, w1 * w2))
    return out


# ---------------- gate containers ----------------

@dataclass
class Gate:
    """Local or cut gate consumed by the exact QPD validator."""

    kind: str                    # "local" | "cut"
    qubits: tuple                # global qubit indices (gate order)
    mat: np.ndarray | None = None   # for "local": 2^k x 2^k dense
    beta: float | None = None       # for "cut": Givens matrix angle


def givens4(beta: float) -> np.ndarray:
    """Manuscript Eq. 8 on (q_i, q_j), basis |q_j q_i> = |00>,|01>,|10>,|11>.

    Index convention: LSB-first within the gate — basis index b = b_i + 2 b_j
    where b_i is the occupation of the FIRST listed qubit. |01> (index 2:
    b_i=0, b_j=1) is 'q occupied'; |10> (index 1) is 'p occupied'.
    x01' = c x01 + s x10 ; x10' = -s x01 + c x10.
    """
    c, s = np.cos(beta), np.sin(beta)
    G = np.eye(4, dtype=complex)
    G[2, 2], G[2, 1], G[1, 2], G[1, 1] = c, s, -s, c
    return G


def pairdouble16(delta: float) -> np.ndarray:
    """Qubit-convention pair-double on gate qubits (pa, qa, pb, qb), LSB-first."""
    G = np.eye(16, dtype=complex)
    c, s = np.cos(delta), np.sin(delta)
    i_p = 0b0101      # pa, pb occupied  (bit0=pa, bit1=qa, bit2=pb, bit3=qb)
    i_q = 0b1010      # qa, qb occupied
    G[i_q, i_q] = c; G[i_p, i_p] = c
    G[i_q, i_p] = s; G[i_p, i_q] = -s
    return G


# ---------------- simulators ----------------

def _apply(psi: np.ndarray, mat: np.ndarray, qubits: tuple, n: int) -> np.ndarray:
    """Apply a 2^k x 2^k gate on `qubits` of an n-qubit statevector.

    Conventions: state index bit q = qubit q (LSB-first register); gate index
    bit i = qubits[i] (LSB-first gate order). numpy's reshape([2]*n) exposes
    axis a = qubit n-1-a, so gate axis order (MSB..LSB) = reversed(qubits).
    """
    k = len(qubits)
    src = tuple(n - 1 - q for q in reversed(qubits))   # axes for gate MSB..LSB
    t = np.moveaxis(psi.reshape([2] * n), src, range(k))
    rest = t.shape[k:]
    t = (mat @ t.reshape(2 ** k, -1)).reshape([2] * k + list(rest))
    return np.moveaxis(t, range(k), src).reshape(-1)


def full_energy(gates: list[Gate], n: int, init_bits: int, H: np.ndarray) -> float:
    """Evaluate the uncut dense reference used by the small validator."""

    psi = np.zeros(2 ** n, dtype=complex)
    psi[init_bits] = 1.0
    for g in gates:
        m = givens4(g.beta) if g.kind == "cut" else g.mat
        psi = _apply(psi, m, g.qubits, n)
    return float((psi.conj() @ (H @ psi)).real)


def pauli_decompose(H: np.ndarray, n: int, tol: float = 1e-12):
    """Decompose a dense n-qubit Hamiltonian into nonzero Pauli terms."""

    terms = []
    for combo in itertools.product("IXYZ", repeat=n):
        P = np.array([[1.0 + 0j]])
        for ch in combo:               # qubit 0 first -> becomes LSB via kron order
            P = np.kron(PAULI[ch], P)
        c = (np.trace(P.conj().T @ H) / (2 ** n)).real
        if abs(c) > tol:
            terms.append((float(c), combo))
    return terms


def _pauli_mat(combo) -> np.ndarray:
    P = np.array([[1.0 + 0j]])
    for ch in combo:
        P = np.kron(PAULI[ch], P)
    return P


def qpd_energy(gates: list[Gate], n: int, side_a: set, init_bits: int,
               pauli_terms: list) -> tuple[float, float]:
    """Exact QPD reconstruction of <H>; returns (energy, gamma_naive)."""
    A = sorted(side_a)
    B = [q for q in range(n) if q not in side_a]
    posA = {q: k for k, q in enumerate(A)}
    posB = {q: k for k, q in enumerate(B)}
    nA, nB = len(A), len(B)

    for g in gates:
        if g.kind == "local":
            sides = {q in side_a for q in g.qubits}
            if len(sides) != 1:
                raise ValueError(f"local gate {g.qubits} crosses the cut")

    cuts = [g for g in gates if g.kind == "cut"]
    branch_sets = [_givens_branches(g.beta) for g in cuts]

    # cache Pauli matrices per side
    pcacheA, pcacheB = {}, {}

    def side_terms():
        out = []
        for c, combo in pauli_terms:
            sA = tuple(combo[q] for q in A)
            sB = tuple(combo[q] for q in B)
            if sA not in pcacheA:
                pcacheA[sA] = _pauli_mat(sA)
            if sB not in pcacheB:
                pcacheB[sB] = _pauli_mat(sB)
            out.append((c, pcacheA[sA], pcacheB[sB]))
        return out

    terms = side_terms()
    iA0 = sum(((init_bits >> q) & 1) << posA[q] for q in A)
    iB0 = sum(((init_bits >> q) & 1) << posB[q] for q in B)

    total, gamma = 0.0, 0.0
    for choice in itertools.product(*[range(len(b)) for b in branch_sets]):
        psiA = np.zeros(2 ** nA, dtype=complex); psiA[iA0] = 1.0
        psiB = np.zeros(2 ** nB, dtype=complex); psiB[iB0] = 1.0
        w = 1.0
        ci = 0
        for g in gates:
            if g.kind == "local":
                if g.qubits[0] in side_a:
                    psiA = _apply(psiA, g.mat, tuple(posA[q] for q in g.qubits), nA)
                else:
                    psiB = _apply(psiB, g.mat, tuple(posB[q] for q in g.qubits), nB)
            else:
                Kf, Ks, wb = branch_sets[ci][choice[ci]]
                ci += 1
                w *= wb
                for q, K in zip(g.qubits, (Kf, Ks)):
                    if q in side_a:
                        psiA = _apply(psiA, K, (posA[q],), nA)
                    else:
                        psiB = _apply(psiB, K, (posB[q],), nB)
        gamma += abs(w)
        acc = 0.0
        for c, PA, PB in terms:
            acc += c * ((psiA.conj() @ (PA @ psiA)) * (psiB.conj() @ (PB @ psiB))).real
        total += w * acc
    return total, gamma
