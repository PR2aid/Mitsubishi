"""Dense Jordan-Wigner reference implementations for surgical cross-validation.

Everything here is exponentially scaling and used ONLY in tests (n <= ~10
qubits): dense fermionic operators with explicit JW strings, the dense
molecular Hamiltonian from CAS integrals, and dense gate matrices for both
gate conventions. Basis convention matches sector.py: basis index i is the
bitstring with bit k = occupation of spin-orbital k; alpha orbitals are
qubits 0..norb-1, beta are norb..2*norb-1; JW strings act on lower qubits.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def annihilation(n_qubits: int, p: int) -> np.ndarray:
    """Dense JW a_p with Z-string on qubits < p."""
    dim = 2 ** n_qubits
    A = np.zeros((dim, dim), dtype=np.float64)
    lower = (1 << p) - 1
    for i in range(dim):
        if (i >> p) & 1:
            j = i & ~(1 << p)
            sign = (-1.0) ** bin(i & lower).count("1")
            A[j, i] = sign
    return A


def dense_hamiltonian(h1e: np.ndarray, eri: np.ndarray, ecore: float = 0.0) -> np.ndarray:
    """Dense H (electronic + ecore) from CAS integrals, blocked-spin JW.

    H = sum_pq,s h1[p,q] a+_ps a_qs
        + 1/2 sum_pqrs,st (pq|rs) a+_ps a+_rt a_st a_qs   (chemist notation)
    """
    no = h1e.shape[0]
    n = 2 * no
    dim = 2 ** n
    a = [annihilation(n, k) for k in range(n)]
    ad = [m.T.copy() for m in a]

    def so(p, spin):  # spin-orbital index
        return p + no * spin

    H = np.zeros((dim, dim), dtype=np.float64)
    for p in range(no):
        for q in range(no):
            if abs(h1e[p, q]) < 1e-14:
                continue
            for s in (0, 1):
                H += h1e[p, q] * ad[so(p, s)] @ a[so(q, s)]
    for p in range(no):
        for q in range(no):
            for r in range(no):
                for s_ in range(no):
                    v = eri[p, q, r, s_]
                    if abs(v) < 1e-14:
                        continue
                    for s1 in (0, 1):
                        for s2 in (0, 1):
                            H += 0.5 * v * (
                                ad[so(p, s1)] @ ad[so(r, s2)] @ a[so(s_, s2)] @ a[so(q, s1)]
                            )
    return H + ecore * np.eye(dim)


# ---------------- dense gates ----------------

def dense_single_qubitconv(n_qubits: int, i: int, j: int, beta: float) -> np.ndarray:
    """Qubit-convention Givens on qubits i<j (manuscript Eq. 8): no JW string."""
    dim = 2 ** n_qubits
    G = np.eye(dim)
    c, s = np.cos(beta), np.sin(beta)
    for k in range(dim):
        bi, bj = (k >> i) & 1, (k >> j) & 1
        if bi == 1 and bj == 0:          # |10> component (p=i occupied)
            k01 = (k & ~(1 << i)) | (1 << j)
            # x01' = c x01 + s x10 ; x10' = -s x01 + c x10
            G[k01, k01] = c; G[k, k] = c
            G[k01, k] = s;  G[k, k01] = -s
    return G


def dense_single_fermconv(n_qubits: int, p: int, q: int, beta: float) -> np.ndarray:
    """exp(beta (a_q^dag a_p - a_p^dag a_q)) with full JW strings."""
    a_p = annihilation(n_qubits, p)
    a_q = annihilation(n_qubits, q)
    K = a_q.T @ a_p - a_p.T @ a_q
    return expm(beta * K)


def dense_pairdouble_qubitconv(n_qubits: int, norb: int, p: int, q: int,
                               delta: float) -> np.ndarray:
    """Qubit-convention pair-double: rotates p-pair-occ <-> q-pair-occ configs."""
    dim = 2 ** n_qubits
    pa, pb, qa, qb = p, p + norb, q, q + norb
    G = np.eye(dim)
    c, s = np.cos(delta), np.sin(delta)
    for k in range(dim):
        occ = lambda b: (k >> b) & 1
        if occ(pa) and occ(pb) and not occ(qa) and not occ(qb):   # "10" = p-pair
            k01 = (k & ~(1 << pa) & ~(1 << pb)) | (1 << qa) | (1 << qb)
            G[k01, k01] = c; G[k, k] = c
            G[k01, k] = s;  G[k, k01] = -s
    return G


def dense_pairdouble_fermconv(n_qubits: int, norb: int, p: int, q: int,
                              delta: float) -> np.ndarray:
    """exp(delta (a_qa^dag a_qb^dag a_pb a_pa - h.c.)) with JW strings."""
    pa, pb, qa, qb = p, p + norb, q, q + norb
    a = {k: annihilation(n_qubits, k) for k in (pa, pb, qa, qb)}
    T = a[qa].T @ a[qb].T @ a[pb] @ a[pa]
    return expm(delta * (T - T.T))
