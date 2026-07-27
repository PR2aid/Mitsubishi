"""Exact (Na, Nb)-sector simulator for particle-conserving Givens circuits.

The state is a real float64 torch tensor c of shape (dimA, dimB), where
dimA = C(norb, na) alpha determinant strings and dimB = C(norb, nb) beta
strings, in PySCF `cistring` address order (so PySCF FCI kernels apply
directly). Qubit p of the 2*norb-qubit register is spin-orbital p:
alpha orbitals are qubits 0..norb-1, beta orbitals qubits norb..2*norb-1
(blocked Jordan-Wigner ordering). A computational basis state is the
pair (alpha string, beta string); the full-register bitstring is
alpha | (beta << norb).

Gates (all real-orthogonal, particle- and Sz-conserving):

* Givens single exchange G_pq(beta) on same-spin orbitals (p, q), p < q.
  Qubit convention ("qubit"): the two-qubit subspace rotation of the
  manuscript's Eq. (8) -- no Jordan-Wigner string, i.e. a hard-core-boson
  hop. Fermionic convention ("fermionic"): additionally multiplies the
  off-diagonal terms by the JW parity of the occupied orbitals strictly
  between p and q, i.e. exp(beta (a_q^dag a_p - a_p^dag a_q)).
  Amplitude convention (fixing the paper's Eq. (8) with i=p, j=q):
      x01' = cos(b) x01 + s sin(b) x10
      x10' = -s sin(b) x01 + cos(b) x10
  where x01 = amplitude with q occupied / p empty, x10 = p occupied / q
  empty, and s is the sign factor (s=+1 in the qubit convention).

* Pair-double exchange D_pq(delta): rotates |p_a p_b occ, q_a q_b empty>
  with |q_a q_b occ, p_a p_b empty> (PennyLane DoubleExcitation on the
  4 qubits (p_a, q_a, p_b, q_b) restricted to paired configurations).
  Same amplitude convention with x01 = q-pair occupied.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from pyscf.fci import cistring

REAL = torch.float64


def _popcount64(x: np.ndarray) -> np.ndarray:
    """Vectorized popcount for int64 arrays (SWAR; numpy-version agnostic)."""
    x = x.astype(np.uint64, copy=True)
    m1 = np.uint64(0x5555555555555555)
    m2 = np.uint64(0x3333333333333333)
    m4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    h01 = np.uint64(0x0101010101010101)
    x = x - ((x >> np.uint64(1)) & m1)
    x = (x & m2) + ((x >> np.uint64(2)) & m2)
    x = (x + (x >> np.uint64(4))) & m4
    return ((x * h01) >> np.uint64(56)).astype(np.int64)


class SpinStrings:
    """Determinant strings for one spin channel, in PySCF address order."""

    def __init__(self, norb: int, nocc: int):
        self.norb = int(norb)
        self.nocc = int(nocc)
        strs = np.asarray(cistring.make_strings(range(norb), nocc), dtype=np.int64)
        # PySCF addresses are ordered by ascending binary value; assert to be safe.
        if not np.all(np.diff(strs) > 0):
            raise AssertionError("cistring order is not ascending; address lookup invalid")
        self.strs = strs
        self.dim = len(strs)

    def address(self, strings: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.strs, strings)
        if not np.all(self.strs[idx] == strings):
            raise AssertionError("string not found in sector")
        return idx

    @lru_cache(maxsize=None)
    def single_map(self, p: int, q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Index map for the excitation p <-> q (p < q) within this spin.

        Returns (idx_pocc, idx_qocc, sign): addresses of strings with
        (p occ, q empty), their partners (q occ, p empty), and the JW/
        fermionic parity sign of the occupied orbitals strictly between
        p and q (evaluated on the p-occupied string; identical for the
        partner since only p, q flip).
        """
        if not (0 <= p < q < self.norb):
            raise ValueError("need 0 <= p < q < norb")
        s = self.strs
        occ_p = (s >> p) & 1 == 1
        occ_q = (s >> q) & 1 == 1
        sel = occ_p & ~occ_q
        src = s[sel]
        dst = (src & ~(np.int64(1) << p)) | (np.int64(1) << q)
        between = (np.int64(1) << q) - (np.int64(1) << (p + 1))  # bits p+1..q-1
        par = _popcount64(src & between) & 1
        sign = 1.0 - 2.0 * par
        return (np.flatnonzero(sel), self.address(dst), sign.astype(np.float64))


@dataclass
class Sector:
    """The (Na, Nb) sector: string tables and torch-side cached index maps."""

    norb: int
    na: int
    nb: int

    def __post_init__(self):
        self.alpha = SpinStrings(self.norb, self.na)
        self.beta = (
            self.alpha if self.nb == self.na else SpinStrings(self.norb, self.nb)
        )
        self.dimA, self.dimB = self.alpha.dim, self.beta.dim
        self._t_cache: dict = {}

    # ---------------- initial states ----------------
    def hf_index(self) -> tuple[int, int]:
        hf_a = int((1 << self.na) - 1)
        hf_b = int((1 << self.nb) - 1)
        return int(self.alpha.address(np.array([hf_a]))[0]), int(
            self.beta.address(np.array([hf_b]))[0]
        )

    def initial_state(self, hdiag: np.ndarray | None = None) -> torch.Tensor:
        """Lowest-diagonal determinant (paper's diag init); HF if hdiag is None."""
        c = torch.zeros(self.dimA, self.dimB, dtype=REAL)
        if hdiag is None:
            ia, ib = self.hf_index()
        else:
            k = int(np.argmin(hdiag.reshape(self.dimA, self.dimB).ravel()))
            ia, ib = divmod(k, self.dimB)
        c[ia, ib] = 1.0
        return c

    # ---------------- torch-side cached maps ----------------
    def _single_t(self, spin: str, p: int, q: int, device):
        key = ("s", spin, p, q, str(device))
        if key not in self._t_cache:
            table = self.alpha if spin == "a" else self.beta
            i_p, i_q, sgn = table.single_map(p, q)
            self._t_cache[key] = (
                torch.from_numpy(i_p).to(device),
                torch.from_numpy(i_q).to(device),
                torch.from_numpy(sgn).to(device=device, dtype=REAL),
            )
        return self._t_cache[key]

    def _pair_t(self, p: int, q: int, device):
        key = ("d", p, q, str(device))
        if key not in self._t_cache:
            ia_p, ia_q, sa = self.alpha.single_map(p, q)
            ib_p, ib_q, sb = self.beta.single_map(p, q)
            dimB = self.dimB
            f = (ia_p[:, None] * dimB + ib_p[None, :]).ravel()
            t = (ia_q[:, None] * dimB + ib_q[None, :]).ravel()
            s = (sa[:, None] * sb[None, :]).ravel()
            self._t_cache[key] = (
                torch.from_numpy(f).to(device),
                torch.from_numpy(t).to(device),
                torch.from_numpy(s).to(device=device, dtype=REAL),
            )
        return self._t_cache[key]

    # ---------------- gates (autograd-safe, out-of-place) ----------------
    def apply_single(self, c: torch.Tensor, spin: str, p: int, q: int,
                     beta: torch.Tensor, convention: str = "qubit") -> torch.Tensor:
        """Apply G_pq(beta) on same-spin orbitals (p<q). spin in {'a','b'}."""
        i_p, i_q, sgn = self._single_t(spin, p, q, c.device)
        if convention == "qubit":
            s = torch.ones_like(sgn)
        elif convention == "fermionic":
            s = sgn
        else:
            raise ValueError(convention)
        beta = beta.to(REAL) if torch.is_tensor(beta) else torch.tensor(beta, dtype=REAL)
        cb, sb_ = torch.cos(beta), torch.sin(beta)
        if spin == "a":
            x10 = c.index_select(0, i_p)          # p occupied rows
            x01 = c.index_select(0, i_q)          # q occupied rows
            new01 = cb * x01 + sb_ * s[:, None] * x10
            new10 = cb * x10 - sb_ * s[:, None] * x01
            return c.index_copy(0, i_q, new01).index_copy(0, i_p, new10)
        else:
            x10 = c.index_select(1, i_p)
            x01 = c.index_select(1, i_q)
            new01 = cb * x01 + sb_ * s[None, :] * x10
            new10 = cb * x10 - sb_ * s[None, :] * x01
            return c.index_copy(1, i_q, new01).index_copy(1, i_p, new10)

    def apply_pair_double(self, c: torch.Tensor, p: int, q: int,
                          delta: torch.Tensor, convention: str = "qubit") -> torch.Tensor:
        """Apply the pair-double exchange D_pq(delta) on paired configs."""
        f, t, sgn = self._pair_t(p, q, c.device)
        s = torch.ones_like(sgn) if convention == "qubit" else sgn
        flat = c.reshape(-1)
        x10 = flat.index_select(0, f)             # p-pair occupied
        x01 = flat.index_select(0, t)             # q-pair occupied
        delta = delta.to(REAL) if torch.is_tensor(delta) else torch.tensor(delta, dtype=REAL)
        cb, sb_ = torch.cos(delta), torch.sin(delta)
        new01 = cb * x01 + sb_ * s * x10
        new10 = cb * x10 - sb_ * s * x01
        out = flat.index_copy(0, t, new01).index_copy(0, f, new10)
        return out.reshape(c.shape)

    # ---------------- embedding (for dense cross-validation) ----------------
    def embed(self, c: np.ndarray) -> np.ndarray:
        """Embed sector state into the full 2^(2*norb) vector (blocked JW)."""
        n = 2 * self.norb
        full = np.zeros(2 ** n, dtype=np.float64)
        for ia, sa in enumerate(self.alpha.strs):
            base = int(sa)
            for ib, sb in enumerate(self.beta.strs):
                full[base | (int(sb) << self.norb)] = c[ia, ib]
        return full
