"""Exact sector energies via PySCF FCI contraction, differentiable in torch.

E(c) = <c|H|c> / <c|c> + ecore, with H|c> computed by
pyscf.fci.direct_spin1.contract_2e on the (dimA, dimB) CI matrix.
The backward pass uses d<c|H|c>/dc = 2 H|c> (H real symmetric), so memory
stays at a few sector vectors regardless of circuit depth.
"""
from __future__ import annotations

import numpy as np
import torch
from pyscf.fci import direct_spin1

from .chemistry import CASProblem


class _SigmaCache:
    """Precomputed effective two-electron integrals for contract_2e."""

    def __init__(self, prob: CASProblem):
        self.norb = prob.norb
        self.nelec = prob.nelec
        self.h2eff = direct_spin1.absorb_h1e(
            prob.h1e, prob.eri, prob.norb, prob.nelec, 0.5
        )

    def sigma(self, c: np.ndarray) -> np.ndarray:
        return np.asarray(
            direct_spin1.contract_2e(self.h2eff, c, self.norb, self.nelec)
        )


class _QuadForm(torch.autograd.Function):
    """<c|H|c> with grad 2*H|c> (unnormalized quadratic form)."""

    @staticmethod
    def forward(ctx, c: torch.Tensor, cache: _SigmaCache):
        c_np = c.detach().cpu().numpy().astype(np.float64, copy=False)
        sig = cache.sigma(np.ascontiguousarray(c_np))
        sig_t = torch.from_numpy(sig).to(device=c.device, dtype=c.dtype)
        ctx.save_for_backward(sig_t)
        return (c * sig_t).sum()

    @staticmethod
    def backward(ctx, grad_out):
        (sig_t,) = ctx.saved_tensors
        return grad_out * 2.0 * sig_t, None


def electronic_energy(c: torch.Tensor, cache: _SigmaCache) -> torch.Tensor:
    """Rayleigh quotient <c|H|c>/<c|c> (electronic, no ecore)."""
    quad = _QuadForm.apply(c, cache)
    return quad / (c * c).sum()


def make_energy_fn(prob: CASProblem):
    """Return a differentiable total-energy closure and contraction cache."""

    cache = _SigmaCache(prob)

    def total_energy(c: torch.Tensor) -> torch.Tensor:
        return electronic_energy(c, cache) + prob.ecore

    return total_energy, cache


def energy_and_variance(c: torch.Tensor, prob: CASProblem,
                        cache: _SigmaCache | None = None) -> tuple[float, float]:
    """Reference-free certificate: (E_total, Var(H)) from ONE contraction.

    Var(H) = <c|H^2|c>/<c|c> - (<c|H|c>/<c|c>)^2 = ||sigma||^2/<c|c> - E_el^2,
    using the same sigma = H|c> that the energy already needs. Var vanishes
    exactly on eigenstates; by Weinstein's inequality some eigenvalue of H
    lies within sqrt(Var) of E. This certifies solution quality with NO
    exact reference -- the verification tool for sectors beyond CASCI reach.
    (Variance is shift-independent: ecore cancels.)
    """
    if cache is None:
        cache = _SigmaCache(prob)
    c_np = np.ascontiguousarray(c.detach().cpu().numpy().astype(np.float64, copy=False))
    sig = cache.sigma(c_np)
    nrm2 = float(np.vdot(c_np, c_np).real)
    e_el = float(np.vdot(c_np, sig).real) / nrm2
    h2 = float(np.vdot(sig, sig).real) / nrm2
    var = max(0.0, h2 - e_el * e_el)
    return e_el + prob.ecore, var
