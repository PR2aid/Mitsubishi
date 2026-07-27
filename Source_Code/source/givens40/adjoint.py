"""Adjoint-mode differentiation of the sector circuit.

Memory: O(3 sector vectors) independent of circuit depth (vs autograd's
O(#gates) intermediates). Time: ~2.5x one forward pass per gradient.
Method: standard reversible-circuit adjoint -- the backward sweep undoes
each orthogonal gate to recover the pre-gate state, accumulates the angle
gradient from the rotation generator (supported only on the touched
amplitudes), and pulls the adjoint vector back through the gate.

Gradient enters torch through a single autograd.Function taking the flat
angle vector, so upstream reparametrizations (tanh caps, spin sharing)
remain ordinary differentiable torch ops.
"""
from __future__ import annotations

import numpy as np
import torch

from .sector import Sector, REAL


class GateSpec:
    """Compact description of one reversible sector-space rotation."""

    __slots__ = ("kind", "spin", "p", "q", "sign_mode")

    def __init__(self, kind, spin, p, q, sign_mode):
        self.kind, self.spin, self.p, self.q, self.sign_mode = kind, spin, p, q, sign_mode


def _maps(sector: Sector, g: GateSpec, device):
    if g.kind == "s":
        return sector._single_t(g.spin, g.p, g.q, device)
    return sector._pair_t(g.p, g.q, device)


def _rot_inplace(c: torch.Tensor, g: GateSpec, maps, angle: float, inverse: bool = False):
    """Apply the gate (or its inverse) in place. No autograd."""
    i_p, i_q, sgn = maps
    s = sgn if g.sign_mode == "fermionic" else None
    cb = float(np.cos(angle))
    sb = float(np.sin(angle)) * (-1.0 if inverse else 1.0)
    if g.kind == "s" and g.spin == "a":
        u = c.index_select(0, i_p)   # x10 (p occupied)
        v = c.index_select(0, i_q)   # x01 (q occupied)
        f = s[:, None] if s is not None else 1.0
        c.index_copy_(0, i_q, cb * v + sb * f * u)
        c.index_copy_(0, i_p, cb * u - sb * f * v)
    elif g.kind == "s":
        u = c.index_select(1, i_p)
        v = c.index_select(1, i_q)
        f = s[None, :] if s is not None else 1.0
        c.index_copy_(1, i_q, cb * v + sb * f * u)
        c.index_copy_(1, i_p, cb * u - sb * f * v)
    else:
        flat = c.view(-1)
        u = flat.index_select(0, i_p)
        v = flat.index_select(0, i_q)
        f = s if s is not None else 1.0
        flat.index_copy_(0, i_q, cb * v + sb * f * u)
        flat.index_copy_(0, i_p, cb * u - sb * f * v)


def _angle_grad(lam: torch.Tensor, c_before: torch.Tensor, g: GateSpec, maps,
                angle: float) -> float:
    """<lam | dR/dangle | c_before>; dR/dangle is supported on touched entries.

    dR/dangle on the pair (x01, x10): [[-sin, s cos], [-s cos, -sin]].
    """
    i_p, i_q, sgn = maps
    cb, sb = float(np.cos(angle)), float(np.sin(angle))
    if g.kind == "s" and g.spin == "a":
        u = c_before.index_select(0, i_p); v = c_before.index_select(0, i_q)
        lu = lam.index_select(0, i_p);     lv = lam.index_select(0, i_q)
        f = sgn[:, None] if g.sign_mode == "fermionic" else 1.0
    elif g.kind == "s":
        u = c_before.index_select(1, i_p); v = c_before.index_select(1, i_q)
        lu = lam.index_select(1, i_p);     lv = lam.index_select(1, i_q)
        f = sgn[None, :] if g.sign_mode == "fermionic" else 1.0
    else:
        cf = c_before.view(-1); lf = lam.view(-1)
        u = cf.index_select(0, i_p); v = cf.index_select(0, i_q)
        lu = lf.index_select(0, i_p); lv = lf.index_select(0, i_q)
        f = sgn if g.sign_mode == "fermionic" else 1.0
    d01 = -sb * v + cb * f * u     # d(new x01)/dangle
    d10 = -cb * f * v - sb * u     # d(new x10)/dangle
    return float((lv * d01).sum() + (lu * d10).sum())


class _AdjointCircuitFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, angles: torch.Tensor, sector: Sector, gates: list, c0: torch.Tensor):
        a = angles.detach().cpu().numpy().astype(np.float64)
        c = c0.clone()
        with torch.no_grad():
            for g, ang in zip(gates, a):
                _rot_inplace(c, g, _maps(sector, g, c.device), float(ang))
        ctx.sector, ctx.gates, ctx.a = sector, gates, a
        ctx.save_for_backward(c)
        return c.clone()

    @staticmethod
    def backward(ctx, grad_out):
        (c_final,) = ctx.saved_tensors
        sector, gates, a = ctx.sector, ctx.gates, ctx.a
        lam = grad_out.clone()
        c = c_final.clone()
        grads = np.zeros(len(gates), dtype=np.float64)
        with torch.no_grad():
            for k in range(len(gates) - 1, -1, -1):
                g, ang = gates[k], float(a[k])
                maps = _maps(sector, g, c.device)
                _rot_inplace(c, g, maps, ang, inverse=True)   # recover c_before
                grads[k] = _angle_grad(lam, c, g, maps, ang)
                _rot_inplace(lam, g, maps, ang, inverse=True)  # lam = R^T lam
        g_t = torch.from_numpy(grads).to(dtype=REAL, device=grad_out.device)
        return g_t, None, None, None


def adjoint_forward(angles: torch.Tensor, sector: Sector, gates: list,
                    c0: torch.Tensor) -> torch.Tensor:
    """Differentiable circuit application with O(1)-depth memory."""
    return _AdjointCircuitFn.apply(angles, sector, gates, c0)
