"""Ansatz assembly + variational optimization for the sector track."""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field

import numpy as np
import torch

from .chemistry import CASProblem
from .sector import Sector, REAL
from .energy import make_energy_fn
from . import overhead as oh


@dataclass
class AnsatzConfig:
    """Frozen circuit, partition, and parameterization choices for one run."""

    layers: int = 2
    gates: str = "sd"              # "s" = singles only, "sd" = singles + pair doubles
    convention: str = "qubit"      # "qubit" (paper-faithful) or "fermionic"
    spin_symmetric: bool = True    # share the alpha/beta single angle per pair
    topology: str = "all"          # "all" | "partitioned"
    phi_max: float = 15.0          # only used for topology="partitioned"
    beta_cap: float = 0.10         # hard cap |angle| on cross-cut gates (tanh reparam)
    init_scale: float = 0.05
    init_state: str = "diag"       # "diag" (argmin hdiag, paper Eq. 12 analogue) | "hf"
    overhead_penalty: float = 0.0  # lambda * sum_u(cross angles) added to loss
    checkpoint_chunk: int = 0      # gates per checkpoint segment (0 = off)
    engine: str = "autograd"       # "autograd" | "adjoint" (O(1)-depth memory)
    pairs_override: list | None = None  # explicit pair list (e.g. after pruning)
    partition_override: list | None = None  # explicit left spatial-orbital block;
                                   # preserves cross masks and cutting accounting
    seed_doubles: str = "random"   # "random" | "mp2": seed layer-0 pair-double
                                   # angles from the molecule's own MP2-style
                                   # pair amplitudes (computed at build time)
    cross_gates: str = "sd"        # gates allowed on cross-cut pairs: "sd" | "s";
                                   # "s" spends the budget on MORE pairs, no cross doubles
    beta_cap_double: float | None = None  # separate cap for cross pair-doubles
                                   # (None -> beta_cap). Empirically the cross double
                                   # carries the inter-block pair correlation; give it
                                   # the larger share of the budget.


@dataclass
class OptConfig:
    """Deterministic optimizer and adaptive-extension settings."""

    steps: int = 200
    lr: float = 0.05
    seed: int = 17
    active_threshold: float = 1e-4
    # Descent-triggered step extension (molecule-agnostic; uses only the
    # variational history). While the best energy improved by more than
    # extend_eps per step over the last `window` steps, keep optimizing in
    # chunks of steps//2, up to extend_max_chunks extra chunks. Adam state
    # is preserved across extensions.
    extend_max_chunks: int = 0
    extend_eps: float = 1e-7          # Ha per step
    window: int = 50
    # Second-order polish after Adam: L-BFGS with strong-Wolfe line search.
    # Parameter counts are small (~L*n_o^2), so this is cheap and typically
    # finishes the narrow-valley descent that first-order methods crawl.
    polish_steps: int = 0


@dataclass
class BuiltAnsatz:
    """Resolved orbital-pair topology and committed cutting budget."""

    pairs: list                    # ordered (p, q) spatial-orbital pairs
    cross_mask: np.ndarray         # bool per pair: crosses the partition cut
    left_block: list | None
    right_block: list | None
    u_budget: float | None
    u_committed: float | None


def build_topology(prob: CASProblem, cfg: AnsatzConfig) -> BuiltAnsatz:
    """Build all-pair or hard-budget partitioned orbital connectivity."""

    no = prob.norb
    if cfg.pairs_override is not None and cfg.partition_override is not None:
        raise ValueError("pairs_override and partition_override are mutually exclusive")
    if cfg.pairs_override is not None:
        pairs = [tuple(pq) for pq in cfg.pairs_override]
        return BuiltAnsatz(pairs, np.zeros(len(pairs), bool), None, None, None, None)
    all_pairs = [(p, q) for p in range(no) for q in range(p + 1, no)]
    if cfg.topology == "all":
        return BuiltAnsatz(all_pairs, np.zeros(len(all_pairs), bool), None, None, None, None)
    if cfg.topology != "partitioned":
        raise ValueError(cfg.topology)
    A = oh.pair_score(prob.h1e, prob.eri)
    if cfg.partition_override is None:
        left, right = oh.kl_bipartition(A)
    else:
        left = sorted({int(x) for x in cfg.partition_override})
        if len(left) != no // 2 or any(x < 0 or x >= no for x in left):
            raise ValueError(
                "partition_override must contain exactly floor(norb/2) "
                "distinct valid spatial-orbital indices"
            )
        right = sorted(set(range(no)) - set(left))
    intra, cross = oh.split_pairs(no, left)
    # Per CROSS pair per layer the circuit applies TWO single gates (alpha and
    # beta copies -- spin_symmetric only shares the angle, not the gate count)
    # plus one pair-double IF cross doubles are enabled. Singles are capped at
    # beta_cap, the double at beta_cap_double (may differ).
    cross_has_d = ("d" in cfg.gates) and ("d" in cfg.cross_gates)
    cap_d = cfg.beta_cap_double if cfg.beta_cap_double is not None else cfg.beta_cap
    per_pair_u = cfg.layers * (
        2.0 * float(oh.u_single(cfg.beta_cap))
        + (float(oh.u_pair_double(cap_d)) if cross_has_d else 0.0)
    )
    u_budget = oh.u_max_from_phi(cfg.phi_max)
    order = sorted(cross, key=lambda pq: -A[pq[0], pq[1]])
    admitted, u_comm = [], 0.0
    for pq in order:
        if u_comm + per_pair_u <= u_budget + 1e-12:
            admitted.append(pq)
            u_comm += per_pair_u
    pairs = sorted(intra + admitted)
    cross_set = set(admitted)
    mask = np.array([pq in cross_set for pq in pairs], dtype=bool)
    return BuiltAnsatz(pairs, mask, left, right, u_budget, u_comm)


class SectorCircuit:
    """Ordered gate list acting on the sector CI matrix, differentiable."""

    def __init__(self, prob: CASProblem, acfg: AnsatzConfig):
        self.prob, self.acfg = prob, acfg
        self.sector = Sector(prob.norb, *prob.nelec)
        self.topo = build_topology(prob, acfg)
        self.n_pairs = len(self.topo.pairs)
        self.cross_idx = np.flatnonzero(self.topo.cross_mask)

    # parameter shapes
    def init_params(self, seed: int) -> dict[str, torch.nn.Parameter]:
        g = torch.Generator().manual_seed(seed)
        L, P = self.acfg.layers, self.n_pairs
        s = self.acfg.init_scale
        shapes = {"singles": (L, P) if self.acfg.spin_symmetric else (L, P, 2)}
        if "d" in self.acfg.gates:
            shapes["doubles"] = (L, P)
        params = {
            k: torch.nn.Parameter(s * torch.randn(*shp, generator=g, dtype=REAL))
            for k, shp in shapes.items()
        }
        if (self.acfg.seed_doubles == "mp2" and "doubles" in params
                and self.prob.meta.get("t_pair") is not None):
            t = self.prob.meta["t_pair"]
            with torch.no_grad():
                for k, (p, q) in enumerate(self.topo.pairs):
                    if abs(t[p, q]) > 1e-6:
                        params["doubles"][0, k] = float(t[p, q])
        return params

    def _effective_angles(self, params):
        """Apply the hard tanh angle clamps on cross-cut gates (per gate kind)."""
        out = {}
        cap_d = (self.acfg.beta_cap_double
                 if self.acfg.beta_cap_double is not None else self.acfg.beta_cap)
        for k, t in params.items():
            if len(self.cross_idx) == 0:
                out[k] = t
                continue
            cap = cap_d if k == "doubles" else self.acfg.beta_cap
            eff = t.clone()
            idx = torch.from_numpy(self.cross_idx)
            if t.dim() == 2:
                eff[:, idx] = cap * torch.tanh(t[:, idx] / cap)
            else:
                eff[:, idx, :] = cap * torch.tanh(t[:, idx, :] / cap)
            out[k] = eff
        return out

    def _emit_double(self, k: int) -> bool:
        """Does pair k carry a pair-double gate?"""
        if "d" not in self.acfg.gates:
            return False
        if self.topo.cross_mask[k] and "d" not in self.acfg.cross_gates:
            return False
        return True

    def _ops(self, eff):
        """Yield (fn(c) -> c) closures in the fixed lexicographic layer order."""
        conv = self.acfg.convention
        sec = self.sector
        for layer in range(self.acfg.layers):
            for k, (p, q) in enumerate(self.topo.pairs):
                if self.acfg.spin_symmetric:
                    ba = bb = eff["singles"][layer, k]
                else:
                    ba, bb = eff["singles"][layer, k, 0], eff["singles"][layer, k, 1]
                yield lambda c, p=p, q=q, b=ba: sec.apply_single(c, "a", p, q, b, conv)
                yield lambda c, p=p, q=q, b=bb: sec.apply_single(c, "b", p, q, b, conv)
                if self._emit_double(k):
                    d = eff["doubles"][layer, k]
                    yield lambda c, p=p, q=q, d=d: sec.apply_pair_double(c, p, q, d, conv)

    def gate_specs(self):
        """Ordered GateSpec list matching the flat-angle enumeration."""
        from .adjoint import GateSpec

        conv = self.acfg.convention
        specs = []
        for _layer in range(self.acfg.layers):
            for k, (p, q) in enumerate(self.topo.pairs):
                specs.append(GateSpec("s", "a", p, q, conv))
                specs.append(GateSpec("s", "b", p, q, conv))
                if self._emit_double(k):
                    specs.append(GateSpec("d", None, p, q, conv))
        return specs

    def flat_angles(self, params) -> torch.Tensor:
        """Effective angles flattened in gate order (differentiable)."""
        eff = self._effective_angles(params)
        vals = []
        for layer in range(self.acfg.layers):
            for k in range(self.n_pairs):
                if self.acfg.spin_symmetric:
                    sa = sb = eff["singles"][layer, k]
                else:
                    sa, sb = eff["singles"][layer, k, 0], eff["singles"][layer, k, 1]
                vals.extend([sa, sb])
                if self._emit_double(k):
                    vals.append(eff["doubles"][layer, k])
        return torch.stack(vals)

    def forward(self, params, hdiag: np.ndarray | None = None,
                c0: torch.Tensor | None = None) -> torch.Tensor:
        """c0: optional custom initial sector state (e.g., a classically
        pre-optimized NOCI/manifold state) — the hybrid-architecture hook."""
        if c0 is None:
            c0 = self.sector.initial_state(hdiag if self.acfg.init_state == "diag" else None)
        else:
            c0 = c0.to(REAL)
        if self.acfg.engine == "adjoint":
            from .adjoint import adjoint_forward

            if not hasattr(self, "_specs"):
                self._specs = self.gate_specs()
            return adjoint_forward(self.flat_angles(params), self.sector, self._specs, c0)
        eff = self._effective_angles(params)
        c = c0
        ops = list(self._ops(eff))
        chunk = self.acfg.checkpoint_chunk
        if chunk and c.numel() > 200_000:
            from torch.utils.checkpoint import checkpoint

            def run_chunk(c_in, lo, hi):
                for f in ops[lo:hi]:
                    c_in = f(c_in)
                return c_in

            c = c.requires_grad_(True)  # checkpoint needs a grad-tracking input
            for lo in range(0, len(ops), chunk):
                hi = min(lo + chunk, len(ops))
                c = checkpoint(lambda x, lo=lo, hi=hi: run_chunk(x, lo, hi), c,
                               use_reentrant=False)
        else:
            for f in ops:
                c = f(c)
        return c

    def cross_angle_arrays(self, params) -> tuple[np.ndarray, np.ndarray]:
        """Final effective cross-cut angles (singles incl. both spins, doubles)."""
        eff = {k: v.detach().numpy() for k, v in self._effective_angles(params).items()}
        if len(self.cross_idx) == 0:
            return np.array([]), np.array([])
        s = eff["singles"][:, self.cross_idx]
        if self.acfg.spin_symmetric:
            s = np.concatenate([s.ravel(), s.ravel()])  # alpha + beta copies
        else:
            s = s.ravel()
        d = eff.get("doubles")
        if d is None or "d" not in self.acfg.cross_gates:
            d = np.array([])
        else:
            d = d[:, self.cross_idx].ravel()
        return s, d


def run_vqe(prob: CASProblem, acfg: AnsatzConfig, ocfg: OptConfig,
            verbose: bool = False, init_params: dict | None = None,
            return_params: bool = False, init_c0: np.ndarray | None = None) -> dict:
    """Optimize one sector-space ansatz and return energies and resources."""

    torch.manual_seed(ocfg.seed)
    np.random.seed(ocfg.seed)
    circ = SectorCircuit(prob, acfg)
    energy_fn, _ = make_energy_fn(prob)
    energy_evaluations = 0

    def evaluated_energy(state):
        nonlocal energy_evaluations
        energy_evaluations += 1
        return energy_fn(state)
    hdiag = prob.hdiag() if acfg.init_state == "diag" else None
    c0_t = None
    if init_c0 is not None:
        c0_t = torch.from_numpy(np.ascontiguousarray(init_c0, dtype=np.float64))
        c0_t = c0_t / torch.linalg.norm(c0_t)

    if init_params is not None:
        params = {k: torch.nn.Parameter(v.detach().clone()) for k, v in init_params.items()}
    else:
        params = circ.init_params(ocfg.seed)
    opt = torch.optim.Adam(params.values(), lr=ocfg.lr)
    t0 = time.perf_counter()
    history = []
    best_hist = []
    best_e, best_state = np.inf, None
    target = ocfg.steps
    chunks_used = 0
    step = 0
    while step < target:
        opt.zero_grad(set_to_none=True)
        c = circ.forward(params, hdiag, c0=c0_t)
        e = evaluated_energy(c)
        loss = e
        if acfg.overhead_penalty > 0 and len(circ.cross_idx):
            eff = circ._effective_angles(params)
            idx = torch.from_numpy(circ.cross_idx)
            b = eff["singles"][:, idx]
            pen = 2.0 * torch.log(torch.cos(b / 2).abs() + torch.sin(b / 2).abs()).sum()
            if acfg.spin_symmetric:
                pen = pen * 2.0
            if "doubles" in eff:
                d = eff["doubles"][:, idx]
                pen = pen + 8.0 * torch.log(
                    torch.cos(d / 8).abs() + torch.sin(d / 8).abs()
                ).sum()
            loss = loss + acfg.overhead_penalty * pen
        # ``e`` describes the current (pre-update) parameters.  Checkpoint
        # those same parameters before Adam mutates them; otherwise the
        # recorded energy and saved state are off by one optimizer step.
        e_val = float(e.detach())
        history.append(e_val)
        if e_val < best_e:
            best_e = e_val
            best_state = {k: v.detach().clone() for k, v in params.items()}
        best_hist.append(best_e)
        loss.backward()
        opt.step()
        if verbose and (step % 25 == 0 or step == target - 1):
            print(f"    step {step:4d}  E = {e_val:.9f}")
        step += 1
        if (step == target and chunks_used < ocfg.extend_max_chunks
                and len(best_hist) > ocfg.window):
            w = ocfg.window
            rate = (best_hist[-w] - best_hist[-1]) / w
            if rate > ocfg.extend_eps:
                target += max(1, ocfg.steps // 2)
                chunks_used += 1

    # The final Adam update has not yet been evaluated by the loop.  Include
    # it as a candidate so a last-step improvement is not discarded (and a
    # last-step regression is never returned as the best result).
    with torch.no_grad():
        e_post = float(evaluated_energy(circ.forward(params, hdiag, c0=c0_t)))
    history.append(e_post)
    if e_post < best_e:
        best_e = e_post
        best_state = {k: v.detach().clone() for k, v in params.items()}

    if ocfg.polish_steps > 0:
        with torch.no_grad():
            for k in params:
                params[k].copy_(best_state[k])
        lbfgs = torch.optim.LBFGS(params.values(), max_iter=ocfg.polish_steps,
                                  history_size=30, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            e_c = evaluated_energy(circ.forward(params, hdiag, c0=c0_t))
            e_c.backward()
            return e_c

        try:
            lbfgs.step(closure)
            with torch.no_grad():
                e_pol = float(evaluated_energy(circ.forward(params, hdiag, c0=c0_t)))
            if e_pol < best_e:  # accept polish only if it improved
                best_e = e_pol
                best_state = {k: v.detach().clone() for k, v in params.items()}
                history.append(e_pol)
        except Exception as exc:  # polish must never break a run
            print(f"    [L-BFGS polish skipped: {exc}]", flush=True)
    wall = time.perf_counter() - t0

    with torch.no_grad():
        c = circ.forward(best_state, hdiag, c0=c0_t)
        e_final = float(evaluated_energy(c))
    from .energy import energy_and_variance

    _, variance = energy_and_variance(c, prob)
    s_cross, d_cross = circ.cross_angle_arrays(best_state)
    phi = oh.circuit_phi(s_cross, d_cross)
    thr = ocfg.active_threshold
    n_active = {
        k: int((v.abs() > thr).sum()) for k, v in best_state.items()
    }
    res = dict(
        problem=prob.name, n_qubits=prob.n_qubits, norb=prob.norb,
        nelec=prob.nelec, sector_dim=int(np.prod(prob.dims)),
        seed=ocfg.seed, steps=int(target), steps_scheduled=int(ocfg.steps),
        extension_chunks_used=int(chunks_used),
        e_vqe=e_final, e_hf=prob.e_hf, e_casci=prob.e_casci,
        error_mha=(1000.0 * (e_final - prob.e_casci) if prob.e_casci is not None else None),
        corr_fraction=(
            (prob.e_hf - e_final) / (prob.e_hf - prob.e_casci)
            if prob.e_casci is not None and abs(prob.e_hf - prob.e_casci) > 1e-12 else None
        ),
        n_pairs=circ.n_pairs, n_cross_pairs=int(len(circ.cross_idx)),
        active_gates=n_active, phi=phi,
        variance_ha2=variance, weinstein_radius_mha=1000.0 * float(np.sqrt(variance)),
        u_budget=circ.topo.u_budget, u_committed=circ.topo.u_committed,
        wall_seconds=wall, history_first=history[0], history=history,
        energy_evaluations=int(energy_evaluations),
        ansatz=asdict(acfg),
    )
    if return_params:
        res["best_params"] = best_state
        res["pairs"] = list(circ.topo.pairs)
    return res


def warm_start_params(prob_small: CASProblem, prob_big: CASProblem,
                      acfg: AnsatzConfig, params_small: dict, seed: int,
                      fresh_scale: float | None = None) -> dict:
    """Embed a smaller-active-space optimum into a larger rung's parameters.

    Pairs present in both rungs inherit the small rung's EFFECTIVE angles
    (inverting the tanh cap where the pair is cross-cut in the big rung);
    new pairs start at N(0, init_scale) (or exactly 0 with fresh_scale=0).
    Legitimate for unseen molecules: everything is computed by the pipeline
    itself on the molecule at hand — nothing external is borrowed. Requires
    both rungs to use the same canonical-MO ordering (guaranteed by
    build_cas: active = MOs [ncore, ncore+no) of one converged RHF).
    """
    c_s = SectorCircuit(prob_small, acfg)
    c_b = SectorCircuit(prob_big, acfg)
    if c_s.acfg.layers != c_b.acfg.layers:
        raise ValueError("layer counts must match for warm start")
    eff_s = {k: v.detach() for k, v in c_s._effective_angles(params_small).items()}
    raw_b = c_b.init_params(seed)
    if fresh_scale is not None:
        for t in raw_b.values():
            with torch.no_grad():
                t.mul_(0.0 if fresh_scale == 0.0 else fresh_scale / acfg.init_scale)
    pos_b = {pq: k for k, pq in enumerate(c_b.topo.pairs)}
    cross_b = set(int(i) for i in c_b.cross_idx)
    cap_d = acfg.beta_cap_double if acfg.beta_cap_double is not None else acfg.beta_cap
    with torch.no_grad():
        for k_s, pq in enumerate(c_s.topo.pairs):
            if pq not in pos_b:
                continue
            k_b = pos_b[pq]
            for key, t_b in raw_b.items():
                val = eff_s[key].select(1, k_s).clone()
                if k_b in cross_b:
                    cap = cap_d if key == "doubles" else acfg.beta_cap
                    val = cap * torch.atanh(torch.clamp(val / cap, -1 + 1e-9, 1 - 1e-9))
                t_b.select(1, k_b).copy_(val)
    return {k: torch.nn.Parameter(v.detach().clone()) for k, v in raw_b.items()}


def deepen_params(params: dict, seed: int, noise_scale: float = 0.0) -> dict:
    """Append one layer initialized at (near-)identity: angles ~ 0.

    At noise_scale=0 the deepened circuit evaluates EXACTLY to the parent
    optimum (new gates are identities), so depth escalation is monotone by
    construction — the deeper optimization can only improve.
    """
    g = torch.Generator().manual_seed(seed + 777_000)
    out = {}
    for k, t in params.items():
        pad_shape = (1,) + tuple(t.shape[1:])
        pad = (noise_scale * torch.randn(*pad_shape, generator=g, dtype=t.dtype)
               if noise_scale > 0 else torch.zeros(*pad_shape, dtype=t.dtype))
        out[k] = torch.nn.Parameter(torch.cat([t.detach(), pad], dim=0).clone())
    return out


def run_adaptive_depth(prob: CASProblem, acfg: AnsatzConfig, ocfg: OptConfig,
                       l_max: int = 4, accept_mha: float = 0.1,
                       init_params: dict | None = None,
                       log=lambda s: None) -> dict:
    """The adaptive-depth protocol rule (molecule-agnostic, reference-free).

    Optimize at acfg.layers (with descent extension); then repeatedly deepen
    by one identity-initialized layer and re-optimize, accepting the deeper
    result iff the VARIATIONAL energy improves by more than accept_mha.
    Stops at the first non-improvement or at l_max. Returns the accepted
    result; res['ansatz']['layers'] records the accepted depth and
    res['depth_trace'] the per-depth energies.
    """
    from dataclasses import replace

    res = run_vqe(prob, acfg, ocfg, init_params=init_params, return_params=True)
    trace = [(acfg.layers, res["e_vqe"])]
    stages = [
        {
            "layers": int(acfg.layers),
            "energy_hartree": float(res["e_vqe"]),
            "energy_evaluations": int(res["energy_evaluations"]),
            "wall_seconds": float(res["wall_seconds"]),
            "accepted": True,
        }
    ]
    while acfg.layers < l_max:
        deeper_init = deepen_params(res["best_params"], seed=ocfg.seed)
        acfg_deep = replace(acfg, layers=acfg.layers + 1)
        res_deep = run_vqe(prob, acfg_deep, ocfg, init_params=deeper_init,
                           return_params=True)
        gain_mha = 1000.0 * (res["e_vqe"] - res_deep["e_vqe"])
        trace.append((acfg_deep.layers, res_deep["e_vqe"]))
        accepted = bool(gain_mha > accept_mha)
        stages.append(
            {
                "layers": int(acfg_deep.layers),
                "energy_hartree": float(res_deep["e_vqe"]),
                "gain_mha": float(gain_mha),
                "energy_evaluations": int(res_deep["energy_evaluations"]),
                "wall_seconds": float(res_deep["wall_seconds"]),
                "accepted": accepted,
            }
        )
        log(f"      depth {acfg.layers}->{acfg_deep.layers}: gain {gain_mha:+.4f} mHa")
        if gain_mha > accept_mha:
            res, acfg = res_deep, acfg_deep
        else:
            break
    res["depth_trace"] = trace
    res["depth_stages"] = stages
    res["complete_cascade_energy_evaluations"] = sum(
        int(stage["energy_evaluations"]) for stage in stages
    )
    res["complete_cascade_wall_seconds"] = sum(
        float(stage["wall_seconds"]) for stage in stages
    )
    return res


def run_adaptive_prune(prob: CASProblem, acfg: AnsatzConfig, ocfg: OptConfig,
                       tau_pair: float = 0.02, phase1_frac: float = 0.4) -> dict:
    """Enhancement 3: optimize -> drop inactive pairs -> re-optimize (warm start).

    A pair is kept iff ANY of its angles (any layer, singles or doubles)
    exceeds tau_pair in magnitude after phase 1. Returns phase-2 result
    plus pruning statistics.
    """
    from dataclasses import replace

    steps1 = max(10, int(ocfg.steps * phase1_frac))
    steps2 = max(10, ocfg.steps - steps1)
    r1 = run_vqe(prob, acfg, replace(ocfg, steps=steps1), return_params=True)
    params1, pairs = r1["best_params"], r1["pairs"]

    act = torch.zeros(len(pairs), dtype=torch.float64)
    for k, t in params1.items():
        a = t.detach().abs()
        a = a.amax(dim=tuple(i for i in range(a.dim()) if i != 1))  # max over layers/chan
        act = torch.maximum(act, a)
    keep = [i for i in range(len(pairs)) if float(act[i]) > tau_pair]
    kept_pairs = [pairs[i] for i in keep]
    if not kept_pairs:  # degenerate: keep the most active pair
        keep = [int(act.argmax())]
        kept_pairs = [pairs[keep[0]]]

    idx = torch.tensor(keep, dtype=torch.long)
    warm = {}
    for k, t in params1.items():
        warm[k] = t.detach().index_select(1, idx).clone()
    acfg2 = replace(acfg, pairs_override=kept_pairs, topology="all")
    r2 = run_vqe(prob, acfg2, replace(ocfg, steps=steps2), init_params=warm)
    r2["prune"] = dict(
        tau_pair=tau_pair, pairs_before=len(pairs), pairs_after=len(kept_pairs),
        reduction_pct=100.0 * (1 - len(kept_pairs) / len(pairs)),
        phase1_error_mha=r1["error_mha"], phase1_steps=steps1, phase2_steps=steps2,
        kept_pairs=kept_pairs,
    )
    return r2
