"""Surgical validation suite: every layer of the stack is cross-checked
against an independent exponential-cost reference at small size.

Run:  python -m tests.test_all   (from the givens40 package root)
"""
from __future__ import annotations

import sys
import numpy as np
import torch

sys.path.insert(0, ".")

from givens40.chemistry import build_cas
from givens40.sector import Sector
from givens40.energy import make_energy_fn, _SigmaCache, electronic_energy
from givens40 import overhead as oh
from givens40 import densecheck as dc
from givens40.runner import AnsatzConfig, OptConfig, run_vqe, SectorCircuit

PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = ""):
    """Record and print one independent scientific validation."""

    global PASS, FAIL
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))


def main():
    """Execute all 48 small-system cross-checks and return a shell status."""

    rng = np.random.default_rng(7)

    # ---- 0. Nakamura overhead constants (their quoted values) ----
    u_se = float(oh.u_single(np.pi / 8))       # SE(theta=pi/4) -> beta=pi/8
    u_de = float(oh.u_pair_double(np.pi / 8))  # DE(theta=pi/4) -> delta=pi/8
    check("u_SE(theta=pi/4) ~ 0.32 (Nakamura Eq. 11)", abs(u_se - 0.324) < 5e-3, f"{u_se:.4f}")
    check("u_DE(theta=pi/4) ~ 0.37 (Nakamura Eq. 9)", abs(u_de - 0.374) < 5e-3, f"{u_de:.4f}")
    check("phi(u_max(15)) == 15", abs(oh.phi_from_u(oh.u_max_from_phi(15.0)) - 15.0) < 1e-9)

    # ---- 1. CAS integrals -> dense JW H reproduces CASCI exactly ----
    prob = build_cas("BeH2", 3)          # 6 qubits, (2,2) electrons, dim 9
    Hd = dc.dense_hamiltonian(prob.h1e, prob.eri, prob.ecore)
    evals = np.linalg.eigvalsh(Hd)
    # dense H lives in full Fock space; CASCI energy is min over the (2,2) sector.
    sec = Sector(prob.norb, *prob.nelec)
    emb_idx = [int(sa) | (int(sb) << prob.norb) for sa in sec.alpha.strs for sb in sec.beta.strs]
    Hsec = Hd[np.ix_(emb_idx, emb_idx)]
    e_sec = np.linalg.eigvalsh(Hsec)[0]
    check("dense JW sector ground == CASCI", abs(e_sec - prob.e_casci) < 1e-8,
          f"{e_sec:.10f} vs {prob.e_casci:.10f}")

    # ---- 2. PySCF hdiag == dense diagonal on sector ----
    hd = prob.hdiag().ravel() + prob.ecore
    check("make_hdiag == dense diag", np.max(np.abs(hd - np.diag(Hsec))) < 1e-9)

    # ---- 3. contract_2e quadratic form == dense quadratic form ----
    c = rng.normal(size=(sec.dimA, sec.dimB))
    c /= np.linalg.norm(c)
    cache = _SigmaCache(prob)
    e_pyscf = float(np.dot(c.ravel(), cache.sigma(np.ascontiguousarray(c)).ravel())) + prob.ecore
    e_dense = float(c.ravel() @ Hsec @ c.ravel())
    check("contract_2e == dense <c|H|c>", abs(e_pyscf - e_dense) < 1e-9,
          f"{e_pyscf:.10f} vs {e_dense:.10f}")

    # ---- 4. gates vs dense references (both conventions, incl. signs) ----
    n = prob.n_qubits
    c0 = rng.normal(size=(sec.dimA, sec.dimB))
    c0 /= np.linalg.norm(c0)
    for conv in ("qubit", "fermionic"):
        for (p, q) in [(0, 1), (0, 2), (1, 2)]:
            beta = float(rng.normal(scale=0.7))
            ct = torch.from_numpy(c0.copy())
            out = sec.apply_single(ct, "a", p, q, torch.tensor(beta, dtype=torch.float64), conv).numpy()
            outb = sec.apply_single(torch.from_numpy(c0.copy()), "b", p, q,
                                    torch.tensor(beta, dtype=torch.float64), conv).numpy()
            if conv == "qubit":
                Ga = dc.dense_single_qubitconv(n, p, q, beta)                 # alpha qubits p,q
                Gb = dc.dense_single_qubitconv(n, p + prob.norb, q + prob.norb, beta)
            else:
                Ga = dc.dense_single_fermconv(n, p, q, beta)
                Gb = dc.dense_single_fermconv(n, p + prob.norb, q + prob.norb, beta)
            emb = sec.embed(c0)
            ref_a = (Ga @ emb)[emb_idx].reshape(sec.dimA, sec.dimB)
            ref_b = (Gb @ emb)[emb_idx].reshape(sec.dimA, sec.dimB)
            ok_a = np.max(np.abs(out - ref_a)) < 1e-12
            ok_b = np.max(np.abs(outb - ref_b)) < 1e-12
            check(f"single({conv}) a-spin pair ({p},{q})", ok_a)
            check(f"single({conv}) b-spin pair ({p},{q})", ok_b)
        for (p, q) in [(0, 2), (1, 2)]:
            delta = float(rng.normal(scale=0.7))
            out = sec.apply_pair_double(torch.from_numpy(c0.copy()), p, q,
                                        torch.tensor(delta, dtype=torch.float64), conv).numpy()
            if conv == "qubit":
                G = dc.dense_pairdouble_qubitconv(n, prob.norb, p, q, delta)
            else:
                G = dc.dense_pairdouble_fermconv(n, prob.norb, p, q, delta)
            ref = (G @ sec.embed(c0))[emb_idx].reshape(sec.dimA, sec.dimB)
            check(f"pair-double({conv}) pair ({p},{q})",
                  np.max(np.abs(out - ref)) < 1e-12)
    # norm preservation through a random circuit
    ct = torch.from_numpy(c0.copy())
    for (p, q) in [(0, 1), (1, 2), (0, 2)]:
        ct = sec.apply_single(ct, "a", p, q, torch.tensor(0.3, dtype=torch.float64), "qubit")
        ct = sec.apply_pair_double(ct, p, q, torch.tensor(-0.4, dtype=torch.float64), "fermionic")
    check("norm preserved", abs(float((ct * ct).sum()) - 1.0) < 1e-12)

    # ---- 5. autograd gradient vs finite differences ----
    acfg = AnsatzConfig(layers=2, gates="sd", convention="qubit")
    circ = SectorCircuit(prob, acfg)
    params = circ.init_params(seed=3)
    efn, _ = make_energy_fn(prob)
    hdiag = prob.hdiag()
    e0 = efn(circ.forward(params, hdiag))
    e0.backward()
    k, idx = "singles", (0, 2)
    g_auto = float(params[k].grad[idx])
    eps = 1e-6
    with torch.no_grad():
        params[k][idx] += eps
        ep = float(efn(circ.forward(params, hdiag)))
        params[k][idx] -= 2 * eps
        em = float(efn(circ.forward(params, hdiag)))
        params[k][idx] += eps
    g_fd = (ep - em) / (2 * eps)
    check("autograd == finite difference", abs(g_auto - g_fd) < 1e-6,
          f"{g_auto:.9f} vs {g_fd:.9f}")

    # ---- 6. checkpointed forward == plain forward ----
    acfg_ck = AnsatzConfig(layers=2, gates="sd", checkpoint_chunk=5)
    circ_ck = SectorCircuit(prob, acfg_ck)
    with torch.no_grad():
        c_plain = circ.forward(params, hdiag)
    c_ck = circ_ck.forward({k: v.detach().clone() for k, v in params.items()}, hdiag)
    check("checkpointed forward identical",
          float((c_plain - c_ck.detach()).abs().max()) < 1e-12)

    # ---- 7. budget guarantee ----
    prob10 = build_cas("BeH2", 5, run_casci=False)   # 10 qubits
    acfg_p = AnsatzConfig(topology="partitioned", phi_max=15.0, beta_cap=0.1)
    circ_p = SectorCircuit(prob10, acfg_p)
    pp = circ_p.init_params(seed=1)
    with torch.no_grad():
        for t in pp.values():
            t += 10.0  # push raw params far outside the cap
    s_cr, d_cr = circ_p.cross_angle_arrays(pp)
    phi_worst = oh.circuit_phi(s_cr, d_cr)
    check("hard budget: phi <= phi_max even at extreme params",
          phi_worst <= acfg_p.phi_max + 1e-9, f"phi={phi_worst:.3f}")
    check("cross angles clamped at beta_cap",
          (np.max(np.abs(s_cr)) if len(s_cr) else 0.0) <= acfg_p.beta_cap + 1e-12)

    # ---- 7b. adjoint engine == autograd engine (state and full gradient) ----
    for conv in ("qubit", "fermionic"):
        acfg_a = AnsatzConfig(layers=2, gates="sd", convention=conv, engine="autograd")
        acfg_j = AnsatzConfig(layers=2, gates="sd", convention=conv, engine="adjoint")
        ca = SectorCircuit(prob, acfg_a)
        cj = SectorCircuit(prob, acfg_j)
        pa = ca.init_params(seed=5)
        pj = {k: torch.nn.Parameter(v.detach().clone()) for k, v in pa.items()}
        ea = efn(ca.forward(pa, hdiag)); ea.backward()
        ej = efn(cj.forward(pj, hdiag)); ej.backward()
        ea_value = float(ea.detach())
        ej_value = float(ej.detach())
        check(f"adjoint == autograd energy ({conv})",
              abs(ea_value - ej_value) < 1e-11,
              f"{ea_value:.12f} vs {ej_value:.12f}")
        gmax = max(
            float((pa[k].grad - pj[k].grad).abs().max()) for k in pa
        )
        check(f"adjoint == autograd full gradient ({conv})", gmax < 1e-9, f"max diff {gmax:.2e}")

    # ---- 8. end-to-end mini-VQE hits chemical accuracy (BeH2 6q) ----
    res = run_vqe(prob, AnsatzConfig(layers=2, gates="sd"), OptConfig(steps=300, seed=17))
    check("mini-VQE BeH2(6q) < 1.6 mHa vs CASCI", abs(res["error_mha"]) < 1.6,
          f"{res['error_mha']:.4f} mHa")

    # The state saved as "best" must correspond to its measured energy.  A
    # one-step run is a regression case for the former pre/post-step mismatch.
    r_one = run_vqe(prob, AnsatzConfig(layers=2, gates="sd"),
                    OptConfig(steps=1, seed=17))
    check("optimizer returns minimum evaluated checkpoint",
          len(r_one["history"]) == 2
          and abs(r_one["e_vqe"] - min(r_one["history"])) < 1e-12,
          f"{len(r_one['history'])} evaluations; returned {r_one['e_vqe']:.12f}, "
          f"min {min(r_one['history']):.12f}")

    # ---- 9. warm-start embedding is exact ----
    # With new-pair angles set to exactly 0, the bigger rung evaluated at the
    # embedded parameters must reproduce the smaller rung's energy exactly:
    # gates on new pairs are identity, old-pair gates act on the same
    # determinant amplitudes inside the larger sector.
    from givens40.runner import warm_start_params

    acfg_w = AnsatzConfig(layers=2, gates="sd")
    r_small = run_vqe(prob, acfg_w, OptConfig(steps=120, seed=17), return_params=True)
    prob5 = build_cas("BeH2", 5, run_casci=False)
    warm = warm_start_params(prob, prob5, acfg_w, r_small["best_params"], seed=17,
                             fresh_scale=0.0)
    circ5 = SectorCircuit(prob5, acfg_w)
    efn5, _ = make_energy_fn(prob5)
    with torch.no_grad():
        e_warm = float(efn5(circ5.forward(warm, prob5.hdiag())))
    diff = abs(e_warm - r_small["e_vqe"])
    check("warm-start embedding exact (new pairs at 0)", diff < 1e-10,
          f"{e_warm:.10f} vs {r_small['e_vqe']:.10f}")

    # ---- 10. depth escalation is monotone at init (identity new layer) ----
    from givens40.runner import deepen_params

    deep = deepen_params(r_small["best_params"], seed=17, noise_scale=0.0)
    from dataclasses import replace as _rp

    circ3 = SectorCircuit(prob, _rp(acfg_w, layers=3))
    with torch.no_grad():
        e_deep0 = float(efn(circ3.forward(deep, hdiag)))
    check("deepened circuit reproduces parent optimum at init",
          abs(e_deep0 - r_small["e_vqe"]) < 1e-10,
          f"{e_deep0:.10f} vs {r_small['e_vqe']:.10f}")

    # ---- 11. descent extension triggers and helps (short budget on purpose) ----
    r_short = run_vqe(prob, acfg_w, OptConfig(steps=40, seed=17))
    r_ext = run_vqe(prob, acfg_w, OptConfig(steps=40, seed=17, extend_max_chunks=6,
                                            window=20))
    check("descent extension used and not worse",
          r_ext["extension_chunks_used"] > 0 and r_ext["e_vqe"] <= r_short["e_vqe"] + 1e-12,
          f"chunks={r_ext['extension_chunks_used']}, "
          f"E {r_short['e_vqe']:.8f} -> {r_ext['e_vqe']:.8f}")

    # ---- 12. variance certificate ----
    from givens40.energy import energy_and_variance

    # (a) HF determinant variance matches the dense reference
    c_hf = sec.initial_state()  # HF det (no hdiag -> hf_index)
    e_v, var_v = energy_and_variance(c_hf, prob)
    v = np.zeros(sec.dimA * sec.dimB); v[np.flatnonzero(c_hf.numpy().ravel())[0]] = 1.0
    e_d = float(v @ Hsec @ v)
    var_d = float(v @ Hsec @ Hsec @ v) - e_d ** 2
    check("variance(HF det) == dense <H^2>-<H>^2",
          abs(var_v - var_d) < 1e-8 and abs(e_v - e_d) < 1e-9,
          f"{var_v:.8f} vs {var_d:.8f}")
    # (b) converged mini-VQE state has (near-)zero variance
    c_conv = SectorCircuit(prob, AnsatzConfig()).forward(
        run_vqe(prob, AnsatzConfig(), OptConfig(steps=300, seed=17),
                return_params=True)["best_params"], prob.hdiag())
    _, var_c = energy_and_variance(c_conv.detach(), prob)
    check("variance ~ 0 at convergence", var_c < 1e-6, f"{var_c:.2e}")

    # ---- 13. L-BFGS polish never worsens ----
    r_adam = run_vqe(prob, AnsatzConfig(), OptConfig(steps=60, seed=17))
    r_pol = run_vqe(prob, AnsatzConfig(), OptConfig(steps=60, seed=17, polish_steps=80))
    check("polish <= Adam-only", r_pol["e_vqe"] <= r_adam["e_vqe"] + 1e-12,
          f"{r_adam['e_vqe']:.9f} -> {r_pol['e_vqe']:.9f}")

    # ---- 14. open-shell sector == dense (OH doublet, (4,3) electrons) ----
    from givens40.chemistry import build_cas_from_spec

    prob_oh = build_cas_from_spec(dict(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g",
                                  spin=1, ncore=1), 5, name="OH")
    sec_oh = Sector(prob_oh.norb, *prob_oh.nelec)
    emb_oh = [int(sa) | (int(sb) << prob_oh.norb)
              for sa in sec_oh.alpha.strs for sb in sec_oh.beta.strs]
    Hd_oh = dc.dense_hamiltonian(prob_oh.h1e, prob_oh.eri, prob_oh.ecore)
    Hs_oh = Hd_oh[np.ix_(emb_oh, emb_oh)]
    c_r = rng.normal(size=(sec_oh.dimA, sec_oh.dimB)); c_r /= np.linalg.norm(c_r)
    cache_oh = _SigmaCache(prob_oh)
    e_p = float(np.dot(c_r.ravel(), cache_oh.sigma(np.ascontiguousarray(c_r)).ravel())) + prob_oh.ecore
    e_dn = float(c_r.ravel() @ Hs_oh @ c_r.ravel())
    check("open-shell contract_2e == dense", abs(e_p - e_dn) < 1e-9,
          f"{e_p:.9f} vs {e_dn:.9f}")
    check("open-shell CASCI ground == dense sector ground",
          abs(np.linalg.eigvalsh(Hs_oh)[0] - prob_oh.e_casci) < 1e-8)
    # open shells need independent alpha/beta angles (unequal occupations)
    r_oh = run_vqe(prob_oh, AnsatzConfig(spin_symmetric=False),
                   OptConfig(steps=300, seed=17, polish_steps=60))
    check("open-shell mini-VQE < 1.6 mHa", abs(r_oh["error_mha"]) < 1.6,
          f"{r_oh['error_mha']:.4f} mHa")

    # ---- 15. MP2 natural orbitals + amplitude seeding ----
    nat = build_cas_from_spec(dict(atom="Be 0 0 0; H 0 0 1.33; H 0 0 -1.33",
                                   basis="cc-pvdz", ncore=1), 3, name="BeH2",
                              orbital_basis="mp2nat")
    check("mp2nat build + CASCI", nat.e_casci is not None and
          nat.meta["orbital_basis"] == "mp2nat")
    check("mp2 seed amplitudes present", float(np.abs(nat.meta["t_pair"]).max()) > 1e-4,
          f"max|t| = {np.abs(nat.meta['t_pair']).max():.4f}")
    r_seed = run_vqe(nat, AnsatzConfig(seed_doubles="mp2"),
                     OptConfig(steps=200, seed=17))
    check("mp2-seeded run chemically accurate", abs(r_seed["error_mha"]) < 1.6,
          f"{r_seed['error_mha']:.4f} mHa")

    # ---- 16. exported PennyLane circuit == sector engine (hardware-faithful) ----
    try:
        import pennylane  # noqa: F401

        from givens40.export import pennylane_statevector

        circ_x = SectorCircuit(prob, AnsatzConfig())
        px = circ_x.init_params(seed=9)
        with torch.no_grad():
            c_x = circ_x.forward(px, prob.hdiag()).numpy()
        emb_x = circ_x.sector.embed(c_x)
        psi_x = pennylane_statevector(circ_x, px)
        if np.vdot(psi_x, emb_x).real < 0:
            psi_x = -psi_x
        check("exported PennyLane circuit == sector engine",
              float(np.max(np.abs(psi_x - emb_x))) < 1e-12)
    except ImportError as error:
        # PennyLane is pinned in the judge environment. Treat its absence as a
        # failed validation rather than allowing a misleading 47-check pass.
        check("exported PennyLane circuit == sector engine", False,
              f"PennyLane import failed: {error}")

    # ---- 17. Aer uses reproducible, genuine finite-shot measurement ----
    from givens40.qiskit_export import estimate_energy_aer

    circ_q = SectorCircuit(prob, AnsatzConfig())
    p_q = circ_q.init_params(seed=9)
    e_q1, _, m_q1 = estimate_energy_aer(circ_q, p_q, shots=256, seed=91,
                                         return_metadata=True)
    e_q2, _, m_q2 = estimate_energy_aer(circ_q, p_q, shots=256, seed=91,
                                         return_metadata=True)
    check("Aer finite-shot estimate deterministic at fixed seed",
          e_q1 == e_q2 and m_q1 == m_q2, f"{e_q1:.12f} vs {e_q2:.12f}")
    check("Aer reports sampled commuting-group accounting",
          m_q1["finite_shot_backend"] and m_q1["commuting_groups"] >= 1
          and m_q1["shots_per_group"] == 256
          and m_q1["total_circuit_shots"]
          == 256 * m_q1["commuting_groups"],
          str(m_q1))

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
