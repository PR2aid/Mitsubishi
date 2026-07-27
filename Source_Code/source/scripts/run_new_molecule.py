"""THE COMPETITION ENTRY POINT: fully molecule-agnostic escalation pipeline.

Given an UNSEEN molecule (geometry spec or FCIDUMP), this script runs the
complete protocol end to end with the frozen constants below — nothing is
borrowed from previously optimized systems:

  1. RHF -> canonical MOs -> CAS(ne, no) ladder (active spaces grow, the
     molecule and electron count stay fixed);
  2. per rung: diagonal-determinant initialization, L=2 Givens singles +
     pair-double ansatz (qubit convention), Adam;
  3. optional (default ON) warm start: rung k+1 inherits rung k's optimized
     angles on shared orbital pairs — the "cheap run first, then enhance"
     protocol; computed on THIS molecule only;
  4. exact CASCI reference and error whenever the sector fits the cap;
  5. optional partitioned topology with hard phi budget (A_pq cut computed
     from THIS molecule's integrals).

Examples:
  python scripts/run_new_molecule.py --name H2O --atom "H -0.021 -0.002 0; O 0.835 0.452 0; H 1.477 -0.273 0" \
      --basis cc-pvdz --ncore 1 --ladder 5 8 12 --seeds 17 11111
  python scripts/run_new_molecule.py --name mystery --fcidump problem.fcidump --seeds 17
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from givens40.chemistry import build_cas_from_spec, build_from_fcidump
from givens40.runner import (AnsatzConfig, OptConfig, run_vqe, warm_start_params,
                             run_adaptive_depth)

# ---------------- FROZEN PROTOCOL CONSTANTS (declared, not tuned per molecule) ---
PROTOCOL = dict(
    layers=2, gates="sd", convention="qubit", spin_symmetric=True,
    init_state="diag", init_scale=0.05, lr=0.05,
    steps_small=300, steps_medium=200, steps_large=150,   # by sector dim
    dim_medium=50_000, dim_large=300_000,
    engine_switch_dim=20_000,
    phi_max=15.0, beta_cap=0.05, beta_cap_double=0.25,    # partitioned runs
    casci_max_dim=5_000_000,
    warm_start=True,
    # adaptive rules (reference-free, use only variational quantities):
    extend_max_chunks=4,      # keep optimizing while still descending
    extend_eps=1e-7,          # Ha/step descent threshold over the window
    window=50,
    adaptive_depth=True,      # deepen L by identity-init layers while it pays
    depth_accept_mha=0.1,     # accept a deeper circuit iff E improves by this
    l_max=4,
)
# ---------------------------------------------------------------------------------

CSV_FIELDS = ["molecule", "no", "n_qubits", "sector_dim", "topology", "seed",
              "layers", "steps", "extension_chunks_used", "warm_started",
              "e_vqe", "e_casci", "e_hf", "error_mha",
              "corr_fraction", "n_pairs", "n_cross_pairs", "phi", "wall_seconds"]


def steps_for(dim):
    """Select the frozen optimization budget from the sector dimension."""

    if dim <= PROTOCOL["dim_medium"]:
        return PROTOCOL["steps_small"]
    if dim <= PROTOCOL["dim_large"]:
        return PROTOCOL["steps_medium"]
    return PROTOCOL["steps_large"]


def main():
    """Execute the molecule-agnostic active-space escalation protocol."""

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="molecule")
    ap.add_argument("--atom", default=None, help="PySCF geometry string")
    ap.add_argument("--basis", default="cc-pvdz")
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--spin", type=int, default=0)
    ap.add_argument("--unit", default="Angstrom")
    ap.add_argument("--ncore", type=int, default=0, help="frozen doubly-occ orbitals")
    ap.add_argument("--fcidump", default=None, help="FCIDUMP path (alternative input)")
    ap.add_argument("--ladder", nargs="+", type=int, default=None,
                    help="active orbital counts; default: auto up to 20 (40 qubits)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[17, 11111, 22222])
    ap.add_argument("--topology", choices=["all", "partitioned"], default="all")
    ap.add_argument("--no-warm-start", action="store_true")
    ap.add_argument("--no-adaptive", action="store_true",
                    help="disable descent extension + depth escalation")
    ap.add_argument("--l-max", type=int, default=None,
                    help="override PROTOCOL l_max (deeper helps strongly "
                         "correlated systems at proportional cost)")
    ap.add_argument("--max-sector-dim", type=int, default=30_000_000,
                    help="skip rungs above this dimension (memory guard)")
    default_out = (
        Path(__file__).resolve().parents[2]
        / "results"
        / "judge_reproduction"
        / "new_molecule"
    )
    ap.add_argument("--out", default=default_out)
    args = ap.parse_args()

    print("FROZEN PROTOCOL:", json.dumps(PROTOCOL), flush=True)
    out = Path(args.out); (out / "json").mkdir(parents=True, exist_ok=True)
    csv_path = out / "ladder.csv"
    write_header = not csv_path.exists()
    warm = PROTOCOL["warm_start"] and not args.no_warm_start

    if args.fcidump:
        rungs = [None]  # single problem, dimensions fixed by the file
    else:
        if args.atom is None:
            ap.error("provide --atom or --fcidump")
        spec = dict(atom=args.atom, basis=args.basis, charge=args.charge,
                    spin=args.spin, unit=args.unit, ncore=args.ncore)
        rungs = args.ladder or [3, 5, 8, 10, 12, 14, 17, 20]

    # Open-shell molecules get independent alpha/beta angles (a doublet's
    # spin channels are inequivalent); closed shells share them.
    spin_sym = PROTOCOL["spin_symmetric"] and args.spin == 0
    acfg_base = AnsatzConfig(
        layers=PROTOCOL["layers"], gates=PROTOCOL["gates"],
        convention=PROTOCOL["convention"], spin_symmetric=spin_sym,
        init_state=PROTOCOL["init_state"], init_scale=PROTOCOL["init_scale"],
        topology=args.topology, phi_max=PROTOCOL["phi_max"],
        beta_cap=PROTOCOL["beta_cap"], beta_cap_double=PROTOCOL["beta_cap_double"],
    )

    prev = {}   # seed -> (prob, params) from the previous rung
    for no in rungs:
        t0 = time.perf_counter()
        try:
            if args.fcidump:
                prob = build_from_fcidump(args.fcidump, name=args.name,
                                          casci_max_dim=PROTOCOL["casci_max_dim"])
            else:
                prob = build_cas_from_spec(spec, no, name=args.name,
                                           casci_max_dim=PROTOCOL["casci_max_dim"])
        except ValueError as e:
            print(f"[skip] no={no}: {e}", flush=True)
            continue
        dim = int(prob.meta["sector_dim"])
        if dim > args.max_sector_dim:
            print(f"[skip] no={no}: sector dim {dim:,} above guard "
                  f"{args.max_sector_dim:,} (raise --max-sector-dim on a larger node)",
                  flush=True)
            continue
        ref = "CASCI" if prob.e_casci is not None else "none (dim above cap)"
        print(f"\n=== {prob.name}: dim {dim:,}, reference: {ref} "
              f"({time.perf_counter()-t0:.1f}s) ===", flush=True)
        acfg = acfg_base
        from dataclasses import replace
        acfg = replace(acfg, engine="adjoint" if dim > PROTOCOL["engine_switch_dim"]
                       else "autograd")
        nxt = {}
        adaptive = PROTOCOL["adaptive_depth"] and not args.no_adaptive
        for seed in args.seeds:
            seed_layers = prev.get(seed, (None, None, PROTOCOL["layers"]))[2]
            acfg_s = replace(acfg, layers=seed_layers)
            init = None
            warm_used = False
            if warm and seed in prev:
                p_small, par_small, _ = prev[seed]
                try:
                    init = warm_start_params(p_small, prob, acfg_s, par_small, seed)
                    warm_used = True
                except Exception as e:  # fall back to cold start, loudly
                    print(f"    [warm-start failed, cold start] {e}", flush=True)
            ocfg = OptConfig(
                steps=steps_for(dim), seed=seed, lr=PROTOCOL["lr"],
                extend_max_chunks=(PROTOCOL["extend_max_chunks"] if adaptive else 0),
                extend_eps=PROTOCOL["extend_eps"], window=PROTOCOL["window"],
            )
            if adaptive:
                res = run_adaptive_depth(prob, acfg_s, ocfg,
                                         l_max=(args.l_max or PROTOCOL["l_max"]),
                                         accept_mha=PROTOCOL["depth_accept_mha"],
                                         init_params=init,
                                         log=lambda s: print(s, flush=True))
            else:
                res = run_vqe(prob, acfg_s, ocfg, init_params=init, return_params=True)
            accepted_layers = res["ansatz"]["layers"]
            res["layers"] = accepted_layers
            nxt[seed] = (prob, res.pop("best_params"), accepted_layers)
            res.pop("pairs", None)
            res["warm_started"] = warm_used
            res["molecule"] = args.name
            res["no"] = prob.norb
            res["topology"] = args.topology
            hist = res.pop("history")
            (out / "json" / f"{prob.name}_{args.topology}_s{seed}.json").write_text(
                json.dumps({**res, "history": hist}, default=str, indent=1))
            row = {k: res.get(k) for k in CSV_FIELDS}
            with csv_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if write_header:
                    w.writeheader(); write_header = False
                w.writerow(row)
            err = res["error_mha"]
            err_s = f"{err:+9.4f} mHa" if err is not None else "  n/a (no ref)"
            print(f"    seed {seed:6d} [{'warm' if warm_used else 'cold'}] "
                  f"E={res['e_vqe']:.8f}  err={err_s}  phi={res['phi']:.3f}  "
                  f"({res['wall_seconds']:.0f}s)", flush=True)
        prev = nxt


if __name__ == "__main__":
    main()
