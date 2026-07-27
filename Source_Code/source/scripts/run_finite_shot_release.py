"""Generate the canonical genuine finite-shot release artifact.

The Aer estimator constructs and samples commuting-group measurement circuits;
it does not add Gaussian noise to an exact expectation value. ``shots`` means
shots per commuting group, and the JSON records the total circuit shots.

Run from ``source/``::

    python scripts/run_finite_shot_release.py \
      --output ../results/judge_reproduction/finite_shot_beh2_6q.json
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, ".")

from givens40.chemistry import build_cas
from givens40.qiskit_export import estimate_energy_aer, transpiled_resources
from givens40.runner import AnsatzConfig, OptConfig, SectorCircuit, run_vqe


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> int:
    """Regenerate the seeded 20,000-shots-per-group BeH2 result."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=20_000,
                        help="shots per commuting Pauli group")
    parser.add_argument("--seed", type=int, default=7,
                        help="Aer measurement seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../results/judge_reproduction/finite_shot_beh2_6q.json"),
    )
    args = parser.parse_args()

    start = time.perf_counter()
    problem = build_cas("BeH2", 3)
    ansatz = AnsatzConfig()
    optimized = run_vqe(
        problem,
        ansatz,
        OptConfig(steps=200, seed=17),
        return_params=True,
    )
    circuit = SectorCircuit(problem, ansatz)
    sampled_energy, qiskit_circuit, sampling = estimate_energy_aer(
        circuit,
        optimized["best_params"],
        shots=args.shots,
        seed=args.seed,
        return_metadata=True,
    )
    resources = transpiled_resources(qiskit_circuit)
    record = {
        "status": "completed",
        "system": {
            "molecule": "BeH2",
            "n_qubits": problem.n_qubits,
            "active_orbitals": problem.norb,
            "sector_dimension": int(problem.meta["sector_dim"]),
        },
        "environment": {
            "python": platform.python_version(),
            "packages": {
                name: _version(name)
                for name in ("numpy", "torch", "pyscf", "qiskit", "qiskit-aer")
            },
        },
        "optimization": {
            "seed": 17,
            "steps": 200,
            "exact_sector_energy_hartree": optimized["e_vqe"],
            "casci_energy_hartree": problem.e_casci,
            "exact_sector_error_mha": optimized["error_mha"],
        },
        "finite_shot": {
            **sampling,
            "energy_hartree": sampled_energy,
            "difference_from_exact_sector_mha": 1000.0 * (
                sampled_energy - optimized["e_vqe"]
            ),
            "error_vs_casci_mha": 1000.0 * (
                sampled_energy - problem.e_casci
            ),
        },
        "transpiled_resources": resources,
        "total_wall_seconds": time.perf_counter() - start,
        "evidence_note": (
            "Qiskit BackendEstimatorV2 wrapped around AerSimulator; sampled "
            "commuting-group measurement circuits; shots are per group"
        ),
    }
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, sort_keys=True), flush=True)
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
