"""End-to-end QPD validation: cut-circuit energy == uncut energy, exactly.

BeH2 (no=3, 6 qubits), orbital partition {0,1} | {2}: side A carries the
alpha+beta qubits of orbitals 0,1; one cross orbital pair (0,2) contributes
two cut Givens gates (alpha and beta channels). The energy is reconstructed
by exact enumeration of the 10^4 quasi-probability branches, simulating
ONLY the 4-qubit and 2-qubit halves.

Run: python -m tests.test_qpd
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

sys.path.insert(0, ".")

from givens40.chemistry import build_cas
from givens40.densecheck import dense_hamiltonian
from givens40.qpd import (Gate, full_energy, givens4, pairdouble16,
                          pauli_decompose, qpd_energy)


def main(argv: Sequence[str] | None = None) -> int:
    """Compare exact cut reconstruction with the uncut dense reference."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional machine-readable QPD metric record",
    )
    args = parser.parse_args(argv)

    prob = build_cas("BeH2", 3)
    no = prob.norb
    n = prob.n_qubits
    H = dense_hamiltonian(prob.h1e, prob.eri, prob.ecore)
    pauli_terms = pauli_decompose(H, n)
    print(f"Pauli terms: {len(pauli_terms)}")

    rng = np.random.default_rng(11)
    a = lambda: float(rng.normal(scale=0.3))

    def q_a(p):  # alpha qubit of orbital p
        return p

    def q_b(p):  # beta qubit
        return p + no

    left = [0, 1]
    side_a = {q_a(p) for p in left} | {q_b(p) for p in left}
    gates = []
    for layer in range(2):
        # intra-block pair (0,1): alpha single, beta single, pair-double
        b1, b2, d1 = a(), a(), a()
        gates.append(Gate("local", (q_a(0), q_a(1)), mat=givens4(b1)))
        gates.append(Gate("local", (q_b(0), q_b(1)), mat=givens4(b2)))
        gates.append(Gate("local", (q_a(0), q_a(1), q_b(0), q_b(1)),
                          mat=pairdouble16(d1)))
        if layer == 0:
            # cross pair (0,2): alpha + beta singles, CUT
            # (one budgeted cross pair => 2 cut gates => 10^4 branches)
            bc = a()
            gates.append(Gate("cut", (q_a(0), q_a(2)), beta=bc))
            gates.append(Gate("cut", (q_b(0), q_b(2)), beta=bc))

    # HF initial state: orbitals 0,1 doubly occupied
    init_bits = (1 << q_a(0)) | (1 << q_a(1)) | (1 << q_b(0)) | (1 << q_b(1))

    e_full = full_energy(gates, n, init_bits, H)
    e_qpd, gamma = qpd_energy(gates, n, side_a, init_bits, pauli_terms)
    n_branches = 100 ** sum(1 for g in gates if g.kind == "cut")
    diff = abs(e_full - e_qpd)
    print(f"uncut  E = {e_full:.12f} Ha")
    print(f"QPD    E = {e_qpd:.12f} Ha   ({n_branches:,} branches enumerated, "
          f"halves of {len(side_a)} and {n - len(side_a)} qubits only)")
    print(f"|diff| = {diff:.3e} Ha   gamma_naive = {gamma:.3f}")
    ok = diff < 1e-9
    print(f"[{'PASS' if ok else 'FAIL'}] QPD reconstruction == uncut energy")
    if args.output is not None:
        record = {
            "status": "passed" if ok else "failed",
            "n_qubits": n,
            "half_qubits": [len(side_a), n - len(side_a)],
            "cut_gates": sum(1 for gate in gates if gate.kind == "cut"),
            "enumerated_branches": n_branches,
            "uncut_energy_hartree": e_full,
            "qpd_energy_hartree": e_qpd,
            "absolute_difference_hartree": diff,
            "gamma_naive": gamma,
        }
        args.output = args.output.expanduser().resolve()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
