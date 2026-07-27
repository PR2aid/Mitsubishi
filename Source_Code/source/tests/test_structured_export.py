#!/usr/bin/env python3
"""Independent exactness and resource tests for the structured exporter."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np


HERE = Path(__file__).resolve()
ADDON_ROOT = HERE.parents[1]
if str(ADDON_ROOT) not in sys.path:
    sys.path.insert(0, str(ADDON_ROOT))

from givens40.structured_qiskit_export import (  # noqa: E402
    append_pair_double,
    append_single_exchange,
    clopper_pearson_expectation,
    structured_pair_double_circuit,
    transpiled_resources,
    witness_from_counts,
)


def _phase_aligned_max_difference(reference: np.ndarray, candidate: np.ndarray) -> float:
    overlap = np.vdot(reference.reshape(-1), candidate.reshape(-1))
    phase = 1.0 + 0.0j if abs(overlap) == 0 else overlap / abs(overlap)
    return float(np.max(np.abs(candidate - phase * reference)))


def _legacy_pair_double(delta: float):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from givens40.qpd import pairdouble16

    qc = QuantumCircuit(4)
    qc.append(UnitaryGate(pairdouble16(delta)), range(4))
    return qc


def _shaped_circuit(n_qubits: int, pairs: list[tuple[int, int]], structured: bool):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from givens40.qpd import pairdouble16

    no = n_qubits // 2
    qc = QuantumCircuit(n_qubits)
    qc.x(0)
    qc.x(no)
    for layer in range(2):
        for k, (p, q) in enumerate(pairs):
            angle = 0.031 * (1 + layer + k)
            append_single_exchange(qc, angle, (p, q))
            append_single_exchange(qc, -0.7 * angle, (p + no, q + no))
            if structured:
                append_pair_double(qc, 0.5 * angle, (p, q, p + no, q + no))
            else:
                qc.append(
                    UnitaryGate(pairdouble16(0.5 * angle)),
                    [p, q, p + no, q + no],
                )
    return qc


class StructuredExportTests(unittest.TestCase):
    def test_pair_double_unitary_exact_for_multiple_angles(self):
        from qiskit.quantum_info import Operator
        from givens40.qpd import pairdouble16

        for delta in (0.0, 1e-9, 0.013, -0.137, 0.711, -1.2):
            with self.subTest(delta=delta):
                actual = np.asarray(Operator(structured_pair_double_circuit(delta)).data)
                expected = np.asarray(pairdouble16(delta))
                self.assertLess(_phase_aligned_max_difference(expected, actual), 2e-12)

    def test_pair_double_resource_reduction(self):
        delta = 0.137
        _, legacy = transpiled_resources(_legacy_pair_double(delta))
        _, structured = transpiled_resources(structured_pair_double_circuit(delta))
        self.assertEqual(legacy["cx"], 92)
        self.assertLessEqual(structured["cx"], 16)
        self.assertLess(structured["depth"], legacy["depth"])

    def test_multigate_state_equivalence_and_reduction_6q_12q(self):
        from qiskit.quantum_info import Statevector, state_fidelity

        cases = {
            6: [(0, 1), (1, 2)],
            12: [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 3), (2, 5)],
        }
        for width, pairs in cases.items():
            with self.subTest(width=width):
                legacy = _shaped_circuit(width, pairs, structured=False)
                structured = _shaped_circuit(width, pairs, structured=True)
                fidelity = float(
                    state_fidelity(
                        Statevector.from_instruction(legacy),
                        Statevector.from_instruction(structured),
                    )
                )
                self.assertGreaterEqual(fidelity, 1.0 - 2e-12)
                _, legacy_r = transpiled_resources(legacy)
                _, structured_r = transpiled_resources(structured)
                self.assertLess(structured_r["cx"], 0.55 * legacy_r["cx"])
                self.assertLess(structured_r["depth"], legacy_r["depth"])

    def test_counts_endianness_and_exact_interval(self):
        # q0 and q2 form the witness support.  Only q0 is set in "0001",
        # giving negative parity; q0+q2 in "0101" gives positive parity.
        expectation, plus, minus = witness_from_counts(
            {"0001": 30, "0101": 70}, "XIXI"
        )
        self.assertEqual((plus, minus), (70, 30))
        self.assertAlmostEqual(expectation, 0.4)
        lo, hi = clopper_pearson_expectation(plus, minus)
        self.assertLess(lo, expectation)
        self.assertGreater(hi, expectation)


def resource_report() -> dict:
    """Return the exact comparison used in the submission report."""

    pair_angle = 0.137
    _, legacy_gate = transpiled_resources(_legacy_pair_double(pair_angle))
    _, structured_gate = transpiled_resources(structured_pair_double_circuit(pair_angle))
    cases = {
        "6q_partitioned_shape": (6, [(0, 1), (1, 2)]),
        "12q_partitioned_shape": (
            12,
            [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 3), (2, 5)],
        ),
    }
    shaped = {}
    for label, (width, pairs) in cases.items():
        legacy = _shaped_circuit(width, pairs, structured=False)
        structured = _shaped_circuit(width, pairs, structured=True)
        _, old = transpiled_resources(legacy)
        _, new = transpiled_resources(structured)
        shaped[label] = {
            "legacy": old,
            "structured": new,
            "cx_reduction_fraction": 1.0 - new["cx"] / old["cx"],
            "depth_reduction_fraction": 1.0 - new["depth"] / old["depth"],
        }
    return {
        "pair_double_angle": pair_angle,
        "single_pair_double": {
            "legacy": legacy_gate,
            "structured": structured_gate,
            "cx_reduction_fraction": 1.0 - structured_gate["cx"] / legacy_gate["cx"],
            "depth_reduction_fraction": 1.0
            - structured_gate["depth"] / legacy_gate["depth"],
        },
        "shaped_circuits": shaped,
    }


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(StructuredExportTests)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    print("RESOURCE_REPORT=" + json.dumps(resource_report(), sort_keys=True))
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
