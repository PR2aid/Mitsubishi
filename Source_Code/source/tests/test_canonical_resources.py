"""Dependency-light tests for canonical Table-3 artifact semantics."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from givens40.canonical_resources import (
    bound_record_differences,
    candidate_manifest,
    compilation_protocol,
    load_canonical_manifest,
    reference_case_paths,
)


def arm(digest: str, cx: int) -> dict:
    return {
        "qasm_sha256": digest,
        "basis_gates": ["rz", "sx", "x", "cx"],
        "optimization_level": 3,
        "seed_transpiler": 3047,
        "logical_qubits": 6,
        "depth": 10,
        "size": 20,
        "cx": cx,
        "operations": {"cx": cx, "rz": 2},
    }


class CanonicalResourceTests(unittest.TestCase):
    def test_protocol_is_fully_declared(self):
        protocol = compilation_protocol()
        self.assertEqual(protocol["seed_transpiler"], 3047)
        self.assertEqual(protocol["basis_gates"], ["rz", "sx", "x", "cx"])
        self.assertEqual(protocol["optimization_level"], 3)

    def test_hash_or_resource_change_is_a_binding_failure(self):
        expected = {
            "seed": 3047,
            "n_qubits": 6,
            "selected_left_block": [1],
            "compilation_protocol": compilation_protocol(),
            "generic_qasm": arm("a" * 64, 20),
            "structured_qasm": arm("b" * 64, 10),
        }
        actual = {
            **expected,
            "generic_qasm": arm("c" * 64, 21),
        }
        differences = bound_record_differences(actual, expected)
        self.assertTrue(any("qasm_sha256" in item for item in differences))
        self.assertTrue(any(".cx" in item for item in differences))

    def test_candidate_is_never_labeled_canonical(self):
        record = candidate_manifest({})
        self.assertIn("NOT_CANONICAL", record["status"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canonical_table3.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not been explicitly promoted"):
                load_canonical_manifest(path)

    def test_reference_paths_must_remain_below_manifest_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "canonical_table3.json"
            valid = {
                "parameter_artifact": {"file": "parameters/case.npz"},
                "generic_qasm": {"qasm_file": "circuits/generic.qasm"},
                "structured_qasm": {
                    "qasm_file": "circuits/structured.qasm"
                },
            }
            paths = reference_case_paths(manifest, valid)
            self.assertTrue(
                all(root.resolve() in path.parents for path in paths.values())
            )

            escaping = {
                **valid,
                "generic_qasm": {"qasm_file": "../outside.qasm"},
            }
            with self.assertRaisesRegex(ValueError, "strictly below"):
                reference_case_paths(manifest, escaping)

            root_alias = {
                **valid,
                "parameter_artifact": {"file": "."},
            }
            with self.assertRaisesRegex(ValueError, "strictly below"):
                reference_case_paths(manifest, root_alias)


if __name__ == "__main__":
    unittest.main()
