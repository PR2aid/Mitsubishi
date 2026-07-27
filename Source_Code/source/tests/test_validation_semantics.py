"""Focused tests for independent Phi and residual-token recomputation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validate_submission_results import (
    Audit,
    REFERENCE_AUDIT_REQUIRED_CHECKS,
    TABLE3_LABELS,
    _resource_artifact,
    audit_token_sequence,
    json_default,
    recompute_phi,
    recompute_token_cost,
    sha256_file,
    validate_expected_table3_resources,
    validate_frozen_reference_audit,
    validate_table3_structure,
)


def _arm(cx: int, depth: int, qasm_sha256: str) -> dict:
    return {
        "cx": cx,
        "depth": depth,
        "qasm_sha256": qasm_sha256,
    }


def _table3_record(
    *,
    generic_cx: int = 10,
    structured_cx: int = 4,
    generic_depth: int = 20,
    structured_depth: int = 8,
) -> dict:
    generic = _arm(generic_cx, generic_depth, "a" * 64)
    structured = _arm(structured_cx, structured_depth, "b" * 64)
    return {
        "generic_qasm": generic,
        "structured_qasm": structured,
        "legacy_generic_unitary": dict(generic),
        "structured_exact_pauli_network": dict(structured),
        "state_fidelity": 1.0,
        "equivalence_scope": (
            "compositional exact primitive tests; no dense 40q state allocated"
        ),
    }


def _table3_advanced() -> dict:
    return {
        "structured_resources": {
            label: _table3_record() for label in TABLE3_LABELS
        }
    }


def _table3_expected() -> dict:
    targets = {}
    for label in TABLE3_LABELS:
        record = {
            "legacy_cx": 10,
            "structured_cx": 4,
            "legacy_depth": 20,
            "structured_depth": 8,
        }
        if label == "LiH-40":
            record["fidelity_scope"] = (
                "compositional exact primitive tests; "
                "no dense 40q state allocated"
            )
        else:
            record["minimum_fidelity"] = 1.0 - 1e-12
        targets[label] = record
    return {
        "structured_resources": targets,
        "table3_resource_contract_provenance": {
            "exact_qasm_sha256": {
                label: {
                    "generic_qasm": "a" * 64,
                    "structured_qasm": "b" * 64,
                }
                for label in TABLE3_LABELS
            }
        },
    }


class ValidationSemanticTests(unittest.TestCase):
    def test_numpy_diagnostics_have_a_narrow_json_serializer(self):
        encoded = json.dumps(
            {
                "array": np.array([1, 2]),
                "scalar": np.float64(1.25),
            },
            default=json_default,
            sort_keys=True,
        )
        self.assertEqual(
            json.loads(encoded),
            {"array": [1, 2], "scalar": 1.25},
        )
        with self.assertRaises(TypeError):
            json_default(object())

    def test_phi_uses_both_single_and_double_angles(self):
        phi_all = recompute_phi([0.05, -0.05], [0.2])
        phi_singles = recompute_phi([0.05, -0.05], [])
        self.assertGreater(phi_all, phi_singles)
        self.assertGreaterEqual(phi_singles, 1.0)

    def test_token_cost_is_even_in_angle(self):
        positive = recompute_token_cost(
            {"kind": "double", "angle": 0.2, "u_cost": 0.0}
        )
        negative = recompute_token_cost(
            {"kind": "double", "angle": -0.2, "u_cost": 0.0}
        )
        self.assertAlmostEqual(positive, negative, places=15)

    def test_recorded_token_cost_tamper_fails(self):
        token = {"kind": "single", "angle": 0.05, "u_cost": 0.0}
        audit = Audit()
        audit_token_sequence(audit, "tamper probe", [token])
        self.assertEqual(audit.close(), "FAIL")

    def test_exact_recorded_token_cost_passes(self):
        token = {"kind": "single", "angle": 0.05}
        token["u_cost"] = recompute_token_cost(token)
        audit = Audit()
        total = audit_token_sequence(audit, "exact probe", [token])
        self.assertEqual(audit.close(), "PASS")
        self.assertAlmostEqual(total, token["u_cost"], places=15)

    def test_empty_token_sequence_and_declared_count_mismatch_fail(self):
        empty_audit = Audit()
        audit_token_sequence(
            empty_audit,
            "empty probe",
            [],
            expected_count=6,
        )
        self.assertEqual(empty_audit.close(), "FAIL")

        token = {"kind": "identity", "angle": 0.0, "u_cost": 0.0}
        count_audit = Audit()
        audit_token_sequence(
            count_audit,
            "count probe",
            [token],
            expected_count=6,
        )
        self.assertEqual(count_audit.close(), "FAIL")

    def test_exact_table3_case_set_is_required(self):
        valid = Audit()
        labels = validate_table3_structure(valid, _table3_advanced())
        self.assertEqual(labels, list(TABLE3_LABELS))
        self.assertEqual(valid.close(), "PASS")

        empty = Audit()
        labels = validate_table3_structure(
            empty, {"structured_resources": {}}
        )
        self.assertEqual(labels, [])
        self.assertEqual(empty.close(), "FAIL")

        extra_record = _table3_advanced()
        extra_record["structured_resources"]["unexpected"] = _table3_record()
        extra = Audit()
        validate_table3_structure(extra, extra_record)
        self.assertEqual(extra.close(), "FAIL")

    def test_table3_compatibility_aliases_must_equal_bound_records(self):
        advanced = _table3_advanced()
        advanced["structured_resources"]["BeH2-6"][
            "legacy_generic_unitary"
        ]["cx"] = 999
        audit = Audit()
        validate_table3_structure(audit, advanced)
        self.assertEqual(audit.close(), "FAIL")

    def test_expected_table3_values_use_qasm_records_not_aliases(self):
        advanced = _table3_advanced()
        expected = _table3_expected()
        passing = Audit()
        validate_expected_table3_resources(passing, advanced, expected)
        self.assertEqual(passing.close(), "PASS")

        record = advanced["structured_resources"]["BeH2-6"]
        record["generic_qasm"]["cx"] = 999
        record["legacy_generic_unitary"]["cx"] = 10
        tampered = Audit()
        validate_expected_table3_resources(tampered, advanced, expected)
        self.assertEqual(tampered.close(), "FAIL")
        legacy_check = next(
            item
            for item in tampered.checks
            if item["name"] == "BeH2-6 legacy_cx"
        )
        self.assertEqual(legacy_check["actual"], 999)

    def test_expected_table3_exact_qasm_hash_tamper_fails(self):
        advanced = _table3_advanced()
        expected = _table3_expected()
        advanced["structured_resources"]["LiH-40"]["structured_qasm"][
            "qasm_sha256"
        ] = "c" * 64
        advanced["structured_resources"]["LiH-40"][
            "structured_exact_pauli_network"
        ]["qasm_sha256"] = "c" * 64
        audit = Audit()
        validate_expected_table3_resources(audit, advanced, expected)
        self.assertEqual(audit.close(), "FAIL")
        hash_check = next(
            item
            for item in audit.checks
            if item["name"]
            == "LiH-40 structured_qasm exact expected QASM hash"
        )
        self.assertEqual(hash_check["actual"], "c" * 64)

    def test_expected_table3_hash_contract_rejects_nonhex_digest(self):
        advanced = _table3_advanced()
        expected = _table3_expected()
        expected["table3_resource_contract_provenance"][
            "exact_qasm_sha256"
        ]["LiH-40"]["structured_qasm"] = "z" * 64
        audit = Audit()
        validate_expected_table3_resources(audit, advanced, expected)
        self.assertEqual(audit.close(), "FAIL")
        contract_check = next(
            item
            for item in audit.checks
            if item["name"]
            == "expected Table-3 exact-QASM hash contract is complete"
        )
        self.assertEqual(contract_check["status"], "FAIL")

    def test_resource_artifact_root_and_qasm_are_strict_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            advanced_root = Path(directory) / "advanced_method"
            record = {
                "artifact_root": "canonical",
                "generic_qasm": {"qasm_file": "circuits/generic.qasm"},
            }
            valid = _resource_artifact(
                advanced_root, record, "generic_qasm"
            )
            self.assertEqual(
                valid,
                (
                    advanced_root
                    / "canonical"
                    / "circuits"
                    / "generic.qasm"
                ).resolve(),
            )

            root_alias = {
                **record,
                "artifact_root": ".",
            }
            with self.assertRaisesRegex(ValueError, "strictly below"):
                _resource_artifact(
                    advanced_root, root_alias, "generic_qasm"
                )

            escaped_root = {
                **record,
                "artifact_root": "../outside",
            }
            with self.assertRaisesRegex(ValueError, "strictly below"):
                _resource_artifact(
                    advanced_root, escaped_root, "generic_qasm"
                )

            escaped_qasm = {
                **record,
                "generic_qasm": {"qasm_file": "../outside.qasm"},
            }
            with self.assertRaisesRegex(ValueError, "escapes"):
                _resource_artifact(
                    advanced_root, escaped_qasm, "generic_qasm"
                )

    def _write_reference_fixture(
        self, directory: Path
    ) -> tuple[Path, Path, dict]:
        manifest_path = directory / "MANIFEST.json"
        problems = {
            f"case_{index}": {
                "norb": 2,
                "nelec_alpha_beta": [1, 1],
                "sector_dimension": 4,
                "source": {"frozen_core_orbitals": 1},
                "array_sha256": {"mo_coeff_active": "a" * 64},
            }
            for index in range(10)
        }
        manifest_path.write_text(
            json.dumps({"problems": problems}),
            encoding="utf-8",
        )
        problem_records = {}
        for name in problems:
            problem_records[name] = {
                "status": "PASS",
                "nelec_alpha_beta": [1, 1],
                "sector_dimension": 4,
                "sector_consistency": {
                    "passed": True,
                    "manifest_nelec_alpha_beta": [1, 1],
                    "bundle_nelec_alpha_beta": [1, 1],
                    "bundle_matches_manifest": True,
                    "declared_sector_dimension": 4,
                    "combinatorial_sector_dimension": 4,
                    "declared_dimension_matches_combinatorial": True,
                },
                "checks": {
                    check: True
                    for check in REFERENCE_AUDIT_REQUIRED_CHECKS
                },
                "frozen_active_basis": {
                    "passed": True,
                    "coefficient_sha256": "a" * 64,
                    "expected_coefficient_sha256": "a" * 64,
                    "coefficient_checksum_match": True,
                    "s_orthonormality": {
                        "passed": True,
                        "max_abs_error": 1e-14,
                        "absolute_tolerance": 2e-8,
                    },
                    "core_active_orthogonality": {
                        "passed": True,
                        "max_abs_overlap": 1e-14,
                        "absolute_tolerance": 2e-8,
                    },
                    "generalized_fock_eigen_residual": {
                        "passed": True,
                        "max_abs_residual": 1e-14,
                        "absolute_tolerance": 2e-8,
                    },
                },
                "active_space_regeneration": {
                    "basis": (
                        "checksum-bound frozen active orbitals inserted after "
                        "the regenerated core"
                    ),
                    "declared_ncore": 1,
                    "pyscf_casci_ncore": 1,
                    "declared_core_count_matches": True,
                },
                "pspace": {
                    "solver": (
                        "numpy.linalg.eigh on PySCF full determinant p-space"
                    ),
                    "converged": True,
                    "all_determinant_addresses_present": True,
                    "full_determinant_dimension": 4,
                    "p_space_dimension": 4,
                    "eigen_residual_norm_hartree": 1e-14,
                    "hermiticity_max_abs_hartree": 0.0,
                    "residual_absolute_tolerance_hartree": 1e-10,
                },
            }
        certificate = {
            "status": "PASS",
            "manifest_sha256": sha256_file(manifest_path),
            "problem_count": 10,
            "passed": 10,
            "failed": 0,
            "problems": problem_records,
        }
        run_dir = directory / "run"
        run_dir.mkdir()
        artifact_path = run_dir / "frozen_reference_audit.json"
        artifact_path.write_text(
            json.dumps(certificate),
            encoding="utf-8",
        )
        return manifest_path, run_dir, certificate

    def test_independent_reference_audit_requires_exact_ten_case_set(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            manifest, run_dir, certificate = self._write_reference_fixture(
                directory
            )
            valid = Audit()
            validate_frozen_reference_audit(
                valid, run_dir, manifest_path=manifest
            )
            self.assertEqual(valid.close(), "PASS")

            certificate["problems"].pop("case_9")
            (run_dir / "frozen_reference_audit.json").write_text(
                json.dumps(certificate),
                encoding="utf-8",
            )
            missing = Audit()
            validate_frozen_reference_audit(
                missing, run_dir, manifest_path=manifest
            )
            self.assertEqual(missing.close(), "FAIL")

    def test_independent_reference_audit_rejects_failed_check_or_residual(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            manifest, run_dir, certificate = self._write_reference_fixture(
                directory
            )
            record = certificate["problems"]["case_0"]
            record["checks"]["bundle_electron_sector"] = False
            record["sector_consistency"]["bundle_nelec_alpha_beta"] = [1, 0]
            record["pspace"]["eigen_residual_norm_hartree"] = 1e-3
            (run_dir / "frozen_reference_audit.json").write_text(
                json.dumps(certificate),
                encoding="utf-8",
            )
            audit = Audit()
            validate_frozen_reference_audit(
                audit, run_dir, manifest_path=manifest
            )
            self.assertEqual(audit.close(), "FAIL")

    def test_independent_reference_audit_rederives_frozen_basis_gates(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            manifest, run_dir, certificate = self._write_reference_fixture(
                directory
            )
            certificate["problems"]["case_0"]["frozen_active_basis"][
                "generalized_fock_eigen_residual"
            ]["max_abs_residual"] = 1e-3
            (run_dir / "frozen_reference_audit.json").write_text(
                json.dumps(certificate),
                encoding="utf-8",
            )
            audit = Audit()
            validate_frozen_reference_audit(
                audit, run_dir, manifest_path=manifest
            )
            self.assertEqual(audit.close(), "FAIL")


if __name__ == "__main__":
    unittest.main()
