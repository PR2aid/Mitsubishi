"""Focused adversarial tests for the pre-promotion Table-3 verifier."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from audit_table3_candidate import (
    Audit,
    CANDIDATE_STATUS,
    CandidateAuditFailure,
    EXPECTED_CANDIDATE_FILES,
    EXPECTED_CASES,
    EXPECTED_STAGE_NAMES,
    REFERENCE_AUDIT_REQUIRED_CHECKS,
    candidate_tree_manifest,
    expected_bootstrap_stage_contracts,
    inspect_parameter_npz,
    inspect_qasm_arm,
    recompute_phi,
    sha256_file,
    validate_bootstrap_status,
    validate_candidate_manifest_structure,
    validate_consumed_parameter_contract,
    validate_dense_record,
    validate_frozen_reference_certificate,
    validate_output_path,
)
from givens40.reference_audit import DEFAULT_TOLERANCES, array_sha256


PROTOCOL = {
    "basis_gates": ["rz", "sx", "x", "cx"],
    "optimization_level": 3,
    "seed_transpiler": 3047,
    "connectivity": "all-to-all",
    "scope": "diagnostic logical compile; not device-native",
}
RESOURCE_TARGETS = {
    "BeH2-6": {
        "legacy_cx": 374,
        "structured_cx": 65,
        "legacy_depth": 1278,
        "structured_depth": 192,
        "minimum_fidelity": 0.9999999999,
    },
    "BeH2-12": {
        "legacy_cx": 1325,
        "structured_cx": 235,
        "legacy_depth": 2316,
        "structured_depth": 343,
        "minimum_fidelity": 0.9999999999,
    },
    "LiH-40": {
        "legacy_cx": 17190,
        "structured_cx": 3091,
        "legacy_depth": 9483,
        "structured_depth": 1537,
        "fidelity_scope": (
            "compositional exact primitive tests; no dense 40q state allocated"
        ),
    },
}
TOPOLOGY_TARGETS = {
    "BeH2-6": {
        "decision": "ACCEPT_CONSTRAINED_PARETO_FRONT_TRADEOFF",
        "static_left": [2],
        "selected_left": [1],
    },
    "BeH2-12": {
        "decision": "ACCEPT_CONSTRAINED_PARETO_FRONT_TRADEOFF",
        "static_left": [0, 1, 5],
        "selected_left": [0, 3, 4],
    },
    "LiH-40": {
        "decision": (
            "RETAIN_STATIC_NO_ADAPTIVE_CANDIDATE_PASSED_ALL_GATES"
        ),
        "static_left": [0, 1, 4, 5, 8, 9, 10, 13, 17, 18],
        "selected_left": [0, 1, 4, 5, 8, 9, 10, 13, 17, 18],
    },
}


def metrics() -> dict:
    return {
        "structured_resources": copy.deepcopy(RESOURCE_TARGETS),
        "adaptive_topology": copy.deepcopy(TOPOLOGY_TARGETS),
    }


def arm(
    *,
    label: str,
    kind: str,
    n_qubits: int,
    cx: int,
    depth: int,
) -> dict:
    return {
        **PROTOCOL,
        "qasm_file": f"circuits/{label}_{kind}.qasm",
        "qasm_sha256": "a" * 64,
        "logical_qubits": n_qubits,
        "depth": depth,
        "size": cx + 20,
        "cx": cx,
        "operations": {"cx": cx, "rz": 20},
        "derived_from_reloaded_qasm": True,
        "compiled_once_before_serialization": True,
    }


def candidate_manifest() -> dict:
    cases = {}
    for label, (molecule, norb, n_qubits) in EXPECTED_CASES.items():
        target = RESOURCE_TARGETS[label]
        selected = TOPOLOGY_TARGETS[label]["selected_left"]
        generic = arm(
            label=label,
            kind="generic",
            n_qubits=n_qubits,
            cx=target["legacy_cx"],
            depth=target["legacy_depth"],
        )
        structured = arm(
            label=label,
            kind="structured",
            n_qubits=n_qubits,
            cx=target["structured_cx"],
            depth=target["structured_depth"],
        )
        cases[label] = {
            "label": label,
            "seed": 3047,
            "molecule": molecule,
            "norb": norb,
            "n_qubits": n_qubits,
            "nelec_alpha_beta": [1, 1],
            "selected_left_block": selected,
            "compilation_protocol": PROTOCOL,
            "ansatz": {
                "topology": "partitioned",
                "partition_override": selected,
                "pairs_override": None,
                "phi_max": 15.0,
            },
            "generic_qasm": generic,
            "structured_qasm": structured,
            "legacy_generic_unitary": copy.deepcopy(generic),
            "structured_exact_pauli_network": copy.deepcopy(structured),
            "parameter_artifact": {
                "file": f"parameters/{label}_seed-3047.npz",
                "sha256": "b" * 64,
                "array_names": ["singles", "doubles"],
                "array_shapes": {
                    "singles": [2, 3],
                    "doubles": [2, 3],
                },
                "array_replay_exact": True,
            },
            "cx_reduction_fraction": (
                1.0 - target["structured_cx"] / target["legacy_cx"]
            ),
            "depth_reduction_fraction": (
                1.0 - target["structured_depth"] / target["legacy_depth"]
            ),
            "state_fidelity": 1.0 if n_qubits <= 12 else None,
            "dense_qasm_audit": (
                {
                    "scope": (
                        "dense validation of the exact reloaded QASM artifacts"
                    ),
                    "generic_vs_structured_state_fidelity": 1.0,
                }
                if n_qubits <= 12
                else None
            ),
            "equivalence_scope": (
                "dense validation of the exact reloaded QASM artifacts"
                if n_qubits <= 12
                else target["fidelity_scope"]
            ),
            "qasm_validation_scope": (
                "dense validation of the exact reloaded QASM artifacts"
                if n_qubits <= 12
                else (
                    "40-qubit dense state/leakage/energy allocation "
                    "intentionally omitted; exact primitive tests plus "
                    "checksum-bound parameter and QASM artifacts define "
                    "this scope"
                )
            ),
            "qasm_file": structured["qasm_file"],
            "qasm_sha256": structured["qasm_sha256"],
            "reference_mode": "FULL_RUN_CANDIDATE",
            "artifact_root": "canonical_table3_candidate",
            "device_native": False,
        }
    return {
        "schema_version": 1,
        "status": CANDIDATE_STATUS,
        "seed": 3047,
        "compilation_protocol": PROTOCOL,
        "cases": cases,
        "claim_boundary": (
            "This file is emitted evidence, not a trusted input. Quick mode "
            "requires a separately promoted checksum-bound canonical manifest."
        ),
    }


class FakeCircuit:
    def __init__(
        self,
        *,
        n_qubits: int = 6,
        depth: int = 10,
        size: int = 25,
        operations: dict[str, int] | None = None,
    ) -> None:
        self.num_qubits = n_qubits
        self._depth = depth
        self._size = size
        self._operations = operations or {"cx": 5, "rz": 20}

    def depth(self) -> int:
        return self._depth

    def size(self) -> int:
        return self._size

    def count_ops(self) -> dict[str, int]:
        return self._operations


class CandidateAuditTests(unittest.TestCase):
    def test_successor_resource_contract_is_bound_to_replay_evidence(
        self,
    ) -> None:
        expected = json.loads(
            (ROOT / "expected_metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            expected.get("structured_resources"),
            RESOURCE_TARGETS,
        )
        provenance = expected.get("table3_resource_contract_provenance")
        self.assertIsInstance(provenance, dict)
        self.assertEqual(
            provenance.get("contract_kind"),
            "EXACT_QASM_REDERIVED_RESOURCE_CONTRACT_V1",
        )
        self.assertEqual(
            provenance.get("predecessor_source_identity_sha256"),
            "a440f8ec60a5547113a6e721c0d6b170a7f5f92dd817992e9cd4f2270b2541ab",
        )
        self.assertEqual(
            provenance.get("predecessor_candidate_artifact_aggregate_sha256"),
            "40d732f3f328b94c7534048fcebfbe6f4a89447f458c1fbb2985c223ca187c9c",
        )
        self.assertEqual(
            provenance.get("predecessor_two_process_replay_canonical_sha256"),
            "84fb21014d1d9ad6cc5fedd149ac85308b4d765a2cf997b334c98a5ce137b36f",
        )
        self.assertEqual(
            provenance.get("locked_qiskit_version"),
            "2.5.0",
        )
        self.assertIs(provenance.get("historical_fixture_origin_proven"), False)
        self.assertEqual(
            provenance.get("exact_qasm_sha256"),
            {
                "BeH2-6": {
                    "generic_qasm": (
                        "7926c8c2868ae358d5a7479307c027b101a963cb3fb18e1886748a0b1cbddf78"
                    ),
                    "structured_qasm": (
                        "7f9d5f6f1f33f81feec2c0033f0723202ed805f2990ec0a67d510f685e1c3d2d"
                    ),
                },
                "BeH2-12": {
                    "generic_qasm": (
                        "27bb6c68bc93f2f073dd6d80ce3e7f5179c46ba3dbcd14f1eee30d2d2113a18d"
                    ),
                    "structured_qasm": (
                        "73ff85a69941b10b93df011a74c364ce31c8292c52a9e078eced65540cf3b188"
                    ),
                },
                "LiH-40": {
                    "generic_qasm": (
                        "201915c8dc6f24f6d3e5db5171c71b1dd3f87cf76acca6db85573a32db5d449b"
                    ),
                    "structured_qasm": (
                        "9ba5b2af3f520e5b538c4ded56b99cb59fb21510fb9c231f8fc8ef9063f72dd2"
                    ),
                },
            },
        )

    def assert_audit_fails(self, callback) -> Audit:
        audit = Audit()
        with self.assertRaises(CandidateAuditFailure):
            callback(audit)
        self.assertEqual(audit.status, "FAIL")
        return audit

    def make_candidate_tree(self, root: Path) -> Path:
        candidate = root / "candidate"
        for relative in EXPECTED_CANDIDATE_FILES:
            path = candidate / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"fixture:{relative}\n".encode("utf-8"))
        return candidate

    def make_frozen_reference_fixture(
        self, root: Path
    ) -> tuple[Path, Path, dict]:
        source_code = root / "Source_Code"
        frozen_inputs = source_code / "frozen_inputs"
        frozen_inputs.mkdir(parents=True)
        invocation = root / "invocation"
        invocation.mkdir()
        frozen_rhf = -1.0
        regenerated_rhf = frozen_rhf + 1e-10
        frozen_ecore = -0.3
        regenerated_ecore = frozen_ecore + 2e-10
        electronic_energy = -0.8
        total_energy = electronic_energy + regenerated_ecore
        frozen_casci = -1.1
        occupations = np.asarray([2.0, 0.0], dtype=np.float64)
        nelec = np.asarray([1, 1], dtype=np.int64)
        problems = {}
        for index in range(10):
            name = f"case_{index}"
            bundle_file = f"{name}.npz"
            bundle_path = frozen_inputs / bundle_file
            np.savez_compressed(
                bundle_path,
                e_hf=np.asarray(frozen_rhf, dtype=np.float64),
                e_casci=np.asarray(frozen_casci, dtype=np.float64),
                ecore=np.asarray(frozen_ecore, dtype=np.float64),
                mo_occ_active=occupations,
                nelec=nelec,
            )
            problems[name] = {
                "schema_version": 1,
                "norb": 2,
                "nelec_alpha_beta": [1, 1],
                "sector_dimension": 4,
                "bundle_file": bundle_file,
                "bundle_sha256": sha256_file(bundle_path),
                "rhf_energy_hartree": frozen_rhf,
                "casci_energy_hartree": frozen_casci,
                "ecore_hartree": frozen_ecore,
                "source": {"frozen_core_orbitals": 1},
                "array_sha256": {
                    "mo_coeff_active": "a" * 64,
                    "mo_occ_active": array_sha256(occupations),
                },
            }
        manifest_path = frozen_inputs / "MANIFEST.json"
        manifest_path.write_text(
            json.dumps({"schema_version": 1, "problems": problems}),
            encoding="utf-8",
        )
        tolerances = dict(DEFAULT_TOLERANCES)
        records = {}
        for name, problem in problems.items():
            bundle_sha256 = problem["bundle_sha256"]
            rhf_difference = abs(frozen_rhf - regenerated_rhf)
            casci_difference = abs(frozen_casci - total_energy)
            ecore_difference = abs(frozen_ecore - regenerated_ecore)
            records[name] = {
                "status": "PASS",
                "nelec_alpha_beta": [1, 1],
                "sector_dimension": 4,
                "bundle": {
                    "e_hf": frozen_rhf,
                    "e_casci": frozen_casci,
                    "ecore": frozen_ecore,
                    "bundle_file": problem["bundle_file"],
                    "bundle_sha256": bundle_sha256,
                    "bundle_checksum_match": True,
                    "frozen_active_orbitals": {
                        "array": "mo_coeff_active",
                        "sha256": "a" * 64,
                        "expected_sha256": "a" * 64,
                        "checksum_match": True,
                    },
                },
                "checks": {
                    check: True
                    for check in REFERENCE_AUDIT_REQUIRED_CHECKS
                },
                "occupation_order": {
                    "passed": True,
                    "shape_match": True,
                    "max_abs_frozen_vs_regenerated": 0.0,
                    "max_abs_frozen_vs_expected_order": 0.0,
                    "max_abs_regenerated_vs_expected_order": 0.0,
                    "expected": occupations.tolist(),
                    "frozen": occupations.tolist(),
                    "regenerated": occupations.tolist(),
                    "absolute_tolerance": tolerances[
                        "occupation_max_abs"
                    ],
                },
                "frozen_active_basis": {
                    "passed": True,
                    "shape_valid": True,
                    "finite": True,
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
                        "core_orbital_count": 1,
                    },
                    "generalized_fock_eigen_residual": {
                        "passed": True,
                        "max_abs_residual": 1e-14,
                        "maximum_column_l2_norm": 1e-14,
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
                "integral_differences": {
                    "basis": (
                        "checksum-bound frozen mo_coeff_active with regenerated "
                        "core orbitals"
                    ),
                    "h1e_max_abs": 1e-14,
                    "eri_max_abs": 1e-14,
                    "ecore_abs": ecore_difference,
                },
                "rhf": {
                    "frozen_energy_hartree": frozen_rhf,
                    "regenerated_energy_hartree": regenerated_rhf,
                    "absolute_difference_hartree": rhf_difference,
                    "absolute_tolerance_hartree": tolerances[
                        "rhf_energy_hartree"
                    ],
                    "converged": True,
                },
                "casci": {
                    "frozen_energy_hartree": frozen_casci,
                    "regenerated_energy_hartree": total_energy,
                    "absolute_difference_hartree": casci_difference,
                    "absolute_tolerance_hartree": tolerances[
                        "casci_energy_hartree"
                    ],
                },
                "sector_consistency": {
                    "passed": True,
                    "bundle_nelec_alpha_beta": [1, 1],
                    "manifest_nelec_alpha_beta": [1, 1],
                    "bundle_matches_manifest": True,
                    "declared_sector_dimension": 4,
                    "combinatorial_sector_dimension": 4,
                    "declared_dimension_matches_combinatorial": True,
                },
                "pspace": {
                    "solver": (
                        "numpy.linalg.eigh on PySCF full determinant p-space"
                    ),
                    "converged": True,
                    "all_determinant_addresses_present": True,
                    "full_determinant_dimension": 4,
                    "p_space_dimension": 4,
                    "electronic_energy_hartree": electronic_energy,
                    "total_energy_hartree": total_energy,
                    "eigen_residual_norm_hartree": 1e-14,
                    "hermiticity_max_abs_hartree": 0.0,
                    "residual_absolute_tolerance_hartree": tolerances[
                        "eigen_residual_hartree"
                    ],
                },
            }
        certificate = {
            "schema_version": 1,
            "status": "PASS",
            "manifest_sha256": sha256_file(manifest_path),
            "problem_count": 10,
            "passed": 10,
            "failed": 0,
            "tolerances": tolerances,
            "problems": records,
        }
        (invocation / "frozen_reference_audit.json").write_text(
            json.dumps(certificate),
            encoding="utf-8",
        )
        return source_code, invocation, certificate

    def test_exact_candidate_tree_passes_and_is_not_mutated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = self.make_candidate_tree(Path(directory))
            before = {
                relative: (candidate / relative).read_bytes()
                for relative in EXPECTED_CANDIDATE_FILES
            }
            audit = Audit()
            manifest = candidate_tree_manifest(audit, candidate)
            self.assertEqual(audit.status, "PASS")
            self.assertEqual(manifest["file_count"], 10)
            self.assertEqual(
                {item["path"] for item in manifest["files"]},
                set(EXPECTED_CANDIDATE_FILES),
            )
            self.assertEqual(
                before,
                {
                    relative: (candidate / relative).read_bytes()
                    for relative in EXPECTED_CANDIDATE_FILES
                },
            )

    def test_unexpected_zero_byte_and_symlink_candidates_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            unexpected = self.make_candidate_tree(root / "unexpected")
            (unexpected / "extra.txt").write_text("extra\n", encoding="utf-8")
            self.assert_audit_fails(
                lambda audit: candidate_tree_manifest(audit, unexpected)
            )

            empty = self.make_candidate_tree(root / "empty")
            (empty / "canonical_table3.json").write_bytes(b"")
            self.assert_audit_fails(
                lambda audit: candidate_tree_manifest(audit, empty)
            )

            linked = self.make_candidate_tree(root / "linked")
            target = linked / "outside.qasm"
            target.write_text("OPENQASM 2.0;\n", encoding="utf-8")
            qasm = linked / "circuits" / "BeH2-6_generic.qasm"
            qasm.unlink()
            qasm.symlink_to(target)
            self.assert_audit_fails(
                lambda audit: candidate_tree_manifest(audit, linked)
            )

            special = self.make_candidate_tree(root / "special")
            os.mkfifo(special / "transport.pipe")
            self.assert_audit_fails(
                lambda audit: candidate_tree_manifest(audit, special)
            )

    def test_output_cannot_enter_candidate_reference_or_source_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_code = root / "Source_Code"
            candidate = source_code / "results" / "candidate"
            candidate.mkdir(parents=True)
            (source_code / "reference").mkdir()

            with self.assertRaisesRegex(ValueError, "candidate"):
                validate_output_path(
                    candidate / "audit.json",
                    source_code=source_code,
                    candidate_directory=candidate,
                )
            with self.assertRaisesRegex(ValueError, "reference"):
                validate_output_path(
                    source_code / "reference" / "audit.json",
                    source_code=source_code,
                    candidate_directory=candidate,
                )
            with self.assertRaisesRegex(ValueError, "only below results"):
                validate_output_path(
                    source_code / "audit.json",
                    source_code=source_code,
                    candidate_directory=candidate,
                )
            accepted = validate_output_path(
                source_code / "results" / "audit.json",
                source_code=source_code,
                candidate_directory=candidate,
            )
            self.assertEqual(
                accepted, (source_code / "results" / "audit.json").resolve()
            )

    def test_promoted_status_wrong_basis_topology_and_counts_fail(self) -> None:
        valid = candidate_manifest()
        audit = Audit()
        validate_candidate_manifest_structure(
            audit,
            valid,
            compilation_protocol=PROTOCOL,
            expected_metrics=metrics(),
        )
        self.assertEqual(audit.status, "PASS")

        promoted = copy.deepcopy(valid)
        promoted["status"] = "CANONICAL_PROMOTED"
        self.assert_audit_fails(
            lambda item: validate_candidate_manifest_structure(
                item,
                promoted,
                compilation_protocol=PROTOCOL,
                expected_metrics=metrics(),
            )
        )

        wrong_basis = copy.deepcopy(valid)
        generic = wrong_basis["cases"]["BeH2-6"]["generic_qasm"]
        generic["basis_gates"] = ["u", "cx"]
        wrong_basis["cases"]["BeH2-6"][
            "legacy_generic_unitary"
        ] = copy.deepcopy(generic)
        self.assert_audit_fails(
            lambda item: validate_candidate_manifest_structure(
                item,
                wrong_basis,
                compilation_protocol=PROTOCOL,
                expected_metrics=metrics(),
            )
        )

        wrong_topology = copy.deepcopy(valid)
        wrong_topology["cases"]["BeH2-12"]["selected_left_block"] = [0, 1, 2]
        wrong_topology["cases"]["BeH2-12"]["ansatz"][
            "partition_override"
        ] = [0, 1, 2]
        self.assert_audit_fails(
            lambda item: validate_candidate_manifest_structure(
                item,
                wrong_topology,
                compilation_protocol=PROTOCOL,
                expected_metrics=metrics(),
            )
        )

        wrong_count = copy.deepcopy(valid)
        generic = wrong_count["cases"]["LiH-40"]["generic_qasm"]
        generic["cx"] += 1
        wrong_count["cases"]["LiH-40"][
            "legacy_generic_unitary"
        ] = copy.deepcopy(generic)
        wrong_count["cases"]["LiH-40"]["cx_reduction_fraction"] = (
            1.0
            - RESOURCE_TARGETS["LiH-40"]["structured_cx"] / generic["cx"]
        )
        self.assert_audit_fails(
            lambda item: validate_candidate_manifest_structure(
                item,
                wrong_count,
                compilation_protocol=PROTOCOL,
                expected_metrics=metrics(),
            )
        )

    def test_npz_dtype_finite_shape_pickle_and_members_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            valid_path = root / "valid.npz"
            np.savez_compressed(
                valid_path,
                singles=np.ones((2, 3), dtype=np.float64),
                doubles=np.zeros((2, 3), dtype=np.float64),
            )
            record = {
                "sha256": sha256_file(valid_path),
                "array_names": ["singles", "doubles"],
                "array_shapes": {
                    "singles": [2, 3],
                    "doubles": [2, 3],
                },
            }
            audit = Audit()
            arrays = inspect_parameter_npz(
                audit, path=valid_path, record=record, label="valid"
            )
            self.assertEqual(audit.status, "PASS")
            self.assertEqual(set(arrays), {"singles", "doubles"})

            probes = {
                "float32": {
                    "singles": np.ones((2, 3), dtype=np.float32),
                    "doubles": np.zeros((2, 3), dtype=np.float64),
                },
                "nonfinite": {
                    "singles": np.full((2, 3), np.nan, dtype=np.float64),
                    "doubles": np.zeros((2, 3), dtype=np.float64),
                },
                "shape": {
                    "singles": np.ones((3, 2), dtype=np.float64),
                    "doubles": np.zeros((2, 3), dtype=np.float64),
                },
                "pickle": {
                    "singles": np.array([{"unsafe": True}], dtype=object),
                    "doubles": np.zeros((2, 3), dtype=np.float64),
                },
            }
            for name, values in probes.items():
                with self.subTest(name=name):
                    path = root / f"{name}.npz"
                    np.savez_compressed(path, **values)
                    tampered_record = {
                        **record,
                        "sha256": sha256_file(path),
                    }
                    self.assert_audit_fails(
                        lambda item, p=path, r=tampered_record, n=name: (
                            inspect_parameter_npz(
                                item, path=p, record=r, label=n
                            )
                        )
                    )

            extra_path = root / "extra.npz"
            np.savez_compressed(
                extra_path,
                singles=np.ones((2, 3), dtype=np.float64),
                doubles=np.zeros((2, 3), dtype=np.float64),
                extra=np.ones((1,), dtype=np.float64),
            )
            extra_record = {**record, "sha256": sha256_file(extra_path)}
            self.assert_audit_fails(
                lambda item: inspect_parameter_npz(
                    item,
                    path=extra_path,
                    record=extra_record,
                    label="extra",
                )
            )

    def test_parameter_contract_rejects_declared_ignored_array(self) -> None:
        consumed = {
            "singles": np.zeros((2, 3), dtype=np.float64),
            "doubles": np.zeros((2, 3), dtype=np.float64),
        }
        record = {
            "array_names": ["doubles", "singles"],
            "array_shapes": {
                "doubles": [2, 3],
                "singles": [2, 3],
            },
        }
        audit = Audit()
        validate_consumed_parameter_contract(
            audit,
            label="valid",
            parameter_record=record,
            artifact_arrays=consumed,
            loaded_params=consumed,
            initialized_params=consumed,
        )
        self.assertEqual(audit.status, "PASS")

        ignored = {
            **consumed,
            "ignored": np.ones((1,), dtype=np.float64),
        }
        ignored_record = {
            "array_names": ["doubles", "ignored", "singles"],
            "array_shapes": {
                "doubles": [2, 3],
                "ignored": [1],
                "singles": [2, 3],
            },
        }
        self.assert_audit_fails(
            lambda item: validate_consumed_parameter_contract(
                item,
                label="ignored",
                parameter_record=ignored_record,
                artifact_arrays=ignored,
                loaded_params=ignored,
                initialized_params=consumed,
            )
        )

    def test_dense_record_binds_all_emitted_fields_and_outer_energy(self) -> None:
        outer_energy = -1.0
        generic_energy = outer_energy + 2e-12
        structured_energy = outer_energy - 3e-12
        record = {
            "scope": (
                "dense validation of the exact reloaded QASM artifacts"
            ),
            "generic_vs_structured_state_fidelity": 1.0,
            "generic_vs_sector_state_fidelity": 1.0,
            "structured_vs_sector_state_fidelity": 1.0,
            "generic_sector_leakage_probability": 0.0,
            "structured_sector_leakage_probability": 0.0,
            "generic_energy_hartree": generic_energy,
            "structured_energy_hartree": structured_energy,
            "sector_replay_energy_hartree": outer_energy,
            "generic_energy_difference_hartree": abs(
                generic_energy - outer_energy
            ),
            "structured_energy_difference_hartree": abs(
                structured_energy - outer_energy
            ),
        }
        audit = Audit()
        validate_dense_record(
            audit,
            label="valid",
            n_qubits=6,
            recorded=copy.deepcopy(record),
            replayed=copy.deepcopy(record),
            outer_replay_energy=outer_energy,
        )
        self.assertEqual(audit.status, "PASS")

        probes = {}
        wrong_scope = copy.deepcopy(record)
        wrong_scope["scope"] = "self-declared scope"
        probes["scope"] = (wrong_scope, record, outer_energy)

        extra = copy.deepcopy(record)
        extra["ignored"] = True
        probes["exact field set"] = (extra, record, outer_energy)

        for key in (
            "generic_vs_structured_state_fidelity",
            "generic_vs_sector_state_fidelity",
            "structured_vs_sector_state_fidelity",
            "generic_sector_leakage_probability",
            "structured_sector_leakage_probability",
            "generic_energy_hartree",
            "structured_energy_hartree",
            "generic_energy_difference_hartree",
            "structured_energy_difference_hartree",
        ):
            tampered = copy.deepcopy(record)
            tampered[key] = float(tampered[key]) + 1e-6
            probes[key] = (tampered, record, outer_energy)

        shifted_recorded = copy.deepcopy(record)
        shifted_replayed = copy.deepcopy(record)
        shifted_outer = outer_energy + 0.25
        for item in (shifted_recorded, shifted_replayed):
            item["sector_replay_energy_hartree"] = shifted_outer
            item["generic_energy_difference_hartree"] = abs(
                item["generic_energy_hartree"] - shifted_outer
            )
            item["structured_energy_difference_hartree"] = abs(
                item["structured_energy_hartree"] - shifted_outer
            )
        probes["sector versus outer energy"] = (
            shifted_recorded,
            shifted_replayed,
            outer_energy,
        )

        for name, (
            recorded,
            replayed,
            expected_outer,
        ) in probes.items():
            with self.subTest(name=name):
                self.assert_audit_fails(
                    lambda item, left=recorded, right=replayed, outer=expected_outer: (
                        validate_dense_record(
                            item,
                            label=name,
                            n_qubits=6,
                            recorded=left,
                            replayed=right,
                            outer_replay_energy=outer,
                        )
                    )
                )

    def test_qasm_hash_resources_and_basis_are_rederived(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "circuit.qasm"
            path.write_text("OPENQASM 2.0;\n", encoding="utf-8")
            record = {
                **PROTOCOL,
                "qasm_sha256": sha256_file(path),
                "logical_qubits": 6,
                "depth": 10,
                "size": 25,
                "cx": 5,
                "operations": {"cx": 5, "rz": 20},
            }
            audit = Audit()
            inspect_qasm_arm(
                audit,
                path=path,
                record=record,
                label="case",
                arm_name="generic",
                qasm_loader=lambda _: FakeCircuit(),
            )
            self.assertEqual(audit.status, "PASS")

            bad_hash = {**record, "qasm_sha256": "0" * 64}
            self.assert_audit_fails(
                lambda item: inspect_qasm_arm(
                    item,
                    path=path,
                    record=bad_hash,
                    label="case",
                    arm_name="generic",
                    qasm_loader=lambda _: FakeCircuit(),
                )
            )
            self.assert_audit_fails(
                lambda item: inspect_qasm_arm(
                    item,
                    path=path,
                    record=record,
                    label="case",
                    arm_name="generic",
                    qasm_loader=lambda _: FakeCircuit(
                        operations={"cx": 5, "u": 20}
                    ),
                )
            )

    def test_bootstrap_identity_stage_and_artifact_tampering_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_code = (root / "Source_Code").resolve()
            source_code.mkdir()
            environment = (root / "locked_env").resolve()
            (environment / "bin").mkdir(parents=True)
            (environment / "bin" / "python").write_text(
                "#!/bin/sh\n", encoding="utf-8"
            )
            helper = (root / "bootstrap_table3_candidate.py").resolve()
            helper.write_text("# trusted helper\n", encoding="utf-8")
            invocation = (root / "run-12345678").resolve()
            invocation.mkdir()
            status_path = invocation / "bootstrap_status.json"
            candidate = (
                invocation
                / "advanced_method"
                / "canonical_table3_candidate"
            )
            candidate.mkdir(parents=True)
            provider_free_home = invocation / ".provider_free_home"
            for name in (
                "",
                "xdg-config",
                "xdg-cache",
                "xdg-data",
                "xdg-state",
            ):
                (provider_free_home / name).mkdir(
                    parents=True,
                    exist_ok=True,
                )
            identity = {
                "algorithm": "sha256-path-hash-size-notebook-source-v2",
                "sha256": "1" * 64,
                "file_count": 77,
            }
            artifact = {
                "algorithm": "sha256-path-hash-size-v1",
                "aggregate_sha256": "2" * 64,
                "file_count": 10,
                "files": [
                    {
                        "path": "a",
                        "sha256": "3" * 64,
                        "size_bytes": 1,
                    }
                ],
            }
            stage_contracts = expected_bootstrap_stage_contracts(
                source_code=source_code,
                invocation_directory=invocation,
                environment=environment,
            )
            stages = []
            for contract in stage_contracts:
                log_path = Path(contract["log"])
                log_path.write_text(
                    f"{contract['name']}: PASS\n", encoding="utf-8"
                )
                stages.append(
                    {
                        **contract,
                        "status": "PASS",
                        "returncode": 0,
                        "log_sha256": sha256_file(log_path),
                        "log_size_bytes": log_path.stat().st_size,
                    }
                )
            status = {
                "schema_version": 1,
                "invocation_id": invocation.name,
                "invocation_directory": str(invocation),
                "output_directory": str(invocation),
                "bootstrap_status": str(status_path),
                "bootstrap_helper": {
                    "path": str(helper),
                    "sha256": sha256_file(helper),
                },
                "status": "CANDIDATE_READY_NOT_CERTIFIED",
                "source_code": str(source_code),
                "environment": str(environment),
                "environment_python": str(
                    environment / "bin" / "python"
                ),
                "candidate_directory": str(candidate),
                "provider_imported": False,
                "qpu_contacted": False,
                "provider_access_boundary": {
                    "credential_home": str(provider_free_home),
                    "xdg_config_home": str(
                        provider_free_home / "xdg-config"
                    ),
                    "xdg_cache_home": str(
                        provider_free_home / "xdg-cache"
                    ),
                    "xdg_data_home": str(
                        provider_free_home / "xdg-data"
                    ),
                    "xdg_state_home": str(
                        provider_free_home / "xdg-state"
                    ),
                    "python_network_audit_guard": True,
                    "aws_metadata_disabled": True,
                    "aws_configuration_files": os.devnull,
                    "pip_no_index": True,
                },
                "stages": stages,
                "source_identity_before": identity,
                "source_identity_after": identity,
                "candidate_artifacts": artifact,
            }
            status_path.write_text(json.dumps(status), encoding="utf-8")

            def validate(item: Audit, value: dict) -> None:
                validate_bootstrap_status(
                    item,
                    value,
                    bootstrap_helper_path=helper,
                    bootstrap_status_path=status_path,
                    source_code=source_code,
                    environment=environment,
                    candidate_directory=candidate,
                    current_source_identity=identity,
                    observed_candidate_manifest=artifact,
                )

            audit = Audit()
            validate(audit, status)
            self.assertEqual(audit.status, "PASS")

            probes = {
                "stale source": (
                    ("source_identity_after", "sha256"),
                    "9" * 64,
                ),
                "artifact size": (
                    (
                        "candidate_artifacts",
                        "files",
                        0,
                        "size_bytes",
                    ),
                    2,
                ),
                "invocation id": (
                    ("invocation_id",),
                    "different-invocation",
                ),
                "stage argv": (
                    ("stages", 2, "argv"),
                    ["python", "untrusted.py"],
                ),
                "recorded log hash": (
                    ("stages", 1, "log_sha256"),
                    "0" * 64,
                ),
            }

            def set_nested(value, path, replacement) -> None:
                target = value
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement

            for name, (path, replacement) in probes.items():
                with self.subTest(name=name):
                    tampered = copy.deepcopy(status)
                    set_nested(tampered, path, replacement)
                    self.assert_audit_fails(
                        lambda item, value=tampered: validate(item, value)
                    )

            helper.write_text("# mutated helper\n", encoding="utf-8")
            self.assert_audit_fails(
                lambda item: validate(item, status)
            )
            helper.write_text("# trusted helper\n", encoding="utf-8")

            log_path = Path(stage_contracts[0]["log"])
            log_path.write_text("mutated log\n", encoding="utf-8")
            self.assert_audit_fails(
                lambda item: validate(item, status)
            )

    def test_phi_uses_both_angle_families_and_is_even(self) -> None:
        singles_only = recompute_phi([0.05, -0.05], [])
        with_doubles = recompute_phi([0.05, -0.05], [0.2])
        with_negative_double = recompute_phi(
            [0.05, -0.05], [-0.2]
        )
        self.assertGreater(with_doubles, singles_only)
        self.assertAlmostEqual(
            with_doubles, with_negative_double, places=15
        )

    def test_frozen_reference_proof_rederives_nested_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_code, invocation, certificate = (
                self.make_frozen_reference_fixture(Path(directory))
            )
            audit = Audit()
            validate_frozen_reference_certificate(
                audit,
                source_code=source_code,
                invocation_directory=invocation,
            )
            self.assertEqual(audit.status, "PASS")

            def set_nested(
                value: dict, path: tuple[str, ...], replacement
            ) -> None:
                target = value
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = replacement

            probes = {
                "unexpected check key": (
                    ("checks", "unexpected"),
                    True,
                ),
                "basis checksum": (
                    ("frozen_active_basis", "coefficient_sha256"),
                    "b" * 64,
                ),
                "basis finite flag": (
                    ("frozen_active_basis", "finite"),
                    False,
                ),
                "S residual": (
                    (
                        "frozen_active_basis",
                        "s_orthonormality",
                        "max_abs_error",
                    ),
                    3e-8,
                ),
                "core count": (
                    (
                        "frozen_active_basis",
                        "core_active_orthogonality",
                        "core_orbital_count",
                    ),
                    2,
                ),
                "Fock residual finite": (
                    (
                        "frozen_active_basis",
                        "generalized_fock_eigen_residual",
                        "max_abs_residual",
                    ),
                    float("inf"),
                ),
                "integral tolerance": (
                    ("integral_differences", "h1e_max_abs"),
                    3e-9,
                ),
                "CASCI core count": (
                    ("active_space_regeneration", "pyscf_casci_ncore"),
                    2,
                ),
                "reported RHF difference": (
                    ("rhf", "absolute_difference_hartree"),
                    0.0,
                ),
                "CASCI versus p-space total": (
                    ("casci", "regenerated_energy_hartree"),
                    -0.75,
                ),
                "p-space electronic/core split": (
                    ("pspace", "electronic_energy_hartree"),
                    -0.7,
                ),
                "negative p-space residual": (
                    ("pspace", "eigen_residual_norm_hartree"),
                    -1e-14,
                ),
            }
            artifact_path = invocation / "frozen_reference_audit.json"
            for name, (path, replacement) in probes.items():
                with self.subTest(name=name):
                    tampered = copy.deepcopy(certificate)
                    set_nested(
                        tampered["problems"]["case_0"],
                        path,
                        replacement,
                    )
                    artifact_path.write_text(
                        json.dumps(tampered),
                        encoding="utf-8",
                    )
                    self.assert_audit_fails(
                        lambda item: validate_frozen_reference_certificate(
                            item,
                            source_code=source_code,
                            invocation_directory=invocation,
                        )
                    )

            top_level_probes = {
                "certificate schema bool": (
                    ("schema_version",),
                    True,
                ),
                "widened trusted tolerance": (
                    ("tolerances", "integral_max_abs"),
                    1.0,
                ),
            }
            for name, (path, replacement) in top_level_probes.items():
                with self.subTest(name=name):
                    tampered = copy.deepcopy(certificate)
                    set_nested(tampered, path, replacement)
                    artifact_path.write_text(
                        json.dumps(tampered),
                        encoding="utf-8",
                    )
                    self.assert_audit_fails(
                        lambda item: validate_frozen_reference_certificate(
                            item,
                            source_code=source_code,
                            invocation_directory=invocation,
                        )
                    )

            reordered = copy.deepcopy(certificate)
            occupation = reordered["problems"]["case_0"][
                "occupation_order"
            ]
            for key in ("expected", "frozen", "regenerated"):
                occupation[key] = [0.0, 2.0]
            occupation["max_abs_frozen_vs_regenerated"] = 0.0
            occupation["max_abs_frozen_vs_expected_order"] = 0.0
            occupation["max_abs_regenerated_vs_expected_order"] = 0.0
            artifact_path.write_text(
                json.dumps(reordered),
                encoding="utf-8",
            )
            self.assert_audit_fails(
                lambda item: validate_frozen_reference_certificate(
                    item,
                    source_code=source_code,
                    invocation_directory=invocation,
                )
            )

            nonfinite = copy.deepcopy(certificate)
            nonfinite["problems"]["case_0"]["occupation_order"][
                "regenerated"
            ][0] = float("nan")
            artifact_path.write_text(
                json.dumps(nonfinite),
                encoding="utf-8",
            )
            self.assert_audit_fails(
                lambda item: validate_frozen_reference_certificate(
                    item,
                    source_code=source_code,
                    invocation_directory=invocation,
                )
            )

            manifest_path = source_code / "frozen_inputs" / "MANIFEST.json"
            original_manifest = manifest_path.read_bytes()
            tampered_manifest = json.loads(
                original_manifest.decode("utf-8")
            )
            tampered_manifest["problems"]["case_0"][
                "rhf_energy_hartree"
            ] += 1e-4
            manifest_path.write_text(
                json.dumps(tampered_manifest),
                encoding="utf-8",
            )
            manifest_bound = copy.deepcopy(certificate)
            manifest_bound["manifest_sha256"] = sha256_file(manifest_path)
            artifact_path.write_text(
                json.dumps(manifest_bound),
                encoding="utf-8",
            )
            self.assert_audit_fails(
                lambda item: validate_frozen_reference_certificate(
                    item,
                    source_code=source_code,
                    invocation_directory=invocation,
                )
            )
            manifest_path.write_bytes(original_manifest)


if __name__ == "__main__":
    unittest.main()
