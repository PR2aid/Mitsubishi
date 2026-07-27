#!/usr/bin/env python3
"""Fail-closed validation of the Phase 3 judge reproduction outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
from scipy.stats import beta


ROOT = Path(__file__).resolve().parent
TABLE3_LABELS = ("BeH2-6", "BeH2-12", "LiH-40")
REFERENCE_AUDIT_REQUIRED_CHECKS = frozenset(
    {
        "bundle_checksum",
        "frozen_active_orbital_checksum",
        "bundle_electron_sector",
        "rhf_converged",
        "rhf_energy",
        "frozen_occupation_order",
        "frozen_active_s_orthonormality",
        "frozen_core_active_orthogonality",
        "generalized_fock_eigen_residual",
        "canonical_orbital_order",
        "mo_energies",
        "h1e",
        "eri",
        "ecore",
        "casci_declared_core_count",
        "declared_sector_dimension",
        "full_determinant_pspace",
        "pspace_sector_dimension",
        "casci_energy",
    }
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def json_default(value: Any) -> Any:
    """Serialize only the NumPy values emitted by scientific diagnostics."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_load_qasm2_legacy_sx(path: str | Path) -> Any:
    """Strictly parse Qiskit legacy QASM2 with only undeclared ``sx`` allowed."""

    from qiskit import qasm2

    sx_only = tuple(
        item
        for item in qasm2.LEGACY_CUSTOM_INSTRUCTIONS
        if item.name == "sx"
    )
    if len(sx_only) != 1:
        raise RuntimeError(
            "Qiskit legacy OpenQASM 2 profile must contain exactly one "
            f"'sx' instruction; found {len(sx_only)}"
        )
    sx_instruction = sx_only[0]
    if (
        sx_instruction.name != "sx"
        or sx_instruction.num_params != 0
        or sx_instruction.num_qubits != 1
        or sx_instruction.builtin is not True
    ):
        raise RuntimeError(
            "Qiskit legacy 'sx' instruction has an unexpected signature"
        )
    return qasm2.load(
        str(path),
        include_path=(),
        include_input_directory=None,
        custom_instructions=sx_only,
        custom_classical=(),
        strict=True,
    )


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(self, name: str, passed: bool, *, actual: Any = None, expected: Any = None) -> None:
        item = {"name": name, "status": "PASS" if passed else "FAIL"}
        if actual is not None:
            item["actual"] = actual
        if expected is not None:
            item["expected"] = expected
        self.checks.append(item)
        print(f"[{item['status']}] {name}")

    def close(self) -> str:
        return "PASS" if all(item["status"] == "PASS" for item in self.checks) else "FAIL"


def close(actual: float, expected: float, tolerance: float) -> bool:
    return math.isfinite(actual) and abs(actual - expected) <= tolerance


def _u_single(angle: float) -> float:
    value = abs(float(angle))
    return 2.0 * math.log(
        abs(math.cos(value / 2.0)) + abs(math.sin(value / 2.0))
    )


def _u_pair_double(angle: float) -> float:
    value = abs(float(angle))
    return 8.0 * math.log(
        abs(math.cos(value / 8.0)) + abs(math.sin(value / 8.0))
    )


def recompute_phi(
    single_cross_angles: list[Any], double_cross_angles: list[Any]
) -> float:
    """Independently recompute Phi from emitted effective cross-cut angles."""

    singles = np.asarray(single_cross_angles, dtype=np.float64).reshape(-1)
    doubles = np.asarray(double_cross_angles, dtype=np.float64).reshape(-1)
    total_u = sum(_u_single(value) for value in singles)
    total_u += sum(_u_pair_double(value) for value in doubles)
    return 2.0 * math.exp(2.0 * total_u) - 1.0


def recompute_token_cost(token: dict[str, Any]) -> float:
    """Recompute one emitted residual token's exact additive log-overhead."""

    kind = token.get("kind")
    angle = float(token.get("angle", 0.0))
    if kind == "identity":
        return 0.0
    if kind == "single":
        return _u_single(angle)
    if kind == "double":
        return _u_pair_double(angle)
    raise ValueError(f"unknown residual token kind: {kind!r}")


def audit_token_sequence(
    audit: Audit,
    name: str,
    tokens: list[dict[str, Any]],
    *,
    expected_count: int | None = None,
) -> float:
    """Check every recorded token cost and return the independently summed cost."""

    audit.check(
        f"{name} residual-token sequence is nonempty",
        isinstance(tokens, list) and len(tokens) > 0,
        actual=len(tokens) if isinstance(tokens, list) else type(tokens).__name__,
        expected="at least one token",
    )
    if not isinstance(tokens, list):
        return 0.0
    if expected_count is not None:
        declared = int(expected_count)
        audit.check(
            f"{name} residual-token count",
            declared > 0 and len(tokens) == declared,
            actual=len(tokens),
            expected=declared,
        )
    discrepancies = []
    recomputed = []
    for index, token in enumerate(tokens):
        value = recompute_token_cost(token)
        recomputed.append(value)
        recorded = float(token.get("u_cost", float("nan")))
        if not close(recorded, value, 1e-12):
            discrepancies.append(
                {
                    "index": index,
                    "kind": token.get("kind"),
                    "angle": token.get("angle"),
                    "recorded": recorded,
                    "recomputed": value,
                }
            )
    audit.check(
        f"{name} residual-token costs recompute",
        not discrepancies,
        actual=discrepancies or f"{len(tokens)} tokens",
        expected="every u_cost equals the formula applied to kind and angle",
    )
    return float(sum(recomputed))


def validate_budget_semantics(
    audit: Audit,
    name: str,
    record: dict[str, Any],
) -> None:
    """Recompute selected residual costs and the final Phi from raw tokens."""

    sequences: list[tuple[str, list[dict[str, Any]]]] = [
        ("sampled", record["sampled_result"]["tokens"]),
        ("guarded", record["guarded_result"]["tokens"]),
    ]
    controls = record["heldout_evaluation"].get("matched_policy_controls", {})
    for control_name in ("identity", "random_feasible", "stratified_pool_greedy"):
        if control_name in controls:
            sequences.append(
                (f"control-{control_name}", controls[control_name]["tokens"])
            )
    declared_count = record.get("gqe_config", {}).get("ngates")
    costs = {
        sequence_name: audit_token_sequence(
            audit,
            f"{name} {sequence_name}",
            tokens,
            expected_count=declared_count,
        )
        for sequence_name, tokens in sequences
    }
    budget = record["cutting_budget"]
    base_phi = float(record["backbone"]["phi"])
    base_u = 0.5 * math.log((base_phi + 1.0) / 2.0)
    guarded_u = costs["guarded"]
    total_u = base_u + guarded_u
    total_phi = 2.0 * math.exp(2.0 * total_u) - 1.0
    audit.check(
        f"{name} selected residual_u recomputes",
        close(guarded_u, float(budget["residual_u"]), 1e-12),
        actual=guarded_u,
        expected=budget["residual_u"],
    )
    audit.check(
        f"{name} selected total Phi recomputes",
        close(total_phi, float(budget["phi"]), 1e-12),
        actual=total_phi,
        expected=budget["phi"],
    )


def _resource_artifact(
    advanced_root: Path, record: dict[str, Any], arm: str
) -> Path:
    advanced_allowed = advanced_root.resolve()
    artifact_root = (
        advanced_allowed / str(record["artifact_root"])
    ).resolve()
    if advanced_allowed not in artifact_root.parents:
        raise ValueError(
            "Table-3 artifact root must be strictly below the advanced run "
            f"directory: {artifact_root}"
        )
    path = (artifact_root / str(record[arm]["qasm_file"])).resolve()
    if artifact_root not in path.parents:
        raise ValueError(f"Table-3 QASM path escapes its artifact root: {path}")
    return path


def validate_table3_structure(
    audit: Audit, advanced: dict[str, Any]
) -> list[str]:
    """Require the exact Table-3 case set and exact compatibility aliases."""

    resources = advanced.get("structured_resources")
    actual_labels = set(resources) if isinstance(resources, dict) else set()
    expected_labels = set(TABLE3_LABELS)
    audit.check(
        "Table-3 structured-resource case set",
        isinstance(resources, dict) and actual_labels == expected_labels,
        actual=sorted(actual_labels),
        expected=list(TABLE3_LABELS),
    )
    if not isinstance(resources, dict):
        return []
    present = [label for label in TABLE3_LABELS if label in resources]
    for label in present:
        record = resources[label]
        generic = record.get("generic_qasm")
        structured = record.get("structured_qasm")
        audit.check(
            f"{label} legacy generic alias equals artifact-bound record",
            isinstance(generic, dict)
            and record.get("legacy_generic_unitary") == generic,
            actual=record.get("legacy_generic_unitary"),
            expected=generic,
        )
        audit.check(
            f"{label} structured alias equals artifact-bound record",
            isinstance(structured, dict)
            and record.get("structured_exact_pauli_network") == structured,
            actual=record.get("structured_exact_pauli_network"),
            expected=structured,
        )
    return present


def validate_expected_table3_resources(
    audit: Audit,
    advanced: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    """Compare declared Table-3 values only with artifact-bound QASM records."""

    resources = advanced.get("structured_resources", {})
    targets = expected.get("structured_resources", {})
    provenance = expected.get("table3_resource_contract_provenance")
    qasm_hash_targets = (
        provenance.get("exact_qasm_sha256")
        if isinstance(provenance, dict)
        else None
    )

    def is_sha256(value: Any) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
        )

    exact_hash_contract = (
        isinstance(qasm_hash_targets, dict)
        and set(qasm_hash_targets) == set(TABLE3_LABELS)
        and all(
            isinstance(qasm_hash_targets.get(label), dict)
            and set(qasm_hash_targets[label])
            == {"generic_qasm", "structured_qasm"}
            and all(
                is_sha256(qasm_hash_targets[label][arm])
                for arm in ("generic_qasm", "structured_qasm")
            )
            for label in TABLE3_LABELS
        )
    )
    audit.check(
        "expected Table-3 exact-QASM hash contract is complete",
        exact_hash_contract,
        actual=qasm_hash_targets,
        expected={
            label: ["generic_qasm", "structured_qasm"]
            for label in TABLE3_LABELS
        },
    )
    for label in TABLE3_LABELS:
        if label not in resources or label not in targets:
            audit.check(
                f"{label} expected Table-3 record is present",
                False,
                actual={
                    "advanced": label in resources,
                    "expected": label in targets,
                },
                expected={"advanced": True, "expected": True},
            )
            continue
        target = targets[label]
        record = resources[label]
        generic = record.get("generic_qasm") or {}
        structured = record.get("structured_qasm") or {}
        expected_hashes = (
            qasm_hash_targets.get(label, {})
            if isinstance(qasm_hash_targets, dict)
            else {}
        )
        for arm, declaration in (
            ("generic_qasm", generic),
            ("structured_qasm", structured),
        ):
            actual_hash = declaration.get("qasm_sha256")
            expected_hash = expected_hashes.get(arm)
            audit.check(
                f"{label} {arm} exact expected QASM hash",
                is_sha256(expected_hash) and actual_hash == expected_hash,
                actual=actual_hash,
                expected=expected_hash,
            )
        for field, actual in (
            ("legacy_cx", generic.get("cx")),
            ("structured_cx", structured.get("cx")),
            ("legacy_depth", generic.get("depth")),
            ("structured_depth", structured.get("depth")),
        ):
            audit.check(
                f"{label} {field}",
                actual is not None and int(actual) == int(target[field]),
                actual=actual,
                expected=target[field],
            )
        if "minimum_fidelity" in target:
            fidelity = record.get("state_fidelity")
            audit.check(
                f"{label} structured state fidelity",
                fidelity is not None
                and float(fidelity) >= float(target["minimum_fidelity"]),
                actual=fidelity,
                expected=target["minimum_fidelity"],
            )
        else:
            audit.check(
                f"{label} structured exactness scope",
                record.get("equivalence_scope") == target["fidelity_scope"],
                actual=record.get("equivalence_scope"),
                expected=target["fidelity_scope"],
            )


def validate_table3_semantics(
    audit: Audit,
    advanced_root: Path,
    advanced: dict[str, Any],
    mode: str,
) -> None:
    """Validate QASM hashes and Phi semantics independently of claimed metrics."""

    labels = validate_table3_structure(audit, advanced)
    for label in labels:
        record = advanced["structured_resources"][label]
        expected_mode = (
            "PROMOTED_CANONICAL_REPLAY"
            if mode == "quick"
            else "FULL_RUN_CANDIDATE"
        )
        audit.check(
            f"{label} Table-3 reference mode",
            record.get("reference_mode") == expected_mode,
            actual=record.get("reference_mode"),
            expected=expected_mode,
        )
        for arm in ("generic_qasm", "structured_qasm"):
            artifact = _resource_artifact(advanced_root, record, arm)
            exists = artifact.is_file()
            actual_hash = sha256_file(artifact) if exists else None
            reloaded = (
                strict_load_qasm2_legacy_sx(artifact) if exists else None
            )
            operations = (
                {}
                if reloaded is None
                else {
                    str(key): int(value)
                    for key, value in reloaded.count_ops().items()
                }
            )
            resource_match = (
                reloaded is not None
                and int(reloaded.depth()) == int(record[arm]["depth"])
                and int(reloaded.size()) == int(record[arm]["size"])
                and int(operations.get("cx", 0)) == int(record[arm]["cx"])
                and operations == record[arm]["operations"]
                and not (
                    set(operations)
                    - set(record[arm]["basis_gates"])
                )
            )
            audit.check(
                f"{label} {arm} exact artifact hash",
                exists
                and actual_hash == record[arm]["qasm_sha256"]
                and record[arm].get("derived_from_reloaded_qasm") is True
                and record[arm].get("compiled_once_before_serialization") is True,
                actual=actual_hash,
                expected=record[arm]["qasm_sha256"],
            )
            audit.check(
                f"{label} {arm} resources rederive from QASM",
                resource_match,
                actual={
                    "depth": None if reloaded is None else reloaded.depth(),
                    "size": None if reloaded is None else reloaded.size(),
                    "operations": operations,
                },
                expected={
                    "depth": record[arm]["depth"],
                    "size": record[arm]["size"],
                    "operations": record[arm]["operations"],
                },
            )
        cutting = record["cutting_accounting"]
        phi = recompute_phi(
            cutting["single_cross_angles"],
            cutting["double_cross_angles"],
        )
        audit.check(
            f"{label} Phi recomputes from emitted cross angles",
            close(phi, float(cutting["reported_phi"]), 1e-12)
            and close(phi, float(cutting["recomputed_phi"]), 1e-12)
            and close(float(cutting["absolute_difference"]), 0.0, 1e-12),
            actual=phi,
            expected=cutting["reported_phi"],
        )
        dense = record.get("dense_qasm_audit")
        if int(record["n_qubits"]) <= 12:
            passed = (
                isinstance(dense, dict)
                and float(dense["generic_vs_structured_state_fidelity"])
                >= 1.0 - 1e-10
                and float(dense["generic_vs_sector_state_fidelity"])
                >= 1.0 - 1e-10
                and float(dense["structured_vs_sector_state_fidelity"])
                >= 1.0 - 1e-10
                and float(dense["generic_sector_leakage_probability"]) <= 1e-10
                and float(dense["structured_sector_leakage_probability"]) <= 1e-10
                and float(dense["generic_energy_difference_hartree"]) <= 1e-9
                and float(dense["structured_energy_difference_hartree"]) <= 1e-9
            )
            audit.check(
                f"{label} exact-QASM state/leakage/energy binding",
                passed,
                actual=dense,
                expected="fidelity >= 1-1e-10, leakage <= 1e-10, |dE| <= 1e-9 Ha",
            )
        else:
            audit.check(
                f"{label} dense-QASM allocation boundary",
                dense is None
                and record["equivalence_scope"]
                == "compositional exact primitive tests; no dense 40q state allocated",
                actual=record.get("equivalence_scope"),
                expected="explicit 40-qubit dense-allocation exclusion",
            )


def validate_table3_claims(
    audit: Audit,
    advanced_root: Path,
    advanced: dict[str, Any],
    expected: dict[str, Any],
    mode: str,
) -> None:
    """Bind semantic and resource claims in both quick and full modes."""

    validate_table3_semantics(audit, advanced_root, advanced, mode)
    validate_expected_table3_resources(audit, advanced, expected)
    if mode == "full":
        validate_full_table3_candidate_binding(audit, advanced_root, advanced)


def _table3_artifact_aggregate(
    root: Path,
    relatives: list[str],
) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    for relative in sorted(relatives):
        path = root / relative
        file_hash = sha256_file(path)
        size = path.stat().st_size
        records.append(
            {"path": relative, "sha256": file_hash, "size_bytes": size}
        )
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), records


def validate_full_table3_candidate_binding(
    audit: Audit,
    advanced_root: Path,
    advanced: dict[str, Any],
) -> None:
    """Bind a full run's fresh candidate without a historical reference."""

    from certify_release import (
        TABLE3_CANDIDATE_STATUS,
        TABLE3_CASES,
        TABLE3_PROTOCOL,
        TABLE3_SEED,
        _validate_parameter_npz,
        _validate_qasm_declaration,
    )

    candidate_root = (
        advanced_root.resolve() / "canonical_table3_candidate"
    )
    candidate_manifest_path = candidate_root / "canonical_table3.json"
    expected_artifacts = [
        *(
            f"parameters/{label}_seed-{TABLE3_SEED}.npz"
            for label in TABLE3_LABELS
        ),
        *(
            f"circuits/{label}_{arm}.qasm"
            for label in TABLE3_LABELS
            for arm in ("generic", "structured")
        ),
    ]
    expected_files = {"canonical_table3.json", *expected_artifacts}
    expected_directories = {"circuits", "parameters"}
    candidate_files: set[str] = set()
    candidate_directories: set[str] = set()
    unsafe_nodes: list[str] = []
    candidate_root_safe = (
        candidate_root.is_dir()
        and not candidate_root.is_symlink()
        and candidate_root.resolve() == candidate_root
    )
    if candidate_root_safe:
        for path in candidate_root.rglob("*"):
            relative = path.relative_to(candidate_root).as_posix()
            if path.is_symlink():
                unsafe_nodes.append(f"symlink:{relative}")
            elif path.is_dir():
                candidate_directories.add(relative)
            elif path.is_file():
                candidate_files.add(relative)
            else:
                unsafe_nodes.append(f"special:{relative}")

    candidate_tree_valid = (
        candidate_root_safe
        and candidate_files == expected_files
        and candidate_directories == expected_directories
        and not unsafe_nodes
    )
    audit.check(
        "full Table-3 candidate exact ten-file boundary",
        candidate_tree_valid,
        actual={
            "root_safe": candidate_root_safe,
            "directories": sorted(candidate_directories),
            "files": sorted(candidate_files),
            "unsafe_nodes": unsafe_nodes,
        },
        expected={
            "root_safe": True,
            "directories": sorted(expected_directories),
            "files": sorted(expected_files),
            "unsafe_nodes": [],
        },
    )

    manifest_error: str | None = None
    manifest_sha256: str | None = None
    candidate_manifest: dict[str, Any] = {}
    if (
        candidate_manifest_path.is_file()
        and not candidate_manifest_path.is_symlink()
    ):
        manifest_sha256 = sha256_file(candidate_manifest_path)
        try:
            candidate_manifest = load(candidate_manifest_path)
        except Exception as error:
            manifest_error = f"{type(error).__name__}: {error}"
    else:
        manifest_error = "manifest is missing or a symlink"
    cases = candidate_manifest.get("cases")
    audit.check(
        "full Table-3 candidate manifest is a fresh exact-contract payload",
        manifest_error is None
        and type(candidate_manifest.get("schema_version")) is int
        and candidate_manifest.get("schema_version") == 1
        and candidate_manifest.get("status") == TABLE3_CANDIDATE_STATUS
        and type(candidate_manifest.get("seed")) is int
        and candidate_manifest.get("seed") == TABLE3_SEED
        and candidate_manifest.get("compilation_protocol")
        == TABLE3_PROTOCOL
        and "promotion_provenance" not in candidate_manifest
        and isinstance(cases, dict)
        and set(cases) == set(TABLE3_LABELS),
        actual={
            "error": manifest_error,
            "schema_version": candidate_manifest.get("schema_version"),
            "status": candidate_manifest.get("status"),
            "seed": candidate_manifest.get("seed"),
            "compilation_protocol": candidate_manifest.get(
                "compilation_protocol"
            ),
            "case_labels": (
                sorted(cases) if isinstance(cases, dict) else None
            ),
            "candidate_manifest_sha256": manifest_sha256,
            "has_promotion_provenance": (
                "promotion_provenance" in candidate_manifest
            ),
        },
        expected={
            "schema_version": 1,
            "status": TABLE3_CANDIDATE_STATUS,
            "seed": TABLE3_SEED,
            "compilation_protocol": TABLE3_PROTOCOL,
            "case_labels": list(TABLE3_LABELS),
            "has_promotion_provenance": False,
        },
    )

    declared_hashes: dict[str, str] = {}
    artifact_checks_passed = True
    if not isinstance(cases, dict):
        cases = {}
    for label in TABLE3_LABELS:
        case = cases.get(label)
        molecule, norb, n_qubits = TABLE3_CASES[label]
        identity_valid = (
            isinstance(case, dict)
            and case.get("label") == label
            and case.get("molecule") == molecule
            and type(case.get("norb")) is int
            and case.get("norb") == norb
            and type(case.get("n_qubits")) is int
            and case.get("n_qubits") == n_qubits
            and type(case.get("seed")) is int
            and case.get("seed") == TABLE3_SEED
            and case.get("compilation_protocol") == TABLE3_PROTOCOL
        )
        audit.check(
            f"{label} fresh candidate identity and protocol",
            identity_valid,
            actual=(
                {
                    key: case.get(key)
                    for key in (
                        "label",
                        "molecule",
                        "norb",
                        "n_qubits",
                        "seed",
                        "compilation_protocol",
                    )
                }
                if isinstance(case, dict)
                else case
            ),
            expected={
                "label": label,
                "molecule": molecule,
                "norb": norb,
                "n_qubits": n_qubits,
                "seed": TABLE3_SEED,
                "compilation_protocol": TABLE3_PROTOCOL,
            },
        )
        artifact_checks_passed = artifact_checks_passed and identity_valid
        if not isinstance(case, dict):
            continue

        declarations = (
            (
                "parameter",
                case.get("parameter_artifact"),
                f"parameters/{label}_seed-{TABLE3_SEED}.npz",
                "file",
                "sha256",
            ),
            (
                "generic QASM",
                case.get("generic_qasm"),
                f"circuits/{label}_generic.qasm",
                "qasm_file",
                "qasm_sha256",
            ),
            (
                "structured QASM",
                case.get("structured_qasm"),
                f"circuits/{label}_structured.qasm",
                "qasm_file",
                "qasm_sha256",
            ),
        )
        for description, declaration, relative, path_key, hash_key in declarations:
            artifact = candidate_root / relative
            regular = artifact.is_file() and not artifact.is_symlink()
            actual_hash = sha256_file(artifact) if regular else None
            declared_hash = (
                declaration.get(hash_key)
                if isinstance(declaration, dict)
                else None
            )
            declaration_valid = (
                isinstance(declaration, dict)
                and declaration.get(path_key) == relative
                and isinstance(declared_hash, str)
                and len(declared_hash) == 64
                and all(
                    character in "0123456789abcdef"
                    for character in declared_hash
                )
                and regular
                and actual_hash == declared_hash
            )
            strict_error: str | None = None
            if declaration_valid:
                try:
                    if path_key == "file":
                        _validate_parameter_npz(
                            artifact,
                            declaration,
                            label=label,
                        )
                    else:
                        _validate_qasm_declaration(
                            artifact,
                            declaration,
                            label=label,
                            description=description,
                            expected_qubits=n_qubits,
                        )
                except Exception as error:
                    strict_error = f"{type(error).__name__}: {error}"
            else:
                strict_error = "path, hash, or regular-file binding failed"
            passed = declaration_valid and strict_error is None
            audit.check(
                f"{label} fresh {description} declaration and bytes",
                passed,
                actual={
                    "declared_path": (
                        declaration.get(path_key)
                        if isinstance(declaration, dict)
                        else None
                    ),
                    "expected_path": relative,
                    "declared_sha256": declared_hash,
                    "actual_sha256": actual_hash,
                    "strict_validation_error": strict_error,
                },
                expected={
                    "path": relative,
                    "sha256": actual_hash,
                    "strict_validation_error": None,
                },
            )
            artifact_checks_passed = artifact_checks_passed and passed
            if isinstance(declared_hash, str):
                declared_hashes[relative] = declared_hash

    candidate_aggregate: str | None = None
    candidate_records: list[dict[str, Any]] = []
    aggregate_binding = False
    if candidate_tree_valid:
        candidate_aggregate, candidate_records = _table3_artifact_aggregate(
            candidate_root,
            ["canonical_table3.json", *expected_artifacts],
        )
        aggregate_binding = (
            artifact_checks_passed
            and len(candidate_records) == 10
            and set(declared_hashes) == set(expected_artifacts)
            and all(
                record["path"] == "canonical_table3.json"
                or record["sha256"] == declared_hashes.get(record["path"])
                for record in candidate_records
            )
        )
    audit.check(
        "full Table-3 fresh ten-file candidate is self-bound",
        aggregate_binding,
        actual={
            "aggregate_sha256": candidate_aggregate,
            "files": candidate_records,
        },
        expected={
            "file_count": 10,
            "manifest": "validated schema/status/protocol/cases",
            "scientific_artifacts": (
                "nine fixed-path files matching their manifest declarations"
            ),
        },
    )

    expected_resources = (
        {
            label: {
                key: value
                for key, value in cases[label].items()
                if key not in {"ansatz", "parameter_artifact"}
            }
            for label in TABLE3_LABELS
        }
        if isinstance(cases, dict) and set(cases) == set(TABLE3_LABELS)
        else None
    )
    audit.check(
        "full advanced resources equal the fresh bound candidate records",
        expected_resources is not None
        and advanced.get("structured_resources") == expected_resources,
        actual=advanced.get("structured_resources"),
        expected=expected_resources,
    )


def validate_frozen_reference_audit(
    audit: Audit,
    run_dir: Path,
    *,
    manifest_path: Path | None = None,
) -> None:
    """Validate the independent ten-problem RHF/full-p-space certificate."""

    using_submission_manifest = manifest_path is None
    manifest_path = (
        ROOT / "frozen_inputs" / "MANIFEST.json"
        if manifest_path is None
        else Path(manifest_path)
    ).resolve()
    manifest = load(manifest_path)
    expected_problems = manifest.get("problems")
    if not isinstance(expected_problems, dict):
        raise TypeError("frozen-input manifest problems must be a JSON object")
    expected_names = set(expected_problems)
    audit.check(
        "frozen-reference manifest declares exactly ten cases",
        len(expected_names) == 10,
        actual=sorted(expected_names),
        expected="exactly ten manifest problem names",
    )

    artifact_path = Path(run_dir).resolve() / "frozen_reference_audit.json"
    exists = artifact_path.is_file()
    audit.check(
        "independent frozen-reference audit artifact exists",
        exists,
        actual=str(artifact_path),
        expected="existing frozen_reference_audit.json",
    )
    if not exists:
        return
    certificate = load(artifact_path)
    problems = certificate.get("problems")
    actual_names = set(problems) if isinstance(problems, dict) else set()
    audit.check(
        "independent frozen-reference audit manifest binding",
        certificate.get("manifest_sha256") == sha256_file(manifest_path),
        actual=certificate.get("manifest_sha256"),
        expected=sha256_file(manifest_path),
    )
    audit.check(
        "independent frozen-reference audit completed status",
        certificate.get("status") == "PASS",
        actual=certificate.get("status"),
        expected="PASS",
    )
    audit.check(
        "independent frozen-reference audit exact case set",
        isinstance(problems, dict) and actual_names == expected_names,
        actual=sorted(actual_names),
        expected=sorted(expected_names),
    )
    audit.check(
        "independent frozen-reference audit summary counts",
        int(certificate.get("problem_count", -1)) == len(expected_names)
        and int(certificate.get("passed", -1)) == len(expected_names)
        and int(certificate.get("failed", -1)) == 0,
        actual={
            "problem_count": certificate.get("problem_count"),
            "passed": certificate.get("passed"),
            "failed": certificate.get("failed"),
        },
        expected={
            "problem_count": len(expected_names),
            "passed": len(expected_names),
            "failed": 0,
        },
    )
    if not isinstance(problems, dict):
        return

    for name in sorted(expected_names & actual_names):
        expected = expected_problems[name]
        record = problems[name]
        checks = record.get("checks")
        check_names = set(checks) if isinstance(checks, dict) else set()
        audit.check(
            f"{name} independent reference status and checks",
            record.get("status") == "PASS"
            and isinstance(checks, dict)
            and REFERENCE_AUDIT_REQUIRED_CHECKS == check_names
            and all(value is True for value in checks.values()),
            actual={"status": record.get("status"), "checks": checks},
            expected={
                "status": "PASS",
                "required_checks": sorted(REFERENCE_AUDIT_REQUIRED_CHECKS),
                "all_checks": True,
            },
        )

        norb = int(expected["norb"])
        nelec = tuple(int(value) for value in expected["nelec_alpha_beta"])
        expected_active_hash = (
            expected.get("array_sha256", {}).get("mo_coeff_active")
            if isinstance(expected.get("array_sha256"), dict)
            else None
        )
        frozen_basis = record.get("frozen_active_basis")
        basis_ok = False
        if isinstance(frozen_basis, dict):
            orthonormality = frozen_basis.get("s_orthonormality")
            core_orthogonality = frozen_basis.get(
                "core_active_orthogonality"
            )
            fock_residual = frozen_basis.get(
                "generalized_fock_eigen_residual"
            )

            def within_declared_tolerance(
                item: Any,
                value_key: str,
            ) -> bool:
                if not isinstance(item, dict):
                    return False
                actual = float(item.get(value_key, float("nan")))
                tolerance = float(
                    item.get("absolute_tolerance", float("nan"))
                )
                return (
                    item.get("passed") is True
                    and math.isfinite(actual)
                    and actual >= 0.0
                    and math.isfinite(tolerance)
                    and tolerance > 0.0
                    and actual <= tolerance
                )

            basis_ok = (
                frozen_basis.get("passed") is True
                and frozen_basis.get("coefficient_checksum_match") is True
                and isinstance(expected_active_hash, str)
                and frozen_basis.get("coefficient_sha256")
                == expected_active_hash
                and frozen_basis.get("expected_coefficient_sha256")
                == expected_active_hash
                and within_declared_tolerance(
                    orthonormality,
                    "max_abs_error",
                )
                and within_declared_tolerance(
                    core_orthogonality,
                    "max_abs_overlap",
                )
                and within_declared_tolerance(
                    fock_residual,
                    "max_abs_residual",
                )
            )
        regeneration = record.get("active_space_regeneration")
        regeneration_ok = (
            isinstance(regeneration, dict)
            and regeneration.get("basis")
            == (
                "checksum-bound frozen active orbitals inserted after "
                "the regenerated core"
            )
            and int(regeneration.get("declared_ncore", -1))
            == int(expected.get("source", {}).get("frozen_core_orbitals", -2))
            and int(regeneration.get("pyscf_casci_ncore", -1))
            == int(regeneration.get("declared_ncore", -2))
            and regeneration.get("declared_core_count_matches") is True
        )
        audit.check(
            f"{name} checksum-bound frozen-basis regeneration",
            basis_ok and regeneration_ok,
            actual={
                "frozen_active_basis": frozen_basis,
                "active_space_regeneration": regeneration,
            },
            expected=(
                "manifest-bound frozen active orbitals with S/core/Fock "
                "residuals within declared tolerances and matching CASCI core"
            ),
        )

        expected_dimension = math.comb(norb, nelec[0]) * math.comb(
            norb, nelec[1]
        )
        declared_dimension = int(expected["sector_dimension"])
        sector = record.get("sector_consistency")
        sector_ok = (
            isinstance(sector, dict)
            and sector.get("passed") is True
            and sector.get("manifest_nelec_alpha_beta") == list(nelec)
            and sector.get("bundle_nelec_alpha_beta") == list(nelec)
            and sector.get("bundle_matches_manifest") is True
            and int(sector.get("declared_sector_dimension", -1))
            == declared_dimension
            and int(sector.get("combinatorial_sector_dimension", -1))
            == expected_dimension
            and sector.get("declared_dimension_matches_combinatorial")
            is True
        )
        pspace = record.get("pspace")
        residual_ok = False
        if isinstance(pspace, dict):
            residual = float(
                pspace.get("eigen_residual_norm_hartree", float("nan"))
            )
            hermiticity = float(
                pspace.get("hermiticity_max_abs_hartree", float("nan"))
            )
            residual_tolerance = float(
                pspace.get(
                    "residual_absolute_tolerance_hartree", float("nan")
                )
            )
            residual_ok = (
                pspace.get("solver")
                == "numpy.linalg.eigh on PySCF full determinant p-space"
                and pspace.get("converged") is True
                and pspace.get("all_determinant_addresses_present") is True
                and int(pspace.get("full_determinant_dimension", -1))
                == expected_dimension
                and int(pspace.get("p_space_dimension", -1))
                == expected_dimension
                and math.isfinite(residual)
                and residual >= 0.0
                and math.isfinite(hermiticity)
                and hermiticity >= 0.0
                and math.isfinite(residual_tolerance)
                and residual_tolerance > 0.0
                and residual <= residual_tolerance
                and hermiticity <= residual_tolerance
            )
        audit.check(
            f"{name} full determinant p-space residual certificate",
            declared_dimension == expected_dimension
            and record.get("nelec_alpha_beta") == list(nelec)
            and int(record.get("sector_dimension", -1))
            == declared_dimension
            and sector_ok
            and residual_ok,
            actual={
                "manifest_sector_dimension": declared_dimension,
                "combinatorial_dimension": expected_dimension,
                "record_nelec_alpha_beta": record.get("nelec_alpha_beta"),
                "record_sector_dimension": record.get("sector_dimension"),
                "sector_consistency": sector,
                "pspace": pspace,
            },
            expected=(
                "manifest-bound electron sector and complete determinant "
                "p-space with residual/hermiticity within declared tolerance"
            ),
        )

    # Apply the stricter pre-promotion proof as a second independent gate.
    # This binds the trusted tolerance map, checksum-verified bundle scalars
    # and occupations, recomputed RHF/CASCI/core-energy arithmetic, and the
    # full p-space record instead of trusting summary booleans.
    from audit_table3_candidate import (
        validate_frozen_reference_certificate as validate_strict_reference,
    )

    class StrictReferenceAdapter:
        def check(
            self,
            name: str,
            passed: bool,
            *,
            actual: Any = None,
            expected: Any = None,
            fatal: bool = True,
        ) -> bool:
            del fatal
            audit.check(
                f"strict {name}",
                passed,
                actual=actual,
                expected=expected,
            )
            return passed

    if using_submission_manifest:
        try:
            validate_strict_reference(
                StrictReferenceAdapter(),
                source_code=manifest_path.parent.parent,
                invocation_directory=Path(run_dir).resolve(),
            )
        except Exception as error:
            audit.check(
                "strict frozen-reference verifier completed",
                False,
                actual=f"{type(error).__name__}: {error}",
                expected="complete fail-closed verification",
            )


def median_run_metric(record: dict[str, Any], key: str) -> float:
    return statistics.median(float(run[key]) for run in record["runs"].values())


def minimum_embedding_fidelity(record: dict[str, Any]) -> float:
    values = []
    for seed in record["seeds"].values():
        for rung in seed["cascade"]:
            fidelity = rung.get("initial_embedded_state_fidelity")
            if fidelity is not None:
                values.append(float(fidelity))
    if not values:
        raise ValueError("No warm-start embedding fidelities were recorded")
    return min(values)


def validate_forte(audit: Audit, expected: dict[str, Any]) -> None:
    result_path = ROOT / "reference" / "h2_forte_result.json"
    submission_path = ROOT / "reference" / "forte_submission.json"
    source_reference_path = ROOT / "reference" / "H2_GQE_REFERENCE.json"
    record = load(result_path)
    submission = load(submission_path)
    counts = record["counts"]
    shots = sum(int(value) for value in counts.values())
    signed = sum(
        (1 if bitstring.count("1") % 2 == 0 else -1) * int(count)
        for bitstring, count in counts.items()
    )
    measured = signed / shots
    plus = (shots + signed) // 2
    minus = shots - plus
    alpha = 0.05
    p_low = 0.0 if plus == 0 else float(beta.ppf(alpha / 2, plus, minus + 1))
    p_high = 1.0 if minus == 0 else float(beta.ppf(1 - alpha / 2, plus + 1, minus))
    interval = [2 * p_low - 1, 2 * p_high - 1]
    standard_error = math.sqrt(max(0.0, 1.0 - measured**2) / shots)
    duration_seconds = float(record["result_details"]["time_stamps"]["executionDuration"]) / 1000.0

    audit.check(
        "Forte immutable result checksum",
        sha256_file(result_path) == expected["result_record_sha256"],
        actual=sha256_file(result_path),
        expected=expected["result_record_sha256"],
    )
    audit.check(
        "Forte immutable submission checksum",
        sha256_file(submission_path) == expected["submission_record_sha256"],
        actual=sha256_file(submission_path),
        expected=expected["submission_record_sha256"],
    )
    audit.check(
        "Forte immutable source-reference checksum",
        sha256_file(source_reference_path) == expected["source_reference_sha256"],
        actual=sha256_file(source_reference_path),
        expected=expected["source_reference_sha256"],
    )
    audit.check(
        "Forte result links to immutable submission",
        record["submission_record_sha256"] == sha256_file(submission_path),
        actual=record["submission_record_sha256"],
        expected=sha256_file(submission_path),
    )
    audit.check("Forte reference shot count", shots == int(expected["shots"]), actual=shots, expected=expected["shots"])
    audit.check("Forte requested and returned shots agree", int(record["shots_requested"]) == shots == int(record["shots_returned"]), actual=[record["shots_requested"], shots, record["shots_returned"]], expected=expected["shots"])
    audit.check("Forte completed status", record["status"] == expected["status"], actual=record["status"], expected=expected["status"])
    audit.check("Forte device identity", record["device_id"] == submission["device"]["id"] == expected["device_id"], actual=[record["device_id"], submission["device"]["id"]], expected=expected["device_id"])
    audit.check("Forte job identity", record["job_id"] == submission["job_id"] == expected["job_id"], actual=[record["job_id"], submission["job_id"]], expected=expected["job_id"])
    audit.check("Forte recorded credits", float(record["result_details"]["cost"]) == float(expected["historical_cost_credits"]), actual=record["result_details"]["cost"], expected=expected["historical_cost_credits"])
    audit.check("Forte provider duration", close(duration_seconds, float(expected["provider_duration_seconds"]), 1e-9), actual=duration_seconds, expected=expected["provider_duration_seconds"])
    audit.check(
        "Forte reference expectation recomputes from counts",
        close(measured, float(expected["measured_expectation"]), 1e-9),
        actual=measured,
        expected=expected["measured_expectation"],
    )
    audit.check(
        "Forte exact 95% interval recomputes",
        all(close(a, float(e), float(expected["interval_absolute_tolerance"])) for a, e in zip(interval, expected["exact_95_interval"])),
        actual=interval,
        expected=expected["exact_95_interval"],
    )
    audit.check(
        "Forte shot-noise standard error recomputes",
        close(standard_error, float(expected["shot_noise_standard_error"]), 1e-9),
        actual=standard_error,
        expected=expected["shot_noise_standard_error"],
    )
    audit.check(
        "Forte ideal value lies in the exact interval",
        interval[0] <= float(record["ideal_expectation"]) <= interval[1],
        actual=record["ideal_expectation"],
        expected=interval,
    )
    audit.check(
        "Forte interval excludes the RHF zero expectation",
        not (interval[0] <= float(expected["rhf_expectation"]) <= interval[1]),
        actual=interval,
        expected=f"exclude {expected['rhf_expectation']}",
    )

    # Reconstruct the exact source-level payload without importing a provider.
    sys.path.insert(0, str(ROOT / "forte"))
    import forte_hardware as fh

    reference = fh.load_reference(ROOT / "forte")
    preflight = fh.qiskit_preflight(reference)
    hardware_circuit = preflight["hardware_circuit"]
    source_cx = int(hardware_circuit.count_ops().get("cx", 0))
    source_depth = int(hardware_circuit.depth())
    hf_state = np.zeros(1 << fh.N_QUBITS, dtype=complex)
    hf_state[(1 << fh.N_ELECTRONS) - 1] = 1.0
    rhf_expectation = fh.pauli_expectation(hf_state, fh.MEASUREMENT_WORD)
    audit.check("Forte reconstructed QASM hash", preflight["hardware_circuit_qasm2_sha256"] == submission["circuit"]["qasm2_sha256"] == expected["qasm2_sha256"], actual=preflight["hardware_circuit_qasm2_sha256"], expected=expected["qasm2_sha256"])
    audit.check("Forte source-level CX count", source_cx == int(expected["source_logical_cx"]), actual=source_cx, expected=expected["source_logical_cx"])
    audit.check("Forte submitted logical depth", source_depth == int(expected["source_logical_depth"]), actual=source_depth, expected=expected["source_logical_depth"])
    audit.check("Forte ideal observable recomputes from the frozen circuit", close(float(preflight["ideal_xxyy_expectation"]), float(expected["ideal_expectation"]), 1e-9), actual=preflight["ideal_xxyy_expectation"], expected=expected["ideal_expectation"])
    audit.check("RHF determinant has zero W expectation", close(rhf_expectation, float(expected["rhf_expectation"]), 1e-12), actual=rhf_expectation, expected=expected["rhf_expectation"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    expected = load(ROOT / "expected_metrics.json")
    audit = Audit()

    advanced_root = run_dir / "advanced_method"
    advanced = load(advanced_root / "enhanced_release_summary.json")
    metadata = load(advanced_root / "run_metadata.json")
    audit.check("advanced simulator release completed", advanced.get("status") == "COMPLETED", actual=advanced.get("status"), expected="COMPLETED")
    audit.check("advanced workflow did not contact a QPU", metadata.get("qpu_contacted") is False, actual=metadata.get("qpu_contacted"), expected=False)
    audit.check("advanced workflow imported no provider", metadata.get("provider_imported") is False, actual=metadata.get("provider_imported"), expected=False)

    validate_frozen_reference_audit(audit, run_dir)
    validate_forte(audit, expected["forte_h2_reference"])
    # Both modes make the same judge-facing Table-3 resource claims.  Quick
    # mode replays the promoted payload, while full mode regenerates a
    # candidate; neither may pass against stale CX/depth expectations.
    validate_table3_claims(audit, advanced_root, advanced, expected, args.mode)

    for label, controls in advanced["qsci_residual_controls"].items():
        for control_name in (
            "identity",
            "random_feasible_best",
            "greedy_feasible",
        ):
            audit_token_sequence(
                audit,
                f"{label} advanced {control_name}",
                controls[control_name]["tokens"],
                expected_count=controls.get("sequence_length"),
            )

    # Exact 10,000-branch QPD is part of both quick and full certification.
    qpd = load(run_dir / "qpd_result.json")
    target = expected["qpd_6"]
    audit.check(
        "6q exact-QPD completed status",
        qpd.get("status") == "passed",
        actual=qpd.get("status"),
        expected="passed",
    )
    audit.check(
        "6q exact-QPD energy agreement",
        abs(float(qpd["absolute_difference_hartree"]))
        <= float(target["maximum_absolute_difference_hartree"]),
        actual=qpd["absolute_difference_hartree"],
        expected=target["maximum_absolute_difference_hartree"],
    )
    audit.check(
        "6q exact-QPD branch count",
        int(qpd["enumerated_branches"]) == int(target["enumerated_branches"]),
        actual=qpd["enumerated_branches"],
        expected=target["enumerated_branches"],
    )
    audit.check(
        "6q exact-QPD fragment widths",
        qpd["half_qubits"] == target["half_qubits"],
        actual=qpd["half_qubits"],
        expected=target["half_qubits"],
    )

    if args.mode == "full":
        for label, target in expected["givens_backbone_median_error_mha"].items():
            record = advanced["baseline"][label]
            actual = median_run_metric(record, "error_mha")
            audit.check(
                f"{label} optimized Givens-backbone median error",
                close(actual, float(target["expected"]), float(target["absolute_tolerance"])),
                actual=actual,
                expected=target,
            )
            sector_dims = {int(run["sector_dim"]) for run in record["runs"].values()}
            audit.check(f"{label} fixed-sector dimension", sector_dims == {int(target["sector_dim"])}, actual=sorted(sector_dims), expected=target["sector_dim"])
            audit.check(f"{label} declared seed count", len(record["runs"]) == int(target["seed_count"]), actual=len(record["runs"]), expected=target["seed_count"])
            median_time = median_run_metric(record, "wall_seconds")
            reference_time = float(target["reference_vqe_time_seconds"])
            audit.check(f"{label} Givens-VQE optimization time recorded", math.isfinite(median_time) and median_time > 0 and 0.1 * reference_time <= median_time <= 10.0 * reference_time, actual=median_time, expected=f"positive platform time within 0.1x-10x of reference {reference_time}")

        for label, target in expected["givens_backbone_rhf_error_mha"].items():
            record = advanced["baseline"][label]
            actual = statistics.median(
                (float(run["e_hf"]) - float(run["e_casci"])) * 1000.0
                for run in record["runs"].values()
            )
            audit.check(
                f"{label} same-instance RHF baseline error",
                close(actual, float(target["expected"]), float(target["absolute_tolerance"])),
                actual=actual,
                expected=target,
            )

        for label, target in expected["adaptive_topology"].items():
            record = advanced["adaptive_topology"][label]
            audit.check(f"{label} topology decision", record["decision"] == target["decision"], actual=record["decision"], expected=target["decision"])
            audit.check(f"{label} static partition", record["static_left_block"] == target["static_left"], actual=record["static_left_block"], expected=target["static_left"])
            audit.check(f"{label} selected partition", record["selected_left_block"] == target["selected_left"], actual=record["selected_left_block"], expected=target["selected_left"])
            required_gates = (
                "candidate_on_screened_pareto_front",
                "differs_from_static_partition",
                "energy_noninferior_within_0p1_mha",
                "improves_at_least_one_declared_pre_screen_metric",
                "within_phi_budget",
            )
            passing = [
                candidate
                for candidate in record["screened_candidates"]
                if all(candidate["acceptance"].get(gate) is True for gate in required_gates)
            ]
            accepted = record["decision"].startswith("ACCEPT_")
            selected_candidates = [
                candidate
                for candidate in passing
                if candidate["pre_screen"]["left_block"] == record["selected_left_block"]
            ]
            audit.check(
                f"{label} topology Pareto/energy/budget gates",
                (accepted and len(selected_candidates) == 1)
                or (not accepted and not passing and record["selected_left_block"] == record["static_left_block"]),
                actual={
                    "accepted": accepted,
                    "passing_candidates": [x["pre_screen"]["left_block"] for x in passing],
                    "selected": record["selected_left_block"],
                },
                expected="one selected all-gates candidate when accepted; otherwise retain static",
            )
            if accepted and selected_candidates:
                tradeoff = selected_candidates[0]["tradeoff_vs_static_pre_screen"]
                audit.check(
                    f"{label} accepted topology is a declared tradeoff",
                    tradeoff["interpretation"] == "pareto_front_tradeoff_not_dominance"
                    and float(tradeoff["schmidt_entropy_delta_nats"]) < 0.0
                    and float(tradeoff["hamiltonian_cross_weight_delta"]) > 0.0,
                    actual=tradeoff,
                    expected="lower entropy with increased cross-Hamiltonian weight",
                )

        for label, target in expected["warm_start"].items():
            record = advanced["warm_start"][label]
            fidelity = minimum_embedding_fidelity(record)
            ratios = [float(seed["cascade_to_direct_call_ratio"]) for seed in record["seeds"].values()]
            ratio = statistics.median(ratios)
            audit.check(f"{label} embedded-state fidelity", fidelity >= float(target["minimum_embedding_fidelity"]), actual=fidelity, expected=target["minimum_embedding_fidelity"])
            audit.check(f"{label} complete cascade/direct calls", close(ratio, float(target["cascade_to_direct_call_ratio"]), float(target["absolute_tolerance"])), actual=ratio, expected=target)

        h2 = load(run_dir / "h2_gqe_full.json")
        target = expected["h2_transformer_gqe"]
        audit.check("H2 Transformer-GQE error", close(float(h2["energies_hartree"]["absolute_error"]), float(target["absolute_error_hartree"]), float(target["absolute_tolerance_hartree"])), actual=h2["energies_hartree"]["absolute_error"], expected=target)
        audit.check("H2 Transformer-GQE two-qubit estimate", int(h2["estimated_two_qubit_gate_count"]) == int(target["estimated_two_qubit_gate_count"]), actual=h2["estimated_two_qubit_gate_count"], expected=target["estimated_two_qubit_gate_count"])
        rhf_error_mha = (float(h2["energies_hartree"]["hartree_fock"]) - float(h2["energies_hartree"]["exact_fci"])) * 1000.0
        gqe_error_mha = float(h2["energies_hartree"]["absolute_error"]) * 1000.0
        reduction_percent = 100.0 * (rhf_error_mha - gqe_error_mha) / rhf_error_mha
        audit.check("H2 RHF baseline error", close(rhf_error_mha, float(target["rhf_error_mha"]), 0.000001), actual=rhf_error_mha, expected=target["rhf_error_mha"])
        audit.check("H2 ideal GQE error", close(gqe_error_mha, float(target["gqe_error_mha"]), 0.000001), actual=gqe_error_mha, expected=target["gqe_error_mha"])
        audit.check("H2 ideal GQE relative error reduction", close(reduction_percent, float(target["relative_error_reduction_percent"]), 0.0001), actual=reduction_percent, expected=target["relative_error_reduction_percent"])
        for field in ("max_iters", "num_samples", "operator_pool_size"):
            audit.check(f"H2 GQE {field}", int(h2["gqe_config"][field]) == int(target[field]), actual=h2["gqe_config"][field], expected=target[field])
        candidate_evaluations = int(h2["gqe_config"]["max_iters"]) * int(h2["gqe_config"]["num_samples"])
        audit.check("H2 GQE generated-circuit budget", candidate_evaluations == int(target["candidate_evaluations"]), actual=candidate_evaluations, expected=target["candidate_evaluations"])
        audit.check("H2 GQE selected ten operators", len(h2["selected_operators"]) == int(target["selected_operator_count"]), actual=len(h2["selected_operators"]), expected=target["selected_operator_count"])
        audit.check("H2 GQE selected nonidentity/nonlocal operators", int(h2["selected_nonlocal_operator_count"]) == int(target["selected_nonlocal_operator_count"]), actual=h2["selected_nonlocal_operator_count"], expected=target["selected_nonlocal_operator_count"])

        finite = load(run_dir / "finite_shot_beh2_6q.json")["finite_shot"]
        target = expected["finite_shot_beh2_6"]
        audit.check("BeH2 6q finite-shot error", close(float(finite["error_vs_casci_mha"]), float(target["error_vs_casci_mha"]), float(target["absolute_tolerance_mha"])), actual=finite["error_vs_casci_mha"], expected=target)
        for field in ("shots_per_group", "commuting_groups", "total_circuit_shots"):
            audit.check(f"BeH2 6q {field}", int(finite[field]) == int(target[field]), actual=finite[field], expected=target[field])
        audit.check("BeH2 6q finite-shot standard error", close(float(finite["reported_standard_error"]), float(target["standard_error_hartree"]), 0.00000001), actual=finite["reported_standard_error"], expected=target["standard_error_hartree"])
        audit.check("BeH2 6q finite-shot seed", int(finite["seed"]) == int(target["seed"]), actual=finite["seed"], expected=target["seed"])
        audit.check("BeH2 6q finite-shot backend", finite["estimator"] == target["backend"] and finite["finite_shot_backend"] is True, actual=[finite["estimator"], finite["finite_shot_backend"]], expected=[target["backend"], True])

        for system, target in expected["gqe_qsci_objective"].items():
            stem = system.replace("-", "_")
            comparison = load(run_dir / "objectives" / f"{stem}_objective_comparison.json")
            exact_record = load(run_dir / "objectives" / f"{stem}_exact_energy.json")
            qsci_record = load(run_dir / "objectives" / f"{stem}_qsci_topk.json")
            audit.check(f"{system} QSCI promotion decision", comparison["decision"] == target["decision"], actual=comparison["decision"], expected=target["decision"])
            audit.check(f"{system} QSCI objective was not promoted", comparison["promoted"] is False, actual=comparison["promoted"], expected=False)
            raw_delta = float(comparison["raw_state_errors_mha"]["qsci_minus_exact"])
            audit.check(f"{system} QSCI raw-state delta", close(raw_delta, float(target["raw_qsci_minus_exact_mha"]), float(target["absolute_tolerance_mha"])), actual=raw_delta, expected=target)
            exact_heldout = float(comparison["heldout_median_errors_mha"]["exact_energy_transformer"])
            qsci_heldout = float(comparison["heldout_median_errors_mha"]["qsci_topk_transformer"])
            audit.check(f"{system} exact-objective held-out median", close(exact_heldout, float(target["exact_heldout_median_mha"]), 0.000001), actual=exact_heldout, expected=target["exact_heldout_median_mha"])
            audit.check(f"{system} QSCI-objective held-out median", close(qsci_heldout, float(target["qsci_heldout_median_mha"]), 0.000001), actual=qsci_heldout, expected=target["qsci_heldout_median_mha"])
            interval = [float(x) for x in comparison["paired_primary"]["bootstrap_95pct_ci_mha"]]
            audit.check(f"{system} paired bootstrap interval", all(close(a, float(e), 0.000001) for a, e in zip(interval, target["bootstrap_95_interval_mha"])), actual=interval, expected=target["bootstrap_95_interval_mha"])
            if "paired_median_delta_mha" in target:
                paired = float(comparison["paired_primary"]["median_delta_mha"])
                audit.check(f"{system} QSCI paired held-out median", close(paired, float(target["paired_median_delta_mha"]), float(target["paired_median_absolute_tolerance_mha"])), actual=paired, expected=target)

            qsci_tokens = qsci_record["guarded_result"]["tokens"]
            qsci_nonidentity = any(token["kind"] != "identity" for token in qsci_tokens)
            audit.check(f"{system} QSCI guarded selection source", qsci_record["guarded_result"]["selection_source"] == target["qsci_selection_source"], actual=qsci_record["guarded_result"]["selection_source"], expected=target["qsci_selection_source"])
            audit.check(f"{system} QSCI identity/nonidentity selection", qsci_nonidentity is bool(target["qsci_selected_nonidentity"]), actual=qsci_nonidentity, expected=target["qsci_selected_nonidentity"])

            for arm_name, arm_record in (("exact-energy", exact_record), ("QSCI", qsci_record)):
                validate_budget_semantics(
                    audit,
                    f"{system} {arm_name}",
                    arm_record,
                )
                budget = arm_record["cutting_budget"]
                audit.check(
                    f"{system} {arm_name} pre-sampling mask and Phi budget",
                    budget["mask_applied_before_sampling"] is True
                    and int(budget["trajectory_mask_violations"]) == 0
                    and budget["within_budget"] is True
                    and close(float(budget["phi_max"]), float(target["phi_max"]), 1e-12)
                    and float(arm_record["cost_statistics"]["max_total_phi"]) <= float(target["phi_max"]),
                    actual={
                        "mask_applied_before_sampling": budget["mask_applied_before_sampling"],
                        "trajectory_mask_violations": budget["trajectory_mask_violations"],
                        "phi_max": budget["phi_max"],
                        "max_total_phi": arm_record["cost_statistics"]["max_total_phi"],
                    },
                    expected=f"mask before sampling, zero violations, total Phi <= {target['phi_max']}",
                )

            parity = comparison["protocol_parity"]
            parity_matches = {
                key: value.get("match")
                for key, value in parity.items()
                if isinstance(value, dict) and "match" in value
            }
            audit.check(f"{system} exact/QSCI protocol parity", bool(parity_matches) and all(value is True for value in parity_matches.values()), actual=parity_matches, expected="all parity fields true")
            protocol = parity["heldout_protocol"]["exact"]
            audit.check(
                f"{system} held-out QSCI protocol budget",
                int(protocol["max_sampled_determinants"]) == int(target["qsci_k"])
                and int(protocol["primary_shots"]) == int(target["primary_shots"])
                and int(protocol["trial_count"]) == int(target["trial_count"])
                and protocol["promotion_endpoint_predeclared_before_policy_runs"] is True,
                actual={"K": protocol["max_sampled_determinants"], "shots": protocol["primary_shots"], "trials": protocol["trial_count"]},
                expected={"K": target["qsci_k"], "shots": target["primary_shots"], "trials": target["trial_count"]},
            )

            rule = comparison["promotion_rule"]
            criteria = rule["criteria"]
            benefit_criteria = (
                "beats_qsci_greedy_by_at_least_0p1_mha",
                "beats_qsci_identity_by_at_least_0p1_mha",
                "beats_qsci_random_by_at_least_0p1_mha",
                "paired_bootstrap_95pct_upper_bound_below_zero",
                "paired_median_improves_by_at_least_0p1_mha",
            )
            audit.check(
                f"{system} predeclared QSCI promotion rule",
                close(float(rule["minimum_meaningful_improvement_mha"]), 0.1, 1e-12)
                and close(float(rule["raw_state_degradation_limit_mha"]), 1.6, 1e-12)
                and criteria["all_protocol_parity_checks_pass"] is True
                and criteria["zero_budget_mask_violations"] is True
                and criteria["raw_state_degradation_within_chemical_accuracy"] is bool(target["raw_state_degradation_within_chemical_accuracy"])
                and all(criteria[key] is False for key in benefit_criteria),
                actual=rule,
                expected="0.1 mHa benefit threshold, 1.6 mHa degradation limit, no promotion criteria met",
            )

            qsci_control_keys = {
                "identity": "qsci_identity",
                "random_feasible": "qsci_random_feasible",
                "stratified_greedy": "qsci_stratified_pool_greedy",
            }
            for expected_key, comparison_key in qsci_control_keys.items():
                actual = float(comparison["heldout_median_errors_mha"][comparison_key])
                expected_value = float(target["qsci_matched_control_medians_mha"][expected_key])
                audit.check(f"{system} QSCI {expected_key} control median", close(actual, expected_value, 0.000001), actual=actual, expected=expected_value)

        for system, target in expected["gqe_residual_exact_energy"].items():
            stem = system.replace("-", "_")
            exact = load(run_dir / "objectives" / f"{stem}_exact_energy.json")
            comparison = load(run_dir / "objectives" / f"{stem}_objective_comparison.json")
            guarded = exact["guarded_result"]
            controls = exact["heldout_evaluation"]["matched_policy_controls"]
            sampled_error = (float(exact["sampled_result"]["raw_state_energy_hartree"]) - float(guarded["exact_casci_hartree"])) * 1000.0
            sampled_tokens = exact["sampled_result"]["tokens"]
            audit.check(f"{system} residual GQE exact-sector candidate budget", int(exact["cost_statistics"]["calls"]) == int(target["candidate_evaluations"]), actual=exact["cost_statistics"]["calls"], expected=target["candidate_evaluations"])
            audit.check(f"{system} residual GQE frozen-backbone error", close(float(exact["backbone"]["error_mha"]), float(target["frozen_backbone_error_mha"]), float(target["absolute_tolerance_mha"])), actual=exact["backbone"]["error_mha"], expected=target["frozen_backbone_error_mha"])
            audit.check(f"{system} residual GQE sampled nonidentity error", close(sampled_error, float(target["sampled_nonidentity_error_mha"]), float(target["absolute_tolerance_mha"])), actual=sampled_error, expected=target["sampled_nonidentity_error_mha"])
            audit.check(f"{system} residual GQE sampled candidate is nonidentity", any(token["kind"] != "identity" for token in sampled_tokens), actual=[token["kind"] for token in sampled_tokens], expected="at least one nonidentity token")
            audit.check(f"{system} residual GQE guarded error", close(float(guarded["error_mha"]), float(target["guarded_error_mha"]), float(target["absolute_tolerance_mha"])), actual=guarded["error_mha"], expected=target["guarded_error_mha"])
            audit.check(f"{system} residual GQE selection source", guarded["selection_source"] == target["selection_source"], actual=guarded["selection_source"], expected=target["selection_source"])
            audit.check(f"{system} residual GQE retained identity", all(token["kind"] == "identity" for token in guarded["tokens"]), actual=[token["kind"] for token in guarded["tokens"]], expected="six identity tokens")
            audit.check(f"{system} residual GQE used shot-free exact candidate costs", exact["backend"]["qpu_used"] is False and "exact PySCF fixed-particle sector" in exact["backend"]["cost_evaluator"], actual=exact["backend"], expected="credential-free exact-sector evaluator")
            audit.check(f"{system} random control candidate budget", int(controls["random_feasible"]["candidate_evaluations"]) == int(target["candidate_evaluations"]), actual=controls["random_feasible"]["candidate_evaluations"], expected=target["candidate_evaluations"])
            audit.check(f"{system} greedy control candidate budget", int(controls["stratified_pool_greedy"]["candidate_evaluations"]) == int(target["candidate_evaluations"]), actual=controls["stratified_pool_greedy"]["candidate_evaluations"], expected=target["candidate_evaluations"])
            audit.check(f"{system} identity reference uses no candidate evaluations", int(controls["identity"]["candidate_evaluations"]) == 0, actual=controls["identity"]["candidate_evaluations"], expected=0)
            exact_heldout = float(comparison["heldout_median_errors_mha"]["exact_energy_transformer"])
            audit.check(f"{system} exact-energy Transformer held-out median", close(exact_heldout, float(target["heldout_median_errors_mha"]["exact_energy_transformer"]), 0.000001), actual=exact_heldout, expected=target["heldout_median_errors_mha"]["exact_energy_transformer"])
            exact_control_keys = {
                "identity": "identity",
                "random_feasible": "random_feasible",
                "stratified_greedy": "stratified_pool_greedy",
            }
            for expected_key, control_key in exact_control_keys.items():
                control = controls[control_key]
                control_median = float(control["heldout"]["finite_shot"]["error_mha_median"])
                target_value = float(target["heldout_median_errors_mha"][expected_key])
                audit.check(f"{system} exact-energy {expected_key} control held-out median", close(control_median, target_value, 0.000001), actual=control_median, expected=target_value)
                audit.check(f"{system} exact-energy {expected_key} control retained identity", all(token["kind"] == "identity" for token in control["tokens"]), actual=[token["kind"] for token in control["tokens"]], expected="six identity tokens")

        projection = advanced["twenty_plus_twenty_projection"]
        target = expected["twenty_plus_twenty_projection"]
        audit.check("20+20 projection execution status", projection["status"] == target["status"], actual=projection["status"], expected=target["status"])
        audit.check("20+20 fragment widths", projection["spin_qubits_per_fragment"] == target["spin_qubits_per_fragment"], actual=projection["spin_qubits_per_fragment"], expected=target["spin_qubits_per_fragment"])
        audit.check("20+20 theoretical phi", close(float(projection["theoretical_optimal_phi"]), float(target["theoretical_phi"]), 1e-9), actual=projection["theoretical_optimal_phi"], expected=target["theoretical_phi"])
        audit.check("20+20 theoretical phi squared", close(float(projection["theoretical_shot_multiplier_phi_squared"]), float(target["theoretical_phi_squared"]), 1e-9), actual=projection["theoretical_shot_multiplier_phi_squared"], expected=target["theoretical_phi_squared"])
        audit.check("20+20 cross-pair-double QPD remains unavailable", projection["executable_cross_pair_double_qpd_available"] is target["cross_pair_double_qpd_available"], actual=projection["executable_cross_pair_double_qpd_available"], expected=target["cross_pair_double_qpd_available"])
        audit.check("20+20 sampled QPD remains unavailable", projection["sampled_qpd_estimator_available"] is target["sampled_qpd_estimator_available"], actual=projection["sampled_qpd_estimator_available"], expected=target["sampled_qpd_estimator_available"])
        audit.check("20+20 joint QSCI reconstruction remains unavailable", projection["joint_cut_fragment_qsci_reconstruction_available"] is target["joint_cut_fragment_qsci_available"], actual=projection["joint_cut_fragment_qsci_reconstruction_available"], expected=target["joint_cut_fragment_qsci_available"])
        audit.check("20+20 paid execution remains disabled", projection["paid_execution_allowed"] is target["paid_execution_allowed"], actual=projection["paid_execution_allowed"], expected=target["paid_execution_allowed"])

        for system in ("beh2_6", "beh2_12"):
            exact = load(run_dir / "objectives" / f"{system}_exact_energy.json")
            crosscheck = exact["cudaq_state_crosscheck"]
            audit.check(f"{system} CUDA-Q state-energy crosscheck", crosscheck is not None and abs(float(crosscheck["absolute_difference_hartree"])) <= 1e-10, actual=crosscheck, expected="absolute difference <= 1e-10 Ha")

    status = audit.close()
    output = {
        "schema_version": 1,
        "status": status,
        "mode": args.mode,
        "run_directory": str(run_dir),
        "checks": audit.checks,
        "passed": sum(item["status"] == "PASS" for item in audit.checks),
        "failed": sum(item["status"] == "FAIL" for item in audit.checks),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"VALIDATION {status}: {output['passed']} passed, {output['failed']} failed")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
