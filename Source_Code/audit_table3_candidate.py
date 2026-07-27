#!/usr/bin/env python3
"""Fail-closed, read-only audit of a generated Table-3 candidate.

This verifier is deliberately separate from promotion.  It accepts only the
noncanonical candidate emitted by ``bootstrap_table3_candidate.py``, binds it
to that invocation's status and source identity, independently reloads every
parameter/QASM artifact, and reconstructs all three cases from the frozen
inputs.  It never writes below the candidate directory or ``reference/`` and
does not contain a promotion operation.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat as stat_module
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
import uuid
import zipfile

import numpy as np


sys.dont_write_bytecode = True

SCHEMA_VERSION = 1
CANDIDATE_STATUS = "CANDIDATE_NOT_CANONICAL_UNTIL_EXPLICITLY_PROMOTED"
READY_STATUS = "CANDIDATE_READY_NOT_CERTIFIED"
EXPECTED_STAGE_NAMES = (
    "locked_environment_smoke",
    "dependency_consistency",
    "independent_frozen_reference_audit",
    "full_advanced_candidate_generation",
)
EXPECTED_CASES: dict[str, tuple[str, int, int]] = {
    "BeH2-6": ("BeH2", 3, 6),
    "BeH2-12": ("BeH2", 6, 12),
    "LiH-40": ("LiH", 20, 40),
}
EXPECTED_CANDIDATE_FILES = frozenset(
    {
        "canonical_table3.json",
        *(
            f"parameters/{label}_seed-3047.npz"
            for label in EXPECTED_CASES
        ),
        *(
            f"circuits/{label}_{arm}.qasm"
            for label in EXPECTED_CASES
            for arm in ("generic", "structured")
        ),
    }
)
EXPECTED_CANDIDATE_DIRECTORIES = frozenset({"circuits", "parameters"})
EXPECTED_CLAIM_BOUNDARY = (
    "This file is emitted evidence, not a trusted input. Quick mode "
    "requires a separately promoted checksum-bound canonical manifest."
)
EXPECTED_40Q_QASM_SCOPE = (
    "40-qubit dense state/leakage/energy allocation intentionally omitted; "
    "exact primitive tests plus checksum-bound parameter and QASM artifacts "
    "define this scope"
)
RESTART_TIMING_CLAIM_BOUNDARY = (
    "Energy-evaluation counts remain comparable. Optimizer wall times retain "
    "the conditions of the attempt that produced each atomic cache entry; "
    "mixed-origin wall-time sums and ratios are provenance only and must not "
    "be interpreted as same-attempt speedup measurements."
)
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


class CandidateAuditFailure(RuntimeError):
    """Raised after recording a fail-closed candidate-audit check."""


class Audit:
    """Collect machine-readable checks while failing at the first unsafe state."""

    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def check(
        self,
        name: str,
        passed: bool,
        *,
        actual: Any = None,
        expected: Any = None,
        fatal: bool = True,
    ) -> bool:
        item: dict[str, Any] = {
            "name": name,
            "status": "PASS" if passed else "FAIL",
        }
        if actual is not None:
            item["actual"] = json_safe(actual)
        if expected is not None:
            item["expected"] = json_safe(expected)
        self.checks.append(item)
        print(f"[{item['status']}] {name}", flush=True)
        if not passed and fatal:
            raise CandidateAuditFailure(name)
        return passed

    @property
    def status(self) -> str:
        return (
            "PASS"
            if self.checks
            and all(item["status"] == "PASS" for item in self.checks)
            else "FAIL"
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def regular_directory_chain(root: Path, *components: str) -> bool:
    """Require each named descendant to be a real directory, never a symlink."""

    current = root
    try:
        mode = current.lstat().st_mode
    except OSError:
        return False
    if stat_module.S_ISLNK(mode) or not stat_module.S_ISDIR(mode):
        return False
    for component in components:
        current = current / component
        try:
            mode = current.lstat().st_mode
        except OSError:
            return False
        if stat_module.S_ISLNK(mode) or not stat_module.S_ISDIR(mode):
            return False
    return True


def validate_output_path(
    output: Path,
    *,
    source_code: Path,
    candidate_directory: Path | None,
) -> Path:
    """Accept an audit destination that cannot contaminate trusted evidence."""

    output = output.expanduser()
    if output.suffix.lower() != ".json":
        raise ValueError("Candidate-audit output must be a .json file")
    resolved = output.resolve()
    if resolved.exists():
        raise FileExistsError(f"Refusing to overwrite audit output: {resolved}")
    if candidate_directory is not None and is_within(
        resolved, candidate_directory
    ):
        raise ValueError("Audit output must remain outside the candidate directory")
    reference = source_code / "reference"
    if is_within(resolved, reference):
        raise ValueError("Audit output must never be written below reference/")
    if is_within(resolved, source_code) and not is_within(
        resolved, source_code / "results"
    ):
        raise ValueError(
            "Audit output inside Source_Code is allowed only below results/"
        )
    return resolved


def _portable_relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError(f"{field} is not a strict portable relative path: {value}")
    return path.as_posix()


def tree_snapshot(root: Path) -> dict[str, Any]:
    """Hash a tree without following symlinks; suitable for mutation checks."""

    if not root.exists() and not root.is_symlink():
        return {"exists": False, "aggregate_sha256": None, "entries": []}
    entries: list[dict[str, Any]] = []
    if root.is_symlink():
        target = os.readlink(root)
        return {
            "exists": True,
            "aggregate_sha256": hashlib.sha256(
                ("symlink\0" + target).encode("utf-8", errors="surrogateescape")
            ).hexdigest(),
            "entries": [
                {"path": ".", "kind": "symlink", "target": target}
            ],
        }
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted(directories):
            path = current_path / name
            mode = path.lstat().st_mode
            if stat_module.S_ISLNK(mode):
                target = os.readlink(path)
                entries.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "kind": "symlink",
                        "target": target,
                    }
                )
            elif not stat_module.S_ISDIR(mode):
                entries.append(
                    {
                        "path": path.relative_to(root).as_posix(),
                        "kind": "special",
                        "mode": stat_module.S_IFMT(mode),
                    }
                )
        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat_module.S_ISLNK(mode):
                entries.append(
                    {
                        "path": relative,
                        "kind": "symlink",
                        "target": os.readlink(path),
                    }
                )
            elif stat_module.S_ISREG(mode):
                entries.append(
                    {
                        "path": relative,
                        "kind": "file",
                        "sha256": sha256_file(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
            else:
                entries.append(
                    {
                        "path": relative,
                        "kind": "special",
                        "mode": stat_module.S_IFMT(mode),
                    }
                )
    entries.sort(key=lambda item: (item["path"], item["kind"]))
    digest = hashlib.sha256(
        (
            json.dumps(entries, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    return {
        "exists": True,
        "aggregate_sha256": digest,
        "entries": entries,
    }


def candidate_tree_manifest(
    audit: Audit, candidate_directory: Path
) -> dict[str, Any]:
    """Require and hash the exact ten-file candidate transport boundary."""

    audit.check(
        "candidate directory exists and is not a symlink",
        candidate_directory.is_dir() and not candidate_directory.is_symlink(),
        actual=str(candidate_directory),
        expected="regular directory",
    )
    files: dict[str, Path] = {}
    directories: set[str] = set()
    symlinks: list[str] = []
    special_nodes: list[str] = []
    for current, child_directories, child_files in os.walk(
        candidate_directory, followlinks=False
    ):
        current_path = Path(current)
        for name in child_directories:
            path = current_path / name
            relative = path.relative_to(candidate_directory).as_posix()
            mode = path.lstat().st_mode
            if stat_module.S_ISLNK(mode):
                symlinks.append(relative)
            elif stat_module.S_ISDIR(mode):
                directories.add(relative)
            else:
                special_nodes.append(relative)
        for name in child_files:
            path = current_path / name
            relative = path.relative_to(candidate_directory).as_posix()
            mode = path.lstat().st_mode
            if stat_module.S_ISLNK(mode):
                symlinks.append(relative)
            elif stat_module.S_ISREG(mode):
                files[relative] = path
            else:
                special_nodes.append(relative)
    audit.check(
        "candidate contains no symlinks",
        not symlinks,
        actual=sorted(symlinks),
        expected=[],
    )
    audit.check(
        "candidate contains no special filesystem nodes",
        not special_nodes,
        actual=sorted(special_nodes),
        expected=[],
    )
    audit.check(
        "candidate directory set is exact",
        directories == set(EXPECTED_CANDIDATE_DIRECTORIES),
        actual=sorted(directories),
        expected=sorted(EXPECTED_CANDIDATE_DIRECTORIES),
    )
    audit.check(
        "candidate file set is exact",
        set(files) == set(EXPECTED_CANDIDATE_FILES),
        actual=sorted(files),
        expected=sorted(EXPECTED_CANDIDATE_FILES),
    )
    zero_byte = sorted(
        relative for relative, path in files.items() if path.stat().st_size <= 0
    )
    audit.check(
        "candidate contains no zero-byte files",
        not zero_byte,
        actual=zero_byte,
        expected=[],
    )

    records = []
    aggregate = hashlib.sha256()
    for relative in sorted(files):
        path = files[relative]
        digest = sha256_file(path)
        size = path.stat().st_size
        records.append(
            {"path": relative, "sha256": digest, "size_bytes": size}
        )
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\n")
    return {
        "algorithm": "sha256-path-hash-size-v1",
        "aggregate_sha256": aggregate.hexdigest(),
        "file_count": len(records),
        "files": records,
    }


def _normalize_artifact_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("candidate_artifacts must be a JSON object")
    files = value.get("files")
    if not isinstance(files, list):
        raise TypeError("candidate_artifacts.files must be a JSON array")
    normalized = {
        "algorithm": value.get("algorithm"),
        "aggregate_sha256": value.get("aggregate_sha256"),
        "file_count": value.get("file_count"),
        "files": sorted(
            [
                {
                    "path": item.get("path"),
                    "sha256": item.get("sha256"),
                    "size_bytes": item.get("size_bytes"),
                }
                for item in files
                if isinstance(item, dict)
            ],
            key=lambda item: str(item["path"]),
        ),
    }
    return normalized


def expected_bootstrap_stage_contracts(
    *,
    source_code: Path,
    invocation_directory: Path,
    environment: Path,
) -> list[dict[str, Any]]:
    """Rebuild the four trusted bootstrap commands and their log paths."""

    from reproduce import _SCRIPT_BOOTSTRAP

    source_code = source_code.resolve()
    invocation_directory = invocation_directory.resolve()
    environment = environment.resolve()
    environment_python = environment / "bin" / "python"

    def isolated_script(
        script: Path, *arguments: str
    ) -> list[str]:
        return [
            str(environment_python),
            "-I",
            "-B",
            "-c",
            _SCRIPT_BOOTSTRAP,
            str(source_code),
            str(source_code / "source"),
            str(script),
            *arguments,
        ]

    commands: tuple[tuple[str, list[str]], ...] = (
        (
            "locked_environment_smoke",
            isolated_script(
                source_code / "verify_environment.py",
                "--smoke",
                "--output",
                str(invocation_directory / "environment.json"),
            ),
        ),
        (
            "dependency_consistency",
            [
                str(environment_python),
                "-I",
                "-B",
                "-m",
                "pip",
                "--isolated",
                "check",
            ],
        ),
        (
            "independent_frozen_reference_audit",
            isolated_script(
                source_code
                / "source"
                / "scripts"
                / "audit_frozen_references.py",
                "--frozen-inputs",
                str(source_code / "frozen_inputs"),
                "--output",
                str(invocation_directory / "frozen_reference_audit.json"),
            ),
        ),
        (
            "full_advanced_candidate_generation",
            isolated_script(
                source_code
                / "source"
                / "scripts"
                / "run_enhanced_release.py",
                "--frozen-inputs",
                str(source_code / "frozen_inputs"),
                "--output",
                str(invocation_directory / "advanced_method"),
            ),
        ),
    )
    return [
        {
            "name": name,
            "argv": argv,
            "log": str(invocation_directory / f"{name}.log"),
        }
        for name, argv in commands
    ]


def validate_bootstrap_status(
    audit: Audit,
    status: dict[str, Any],
    *,
    bootstrap_helper_path: Path,
    bootstrap_status_path: Path,
    source_code: Path,
    environment: Path,
    candidate_directory: Path,
    current_source_identity: dict[str, Any],
    observed_candidate_manifest: dict[str, Any],
) -> None:
    bootstrap_helper_path = bootstrap_helper_path.resolve()
    bootstrap_status_path = bootstrap_status_path.resolve()
    source_code = source_code.resolve()
    environment = environment.resolve()
    candidate_directory = candidate_directory.resolve()
    invocation_directory = bootstrap_status_path.parent
    audit.check(
        "bootstrap schema",
        type(status.get("schema_version")) is int
        and status.get("schema_version") == SCHEMA_VERSION,
        actual=status.get("schema_version"),
        expected=SCHEMA_VERSION,
    )
    helper_regular = (
        bootstrap_helper_path.is_file()
        and not bootstrap_helper_path.is_symlink()
    )
    helper_sha256 = (
        sha256_file(bootstrap_helper_path) if helper_regular else None
    )
    expected_helper = {
        "path": str(bootstrap_helper_path),
        "sha256": helper_sha256,
    }
    audit.check(
        "bootstrap helper path and bytes are explicitly bound",
        helper_regular and status.get("bootstrap_helper") == expected_helper,
        actual={
            "current": expected_helper,
            "recorded": status.get("bootstrap_helper"),
        },
        expected=expected_helper,
    )
    expected_paths = {
        "invocation_id": invocation_directory.name,
        "invocation_directory": str(invocation_directory),
        "output_directory": str(invocation_directory),
        "bootstrap_status": str(bootstrap_status_path),
        "source_code": str(source_code),
        "environment": str(environment),
        "environment_python": str(environment / "bin" / "python"),
        "candidate_directory": str(
            invocation_directory
            / "advanced_method"
            / "canonical_table3_candidate"
        ),
    }
    actual_paths = {
        key: status.get(key) for key in expected_paths
    }
    audit.check(
        "bootstrap invocation and execution paths are exact",
        bootstrap_status_path.is_file()
        and not bootstrap_status_path.is_symlink()
        and (environment / "bin" / "python").is_file()
        and actual_paths == expected_paths
        and candidate_directory
        == (
            invocation_directory
            / "advanced_method"
            / "canonical_table3_candidate"
        ),
        actual={
            **actual_paths,
            "audited_candidate_directory": str(candidate_directory),
        },
        expected=expected_paths,
    )
    audit.check(
        "bootstrap candidate-ready status remains noncertifying",
        status.get("status") == READY_STATUS,
        actual=status.get("status"),
        expected=READY_STATUS,
    )
    audit.check(
        "bootstrap provider/QPU boundary",
        status.get("provider_imported") is False
        and status.get("qpu_contacted") is False,
        actual={
            "provider_imported": status.get("provider_imported"),
            "qpu_contacted": status.get("qpu_contacted"),
        },
        expected={"provider_imported": False, "qpu_contacted": False},
    )
    provider_free_home = invocation_directory / ".provider_free_home"
    expected_access_boundary = {
        "credential_home": str(provider_free_home),
        "xdg_config_home": str(provider_free_home / "xdg-config"),
        "xdg_cache_home": str(provider_free_home / "xdg-cache"),
        "xdg_data_home": str(provider_free_home / "xdg-data"),
        "xdg_state_home": str(provider_free_home / "xdg-state"),
        "python_network_audit_guard": True,
        "aws_metadata_disabled": True,
        "aws_configuration_files": os.devnull,
        "pip_no_index": True,
    }
    access_directories = (
        provider_free_home,
        provider_free_home / "xdg-config",
        provider_free_home / "xdg-cache",
        provider_free_home / "xdg-data",
        provider_free_home / "xdg-state",
    )
    audit.check(
        "bootstrap isolates credentials and denies Python network access",
        status.get("provider_access_boundary")
        == expected_access_boundary
        and all(
            path.is_dir() and not path.is_symlink()
            for path in access_directories
        ),
        actual=status.get("provider_access_boundary"),
        expected=expected_access_boundary,
    )
    stages = status.get("stages")
    audit.check(
        "bootstrap exact four-stage sequence is declared",
        isinstance(stages, list)
        and len(stages) == len(EXPECTED_STAGE_NAMES)
        and tuple(
            stage.get("name")
            for stage in stages
            if isinstance(stage, dict)
        )
        == EXPECTED_STAGE_NAMES,
        actual=(
            [
                stage.get("name")
                for stage in stages
                if isinstance(stage, dict)
            ]
            if isinstance(stages, list)
            else stages
        ),
        expected=list(EXPECTED_STAGE_NAMES),
    )
    assert isinstance(stages, list)
    expected_stages = expected_bootstrap_stage_contracts(
        source_code=source_code,
        invocation_directory=invocation_directory,
        environment=environment,
    )
    for stage, expected_stage in zip(stages, expected_stages, strict=True):
        audit.check(
            f"bootstrap {expected_stage['name']} stage is an object",
            isinstance(stage, dict),
            actual=type(stage).__name__,
            expected="object",
        )
        assert isinstance(stage, dict)
        log_path = Path(expected_stage["log"])
        log_regular = log_path.is_file() and not log_path.is_symlink()
        log_sha256 = sha256_file(log_path) if log_regular else None
        log_size = log_path.stat().st_size if log_regular else None
        stage_pass = (
            stage.get("name") == expected_stage["name"]
            and stage.get("status") == "PASS"
            and type(stage.get("returncode")) is int
            and stage.get("returncode") == 0
            and stage.get("argv") == expected_stage["argv"]
            and stage.get("log") == expected_stage["log"]
            and log_regular
            and stage.get("log_sha256") == log_sha256
            and type(stage.get("log_size_bytes")) is int
            and stage.get("log_size_bytes") == log_size
        )
        audit.check(
            f"bootstrap {expected_stage['name']} argv and final log bind",
            stage_pass,
            actual={
                "status": stage.get("status"),
                "returncode": stage.get("returncode"),
                "argv": stage.get("argv"),
                "log": stage.get("log"),
                "recorded_log_sha256": stage.get("log_sha256"),
                "current_log_sha256": log_sha256,
                "recorded_log_size_bytes": stage.get("log_size_bytes"),
                "current_log_size_bytes": log_size,
            },
            expected={
                **expected_stage,
                "status": "PASS",
                "returncode": 0,
                "log_sha256": log_sha256,
                "log_size_bytes": log_size,
            },
        )
    before = status.get("source_identity_before")
    after = status.get("source_identity_after")
    audit.check(
        "bootstrap source identity is unchanged and current",
        before == after == current_source_identity,
        actual={"before": before, "after": after},
        expected=current_source_identity,
    )
    recorded_manifest = _normalize_artifact_manifest(
        status.get("candidate_artifacts")
    )
    audit.check(
        "downloaded candidate matches bootstrap artifact manifest",
        recorded_manifest == observed_candidate_manifest,
        actual=observed_candidate_manifest,
        expected=recorded_manifest,
    )


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
    single_cross_angles: Sequence[Any],
    double_cross_angles: Sequence[Any],
) -> float:
    singles = np.asarray(single_cross_angles, dtype=np.float64).reshape(-1)
    doubles = np.asarray(double_cross_angles, dtype=np.float64).reshape(-1)
    total_u = sum(_u_single(value) for value in singles)
    total_u += sum(_u_pair_double(value) for value in doubles)
    return 2.0 * math.exp(2.0 * total_u) - 1.0


def close(actual: float, expected: float, tolerance: float) -> bool:
    return math.isfinite(actual) and abs(actual - expected) <= tolerance


def validate_candidate_manifest_structure(
    audit: Audit,
    manifest: dict[str, Any],
    *,
    compilation_protocol: dict[str, Any],
    expected_metrics: dict[str, Any],
) -> None:
    audit.check(
        "candidate manifest schema",
        int(manifest.get("schema_version", -1)) == SCHEMA_VERSION,
        actual=manifest.get("schema_version"),
        expected=SCHEMA_VERSION,
    )
    audit.check(
        "candidate status is explicitly noncanonical",
        manifest.get("status") == CANDIDATE_STATUS,
        actual=manifest.get("status"),
        expected=CANDIDATE_STATUS,
    )
    audit.check(
        "candidate seed and compilation protocol",
        int(manifest.get("seed", -1)) == 3047
        and manifest.get("compilation_protocol") == compilation_protocol
        and manifest.get("claim_boundary") == EXPECTED_CLAIM_BOUNDARY,
        actual={
            "seed": manifest.get("seed"),
            "compilation_protocol": manifest.get("compilation_protocol"),
            "claim_boundary": manifest.get("claim_boundary"),
        },
        expected={
            "seed": 3047,
            "compilation_protocol": compilation_protocol,
            "claim_boundary": EXPECTED_CLAIM_BOUNDARY,
        },
    )
    cases = manifest.get("cases")
    actual_labels = set(cases) if isinstance(cases, dict) else set()
    audit.check(
        "candidate exact Table-3 case set",
        isinstance(cases, dict) and actual_labels == set(EXPECTED_CASES),
        actual=sorted(actual_labels),
        expected=sorted(EXPECTED_CASES),
    )

    resource_targets = expected_metrics.get("structured_resources")
    topology_targets = expected_metrics.get("adaptive_topology")
    audit.check(
        "expected metrics contain every Table-3 claim",
        isinstance(resource_targets, dict)
        and isinstance(topology_targets, dict)
        and set(EXPECTED_CASES) <= set(resource_targets)
        and set(EXPECTED_CASES) <= set(topology_targets),
        actual={
            "resource_labels": (
                sorted(resource_targets) if isinstance(resource_targets, dict) else []
            ),
            "topology_labels": (
                sorted(topology_targets) if isinstance(topology_targets, dict) else []
            ),
        },
        expected=sorted(EXPECTED_CASES),
    )
    assert isinstance(cases, dict)
    assert isinstance(resource_targets, dict)
    assert isinstance(topology_targets, dict)

    for label, (molecule, norb, n_qubits) in EXPECTED_CASES.items():
        case = cases[label]
        audit.check(
            f"{label} candidate record is an object",
            isinstance(case, dict),
            actual=type(case).__name__,
            expected="object",
        )
        assert isinstance(case, dict)
        target = resource_targets[label]
        selected = topology_targets[label]["selected_left"]
        audit.check(
            f"{label} molecular/topology identity",
            case.get("label") == label
            and case.get("molecule") == molecule
            and int(case.get("norb", -1)) == norb
            and int(case.get("n_qubits", -1)) == n_qubits
            and int(case.get("seed", -1)) == 3047
            and case.get("selected_left_block") == selected
            and case.get("compilation_protocol") == compilation_protocol,
            actual={
                key: case.get(key)
                for key in (
                    "label",
                    "molecule",
                    "norb",
                    "n_qubits",
                    "seed",
                    "selected_left_block",
                    "compilation_protocol",
                )
            },
            expected={
                "label": label,
                "molecule": molecule,
                "norb": norb,
                "n_qubits": n_qubits,
                "seed": 3047,
                "selected_left_block": selected,
                "compilation_protocol": compilation_protocol,
            },
        )
        ansatz = case.get("ansatz")
        audit.check(
            f"{label} partitioned ansatz declaration",
            isinstance(ansatz, dict)
            and ansatz.get("topology") == "partitioned"
            and ansatz.get("partition_override") == selected
            and ansatz.get("pairs_override") is None
            and close(float(ansatz.get("phi_max", float("nan"))), 15.0, 1e-12),
            actual=ansatz,
            expected={
                "topology": "partitioned",
                "partition_override": selected,
                "pairs_override": None,
                "phi_max": 15.0,
            },
        )
        generic = case.get("generic_qasm")
        structured = case.get("structured_qasm")
        audit.check(
            f"{label} compatibility aliases equal artifact-bound records",
            isinstance(generic, dict)
            and isinstance(structured, dict)
            and case.get("legacy_generic_unitary") == generic
            and case.get("structured_exact_pauli_network") == structured,
            actual={
                "legacy_alias_matches": (
                    case.get("legacy_generic_unitary") == generic
                ),
                "structured_alias_matches": (
                    case.get("structured_exact_pauli_network") == structured
                ),
            },
            expected={
                "legacy_alias_matches": True,
                "structured_alias_matches": True,
            },
        )
        assert isinstance(generic, dict)
        assert isinstance(structured, dict)
        for arm_name, arm, expected_file, expected_cx, expected_depth in (
            (
                "generic_qasm",
                generic,
                f"circuits/{label}_generic.qasm",
                target["legacy_cx"],
                target["legacy_depth"],
            ),
            (
                "structured_qasm",
                structured,
                f"circuits/{label}_structured.qasm",
                target["structured_cx"],
                target["structured_depth"],
            ),
        ):
            actual_file = _portable_relative_path(
                arm.get("qasm_file"), field=f"{label}.{arm_name}.qasm_file"
            )
            audit.check(
                f"{label} {arm_name} declared binding",
                actual_file == expected_file
                and arm.get("basis_gates")
                == compilation_protocol["basis_gates"]
                and int(arm.get("optimization_level", -1)) == 3
                and int(arm.get("seed_transpiler", -1)) == 3047
                and arm.get("connectivity") == "all-to-all"
                and arm.get("scope") == compilation_protocol["scope"]
                and int(arm.get("logical_qubits", -1)) == n_qubits
                and int(arm.get("cx", -1)) == int(expected_cx)
                and int(arm.get("depth", -1)) == int(expected_depth)
                and arm.get("derived_from_reloaded_qasm") is True
                and arm.get("compiled_once_before_serialization") is True,
                actual=arm,
                expected={
                    "qasm_file": expected_file,
                    "basis_gates": compilation_protocol["basis_gates"],
                    "optimization_level": 3,
                    "seed_transpiler": 3047,
                    "connectivity": "all-to-all",
                    "scope": compilation_protocol["scope"],
                    "logical_qubits": n_qubits,
                    "cx": expected_cx,
                    "depth": expected_depth,
                    "derived_from_reloaded_qasm": True,
                    "compiled_once_before_serialization": True,
                },
            )
        parameter = case.get("parameter_artifact")
        expected_parameter = f"parameters/{label}_seed-3047.npz"
        audit.check(
            f"{label} parameter declaration",
            isinstance(parameter, dict)
            and _portable_relative_path(
                parameter.get("file"), field=f"{label}.parameter_artifact.file"
            )
            == expected_parameter
            and parameter.get("array_replay_exact") is True
            and isinstance(parameter.get("array_names"), list)
            and bool(parameter.get("array_names"))
            and isinstance(parameter.get("array_shapes"), dict),
            actual=parameter,
            expected={
                "file": expected_parameter,
                "array_replay_exact": True,
                "array_names": "nonempty list",
                "array_shapes": "object",
            },
        )
        audit.check(
            f"{label} candidate-only reference mode and claim boundary",
            case.get("reference_mode") == "FULL_RUN_CANDIDATE"
            and case.get("artifact_root") == "canonical_table3_candidate"
            and case.get("device_native") is False
            and case.get("qasm_file") == structured.get("qasm_file")
            and case.get("qasm_sha256") == structured.get("qasm_sha256"),
            actual={
                "reference_mode": case.get("reference_mode"),
                "artifact_root": case.get("artifact_root"),
                "device_native": case.get("device_native"),
                "qasm_file": case.get("qasm_file"),
                "qasm_sha256": case.get("qasm_sha256"),
            },
            expected={
                "reference_mode": "FULL_RUN_CANDIDATE",
                "artifact_root": "canonical_table3_candidate",
                "device_native": False,
                "qasm_file": structured.get("qasm_file"),
                "qasm_sha256": structured.get("qasm_sha256"),
            },
        )
        generic_cx = int(generic["cx"])
        structured_cx = int(structured["cx"])
        generic_depth = int(generic["depth"])
        structured_depth = int(structured["depth"])
        audit.check(
            f"{label} resource reductions rederive",
            close(
                float(case.get("cx_reduction_fraction", float("nan"))),
                1.0 - structured_cx / max(1, generic_cx),
                1e-15,
            )
            and close(
                float(case.get("depth_reduction_fraction", float("nan"))),
                1.0 - structured_depth / max(1, generic_depth),
                1e-15,
            ),
            actual={
                "cx_reduction_fraction": case.get("cx_reduction_fraction"),
                "depth_reduction_fraction": case.get(
                    "depth_reduction_fraction"
                ),
            },
            expected={
                "cx_reduction_fraction": 1.0
                - structured_cx / max(1, generic_cx),
                "depth_reduction_fraction": 1.0
                - structured_depth / max(1, generic_depth),
            },
        )
        if n_qubits <= 12:
            dense = case.get("dense_qasm_audit")
            fidelity = case.get("state_fidelity")
            audit.check(
                f"{label} declared dense fidelity claim",
                isinstance(dense, dict)
                and fidelity is not None
                and dense.get("scope")
                == "dense validation of the exact reloaded QASM artifacts"
                and case.get("equivalence_scope") == dense.get("scope")
                and case.get("qasm_validation_scope") == dense.get("scope")
                and close(
                    float(fidelity),
                    float(dense["generic_vs_structured_state_fidelity"]),
                    1e-15,
                )
                and float(fidelity) >= float(target["minimum_fidelity"]),
                actual={
                    "state_fidelity": fidelity,
                    "dense_qasm_audit": dense,
                },
                expected={
                    "minimum_fidelity": target["minimum_fidelity"],
                    "state_fidelity_equals_dense_generic_vs_structured": True,
                    "scope": (
                        "dense validation of the exact reloaded QASM artifacts"
                    ),
                },
            )
        else:
            expected_scope = target["fidelity_scope"]
            audit.check(
                f"{label} declared compositional scope",
                case.get("state_fidelity") is None
                and case.get("dense_qasm_audit") is None
                and case.get("equivalence_scope") == expected_scope
                and case.get("qasm_validation_scope")
                == EXPECTED_40Q_QASM_SCOPE,
                actual={
                    "state_fidelity": case.get("state_fidelity"),
                    "dense_qasm_audit": case.get("dense_qasm_audit"),
                    "equivalence_scope": case.get("equivalence_scope"),
                    "qasm_validation_scope": case.get(
                        "qasm_validation_scope"
                    ),
                },
                expected={
                    "state_fidelity": None,
                    "dense_qasm_audit": None,
                    "equivalence_scope": expected_scope,
                    "qasm_validation_scope": EXPECTED_40Q_QASM_SCOPE,
                },
            )


def inspect_parameter_npz(
    audit: Audit,
    *,
    path: Path,
    record: dict[str, Any],
    label: str,
) -> dict[str, np.ndarray]:
    audit.check(
        f"{label} parameter checksum",
        path.is_file() and sha256_file(path) == record.get("sha256"),
        actual=sha256_file(path) if path.is_file() else None,
        expected=record.get("sha256"),
    )
    declared_names = record.get("array_names")
    declared_shapes = record.get("array_shapes")
    audit.check(
        f"{label} parameter metadata types",
        isinstance(declared_names, list)
        and bool(declared_names)
        and len(set(declared_names)) == len(declared_names)
        and isinstance(declared_shapes, dict)
        and set(declared_shapes) == set(declared_names),
        actual={
            "array_names": declared_names,
            "array_shapes": declared_shapes,
        },
        expected="unique nonempty names with one declared shape each",
    )
    assert isinstance(declared_names, list)
    assert isinstance(declared_shapes, dict)

    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        member_names = [member.filename for member in members]
        expected_members = {f"{name}.npy" for name in declared_names}
        safe_members = all(
            not PurePosixPath(name).is_absolute()
            and "." not in PurePosixPath(name).parts
            and ".." not in PurePosixPath(name).parts
            and PurePosixPath(name).name == name
            for name in member_names
        )
        audit.check(
            f"{label} NPZ member boundary",
            len(member_names) == len(set(member_names))
            and set(member_names) == expected_members
            and safe_members
            and all(not member.is_dir() and member.file_size > 0 for member in members),
            actual=member_names,
            expected=sorted(expected_members),
        )

    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            audit.check(
                f"{label} NPZ array-name set",
                set(archive.files) == set(declared_names),
                actual=sorted(archive.files),
                expected=sorted(declared_names),
            )
            for name in declared_names:
                value = np.asarray(archive[name])
                expected_shape = tuple(int(item) for item in declared_shapes[name])
                audit.check(
                    f"{label} {name} float64 finite declared shape",
                    value.dtype == np.dtype(np.float64)
                    and value.shape == expected_shape
                    and value.size > 0
                    and bool(np.isfinite(value).all()),
                    actual={
                        "dtype": str(value.dtype),
                        "shape": list(value.shape),
                        "size": int(value.size),
                        "finite": bool(np.isfinite(value).all()),
                    },
                    expected={
                        "dtype": "float64",
                        "shape": list(expected_shape),
                        "size": "positive",
                        "finite": True,
                    },
                )
                arrays[name] = value.copy()
    except ValueError as error:
        audit.check(
            f"{label} NPZ loads with allow_pickle=False",
            False,
            actual=f"{type(error).__name__}: {error}",
            expected="non-object NumPy arrays",
        )
    return arrays


def validate_consumed_parameter_contract(
    audit: Audit,
    *,
    label: str,
    parameter_record: Mapping[str, Any],
    artifact_arrays: Mapping[str, Any],
    loaded_params: Mapping[str, Any],
    initialized_params: Mapping[str, Any],
) -> None:
    """Reject parameter bytes that the reconstructed circuit would ignore."""

    consumed_shapes = {
        str(name): list(value.shape)
        for name, value in sorted(initialized_params.items())
    }
    artifact_shapes = {
        str(name): list(value.shape)
        for name, value in sorted(artifact_arrays.items())
    }
    loaded_shapes = {
        str(name): list(value.shape)
        for name, value in sorted(loaded_params.items())
    }
    consumed_names = set(consumed_shapes)
    audit.check(
        f"{label} parameter artifact exactly matches consumed circuit inputs",
        consumed_names in ({"singles"}, {"singles", "doubles"})
        and artifact_shapes == consumed_shapes
        and loaded_shapes == consumed_shapes
        and parameter_record.get("array_names")
        == sorted(consumed_shapes)
        and parameter_record.get("array_shapes") == consumed_shapes,
        actual={
            "artifact": artifact_shapes,
            "loaded": loaded_shapes,
            "declared_names": parameter_record.get("array_names"),
            "declared_shapes": parameter_record.get("array_shapes"),
        },
        expected={
            "consumed_names": sorted(consumed_shapes),
            "consumed_shapes": consumed_shapes,
            "initialization_seed": 3047,
        },
    )


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


def inspect_qasm_arm(
    audit: Audit,
    *,
    path: Path,
    record: dict[str, Any],
    label: str,
    arm_name: str,
    qasm_loader: Callable[[str], Any] | None = None,
) -> Any:
    exists = path.is_file()
    actual_hash = sha256_file(path) if exists else None
    audit.check(
        f"{label} {arm_name} exact QASM hash",
        exists and actual_hash == record.get("qasm_sha256"),
        actual=actual_hash,
        expected=record.get("qasm_sha256"),
    )
    if qasm_loader is None:
        qasm_loader = strict_load_qasm2_legacy_sx
    circuit = qasm_loader(str(path))
    operations = {
        str(name): int(count) for name, count in circuit.count_ops().items()
    }
    expected_basis = {"rz", "sx", "x", "cx"}
    resource_match = (
        int(circuit.num_qubits) == int(record["logical_qubits"])
        and int(circuit.depth()) == int(record["depth"])
        and int(circuit.size()) == int(record["size"])
        and operations == record["operations"]
        and int(operations.get("cx", 0)) == int(record["cx"])
        and not (set(operations) - expected_basis)
        and record.get("basis_gates") == ["rz", "sx", "x", "cx"]
        and int(record.get("optimization_level", -1)) == 3
        and int(record.get("seed_transpiler", -1)) == 3047
        and record.get("connectivity") == "all-to-all"
    )
    audit.check(
        f"{label} {arm_name} resources and canonical basis rederive",
        resource_match,
        actual={
            "logical_qubits": circuit.num_qubits,
            "depth": circuit.depth(),
            "size": circuit.size(),
            "operations": operations,
        },
        expected={
            "logical_qubits": record["logical_qubits"],
            "depth": record["depth"],
            "size": record["size"],
            "operations": record["operations"],
            "basis_gates": ["rz", "sx", "x", "cx"],
        },
    )
    return circuit


def validate_dense_record(
    audit: Audit,
    *,
    label: str,
    n_qubits: int,
    recorded: Any,
    replayed: Any,
    outer_replay_energy: float,
) -> None:
    if n_qubits > 12:
        audit.check(
            f"{label} explicit dense-allocation boundary",
            recorded is None
            and replayed is None,
            actual={"recorded": recorded, "replayed": replayed},
            expected=None,
        )
        return
    expected_scope = "dense validation of the exact reloaded QASM artifacts"
    numeric_fields = (
        "generic_vs_structured_state_fidelity",
        "generic_vs_sector_state_fidelity",
        "structured_vs_sector_state_fidelity",
        "generic_sector_leakage_probability",
        "structured_sector_leakage_probability",
        "generic_energy_hartree",
        "structured_energy_hartree",
        "sector_replay_energy_hartree",
        "generic_energy_difference_hartree",
        "structured_energy_difference_hartree",
    )
    required = ("scope", *numeric_fields)
    audit.check(
        f"{label} dense-QASM audit records are complete",
        isinstance(recorded, dict)
        and isinstance(replayed, dict)
        and set(recorded) == set(required)
        and set(replayed) == set(required),
        actual={"recorded": recorded, "replayed": replayed},
        expected=list(required),
    )
    assert isinstance(recorded, dict)
    assert isinstance(replayed, dict)
    finite_values = all(
        type(item[key]) in (int, float)
        and not isinstance(item[key], bool)
        and math.isfinite(float(item[key]))
        for item in (recorded, replayed)
        for key in numeric_fields
    ) and math.isfinite(float(outer_replay_energy))
    audit.check(
        f"{label} dense-QASM scope and numeric fields are trusted types",
        recorded["scope"] == expected_scope
        and replayed["scope"] == expected_scope
        and finite_values,
        actual={
            "recorded_scope": recorded["scope"],
            "replayed_scope": replayed["scope"],
            "finite_values": finite_values,
        },
        expected={"scope": expected_scope, "finite_values": True},
    )
    fidelity_keys = numeric_fields[:3]
    leakage_keys = numeric_fields[3:5]
    absolute_energy_keys = numeric_fields[5:8]
    difference_keys = numeric_fields[8:]
    thresholds_pass = (
        all(
            1.0 - 1e-10 <= float(replayed[key]) <= 1.0 + 1e-12
            for key in fidelity_keys
        )
        and all(
            0.0 <= float(replayed[key]) <= 1e-10
            for key in leakage_keys
        )
        and all(
            0.0 <= float(replayed[key]) <= 1e-9
            for key in difference_keys
        )
    )
    agreement = all(
        close(float(recorded[key]), float(replayed[key]), 1e-12)
        for key in numeric_fields
    )
    rederived = all(
        close(
            float(item["generic_energy_difference_hartree"]),
            abs(
                float(item["generic_energy_hartree"])
                - float(item["sector_replay_energy_hartree"])
            ),
            1e-15,
        )
        and close(
            float(item["structured_energy_difference_hartree"]),
            abs(
                float(item["structured_energy_hartree"])
                - float(item["sector_replay_energy_hartree"])
            ),
            1e-15,
        )
        and close(
            float(item["sector_replay_energy_hartree"]),
            float(outer_replay_energy),
            1e-12,
        )
        for item in (recorded, replayed)
    )
    audit.check(
        f"{label} dense state/leakage/energy replay",
        finite_values
        and thresholds_pass
        and agreement
        and rederived
        and all(
            math.isfinite(float(replayed[key]))
            for key in absolute_energy_keys
        ),
        actual={
            "recorded": recorded,
            "replayed": replayed,
            "outer_replay_energy_hartree": outer_replay_energy,
            "agreement": agreement,
            "rederived": rederived,
        },
        expected=(
            "all 11 emitted fields are bound; candidate/replay numerics agree "
            "within 1e-12; both energy differences rederive; the sector "
            "energy equals the outer replay; fidelity >= 1-1e-10, leakage "
            "<= 1e-10, and |dE| <= 1e-9 Ha"
        ),
    )


def replay_candidate_case(
    audit: Audit,
    *,
    source_code: Path,
    candidate_directory: Path,
    label: str,
    case: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild one case from NPZ/frozen input and compare a fresh compilation."""

    source = source_code / "source"
    if str(source_code) not in sys.path:
        sys.path.insert(0, str(source_code))
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    import torch

    from givens40.canonical_resources import (
        bind_table3_circuits,
        bound_record_differences,
        load_parameter_artifact,
    )
    from givens40.energy import make_energy_fn
    from givens40.frozen_problem import load_named_problem
    from givens40.runner import AnsatzConfig, SectorCircuit

    molecule, norb, n_qubits = EXPECTED_CASES[label]
    parameter_record = case["parameter_artifact"]
    parameter_path = candidate_directory / parameter_record["file"]
    artifact_arrays = inspect_parameter_npz(
        audit,
        path=parameter_path,
        record=parameter_record,
        label=label,
    )
    params = load_parameter_artifact(
        parameter_path, str(parameter_record["sha256"])
    )

    config = AnsatzConfig(**case["ansatz"])
    problem = load_named_problem(
        source_code / "frozen_inputs", molecule, norb
    )
    circuit = SectorCircuit(problem, config)
    selected = [int(item) for item in case["selected_left_block"]]
    audit.check(
        f"{label} reconstructed frozen problem and topology",
        int(problem.n_qubits) == n_qubits
        and int(problem.norb) == norb
        and [int(item) for item in problem.nelec]
        == [int(item) for item in case["nelec_alpha_beta"]]
        and circuit.topo.left_block == selected
        and list(config.partition_override or []) == selected
        and config.topology == "partitioned",
        actual={
            "n_qubits": problem.n_qubits,
            "norb": problem.norb,
            "nelec_alpha_beta": problem.nelec,
            "left_block": circuit.topo.left_block,
            "config_topology": config.topology,
            "partition_override": config.partition_override,
        },
        expected={
            "n_qubits": n_qubits,
            "norb": norb,
            "nelec_alpha_beta": case["nelec_alpha_beta"],
            "left_block": selected,
            "config_topology": "partitioned",
            "partition_override": selected,
        },
    )

    validate_consumed_parameter_contract(
        audit,
        label=label,
        parameter_record=parameter_record,
        artifact_arrays=artifact_arrays,
        loaded_params=params,
        initialized_params=circuit.init_params(3047),
    )

    hdiag = problem.hdiag() if config.init_state == "diag" else None
    energy_fn, _ = make_energy_fn(problem)
    with torch.no_grad():
        state = circuit.forward(params, hdiag)
        energy = float(energy_fn(state))
    declared_energy = float(case["sector_replay_energy_hartree"])
    audit.check(
        f"{label} parameter energy replay",
        close(energy, declared_energy, 1e-10),
        actual=energy,
        expected={"value": declared_energy, "absolute_tolerance": 1e-10},
    )

    actual_singles, actual_doubles = circuit.cross_angle_arrays(params)
    cutting = case["cutting_accounting"]
    recorded_singles = np.asarray(
        cutting["single_cross_angles"], dtype=np.float64
    ).reshape(-1)
    recorded_doubles = np.asarray(
        cutting["double_cross_angles"], dtype=np.float64
    ).reshape(-1)
    expected_single_count = (
        2 * int(config.layers) * int(len(circuit.cross_idx))
    )
    expected_double_count = (
        int(config.layers) * int(len(circuit.cross_idx))
        if "d" in config.gates and "d" in config.cross_gates
        else 0
    )
    audit.check(
        f"{label} effective cross-angle arrays replay exactly",
        np.array_equal(actual_singles.reshape(-1), recorded_singles)
        and np.array_equal(actual_doubles.reshape(-1), recorded_doubles)
        and len(recorded_singles) == expected_single_count
        and len(recorded_doubles) == expected_double_count
        and expected_single_count > 0,
        actual={
            "single_count": len(recorded_singles),
            "double_count": len(recorded_doubles),
            "single_exact": np.array_equal(
                actual_singles.reshape(-1), recorded_singles
            ),
            "double_exact": np.array_equal(
                actual_doubles.reshape(-1), recorded_doubles
            ),
        },
        expected={
            "single_count": expected_single_count,
            "double_count": expected_double_count,
            "single_exact": True,
            "double_exact": True,
        },
    )
    phi = recompute_phi(recorded_singles, recorded_doubles)
    audit.check(
        f"{label} semantic Phi independently recomputes within budget",
        close(phi, float(cutting["reported_phi"]), 1e-12)
        and close(phi, float(cutting["recomputed_phi"]), 1e-12)
        and close(float(cutting["absolute_difference"]), 0.0, 1e-12)
        and phi <= float(config.phi_max) + 1e-12,
        actual=phi,
        expected={
            "reported_phi": cutting["reported_phi"],
            "recomputed_phi": cutting["recomputed_phi"],
            "absolute_difference": 0.0,
            "phi_max": config.phi_max,
            "absolute_tolerance": 1e-12,
        },
    )

    generic_path = candidate_directory / case["generic_qasm"]["qasm_file"]
    structured_path = (
        candidate_directory / case["structured_qasm"]["qasm_file"]
    )
    inspect_qasm_arm(
        audit,
        path=generic_path,
        record=case["generic_qasm"],
        label=label,
        arm_name="generic_qasm",
    )
    inspect_qasm_arm(
        audit,
        path=structured_path,
        record=case["structured_qasm"],
        label=label,
        arm_name="structured_qasm",
    )

    with tempfile.TemporaryDirectory(prefix="table3-candidate-audit-") as temp:
        replay_root = Path(temp)
        replayed = bind_table3_circuits(
            label,
            circuit,
            params,
            energy,
            replay_root,
            relative_root=replay_root,
        )
    differences = bound_record_differences(replayed, case)
    audit.check(
        f"{label} fresh parameter-to-QASM binding is exact",
        not differences,
        actual=differences,
        expected=[],
    )
    replay_cutting = replayed["cutting_accounting"]
    audit.check(
        f"{label} fresh binding cross angles and Phi agree",
        np.array_equal(
            np.asarray(replay_cutting["single_cross_angles"]),
            recorded_singles,
        )
        and np.array_equal(
            np.asarray(replay_cutting["double_cross_angles"]),
            recorded_doubles,
        )
        and close(
            float(replay_cutting["recomputed_phi"]),
            float(cutting["recomputed_phi"]),
            1e-12,
        ),
        actual=replay_cutting,
        expected=cutting,
    )
    validate_dense_record(
        audit,
        label=label,
        n_qubits=n_qubits,
        recorded=case.get("dense_qasm_audit"),
        replayed=replayed.get("dense_qasm_audit"),
        outer_replay_energy=energy,
    )
    if n_qubits > 12:
        expected_scope = (
            "compositional exact primitive tests; no dense 40q state allocated"
        )
        audit.check(
            f"{label} compositional exactness scope",
            case.get("equivalence_scope") == expected_scope,
            actual=case.get("equivalence_scope"),
            expected=expected_scope,
        )

    return {
        "energy_hartree": energy,
        "phi": phi,
        "selected_left_block": selected,
        "generic": {
            "sha256": case["generic_qasm"]["qasm_sha256"],
            "cx": case["generic_qasm"]["cx"],
            "depth": case["generic_qasm"]["depth"],
            "size": case["generic_qasm"]["size"],
        },
        "structured": {
            "sha256": case["structured_qasm"]["qasm_sha256"],
            "cx": case["structured_qasm"]["cx"],
            "depth": case["structured_qasm"]["depth"],
            "size": case["structured_qasm"]["size"],
        },
    }


def validate_frozen_reference_certificate(
    audit: Audit,
    *,
    source_code: Path,
    invocation_directory: Path,
) -> None:
    from givens40.reference_audit import DEFAULT_TOLERANCES, array_sha256

    trusted_tolerances = {
        str(key): float(value)
        for key, value in DEFAULT_TOLERANCES.items()
    }
    manifest_path = source_code / "frozen_inputs" / "MANIFEST.json"
    manifest = load_json_object(manifest_path)
    expected_problems = manifest.get("problems")
    audit.check(
        "frozen manifest schema and exact ten-problem set",
        type(manifest.get("schema_version")) is int
        and manifest.get("schema_version") == 1
        and isinstance(expected_problems, dict)
        and len(expected_problems) == 10
        and all(
            isinstance(problem, dict)
            and type(problem.get("schema_version")) is int
            and problem.get("schema_version") == 1
            for problem in (
                expected_problems.values()
                if isinstance(expected_problems, dict)
                else ()
            )
        ),
        actual=(
            {
                "manifest_schema_version": manifest.get("schema_version"),
                "problem_names": sorted(expected_problems),
                "problem_schema_versions": {
                    name: problem.get("schema_version")
                    for name, problem in expected_problems.items()
                    if isinstance(problem, dict)
                },
            }
            if isinstance(expected_problems, dict)
            else type(expected_problems).__name__
        ),
        expected={
            "manifest_schema_version": 1,
            "problem_count": 10,
            "problem_schema_version": 1,
        },
    )
    assert isinstance(expected_problems, dict)
    certificate_path = (
        invocation_directory / "frozen_reference_audit.json"
    )
    certificate = load_json_object(certificate_path)
    tolerances = certificate.get("tolerances")
    valid_tolerances = (
        isinstance(tolerances, dict)
        and set(tolerances) == set(trusted_tolerances)
        and all(
            type(tolerances[key]) in (int, float)
            and not isinstance(tolerances[key], bool)
            and math.isfinite(float(tolerances[key]))
            and float(tolerances[key]) == trusted_tolerances[key]
            for key in trusted_tolerances
        )
    )
    audit.check(
        "independent frozen-reference schema and trusted tolerances",
        type(certificate.get("schema_version")) is int
        and certificate.get("schema_version") == 1
        and valid_tolerances,
        actual={
            "schema_version": certificate.get("schema_version"),
            "tolerances": tolerances,
        },
        expected={
            "schema_version": 1,
            "tolerances": trusted_tolerances,
        },
    )
    assert isinstance(tolerances, dict)
    problems = certificate.get("problems")
    expected_names = set(expected_problems)
    actual_names = set(problems) if isinstance(problems, dict) else set()
    audit.check(
        "independent frozen-reference certificate summary",
        certificate.get("status") == "PASS"
        and certificate.get("manifest_sha256") == sha256_file(manifest_path)
        and type(certificate.get("problem_count")) is int
        and certificate.get("problem_count") == 10
        and type(certificate.get("passed")) is int
        and certificate.get("passed") == 10
        and type(certificate.get("failed")) is int
        and certificate.get("failed") == 0
        and isinstance(problems, dict)
        and actual_names == expected_names,
        actual={
            "status": certificate.get("status"),
            "manifest_sha256": certificate.get("manifest_sha256"),
            "problem_count": certificate.get("problem_count"),
            "passed": certificate.get("passed"),
            "failed": certificate.get("failed"),
            "problem_names": sorted(actual_names),
        },
        expected={
            "status": "PASS",
            "manifest_sha256": sha256_file(manifest_path),
            "problem_count": 10,
            "passed": 10,
            "failed": 0,
            "problem_names": sorted(expected_names),
        },
    )
    assert isinstance(problems, dict)
    for name in sorted(expected_names):
        expected = expected_problems[name]
        record = problems[name]
        checks = record.get("checks")
        norb = int(expected["norb"])
        nelec = tuple(int(item) for item in expected["nelec_alpha_beta"])
        dimension = math.comb(norb, nelec[0]) * math.comb(norb, nelec[1])
        expected_active_hash = (
            expected.get("array_sha256", {}).get("mo_coeff_active")
            if isinstance(expected.get("array_sha256"), dict)
            else None
        )
        declared_ncore = int(
            expected.get("source", {}).get("frozen_core_orbitals", -1)
        )
        bundle_relative = _portable_relative_path(
            expected.get("bundle_file"),
            field=f"{name}.bundle_file",
        )
        audit.check(
            f"{name} frozen bundle path is one direct file",
            len(PurePosixPath(bundle_relative).parts) == 1,
            actual=bundle_relative,
            expected="one portable filename",
        )
        bundle_path = source_code / "frozen_inputs" / bundle_relative
        bundle_regular = (
            bundle_path.is_file() and not bundle_path.is_symlink()
        )
        bundle_sha256 = (
            sha256_file(bundle_path) if bundle_regular else None
        )
        bundle_scalars: dict[str, float] = {}
        bundle_occupations = np.asarray([], dtype=np.float64)
        bundle_nelec = np.asarray([], dtype=np.int64)
        if bundle_regular:
            with np.load(bundle_path, allow_pickle=False) as archive:
                bundle_scalars = {
                    key: float(np.asarray(archive[key]).item())
                    for key in ("e_hf", "e_casci", "ecore")
                }
                bundle_occupations = np.asarray(
                    archive["mo_occ_active"]
                ).copy()
                bundle_nelec = np.asarray(archive["nelec"]).copy()
        bundle_record = record.get("bundle")
        manifest_scalars = {
            "e_hf": expected.get("rhf_energy_hartree"),
            "e_casci": expected.get("casci_energy_hartree"),
            "ecore": expected.get("ecore_hartree"),
        }
        occupation_hash = (
            array_sha256(bundle_occupations)
            if bundle_occupations.size
            else None
        )
        expected_occupation_hash = (
            expected.get("array_sha256", {}).get("mo_occ_active")
            if isinstance(expected.get("array_sha256"), dict)
            else None
        )
        bundle_ok = (
            bundle_regular
            and isinstance(bundle_record, dict)
            and isinstance(expected.get("bundle_sha256"), str)
            and bundle_sha256 == expected.get("bundle_sha256")
            and bundle_record.get("bundle_file") == bundle_relative
            and bundle_record.get("bundle_sha256") == bundle_sha256
            and bundle_record.get("bundle_checksum_match") is True
            and set(bundle_scalars) == {"e_hf", "e_casci", "ecore"}
            and all(
                type(manifest_scalars[key]) in (int, float)
                and not isinstance(manifest_scalars[key], bool)
                and math.isfinite(float(manifest_scalars[key]))
                and math.isfinite(bundle_scalars[key])
                and bundle_scalars[key] == float(manifest_scalars[key])
                and type(bundle_record.get(key)) in (int, float)
                and not isinstance(bundle_record.get(key), bool)
                and float(bundle_record[key]) == bundle_scalars[key]
                for key in manifest_scalars
            )
            and bundle_occupations.dtype == np.dtype(np.float64)
            and bundle_occupations.shape == (norb,)
            and bool(np.isfinite(bundle_occupations).all())
            and occupation_hash == expected_occupation_hash
            and bundle_nelec.shape == (2,)
            and np.array_equal(
                bundle_nelec,
                np.asarray(nelec, dtype=bundle_nelec.dtype),
            )
        )
        audit.check(
            f"{name} checksum-bound bundle scalars and occupations",
            bundle_ok,
            actual={
                "path": str(bundle_path),
                "sha256": bundle_sha256,
                "manifest_scalars": manifest_scalars,
                "bundle_scalars": bundle_scalars,
                "recorded_bundle": bundle_record,
                "occupation_sha256": occupation_hash,
                "expected_occupation_sha256": expected_occupation_hash,
                "occupations": bundle_occupations,
                "nelec": bundle_nelec,
            },
            expected=(
                "regular checksum-bound NPZ whose RHF/CASCI/ecore scalars, "
                "electron sector, and float64 active occupations exactly "
                "match the manifest and certificate"
            ),
        )
        occupation = record.get("occupation_order")
        expected_occupations = np.zeros(norb, dtype=np.float64)
        expected_occupations[: nelec[1]] = 2.0
        expected_occupations[nelec[1] : nelec[0]] = 1.0
        occupation_ok = False
        occupation_recomputed: dict[str, float] = {}
        if isinstance(occupation, dict):
            frozen_occupations = np.asarray(
                occupation.get("frozen"), dtype=np.float64
            )
            regenerated_occupations = np.asarray(
                occupation.get("regenerated"), dtype=np.float64
            )
            reported_expected = np.asarray(
                occupation.get("expected"), dtype=np.float64
            )
            occupation_arrays_valid = (
                frozen_occupations.shape == (norb,)
                and regenerated_occupations.shape == (norb,)
                and reported_expected.shape == (norb,)
                and bool(np.isfinite(frozen_occupations).all())
                and bool(np.isfinite(regenerated_occupations).all())
                and bool(np.isfinite(reported_expected).all())
            )
            if occupation_arrays_valid:
                occupation_recomputed = {
                    "max_abs_frozen_vs_regenerated": float(
                        np.max(
                            np.abs(
                                frozen_occupations
                                - regenerated_occupations
                            ),
                            initial=0.0,
                        )
                    ),
                    "max_abs_frozen_vs_expected_order": float(
                        np.max(
                            np.abs(
                                frozen_occupations
                                - expected_occupations
                            ),
                            initial=0.0,
                        )
                    ),
                    "max_abs_regenerated_vs_expected_order": float(
                        np.max(
                            np.abs(
                                regenerated_occupations
                                - expected_occupations
                            ),
                            initial=0.0,
                        )
                    ),
                }
                occupation_tolerance = float(
                    occupation.get(
                        "absolute_tolerance", float("nan")
                    )
                )
                occupation_ok = (
                    occupation.get("passed") is True
                    and occupation.get("shape_match") is True
                    and 0 <= nelec[1] <= nelec[0] <= norb
                    and np.array_equal(
                        reported_expected, expected_occupations
                    )
                    and np.array_equal(
                        frozen_occupations, bundle_occupations
                    )
                    and math.isfinite(occupation_tolerance)
                    and occupation_tolerance
                    == trusted_tolerances["occupation_max_abs"]
                    and all(
                        type(occupation.get(key)) in (int, float)
                        and not isinstance(occupation.get(key), bool)
                        and math.isfinite(float(occupation[key]))
                        and float(occupation[key]) >= 0.0
                        and close(
                            float(occupation[key]),
                            recomputed,
                            1e-15,
                        )
                        and recomputed <= occupation_tolerance
                        for key, recomputed in occupation_recomputed.items()
                    )
                    and isinstance(checks, dict)
                    and checks.get("frozen_occupation_order") is True
                )
        audit.check(
            f"{name} frozen occupation ordering independently rederives",
            occupation_ok,
            actual={
                "occupation_order": occupation,
                "recomputed": occupation_recomputed,
                "bundle_occupations": bundle_occupations,
                "expected_occupations": expected_occupations,
            },
            expected=(
                "finite exact-length bundle-bound occupations in reconstructed "
                "RHF/ROHF order, with all three maxima independently "
                f"recomputed <= {trusted_tolerances['occupation_max_abs']}"
            ),
        )
        frozen_basis = record.get("frozen_active_basis")
        regeneration = record.get("active_space_regeneration")

        def within_declared_tolerance(
            item: Any,
            value_key: str,
            certificate_tolerance_key: str,
        ) -> bool:
            if not isinstance(item, dict):
                return False
            actual = float(item.get(value_key, float("nan")))
            tolerance = float(
                item.get("absolute_tolerance", float("nan"))
            )
            expected_tolerance = float(tolerances[certificate_tolerance_key])
            return (
                item.get("passed") is True
                and math.isfinite(actual)
                and actual >= 0.0
                and math.isfinite(tolerance)
                and tolerance > 0.0
                and close(tolerance, expected_tolerance, 0.0)
                and actual <= tolerance
            )

        basis_ok = False
        if isinstance(frozen_basis, dict):
            orthonormality = frozen_basis.get("s_orthonormality")
            core_orthogonality = frozen_basis.get(
                "core_active_orthogonality"
            )
            fock_residual = frozen_basis.get(
                "generalized_fock_eigen_residual"
            )
            basis_ok = (
                frozen_basis.get("passed") is True
                and frozen_basis.get("shape_valid") is True
                and frozen_basis.get("finite") is True
                and frozen_basis.get("coefficient_checksum_match") is True
                and isinstance(expected_active_hash, str)
                and frozen_basis.get("coefficient_sha256")
                == expected_active_hash
                and frozen_basis.get("expected_coefficient_sha256")
                == expected_active_hash
                and within_declared_tolerance(
                    orthonormality,
                    "max_abs_error",
                    "orbital_orthonormality",
                )
                and within_declared_tolerance(
                    core_orthogonality,
                    "max_abs_overlap",
                    "core_active_orthogonality",
                )
                and isinstance(core_orthogonality, dict)
                and int(core_orthogonality.get("core_orbital_count", -2))
                == declared_ncore
                and within_declared_tolerance(
                    fock_residual,
                    "max_abs_residual",
                    "generalized_fock_residual",
                )
                and isinstance(fock_residual, dict)
                and math.isfinite(
                    float(
                        fock_residual.get(
                            "maximum_column_l2_norm", float("nan")
                        )
                    )
                )
            )
        regeneration_ok = (
            isinstance(regeneration, dict)
            and regeneration.get("basis")
            == (
                "checksum-bound frozen active orbitals inserted after "
                "the regenerated core"
            )
            and int(regeneration.get("declared_ncore", -2))
            == declared_ncore
            and int(regeneration.get("pyscf_casci_ncore", -3))
            == declared_ncore
            and regeneration.get("declared_core_count_matches") is True
        )
        audit.check(
            f"{name} checksum-bound frozen-basis invariants",
            basis_ok and regeneration_ok,
            actual={
                "frozen_active_basis": frozen_basis,
                "active_space_regeneration": regeneration,
            },
            expected=(
                "manifest hash, finite S/core/Fock residuals within the "
                "certificate tolerances, and exact declared/CASCI core counts"
            ),
        )

        integral_differences = record.get("integral_differences")
        integral_ok = (
            isinstance(integral_differences, dict)
            and integral_differences.get("basis")
            == (
                "checksum-bound frozen mo_coeff_active with regenerated "
                "core orbitals"
            )
            and math.isfinite(
                float(
                    integral_differences.get(
                        "h1e_max_abs", float("nan")
                    )
                )
            )
            and 0.0
            <= float(integral_differences["h1e_max_abs"])
            <= float(tolerances["integral_max_abs"])
            and math.isfinite(
                float(
                    integral_differences.get(
                        "eri_max_abs", float("nan")
                    )
                )
            )
            and 0.0
            <= float(integral_differences["eri_max_abs"])
            <= float(tolerances["integral_max_abs"])
            and math.isfinite(
                float(
                    integral_differences.get(
                        "ecore_abs", float("nan")
                    )
                )
            )
            and 0.0
            <= float(integral_differences["ecore_abs"])
            <= float(tolerances["ecore_hartree"])
        )
        audit.check(
            f"{name} frozen-basis integral regeneration",
            integral_ok,
            actual=integral_differences,
            expected={
                "h1e_max_abs": f"<= {tolerances['integral_max_abs']}",
                "eri_max_abs": f"<= {tolerances['integral_max_abs']}",
                "ecore_abs": f"<= {tolerances['ecore_hartree']}",
            },
        )
        rhf = record.get("rhf")
        casci = record.get("casci")
        pspace = record.get("pspace")
        pspace_total = (
            float(pspace.get("total_energy_hartree", float("nan")))
            if isinstance(pspace, dict)
            else float("nan")
        )

        def finite_energy_proof(
            item: Any,
            certificate_tolerance_key: str,
            *,
            expected_frozen: float,
            expected_regenerated: float | None = None,
            require_converged: bool = False,
        ) -> bool:
            if not isinstance(item, dict):
                return False
            frozen_energy = float(
                item.get("frozen_energy_hartree", float("nan"))
            )
            regenerated_energy = float(
                item.get("regenerated_energy_hartree", float("nan"))
            )
            difference = float(
                item.get("absolute_difference_hartree", float("nan"))
            )
            tolerance = float(
                item.get("absolute_tolerance_hartree", float("nan"))
            )
            expected_tolerance = float(
                trusted_tolerances[certificate_tolerance_key]
            )
            recomputed_difference = abs(
                frozen_energy - regenerated_energy
            )
            return (
                (not require_converged or item.get("converged") is True)
                and math.isfinite(frozen_energy)
                and math.isfinite(regenerated_energy)
                and math.isfinite(difference)
                and difference >= 0.0
                and frozen_energy == expected_frozen
                and (
                    expected_regenerated is None
                    or close(
                        regenerated_energy,
                        expected_regenerated,
                        1e-15,
                    )
                )
                and close(difference, recomputed_difference, 1e-15)
                and math.isfinite(tolerance)
                and tolerance > 0.0
                and close(tolerance, expected_tolerance, 0.0)
                and difference <= tolerance
            )

        energy_proofs_ok = finite_energy_proof(
            rhf,
            "rhf_energy_hartree",
            expected_frozen=bundle_scalars["e_hf"],
            require_converged=True,
        ) and finite_energy_proof(
            casci,
            "casci_energy_hartree",
            expected_frozen=bundle_scalars["e_casci"],
            expected_regenerated=pspace_total,
        )
        audit.check(
            f"{name} regenerated RHF/CASCI energy proofs",
            energy_proofs_ok,
            actual={"rhf": rhf, "casci": casci},
            expected=(
                "finite frozen/regenerated energies and nonnegative absolute "
                "differences within the exact certificate tolerances"
            ),
        )

        sector = record.get("sector_consistency")
        pspace_ok = False
        ecore_binding_ok = False
        pspace_energy_actual: dict[str, Any] = {
            "pspace": pspace,
            "bundle_ecore_hartree": bundle_scalars["ecore"],
            "recorded_ecore_difference_hartree": (
                integral_differences.get("ecore_abs")
                if isinstance(integral_differences, dict)
                else None
            ),
        }
        if isinstance(pspace, dict):
            eigen_residual = float(
                pspace.get(
                    "eigen_residual_norm_hartree", float("nan")
                )
            )
            hermiticity = float(
                pspace.get(
                    "hermiticity_max_abs_hartree", float("nan")
                )
            )
            residual_tolerance = float(
                pspace.get(
                    "residual_absolute_tolerance_hartree", float("nan")
                )
            )
            electronic_energy = float(
                pspace.get(
                    "electronic_energy_hartree", float("nan")
                )
            )
            total_energy = float(
                pspace.get("total_energy_hartree", float("nan"))
            )
            regenerated_ecore = total_energy - electronic_energy
            recomputed_ecore_difference = abs(
                bundle_scalars["ecore"] - regenerated_ecore
            )
            recorded_ecore_difference = float(
                integral_differences.get("ecore_abs", float("nan"))
                if isinstance(integral_differences, dict)
                else float("nan")
            )
            ecore_rounding_tolerance = max(
                1e-15,
                8.0
                * max(
                    math.ulp(bundle_scalars["ecore"]),
                    (
                        math.ulp(regenerated_ecore)
                        if math.isfinite(regenerated_ecore)
                        else 0.0
                    ),
                    (
                        math.ulp(recorded_ecore_difference)
                        if math.isfinite(recorded_ecore_difference)
                        else 0.0
                    ),
                ),
            )
            ecore_binding_ok = (
                math.isfinite(electronic_energy)
                and math.isfinite(total_energy)
                and math.isfinite(regenerated_ecore)
                and math.isfinite(recomputed_ecore_difference)
                and math.isfinite(recorded_ecore_difference)
                and recorded_ecore_difference >= 0.0
                and close(
                    recorded_ecore_difference,
                    recomputed_ecore_difference,
                    ecore_rounding_tolerance,
                )
                and recomputed_ecore_difference
                <= trusted_tolerances["ecore_hartree"]
                and isinstance(casci, dict)
                and close(
                    float(
                        casci.get(
                            "regenerated_energy_hartree",
                            float("nan"),
                        )
                    ),
                    total_energy,
                    1e-15,
                )
            )
            pspace_energy_actual.update(
                {
                    "electronic_energy_hartree": electronic_energy,
                    "total_energy_hartree": total_energy,
                    "regenerated_ecore_hartree": regenerated_ecore,
                    "recomputed_ecore_difference_hartree": (
                        recomputed_ecore_difference
                    ),
                    "ecore_rounding_tolerance": (
                        ecore_rounding_tolerance
                    ),
                    "casci_regenerated_energy_hartree": (
                        casci.get("regenerated_energy_hartree")
                        if isinstance(casci, dict)
                        else None
                    ),
                }
            )
            pspace_ok = (
                pspace.get("solver")
                == "numpy.linalg.eigh on PySCF full determinant p-space"
                and pspace.get("converged") is True
                and pspace.get("all_determinant_addresses_present") is True
                and int(pspace.get("full_determinant_dimension", -1))
                == dimension
                and int(pspace.get("p_space_dimension", -1)) == dimension
                and math.isfinite(eigen_residual)
                and eigen_residual >= 0.0
                and math.isfinite(hermiticity)
                and hermiticity >= 0.0
                and math.isfinite(residual_tolerance)
                and residual_tolerance > 0.0
                and close(
                    residual_tolerance,
                    trusted_tolerances["eigen_residual_hartree"],
                    0.0,
                )
                and eigen_residual <= residual_tolerance
                and hermiticity <= residual_tolerance
                and ecore_binding_ok
            )
        audit.check(
            f"{name} p-space total/electronic/core energy binding",
            ecore_binding_ok,
            actual=pspace_energy_actual,
            expected=(
                "finite p-space electronic/total energies; reconstructed core "
                "energy agrees with the checksum-bound frozen core within the "
                "trusted tolerance; CASCI regenerated energy equals p-space "
                "total energy"
            ),
        )
        sector_ok = (
            isinstance(sector, dict)
            and sector.get("passed") is True
            and sector.get("bundle_nelec_alpha_beta") == list(nelec)
            and sector.get("manifest_nelec_alpha_beta") == list(nelec)
            and sector.get("bundle_matches_manifest") is True
            and int(sector.get("declared_sector_dimension", -1))
            == int(expected["sector_dimension"])
            and int(sector.get("combinatorial_sector_dimension", -1))
            == dimension
            and sector.get("declared_dimension_matches_combinatorial")
            is True
        )
        valid = (
            record.get("status") == "PASS"
            and isinstance(checks, dict)
            and set(checks) == REFERENCE_AUDIT_REQUIRED_CHECKS
            and all(value is True for value in checks.values())
            and bundle_ok
            and occupation_ok
            and basis_ok
            and regeneration_ok
            and integral_ok
            and energy_proofs_ok
            and int(expected["sector_dimension"]) == dimension
            and record.get("nelec_alpha_beta") == list(nelec)
            and int(record.get("sector_dimension", -1)) == dimension
            and sector_ok
            and pspace_ok
            and ecore_binding_ok
        )
        audit.check(
            f"{name} independent frozen-reference proof",
            valid,
            actual=record,
            expected=(
                "PASS with all required checks, exact electron sector, complete "
                "determinant p-space and residual/hermiticity within tolerance"
            ),
        )


def validate_generation_provenance(
    audit: Audit,
    *,
    source_code: Path,
    invocation_directory: Path,
    advanced: dict[str, Any],
    run_metadata: dict[str, Any],
) -> None:
    environment = load_json_object(invocation_directory / "environment.json")
    lock_hash = sha256_file(source_code / "requirements.lock")
    smoke = environment.get("runtime_smoke")
    audit.check(
        "locked generation environment",
        environment.get("status") == "PASS"
        and environment.get("python") == "3.12.3"
        and int(environment.get("locked_distributions", -1)) == 126
        and environment.get("requirements_lock_sha256") == lock_hash
        and isinstance(smoke, dict)
        and smoke.get("target") == "qpp-cpu"
        and int(smoke.get("sampled_shots", -1)) == 8,
        actual=environment,
        expected={
            "status": "PASS",
            "python": "3.12.3",
            "locked_distributions": 126,
            "requirements_lock_sha256": lock_hash,
            "runtime_smoke": {"target": "qpp-cpu", "sampled_shots": 8},
        },
    )
    audit.check(
        "advanced generation completed provider-free",
        advanced.get("status") == "COMPLETED"
        and run_metadata.get("quick") is False
        and run_metadata.get("qpu_contacted") is False
        and run_metadata.get("provider_imported") is False
        and run_metadata.get("seeds") == [17, 42, 3047],
        actual={
            "advanced_status": advanced.get("status"),
            "quick": run_metadata.get("quick"),
            "qpu_contacted": run_metadata.get("qpu_contacted"),
            "provider_imported": run_metadata.get("provider_imported"),
            "seeds": run_metadata.get("seeds"),
        },
        expected={
            "advanced_status": "COMPLETED",
            "quick": False,
            "qpu_contacted": False,
            "provider_imported": False,
            "seeds": [17, 42, 3047],
        },
    )
    frozen_hash = sha256_file(
        source_code / "frozen_inputs" / "MANIFEST.json"
    )
    audit.check(
        "advanced generation frozen-manifest binding",
        run_metadata.get("frozen_manifest_sha256") == frozen_hash
        and run_metadata.get("frozen_repeat_fingerprints_identical") is True,
        actual={
            "frozen_manifest_sha256": run_metadata.get(
                "frozen_manifest_sha256"
            ),
            "frozen_repeat_fingerprints_identical": run_metadata.get(
                "frozen_repeat_fingerprints_identical"
            ),
        },
        expected={
            "frozen_manifest_sha256": frozen_hash,
            "frozen_repeat_fingerprints_identical": True,
        },
    )
    implementation = run_metadata.get("implementation_sha256")
    mismatches = []
    if not isinstance(implementation, dict) or not implementation:
        mismatches.append("implementation_sha256 is missing or empty")
    else:
        for relative, expected_hash in implementation.items():
            try:
                portable = _portable_relative_path(
                    relative, field="implementation_sha256 path"
                )
            except ValueError as error:
                mismatches.append(str(error))
                continue
            path = source_code / "source" / portable
            actual = sha256_file(path) if path.is_file() else None
            if actual != expected_hash:
                mismatches.append(
                    f"{portable}: {actual} != {expected_hash}"
                )
    audit.check(
        "advanced implementation hashes match current source",
        not mismatches,
        actual=mismatches,
        expected=[],
    )


def validate_restart_cache_provenance(
    audit: Audit,
    *,
    status: dict[str, Any],
    advanced: dict[str, Any],
    run_metadata: dict[str, Any],
    source_code: Path,
    invocation_directory: Path,
    current_source_identity: dict[str, Any],
) -> None:
    """Bind an optional restart cache without trusting it as result evidence."""

    status_cache = status.get("restart_cache")
    metadata_cache = run_metadata.get("restart_cache")
    summary_cache = advanced.get("restart_cache")
    enabled = (
        isinstance(status_cache, dict)
        and status_cache.get("enabled") is True
    )
    if not enabled:
        audit.check(
            "restart cache is consistently disabled",
            (
                (
                    status_cache is None
                    or (
                        isinstance(status_cache, dict)
                        and status_cache.get("enabled") is False
                    )
                )
                and metadata_cache is None
                and summary_cache is None
            ),
            actual={
                "bootstrap": status_cache,
                "metadata": metadata_cache,
                "summary": summary_cache,
            },
            expected={
                "bootstrap": "absent or {'enabled': False}",
                "metadata": None,
                "summary": None,
            },
        )
        return

    audit.check(
        "restart cache summary and metadata records agree",
        isinstance(metadata_cache, dict)
        and isinstance(summary_cache, dict)
        and metadata_cache == summary_cache,
        actual={"metadata": metadata_cache, "summary": summary_cache},
        expected="identical restart-cache records",
    )
    assert isinstance(metadata_cache, dict)
    cache_path = Path(str(metadata_cache.get("path", ""))).resolve(
        strict=False
    )
    expected_status_fields = {
        "path": metadata_cache.get("path"),
        "identity_sha256": metadata_cache.get("identity_sha256"),
        "completed_entry_count": metadata_cache.get(
            "completed_entry_count"
        ),
        "cache_total_completed_entry_count": metadata_cache.get(
            "cache_total_completed_entry_count"
        ),
        "hits": metadata_cache.get("hits"),
        "misses": metadata_cache.get("misses"),
        "commits": metadata_cache.get("commits"),
        "snapshot": metadata_cache.get("snapshot"),
    }
    observed_status_fields = {
        key: status_cache.get(key) for key in expected_status_fields
    }
    audit.check(
        "bootstrap binds the final restart-cache record",
        observed_status_fields == expected_status_fields,
        actual=observed_status_fields,
        expected=expected_status_fields,
    )

    external_path_valid = (
        cache_path.is_absolute()
        and not is_within(cache_path, source_code)
        and not is_within(source_code, cache_path)
        and not is_within(cache_path, invocation_directory)
        and not is_within(invocation_directory, cache_path)
    )
    snapshot_record = metadata_cache.get("snapshot")
    snapshot_relative = (
        snapshot_record.get("relative_path")
        if isinstance(snapshot_record, dict)
        else None
    )
    snapshot_candidate = (
        invocation_directory / "advanced_method" / str(snapshot_relative)
    )
    snapshot_path = snapshot_candidate.resolve(strict=False)
    snapshot_regular = (
        snapshot_relative == "restart_cache_snapshot"
        and regular_directory_chain(
            invocation_directory,
            "advanced_method",
            "restart_cache_snapshot",
        )
        and not snapshot_candidate.is_symlink()
        and snapshot_path.is_dir()
        and not snapshot_path.is_symlink()
        and is_within(snapshot_path, invocation_directory / "advanced_method")
    )
    snapshot_manifest_path = snapshot_path / "SNAPSHOT_MANIFEST.json"
    snapshot_ready_path = snapshot_path / "SNAPSHOT_READY.json"
    snapshot_control_regular = (
        snapshot_manifest_path.is_file()
        and not snapshot_manifest_path.is_symlink()
        and snapshot_ready_path.is_file()
        and not snapshot_ready_path.is_symlink()
    )
    snapshot_manifest_hash = (
        sha256_file(snapshot_manifest_path)
        if snapshot_control_regular
        else None
    )
    snapshot_ready_hash = (
        sha256_file(snapshot_ready_path)
        if snapshot_control_regular
        else None
    )
    audit.check(
        "restart cache uses an invocation-local immutable snapshot",
        external_path_valid
        and snapshot_regular
        and snapshot_control_regular
        and isinstance(snapshot_record, dict)
        and snapshot_manifest_hash
        == snapshot_record.get("manifest_sha256")
        and snapshot_ready_hash == snapshot_record.get("ready_sha256"),
        actual={
            "external_cache_path": str(cache_path),
            "external_path_valid": external_path_valid,
            "snapshot_path": str(snapshot_path),
            "snapshot_regular": snapshot_regular,
            "snapshot_manifest_sha256": snapshot_manifest_hash,
            "snapshot_ready_sha256": snapshot_ready_hash,
            "recorded_snapshot": snapshot_record,
        },
        expected=(
            "external operational path is disjoint; all audited provenance "
            "comes from the hash-bound invocation-local snapshot"
        ),
    )
    snapshot_manifest = load_json_object(snapshot_manifest_path)
    snapshot_ready = load_json_object(snapshot_ready_path)
    declared_files = snapshot_manifest.get("files")
    declared_file_map: dict[str, dict[str, Any]] = {}
    declared_files_valid = isinstance(declared_files, list)
    if declared_files_valid:
        for item in declared_files:
            if not isinstance(item, dict):
                declared_files_valid = False
                break
            relative = item.get("path")
            try:
                portable = _portable_relative_path(
                    relative,
                    field="restart cache snapshot file",
                )
            except (TypeError, ValueError):
                declared_files_valid = False
                break
            key = portable
            if (
                key in declared_file_map
                or key in {"SNAPSHOT_MANIFEST.json", "SNAPSHOT_READY.json"}
                or type(item.get("size_bytes")) is not int
                or item["size_bytes"] < 0
                or not isinstance(item.get("sha256"), str)
                or len(item["sha256"]) != 64
            ):
                declared_files_valid = False
                break
            declared_file_map[key] = item

    observed_payload_files: dict[str, dict[str, Any]] = {}
    observed_directories: set[str] = set()
    snapshot_tree_valid = True
    if snapshot_regular:
        for path in snapshot_path.rglob("*"):
            relative = path.relative_to(snapshot_path).as_posix()
            if path.is_symlink():
                snapshot_tree_valid = False
                continue
            if path.is_dir():
                observed_directories.add(relative)
                continue
            if not path.is_file():
                snapshot_tree_valid = False
                continue
            if relative in {"SNAPSHOT_MANIFEST.json", "SNAPSHOT_READY.json"}:
                continue
            observed_payload_files[relative] = {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    snapshot_files_valid = (
        declared_files_valid
        and snapshot_tree_valid
        and observed_payload_files == declared_file_map
    )
    entry_keys = snapshot_manifest.get("entry_keys")
    entry_keys_valid = (
        isinstance(entry_keys, list)
        and all(
            isinstance(value, str)
            and len(value) == 64
            and all(
                character in "0123456789abcdef"
                for character in value
            )
            for value in entry_keys
        )
    )
    expected_directories = (
        {
            "entries",
            *(f"entries/{value}" for value in entry_keys),
        }
        if entry_keys_valid
        else set()
    )
    snapshot_directories_valid = (
        entry_keys_valid and observed_directories == expected_directories
    )
    ready_expected = {
        "schema_version": 1,
        "kind": "GQE_RESTART_CACHE_SNAPSHOT_READY",
        "manifest_sha256": snapshot_manifest_hash,
        "entry_count": (
            len(entry_keys) if entry_keys_valid else None
        ),
        "file_count": (
            len(declared_file_map) if declared_files_valid else None
        ),
    }
    audit.check(
        "restart cache snapshot manifest binds its exact file tree",
        snapshot_manifest.get("schema_version") == 1
        and snapshot_manifest.get("kind") == "GQE_RESTART_CACHE_SNAPSHOT"
        and snapshot_manifest.get("cache_identity_sha256")
        == metadata_cache.get("identity_sha256")
        and entry_keys_valid
        and len(entry_keys) == len(set(entry_keys))
        and snapshot_ready == ready_expected
        and snapshot_files_valid
        and snapshot_directories_valid
        and isinstance(snapshot_record, dict)
        and snapshot_record.get("entry_count") == len(entry_keys)
        and snapshot_record.get("file_count") == len(declared_file_map),
        actual={
            "snapshot_manifest": snapshot_manifest,
            "snapshot_ready": snapshot_ready,
            "observed_payload_files": observed_payload_files,
            "declared_files_valid": declared_files_valid,
            "snapshot_tree_valid": snapshot_tree_valid,
            "observed_directories": sorted(observed_directories),
            "expected_directories": sorted(expected_directories),
        },
        expected=ready_expected,
    )

    identity_path = snapshot_path / "CACHE_IDENTITY.json"
    entries_path = snapshot_path / "entries"
    identity_regular = (
        identity_path.is_file() and not identity_path.is_symlink()
    )
    entries_regular = (
        entries_path.is_dir() and not entries_path.is_symlink()
    )
    identity_hash = (
        sha256_file(identity_path) if identity_regular else None
    )
    audit.check(
        "restart cache snapshot is regular and identity-bound",
        identity_regular
        and entries_regular
        and identity_hash == metadata_cache.get("identity_sha256"),
        actual={
            "path": str(snapshot_path),
            "identity_regular": identity_regular,
            "entries_regular": entries_regular,
            "identity_sha256": identity_hash,
        },
        expected={
            "snapshot_regular": True,
            "identity_sha256": metadata_cache.get("identity_sha256"),
        },
    )
    identity = load_json_object(identity_path)
    frozen_hash = sha256_file(
        source_code / "frozen_inputs" / "MANIFEST.json"
    )
    audit.check(
        "restart cache identity binds source, frozen inputs and provider boundary",
        identity.get("schema_version") == 1
        and identity.get("kind") == "GQE_RESTART_SAFE_OPTIMIZER_CACHE"
        and identity.get("source_identity") == current_source_identity
        and identity.get("frozen_manifest_sha256") == frozen_hash
        and identity.get("python") == run_metadata.get("python")
        and identity.get("packages") == run_metadata.get("packages")
        and identity.get("threads") == run_metadata.get("threads")
        and identity.get("provider_imported") is False
        and identity.get("qpu_contacted") is False
        and metadata_cache.get("provider_imported") is False
        and metadata_cache.get("qpu_contacted") is False,
        actual={
            "identity": identity,
            "run_metadata_runtime": {
                "python": run_metadata.get("python"),
                "packages": run_metadata.get("packages"),
                "threads": run_metadata.get("threads"),
            },
            "metadata_provider_imported": metadata_cache.get(
                "provider_imported"
            ),
            "metadata_qpu_contacted": metadata_cache.get("qpu_contacted"),
        },
        expected={
            "source_identity": current_source_identity,
            "frozen_manifest_sha256": frozen_hash,
            "provider_imported": False,
            "qpu_contacted": False,
        },
    )

    complete_entries: list[str] = []
    entry_manifests: dict[str, dict[str, Any]] = {}
    invalid_entries: list[str] = []
    for entry in entries_path.iterdir():
        if (
            not entry.is_dir()
            or entry.is_symlink()
            or len(entry.name) != 64
            or any(character not in "0123456789abcdef" for character in entry.name)
        ):
            invalid_entries.append(entry.name)
            continue
        expected_files = {"DONE.json", "manifest.json", "parameters.npz"}
        observed_files = {path.name for path in entry.iterdir()}
        if observed_files != expected_files:
            invalid_entries.append(entry.name)
            continue
        done_path = entry / "DONE.json"
        manifest_path = entry / "manifest.json"
        parameters_path = entry / "parameters.npz"
        if any(
            not path.is_file() or path.is_symlink()
            for path in (done_path, manifest_path, parameters_path)
        ):
            invalid_entries.append(entry.name)
            continue
        try:
            done = load_json_object(done_path)
            manifest = load_json_object(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            invalid_entries.append(entry.name)
            continue
        manifest_hash = sha256_file(manifest_path)
        parameters_hash = sha256_file(parameters_path)
        spec = manifest.get("spec")
        expected_entry_key = (
            hashlib.sha256(
                json.dumps(
                    {
                        "cache_identity_sha256": identity_hash,
                        "spec": spec,
                    },
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if isinstance(spec, dict)
            else None
        )
        valid = (
            done
            == {
                "schema_version": 1,
                "kind": "GQE_RESTART_CACHE_ENTRY_DONE",
                "key": entry.name,
                "manifest_sha256": manifest_hash,
                "parameters_sha256": parameters_hash,
            }
            and manifest.get("schema_version") == 1
            and manifest.get("kind") == "GQE_RESTART_CACHE_ENTRY"
            and manifest.get("key") == entry.name
            and manifest.get("cache_identity_sha256") == identity_hash
            and manifest.get("parameters_sha256") == parameters_hash
            and expected_entry_key == entry.name
            and isinstance(manifest.get("result"), dict)
        )
        if valid:
            complete_entries.append(entry.name)
            entry_manifests[entry.name] = manifest
        else:
            invalid_entries.append(entry.name)

    expected_incomplete = sorted(
        str(value)
        for value in metadata_cache.get("ignored_incomplete_entries", [])
    )
    incomplete_record_valid = all(
        value.startswith(".") and ".tmp-" in value
        for value in expected_incomplete
    )
    hits = metadata_cache.get("hits")
    misses = metadata_cache.get("misses")
    commits = metadata_cache.get("commits")
    recorded_entry_count = metadata_cache.get("completed_entry_count")
    cache_total_entry_count = metadata_cache.get(
        "cache_total_completed_entry_count"
    )
    used_entries = metadata_cache.get("used_entries")
    counts_valid = all(type(value) is int and value >= 0 for value in (
        hits,
        misses,
        commits,
        recorded_entry_count,
        cache_total_entry_count,
    ))
    usage_valid = isinstance(used_entries, list)
    usage_keys: list[str] = []
    usage_hit_wall = 0.0
    usage_commit_wall = 0.0
    usage_hit_count = 0
    usage_commit_count = 0
    if usage_valid:
        for item in used_entries:
            if not isinstance(item, dict):
                usage_valid = False
                break
            key = item.get("key")
            disposition = item.get("disposition")
            wall = item.get("optimizer_wall_seconds")
            manifest = entry_manifests.get(str(key))
            spec = manifest.get("spec") if isinstance(manifest, dict) else None
            result = (
                manifest.get("result")
                if isinstance(manifest, dict)
                else None
            )
            expected_semantics = (
                "COMPLETE_ADAPTIVE_DEPTH_CASCADE"
                if isinstance(spec, dict)
                and spec.get("kind") == "run_adaptive_depth"
                else "COMPLETED_RUN_ONE_CALL"
            )
            expected_wall = (
                result.get("complete_cascade_wall_seconds")
                if expected_semantics == "COMPLETE_ADAPTIVE_DEPTH_CASCADE"
                and isinstance(result, dict)
                else (
                    result.get("wall_seconds")
                    if isinstance(result, dict)
                    else None
                )
            )
            if (
                not isinstance(key, str)
                or len(key) != 64
                or any(character not in "0123456789abcdef" for character in key)
                or type(wall) not in (int, float)
                or not math.isfinite(float(wall))
                or float(wall) < 0.0
                or not isinstance(spec, dict)
                or item.get("kind") != spec.get("kind")
                or item.get("role") != spec.get("role")
                or item.get("wall_time_semantics") != expected_semantics
                or type(expected_wall) not in (int, float)
                or not math.isclose(
                    float(wall),
                    float(expected_wall),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                usage_valid = False
                break
            usage_keys.append(key)
            if disposition == "REPLAYED_VALID_CACHE_HIT":
                usage_hit_count += 1
                usage_hit_wall += float(wall)
            elif disposition == "COMPUTED_AND_ATOMICALLY_COMMITTED":
                usage_commit_count += 1
                usage_commit_wall += float(wall)
            else:
                usage_valid = False
                break
    recorded_hit_wall = metadata_cache.get(
        "replayed_optimizer_wall_seconds"
    )
    recorded_commit_wall = metadata_cache.get(
        "committed_optimizer_wall_seconds"
    )
    usage_valid = (
        usage_valid
        and type(recorded_hit_wall) in (int, float)
        and type(recorded_commit_wall) in (int, float)
        and math.isclose(
            usage_hit_wall,
            float(recorded_hit_wall),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            usage_commit_wall,
            float(recorded_commit_wall),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )
    audit.check(
        "restart cache contains only hash-bound complete entries",
        not invalid_entries
        and incomplete_record_valid
        and counts_valid
        and len(complete_entries) == recorded_entry_count
        and cache_total_entry_count >= recorded_entry_count
        and set(complete_entries) == set(entry_keys)
        and misses == commits
        and hits + commits == len(complete_entries),
        actual={
            "complete_entry_count": len(complete_entries),
            "recorded_complete_entry_count": recorded_entry_count,
            "cache_total_complete_entry_count": cache_total_entry_count,
            "recorded_incomplete_entries": expected_incomplete,
            "incomplete_record_valid": incomplete_record_valid,
            "snapshot_entry_keys": entry_keys,
            "invalid_entries": sorted(invalid_entries),
            "hits": hits,
            "misses": misses,
            "commits": commits,
        },
        expected=(
            "snapshot contains exactly the hash-bound used entries; external "
            "incomplete temp names are recorded but not trusted; misses=commits"
        ),
    )
    audit.check(
        "restart cache usage and timing lineage is complete",
        usage_valid
        and usage_hit_count == hits
        and usage_commit_count == commits
        and len(usage_keys) == len(set(usage_keys))
        and set(usage_keys) == set(complete_entries),
        actual={
            "usage_valid": usage_valid,
            "usage_hit_count": usage_hit_count,
            "recorded_hits": hits,
            "usage_commit_count": usage_commit_count,
            "recorded_commits": commits,
            "usage_keys": sorted(usage_keys),
            "complete_entries": sorted(complete_entries),
            "usage_hit_wall_seconds": usage_hit_wall,
            "recorded_hit_wall_seconds": recorded_hit_wall,
            "usage_commit_wall_seconds": usage_commit_wall,
            "recorded_commit_wall_seconds": recorded_commit_wall,
        },
        expected=(
            "one unique usage record per complete cache entry with exact "
            "hit/commit counts and optimizer wall-time sums"
        ),
    )
    runtime = advanced.get("runtime_accounting")
    attempt_wall = advanced.get("runtime_seconds")
    runtime_valid = (
        isinstance(runtime, dict)
        and type(attempt_wall) in (int, float)
        and type(runtime.get("current_attempt_wall_seconds")) in (int, float)
        and type(
            runtime.get("replayed_completed_optimizer_wall_seconds")
        )
        in (int, float)
        and type(
            runtime.get("current_attempt_completed_optimizer_wall_seconds")
        )
        in (int, float)
        and type(
            runtime.get("effective_completed_work_plus_current_attempt_seconds")
        )
        in (int, float)
    )
    if runtime_valid:
        runtime_valid = (
            math.isclose(
                float(runtime["current_attempt_wall_seconds"]),
                float(attempt_wall),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(runtime["replayed_completed_optimizer_wall_seconds"]),
                float(recorded_hit_wall),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(
                    runtime[
                        "current_attempt_completed_optimizer_wall_seconds"
                    ]
                ),
                float(recorded_commit_wall),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and math.isclose(
                float(
                    runtime[
                        "effective_completed_work_plus_current_attempt_seconds"
                    ]
                ),
                float(attempt_wall) + float(recorded_hit_wall),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and runtime.get("interrupted_incomplete_optimizer_work_excluded")
            is True
        )
    audit.check(
        "restart recovery runtime accounting is explicit",
        runtime_valid,
        actual=runtime,
        expected={
            "current_attempt_wall_seconds": attempt_wall,
            "replayed_completed_optimizer_wall_seconds": recorded_hit_wall,
            "current_attempt_completed_optimizer_wall_seconds": (
                recorded_commit_wall
            ),
            "effective_completed_work_plus_current_attempt_seconds": (
                float(attempt_wall) + float(recorded_hit_wall)
                if type(attempt_wall) in (int, float)
                and type(recorded_hit_wall) in (int, float)
                else None
            ),
            "interrupted_incomplete_optimizer_work_excluded": True,
        },
    )
    timing_boundary = advanced.get("timing_claim_boundary")
    audit.check(
        "restart recovery forbids mixed-origin wall-time speedup claims",
        timing_boundary
        == {
            "energy_evaluation_counts_comparable": True,
            "wall_time_speedup_claim_permitted": False,
            "interpretation": RESTART_TIMING_CLAIM_BOUNDARY,
        },
        actual=timing_boundary,
        expected={
            "energy_evaluation_counts_comparable": True,
            "wall_time_speedup_claim_permitted": False,
            "interpretation": RESTART_TIMING_CLAIM_BOUNDARY,
        },
    )
    local_timing_records: list[dict[str, Any]] = []
    adaptive_topology = advanced.get("adaptive_topology")
    if isinstance(adaptive_topology, dict):
        for record in adaptive_topology.values():
            if not isinstance(record, dict):
                continue
            search_cost = record.get("search_cost")
            if isinstance(search_cost, dict):
                local_timing_records.append(search_cost)
            matched = record.get("matched_controls")
            if isinstance(matched, dict):
                for control in matched.values():
                    if (
                        isinstance(control, dict)
                        and isinstance(
                            control.get("cost_accounting"),
                            dict,
                        )
                    ):
                        local_timing_records.append(
                            control["cost_accounting"]
                        )
    warm_start = advanced.get("warm_start")
    if isinstance(warm_start, dict):
        for ladder in warm_start.values():
            seeds = (
                ladder.get("seeds")
                if isinstance(ladder, dict)
                else None
            )
            if isinstance(seeds, dict):
                local_timing_records.extend(
                    record
                    for record in seeds.values()
                    if isinstance(record, dict)
                )
    audit.check(
        "all emitted cached wall-time comparisons carry the claim boundary",
        bool(local_timing_records)
        and all(
            record.get("wall_time_speedup_claim_permitted") is False
            and record.get("wall_time_interpretation")
            == RESTART_TIMING_CLAIM_BOUNDARY
            for record in local_timing_records
        ),
        actual={
            "record_count": len(local_timing_records),
            "invalid_record_count": sum(
                1
                for record in local_timing_records
                if (
                    record.get("wall_time_speedup_claim_permitted")
                    is not False
                    or record.get("wall_time_interpretation")
                    != RESTART_TIMING_CLAIM_BOUNDARY
                )
            ),
        },
        expected=(
            "every adaptive search/matched-control and warm-ladder wall-time "
            "record explicitly forbids a mixed-origin speedup claim"
        ),
    )
    timing_csv_paths = (
        invocation_directory
        / "advanced_method"
        / "tables"
        / "deterministic_baseline.csv",
        invocation_directory
        / "advanced_method"
        / "tables"
        / "adaptive_topology_matched.csv",
        invocation_directory
        / "advanced_method"
        / "tables"
        / "warm_start_controls.csv",
    )
    invalid_timing_csvs: list[str] = []
    timing_csv_row_count = 0
    for path in timing_csv_paths:
        if not path.is_file() or path.is_symlink():
            invalid_timing_csvs.append(f"{path.name}: not a regular file")
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = set(reader.fieldnames or ())
            if not {
                "wall_time_speedup_claim_permitted",
                "wall_time_interpretation",
            }.issubset(fields):
                invalid_timing_csvs.append(
                    f"{path.name}: timing boundary columns missing"
                )
                continue
            rows = list(reader)
        if not rows:
            invalid_timing_csvs.append(f"{path.name}: no rows")
            continue
        timing_csv_row_count += len(rows)
        if any(
            row.get("wall_time_speedup_claim_permitted") != "False"
            or row.get("wall_time_interpretation")
            != RESTART_TIMING_CLAIM_BOUNDARY
            for row in rows
        ):
            invalid_timing_csvs.append(
                f"{path.name}: unbounded wall-time row"
            )
    audit.check(
        "standalone timing CSVs preserve the restart claim boundary",
        not invalid_timing_csvs and timing_csv_row_count > 0,
        actual={
            "row_count": timing_csv_row_count,
            "invalid": invalid_timing_csvs,
        },
        expected=(
            "all baseline, adaptive and warm-start CSV rows explicitly forbid "
            "mixed-origin wall-time speedup claims"
        ),
    )


def validate_advanced_candidate_binding(
    audit: Audit,
    *,
    manifest: dict[str, Any],
    advanced: dict[str, Any],
    expected_metrics: dict[str, Any],
) -> None:
    resources = advanced.get("structured_resources")
    audit.check(
        "advanced summary exact Table-3 resource set",
        isinstance(resources, dict)
        and set(resources) == set(EXPECTED_CASES),
        actual=sorted(resources) if isinstance(resources, dict) else None,
        expected=sorted(EXPECTED_CASES),
    )
    assert isinstance(resources, dict)
    for label in EXPECTED_CASES:
        case = manifest["cases"][label]
        from_manifest = {
            key: value
            for key, value in case.items()
            if key not in {"ansatz", "parameter_artifact"}
        }
        audit.check(
            f"{label} candidate manifest equals advanced emitted record",
            from_manifest == resources[label],
            actual=from_manifest,
            expected=resources[label],
        )

    adaptive = advanced.get("adaptive_topology")
    targets = expected_metrics["adaptive_topology"]
    audit.check(
        "advanced topology records cover Table-3 cases",
        isinstance(adaptive, dict) and set(EXPECTED_CASES) <= set(adaptive),
        actual=sorted(adaptive) if isinstance(adaptive, dict) else None,
        expected=sorted(EXPECTED_CASES),
    )
    assert isinstance(adaptive, dict)
    required_gates = (
        "candidate_on_screened_pareto_front",
        "differs_from_static_partition",
        "energy_noninferior_within_0p1_mha",
        "improves_at_least_one_declared_pre_screen_metric",
        "within_phi_budget",
    )
    for label in EXPECTED_CASES:
        record = adaptive[label]
        target = targets[label]
        passing = [
            candidate
            for candidate in record["screened_candidates"]
            if all(
                candidate["acceptance"].get(gate) is True
                for gate in required_gates
            )
        ]
        accepted = str(record["decision"]).startswith("ACCEPT_")
        selected_candidates = [
            candidate
            for candidate in passing
            if candidate["pre_screen"]["left_block"]
            == record["selected_left_block"]
        ]
        gate_consistency = (
            accepted and len(selected_candidates) == 1
        ) or (
            not accepted
            and not passing
            and record["selected_left_block"] == record["static_left_block"]
        )
        audit.check(
            f"{label} topology decision and all predeclared gates",
            record["decision"] == target["decision"]
            and record["static_left_block"] == target["static_left"]
            and record["selected_left_block"] == target["selected_left"]
            and record["selected_left_block"]
            == manifest["cases"][label]["selected_left_block"]
            and gate_consistency,
            actual={
                "decision": record["decision"],
                "static_left_block": record["static_left_block"],
                "selected_left_block": record["selected_left_block"],
                "passing_candidates": [
                    candidate["pre_screen"]["left_block"]
                    for candidate in passing
                ],
            },
            expected={
                **target,
                "gate_consistency": True,
            },
        )
        if accepted:
            tradeoff = selected_candidates[0][
                "tradeoff_vs_static_pre_screen"
            ]
            audit.check(
                f"{label} accepted topology is the declared Pareto tradeoff",
                tradeoff.get("interpretation")
                == "pareto_front_tradeoff_not_dominance"
                and float(tradeoff["schmidt_entropy_delta_nats"]) < 0.0
                and float(tradeoff["hamiltonian_cross_weight_delta"]) > 0.0,
                actual=tradeoff,
                expected=(
                    "Pareto-front tradeoff with lower Schmidt entropy and "
                    "increased cross-Hamiltonian weight"
                ),
            )


def run_audit(
    *,
    source_code: Path,
    bootstrap_helper_path: Path,
    bootstrap_status_path: Path,
    candidate_directory: Path,
    output: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    audit = Audit()
    if str(source_code) not in sys.path:
        sys.path.insert(0, str(source_code))
    if str(source_code / "source") not in sys.path:
        sys.path.insert(0, str(source_code / "source"))
    from certify_release import source_identity
    from environment_contract import resolve_environment
    from givens40.canonical_resources import compilation_protocol

    invocation_directory = bootstrap_status_path.resolve().parent
    environment = resolve_environment(source_code).resolve()
    source_before = source_identity(source_code)
    reference_before = tree_snapshot(source_code / "reference")
    invocation_before = tree_snapshot(invocation_directory)
    candidate_before: dict[str, Any] | None = None
    observed: dict[str, Any] = {}
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "audit_kind": "TABLE3_PRE_PROMOTION_READ_ONLY",
        "status": "RUNNING",
        "started_utc": utc_now(),
        "completed_utc": None,
        "wall_seconds": None,
        "source_code": str(source_code),
        "bootstrap_helper": str(bootstrap_helper_path),
        "bootstrap_helper_sha256": (
            sha256_file(bootstrap_helper_path)
            if bootstrap_helper_path.is_file()
            and not bootstrap_helper_path.is_symlink()
            else None
        ),
        "bootstrap_status": str(bootstrap_status_path),
        "bootstrap_status_sha256": (
            sha256_file(bootstrap_status_path)
            if bootstrap_status_path.is_file()
            else None
        ),
        "candidate_directory": str(candidate_directory),
        "output": str(output),
        "candidate_expected_status": CANDIDATE_STATUS,
        "candidate_promoted": False,
        "provider_imported": False,
        "qpu_contacted": False,
        "source_identity_before": source_before,
        "source_identity_after": None,
        "reference_tree_before": reference_before,
        "reference_tree_after": None,
        "invocation_tree_before": invocation_before,
        "invocation_tree_after": None,
        "candidate_tree_before": None,
        "candidate_tree_after": None,
        "candidate_mutated": None,
        "reference_mutated": None,
        "invocation_mutated": None,
        "observed_cases": observed,
        "checks": audit.checks,
    }
    error: BaseException | None = None
    try:
        audit.check(
            "Source_Code is hardened and reference/table3 is absent",
            (source_code / "certify_release.py").is_file()
            and not (source_code / "reference" / "table3").exists()
            and not (source_code / "reference" / "table3").is_symlink(),
            actual={
                "certify_release": (
                    source_code / "certify_release.py"
                ).is_file(),
                "reference_table3_exists": (
                    source_code / "reference" / "table3"
                ).exists(),
            },
            expected={
                "certify_release": True,
                "reference_table3_exists": False,
            },
        )
        candidate_manifest = candidate_tree_manifest(
            audit, candidate_directory
        )
        candidate_before = tree_snapshot(candidate_directory)
        report["candidate_tree_before"] = candidate_before
        status = load_json_object(bootstrap_status_path)
        validate_bootstrap_status(
            audit,
            status,
            bootstrap_helper_path=bootstrap_helper_path,
            bootstrap_status_path=bootstrap_status_path,
            source_code=source_code,
            environment=environment,
            candidate_directory=candidate_directory,
            current_source_identity=source_before,
            observed_candidate_manifest=candidate_manifest,
        )
        manifest_path = candidate_directory / "canonical_table3.json"
        manifest = load_json_object(manifest_path)
        expected_metrics = load_json_object(
            source_code / "expected_metrics.json"
        )
        validate_candidate_manifest_structure(
            audit,
            manifest,
            compilation_protocol=compilation_protocol(),
            expected_metrics=expected_metrics,
        )

        advanced_root = invocation_directory / "advanced_method"
        advanced = load_json_object(
            advanced_root / "enhanced_release_summary.json"
        )
        run_metadata = load_json_object(
            advanced_root / "run_metadata.json"
        )
        validate_generation_provenance(
            audit,
            source_code=source_code,
            invocation_directory=invocation_directory,
            advanced=advanced,
            run_metadata=run_metadata,
        )
        validate_restart_cache_provenance(
            audit,
            status=status,
            advanced=advanced,
            run_metadata=run_metadata,
            source_code=source_code,
            invocation_directory=invocation_directory,
            current_source_identity=source_before,
        )
        validate_frozen_reference_certificate(
            audit,
            source_code=source_code,
            invocation_directory=invocation_directory,
        )
        validate_advanced_candidate_binding(
            audit,
            manifest=manifest,
            advanced=advanced,
            expected_metrics=expected_metrics,
        )
        for label in EXPECTED_CASES:
            observed[label] = replay_candidate_case(
                audit,
                source_code=source_code,
                candidate_directory=candidate_directory,
                label=label,
                case=manifest["cases"][label],
            )
    except BaseException as caught:
        error = caught
        if not isinstance(caught, CandidateAuditFailure):
            try:
                audit.check(
                    "candidate audit completed without an unexpected exception",
                    False,
                    actual=f"{type(caught).__name__}: {caught}",
                    expected="no exception",
                )
            except CandidateAuditFailure:
                pass
    finally:
        try:
            candidate_after = tree_snapshot(candidate_directory)
            reference_after = tree_snapshot(source_code / "reference")
            invocation_after = tree_snapshot(invocation_directory)
            source_after = source_identity(source_code)
            report["candidate_tree_after"] = candidate_after
            report["reference_tree_after"] = reference_after
            report["invocation_tree_after"] = invocation_after
            report["source_identity_after"] = source_after
            report["candidate_mutated"] = candidate_before != candidate_after
            report["reference_mutated"] = reference_before != reference_after
            report["invocation_mutated"] = (
                invocation_before != invocation_after
            )
            try:
                audit.check(
                    "candidate remained byte-for-byte unchanged during audit",
                    candidate_before is not None
                    and candidate_before == candidate_after,
                    actual=report["candidate_mutated"],
                    expected=False,
                    fatal=False,
                )
                audit.check(
                    "reference tree remained byte-for-byte unchanged during audit",
                    reference_before == reference_after,
                    actual=report["reference_mutated"],
                    expected=False,
                    fatal=False,
                )
                audit.check(
                    "bootstrap invocation evidence remained byte-for-byte "
                    "unchanged during audit",
                    invocation_before == invocation_after,
                    actual=report["invocation_mutated"],
                    expected=False,
                    fatal=False,
                )
                audit.check(
                    "Source_Code identity remained unchanged during audit",
                    source_before == source_after,
                    actual=source_after,
                    expected=source_before,
                    fatal=False,
                )
                reloaded_status = load_json_object(
                    candidate_directory / "canonical_table3.json"
                ).get("status")
                audit.check(
                    "candidate status remained noncanonical after audit",
                    reloaded_status == CANDIDATE_STATUS,
                    actual=reloaded_status,
                    expected=CANDIDATE_STATUS,
                    fatal=False,
                )
            except CandidateAuditFailure as caught:
                if error is None:
                    error = caught
        except BaseException as caught:
            error = error or caught
            try:
                audit.check(
                    "post-audit immutability checks completed",
                    False,
                    actual=f"{type(caught).__name__}: {caught}",
                    expected="complete immutable snapshots",
                )
            except CandidateAuditFailure:
                pass

    report["status"] = audit.status if error is None else "FAIL"
    report["checks"] = audit.checks
    report["passed"] = sum(
        item["status"] == "PASS" for item in audit.checks
    )
    report["failed"] = sum(
        item["status"] == "FAIL" for item in audit.checks
    )
    if error is not None:
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    report["completed_utc"] = utc_now()
    report["wall_seconds"] = time.monotonic() - started
    atomic_write_json(output, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-code", type=Path, required=True)
    parser.add_argument("--bootstrap-helper", type=Path, required=True)
    parser.add_argument("--bootstrap-status", type=Path, required=True)
    parser.add_argument(
        "--candidate-directory",
        type=Path,
        help=(
            "candidate location after download; defaults to candidate_directory "
            "recorded in bootstrap_status.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_code = args.source_code.expanduser().resolve()
    bootstrap_helper = args.bootstrap_helper.expanduser().resolve()
    bootstrap_status = args.bootstrap_status.expanduser().resolve()
    try:
        status = load_json_object(bootstrap_status)
        if args.candidate_directory is None:
            candidate_value = status.get("candidate_directory")
            if not isinstance(candidate_value, str) or not candidate_value:
                raise ValueError(
                    "bootstrap status does not declare candidate_directory"
                )
            candidate_directory = Path(candidate_value).expanduser().resolve()
        else:
            candidate_directory = (
                args.candidate_directory.expanduser().resolve()
            )
        output = validate_output_path(
            args.output,
            source_code=source_code,
            candidate_directory=candidate_directory,
        )
        if is_within(output, bootstrap_status.parent):
            raise ValueError(
                "Audit output must remain outside the bootstrap invocation "
                "directory so its evidence snapshot stays immutable"
            )
    except BaseException as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 1

    report = run_audit(
        source_code=source_code,
        bootstrap_helper_path=bootstrap_helper,
        bootstrap_status_path=bootstrap_status,
        candidate_directory=candidate_directory,
        output=output,
    )
    print(
        f"TABLE-3 CANDIDATE AUDIT {report['status']}: "
        f"{report['passed']} passed, {report['failed']} failed",
        flush=True,
    )
    print(output, flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
