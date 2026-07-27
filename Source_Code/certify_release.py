#!/usr/bin/env python3
"""Invocation-bound, fail-closed wrapper around the judge reproduction."""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence
import uuid
import zipfile


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
RESULTS = ROOT / "results" / "judge_reproduction"
CERTIFICATES = RESULTS / "certificates"
LATEST_CERTIFICATE = RESULTS / "latest_certificate.json"
INVOCATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{7,127}$")
SOURCE_EXCLUDED_TOP_LEVEL = {"results", ".git"}
SOURCE_EXCLUDED_DIRECTORY_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
}
TABLE3_LABELS = ("BeH2-6", "BeH2-12", "LiH-40")
TABLE3_CASES = {
    "BeH2-6": ("BeH2", 3, 6),
    "BeH2-12": ("BeH2", 6, 12),
    "LiH-40": ("LiH", 20, 40),
}
TABLE3_SEED = 3047
TABLE3_PROTOCOL = {
    "basis_gates": ["rz", "sx", "x", "cx"],
    "optimization_level": 3,
    "seed_transpiler": TABLE3_SEED,
    "connectivity": "all-to-all",
    "scope": "diagnostic logical compile; not device-native",
}
TABLE3_MANIFEST = ROOT / "reference" / "table3" / "canonical_table3.json"
TABLE3_CANDIDATE_STATUS = (
    "CANDIDATE_NOT_CANONICAL_UNTIL_EXPLICITLY_PROMOTED"
)
TABLE3_PROMOTION_IDENTITY_ALGORITHM = (
    "sha256-path-hash-size-notebook-source-v4-excludes-reference-table3"
)
TABLE3_PROVENANCE_PATHS = {
    "audit_report_path": "provenance/audit_report.json",
    "bootstrap_status_path": "provenance/bootstrap_status.json",
    "bootstrap_helper_path": "provenance/bootstrap_helper.py",
}
TABLE3_TOOL_PATHS = {
    "promotion_tool_path": "promote_table3_candidate.py",
    "audit_tool_path": "audit_table3_candidate.py",
}
TABLE3_BOOTSTRAP_STAGE_NAMES = (
    "locked_environment_smoke",
    "dependency_consistency",
    "independent_frozen_reference_audit",
    "full_advanced_candidate_generation",
)
TABLE3_REQUIRED_AUDIT_CHECKS = frozenset(
    {
        "Source_Code is hardened and reference/table3 is absent",
        "bootstrap helper path and bytes are explicitly bound",
        "bootstrap invocation and execution paths are exact",
        "bootstrap candidate-ready status remains noncertifying",
        "bootstrap provider/QPU boundary",
        "bootstrap exact four-stage sequence is declared",
        *(
            f"bootstrap {name} argv and final log bind"
            for name in TABLE3_BOOTSTRAP_STAGE_NAMES
        ),
        "bootstrap source identity is unchanged and current",
        "downloaded candidate matches bootstrap artifact manifest",
        "candidate remained byte-for-byte unchanged during audit",
        "reference tree remained byte-for-byte unchanged during audit",
        (
            "bootstrap invocation evidence remained byte-for-byte "
            "unchanged during audit"
        ),
        "Source_Code identity remained unchanged during audit",
        "candidate status remained noncanonical after audit",
    }
)

# Prevent a direct, non-``-B`` invocation from creating bytecode in Source_Code
# before the immutable source identity is measured.
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment_contract import (  # noqa: E402
    check_disk_space,
    is_within,
    resolve_environment,
    sanitized_environment,
)


class ReleaseInterrupted(Exception):
    """Raised by the wrapper's SIGINT/SIGTERM handlers."""

    def __init__(self, signum: int):
        super().__init__(f"release certification interrupted by signal {signum}")
        self.signum = signum


@dataclass(frozen=True)
class ManagedProcessResult:
    """Outcome of one process-group-managed child invocation."""

    status: str
    returncode: int | None
    child_pid: int | None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def exit_code(self) -> int:
        if self.status == "INTERRUPTED":
            return 130
        if self.status != "COMPLETED" or self.returncode != 0:
            return 1
        return 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Replace *path* atomically with one fully flushed JSON object."""

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


def write_certificate(
    record: Mapping[str, Any],
    *,
    invocation_path: Path,
    latest_path: Path | None = None,
) -> None:
    """Persist both invocation-local and current status atomically."""

    if latest_path is None:
        latest_path = LATEST_CERTIFICATE
    atomic_write_json(invocation_path, record)
    atomic_write_json(latest_path, record)


def _normalized_notebook_bytes(path: Path) -> bytes:
    """Bind notebook cell sources while ignoring Jupyter runtime autosave state."""

    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise RuntimeError(f"Notebook has no cell list: {path}")
    normalized_cells = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise RuntimeError(f"Notebook cell {index} is not an object: {path}")
        cell_type = cell.get("cell_type")
        source = cell.get("source")
        if cell_type not in {"code", "markdown", "raw"} or not isinstance(
            source, (str, list)
        ):
            raise RuntimeError(f"Notebook cell {index} is malformed: {path}")
        metadata = cell.get("metadata")
        stable_metadata = {}
        if isinstance(metadata, dict) and "tags" in metadata:
            stable_metadata["tags"] = metadata["tags"]
        normalized_cells.append(
            {
                "cell_type": cell_type,
                "source": source,
                "metadata": stable_metadata,
            }
        )
    normalized = {
        "nbformat": notebook.get("nbformat"),
        "nbformat_minor": notebook.get("nbformat_minor"),
        "cells": normalized_cells,
    }
    return json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _source_items(
    root: Path,
    *,
    exclude_promoted_table3: bool = False,
) -> list[tuple[str, str, int]]:
    items: list[tuple[str, str, int]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(root)
        if not relative.parts:
            continue
        if relative.parts[0] in SOURCE_EXCLUDED_TOP_LEVEL:
            continue
        if any(part in SOURCE_EXCLUDED_DIRECTORY_NAMES for part in relative.parts):
            continue
        if path.name.startswith(".~") or path.name.endswith(".ipynb.~"):
            continue
        if (
            exclude_promoted_table3
            and len(relative.parts) >= 2
            and relative.parts[:2] == ("reference", "table3")
        ):
            continue
        if path.is_symlink():
            raise RuntimeError(
                f"Immutable Source_Code must not contain symlinks: {path}"
            )
        if path.is_file():
            if path.suffix == ".ipynb":
                content = _normalized_notebook_bytes(path)
                items.append(
                    (
                        relative.as_posix(),
                        hashlib.sha256(content).hexdigest(),
                        len(content),
                    )
                )
            else:
                items.append(
                    (relative.as_posix(), sha256_file(path), path.stat().st_size)
                )
        elif not path.is_dir():
            raise RuntimeError(
                "Immutable Source_Code must not contain special filesystem "
                f"nodes: {path}"
            )
    return items


def source_identity(root: Path | None = None) -> dict[str, Any]:
    """Return a deterministic identity for immutable release inputs."""

    if root is None:
        root = ROOT
    digest = hashlib.sha256()
    items = _source_items(root)
    for relative, file_hash, size in items:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
    return {
        "algorithm": "sha256-path-hash-size-notebook-source-v3",
        "sha256": digest.hexdigest(),
        "file_count": len(items),
    }


def promotion_source_identity(root: Path | None = None) -> dict[str, Any]:
    """Bind audited source while excluding the separately bound Table-3 payload."""

    if root is None:
        root = ROOT
    digest = hashlib.sha256()
    items = _source_items(root, exclude_promoted_table3=True)
    for relative, file_hash, size in items:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
    return {
        "algorithm": TABLE3_PROMOTION_IDENTITY_ALGORITHM,
        "sha256": digest.hexdigest(),
        "file_count": len(items),
    }


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read_float64_npy(
    payload: bytes,
    *,
    expected_shape: tuple[int, ...],
    description: str,
) -> None:
    """Validate one little-endian, C-order, finite float64 NPY member."""

    if not payload.startswith(b"\x93NUMPY") or len(payload) < 10:
        raise RuntimeError(f"{description} is not an NPY payload")
    major, minor = payload[6], payload[7]
    if (major, minor) == (1, 0):
        header_offset = 10
        header_length = struct.unpack("<H", payload[8:10])[0]
        encoding = "latin1"
    elif major in (2, 3) and minor == 0:
        if len(payload) < 12:
            raise RuntimeError(f"{description} has a truncated NPY header")
        header_offset = 12
        header_length = struct.unpack("<I", payload[8:12])[0]
        encoding = "utf-8" if major == 3 else "latin1"
    else:
        raise RuntimeError(
            f"{description} uses unsupported NPY version {major}.{minor}"
        )
    data_offset = header_offset + header_length
    if data_offset > len(payload):
        raise RuntimeError(f"{description} has a truncated NPY header")
    try:
        header = ast.literal_eval(
            payload[header_offset:data_offset].decode(encoding).strip()
        )
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise RuntimeError(f"{description} has an invalid NPY header") from error
    if not isinstance(header, dict) or set(header) != {
        "descr",
        "fortran_order",
        "shape",
    }:
        raise RuntimeError(f"{description} has an unexpected NPY header schema")
    shape = header.get("shape")
    if (
        header.get("descr") != "<f8"
        or header.get("fortran_order") is not False
        or not isinstance(shape, tuple)
        or any(type(item) is not int or item <= 0 for item in shape)
        or shape != expected_shape
    ):
        raise RuntimeError(
            f"{description} must be little-endian C-order float64 with shape "
            f"{expected_shape}; got descr={header.get('descr')!r}, "
            f"fortran_order={header.get('fortran_order')!r}, shape={shape!r}"
        )
    element_count = math.prod(shape)
    raw = payload[data_offset:]
    if len(raw) != element_count * 8:
        raise RuntimeError(
            f"{description} has {len(raw)} data bytes; expected {element_count * 8}"
        )
    if any(not math.isfinite(value[0]) for value in struct.iter_unpack("<d", raw)):
        raise RuntimeError(f"{description} contains a nonfinite value")


def _validate_parameter_npz(
    path: Path,
    declaration: Mapping[str, Any],
    *,
    label: str,
) -> None:
    names = declaration.get("array_names")
    shapes = declaration.get("array_shapes")
    if (
        declaration.get("array_replay_exact") is not True
        or not isinstance(names, list)
        or not names
        or len(names) != len(set(names))
        or not all(
            isinstance(name, str)
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
            for name in names
        )
        or not isinstance(shapes, dict)
        or set(shapes) != set(names)
    ):
        raise RuntimeError(f"canonical parameter declaration is invalid for {label}")
    normalized_shapes: dict[str, tuple[int, ...]] = {}
    for name in names:
        shape = shapes[name]
        if (
            not isinstance(shape, list)
            or not shape
            or any(type(item) is not int or item <= 0 for item in shape)
        ):
            raise RuntimeError(
                f"canonical parameter shape is invalid for {label}:{name}"
            )
        normalized_shapes[name] = tuple(shape)

    try:
        with zipfile.ZipFile(path) as archive:
            if archive.comment:
                raise RuntimeError(f"canonical NPZ has an archive comment: {path}")
            members = archive.infolist()
            member_names = [item.filename for item in members]
            expected_members = {f"{name}.npy" for name in names}
            if (
                len(member_names) != len(set(member_names))
                or set(member_names) != expected_members
            ):
                raise RuntimeError(
                    f"canonical NPZ member set changed for {label}: "
                    f"{member_names!r}"
                )
            for member in members:
                if (
                    member.is_dir()
                    or member.flag_bits & 0x1
                    or member.compress_type
                    not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                    or member.file_size <= 0
                    or member.file_size > 64 * 1024 * 1024
                ):
                    raise RuntimeError(
                        f"canonical NPZ member is unsafe for {label}: "
                        f"{member.filename}"
                    )
                name = member.filename[:-4]
                _read_float64_npy(
                    archive.read(member),
                    expected_shape=normalized_shapes[name],
                    description=f"{label}:{member.filename}",
                )
            if archive.testzip() is not None:
                raise RuntimeError(f"canonical NPZ CRC validation failed: {path}")
    except (OSError, zipfile.BadZipFile) as error:
        raise RuntimeError(f"canonical parameter artifact is not a safe NPZ: {path}") from error


def _qasm_angle_value(expression: str) -> float:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise RuntimeError(f"invalid rz angle expression: {expression!r}") from error
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Name,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.UAdd,
        ast.USub,
        ast.Load,
    )
    if any(
        not isinstance(node, allowed)
        or isinstance(node, ast.Name)
        and node.id != "pi"
        or isinstance(node, ast.Constant)
        and (type(node.value) not in {int, float})
        for node in ast.walk(parsed)
    ):
        raise RuntimeError(f"unsafe rz angle expression: {expression!r}")
    try:
        value = float(eval(compile(parsed, "<qasm-angle>", "eval"), {"__builtins__": {}}, {"pi": math.pi}))
    except (ArithmeticError, ValueError, TypeError) as error:
        raise RuntimeError(f"invalid rz angle expression: {expression!r}") from error
    if not math.isfinite(value):
        raise RuntimeError(f"nonfinite rz angle expression: {expression!r}")
    return value


def _inspect_constrained_qasm(path: Path) -> dict[str, Any]:
    """Parse the exact OPENQASM-2 subset emitted by the Table-3 compiler."""

    if path.stat().st_size > 32 * 1024 * 1024:
        raise RuntimeError(f"canonical QASM is unexpectedly large: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"canonical QASM is not UTF-8: {path}") from error
    if "\x00" in text or "\r" in text:
        raise RuntimeError(f"canonical QASM contains forbidden bytes: {path}")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4 or lines[:2] != [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
    ]:
        raise RuntimeError(f"canonical QASM header is invalid: {path}")
    register = re.fullmatch(r"qreg q\[([1-9][0-9]*)\];", lines[2])
    if register is None:
        raise RuntimeError(f"canonical QASM must declare exactly qreg q: {path}")
    qubits = int(register.group(1))
    levels = [0] * qubits
    operations: dict[str, int] = {}
    size = 0
    for line in lines[3:]:
        gate: str
        operands: tuple[int, ...]
        match = re.fullmatch(r"(sx|x) q\[([0-9]+)\];", line)
        if match is not None:
            gate = match.group(1)
            operands = (int(match.group(2)),)
        else:
            match = re.fullmatch(r"rz\(([^()\n;]+)\) q\[([0-9]+)\];", line)
            if match is not None:
                _qasm_angle_value(match.group(1))
                gate = "rz"
                operands = (int(match.group(2)),)
            else:
                match = re.fullmatch(
                    r"cx q\[([0-9]+)\],q\[([0-9]+)\];", line
                )
                if match is None:
                    raise RuntimeError(
                        f"canonical QASM contains a forbidden statement: {line!r}"
                    )
                gate = "cx"
                operands = (int(match.group(1)), int(match.group(2)))
        if (
            any(index < 0 or index >= qubits for index in operands)
            or len(set(operands)) != len(operands)
        ):
            raise RuntimeError(f"canonical QASM has invalid operands: {line!r}")
        level = 1 + max(levels[index] for index in operands)
        for index in operands:
            levels[index] = level
        operations[gate] = operations.get(gate, 0) + 1
        size += 1
    if size <= 0:
        raise RuntimeError(f"canonical QASM has no operations: {path}")
    return {
        "logical_qubits": qubits,
        "depth": max(levels),
        "size": size,
        "cx": operations.get("cx", 0),
        "operations": operations,
    }


def _validate_qasm_declaration(
    path: Path,
    declaration: Mapping[str, Any],
    *,
    label: str,
    description: str,
    expected_qubits: int,
) -> None:
    operations = declaration.get("operations")
    declared_resources_valid = (
        declaration.get("basis_gates") == TABLE3_PROTOCOL["basis_gates"]
        and int(declaration.get("optimization_level", -1))
        == TABLE3_PROTOCOL["optimization_level"]
        and int(declaration.get("seed_transpiler", -1)) == TABLE3_SEED
        and declaration.get("connectivity") == "all-to-all"
        and declaration.get("scope") == TABLE3_PROTOCOL["scope"]
        and declaration.get("derived_from_reloaded_qasm") is True
        and declaration.get("compiled_once_before_serialization") is True
        and isinstance(operations, dict)
        and bool(operations)
        and set(operations) <= set(TABLE3_PROTOCOL["basis_gates"])
        and all(
            isinstance(key, str)
            and type(value) is int
            and value > 0
            for key, value in operations.items()
        )
    )
    if not declared_resources_valid:
        raise RuntimeError(
            f"canonical {description} resource declaration is invalid for {label}"
        )
    observed = _inspect_constrained_qasm(path)
    declared = {
        key: declaration.get(key)
        for key in ("logical_qubits", "depth", "size", "cx", "operations")
    }
    if (
        observed["logical_qubits"] != expected_qubits
        or declared != observed
    ):
        raise RuntimeError(
            f"canonical {description} resources do not match QASM for {label}: "
            f"declared={declared!r}, observed={observed!r}"
        )


def _validated_identity(
    value: Any,
    *,
    algorithm: str,
    description: str,
) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"algorithm", "sha256", "file_count"}
        or value.get("algorithm") != algorithm
        or not _valid_sha256(value.get("sha256"))
        or type(value.get("file_count")) is not int
        or value["file_count"] <= 0
    ):
        raise RuntimeError(f"{description} is invalid")
    return value


def _validated_snapshot(value: Any, *, description: str) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"exists", "aggregate_sha256", "entries"}
        or value.get("exists") is not True
        or not _valid_sha256(value.get("aggregate_sha256"))
        or not isinstance(value.get("entries"), list)
    ):
        raise RuntimeError(f"{description} is invalid")
    entries = value["entries"]
    if any(
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or not isinstance(item.get("kind"), str)
        for item in entries
    ):
        raise RuntimeError(f"{description} entries are invalid")
    if entries != sorted(
        entries,
        key=lambda item: (item["path"], item["kind"]),
    ):
        raise RuntimeError(f"{description} entries are not canonical")
    observed = hashlib.sha256(
        (
            json.dumps(entries, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    if observed != value["aggregate_sha256"]:
        raise RuntimeError(f"{description} aggregate checksum is invalid")
    return value


def _validated_candidate_artifacts(
    value: Any,
    *,
    expected_scientific_artifacts: Mapping[str, tuple[str, int]],
    candidate_manifest_sha256: str,
) -> dict[str, dict[str, Any]]:
    if (
        not isinstance(value, dict)
        or set(value) != {
            "algorithm",
            "aggregate_sha256",
            "file_count",
            "files",
        }
        or value.get("algorithm") != "sha256-path-hash-size-v1"
        or not _valid_sha256(value.get("aggregate_sha256"))
        or type(value.get("file_count")) is not int
        or not isinstance(value.get("files"), list)
    ):
        raise RuntimeError("preserved bootstrap candidate manifest is invalid")
    records: dict[str, dict[str, Any]] = {}
    aggregate = hashlib.sha256()
    ordered = sorted(
        value["files"],
        key=lambda item: str(item.get("path")) if isinstance(item, dict) else "",
    )
    for record in ordered:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256", "size_bytes"}
            or not isinstance(record.get("path"), str)
            or record["path"] in records
            or not _valid_sha256(record.get("sha256"))
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] <= 0
        ):
            raise RuntimeError(
                "preserved bootstrap candidate file record is invalid"
            )
        records[record["path"]] = record
        aggregate.update(record["path"].encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(record["sha256"].encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(record["size_bytes"]).encode("ascii"))
        aggregate.update(b"\n")
    expected_paths = {
        "canonical_table3.json",
        *expected_scientific_artifacts,
    }
    if (
        set(records) != expected_paths
        or value["file_count"] != len(records)
        or value["file_count"] != 10
        or aggregate.hexdigest() != value["aggregate_sha256"]
        or records["canonical_table3.json"]["sha256"]
        != candidate_manifest_sha256
    ):
        raise RuntimeError(
            "preserved bootstrap candidate manifest is not the promoted payload"
        )
    for relative, (digest, size) in expected_scientific_artifacts.items():
        record = records[relative]
        if record["sha256"] != digest or record["size_bytes"] != size:
            raise RuntimeError(
                "preserved bootstrap candidate artifact changed during "
                f"promotion: {relative}"
            )
    return records


def _validate_promotion_evidence(
    *,
    root: Path,
    source_code: Path,
    provenance: Mapping[str, Any],
    scientific_artifacts: Mapping[str, tuple[str, int]],
) -> None:
    evidence: dict[str, Path] = {}
    for path_key, expected_relative in TABLE3_PROVENANCE_PATHS.items():
        if provenance.get(path_key) != expected_relative:
            raise RuntimeError(
                f"canonical Table-3 {path_key} is not the fixed provenance path"
            )
        path = root / expected_relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(
                f"canonical Table-3 provenance evidence is invalid: {path}"
            )
        evidence[path_key] = path
    for path_key, expected_relative in TABLE3_TOOL_PATHS.items():
        if provenance.get(path_key) != expected_relative:
            raise RuntimeError(
                f"canonical Table-3 {path_key} is not the fixed source path"
            )
        path = source_code / expected_relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(
                f"canonical Table-3 provenance tool is invalid: {path}"
            )
        expected_hash_key = path_key.replace("_path", "_sha256")
        if sha256_file(path) != provenance.get(expected_hash_key):
            raise RuntimeError(
                f"canonical Table-3 {expected_hash_key} does not match current code"
            )

    hash_bindings = {
        "audit_report_path": "audit_report_sha256",
        "bootstrap_status_path": "bootstrap_status_sha256",
        "bootstrap_helper_path": "bootstrap_helper_sha256",
    }
    for path_key, hash_key in hash_bindings.items():
        if sha256_file(evidence[path_key]) != provenance.get(hash_key):
            raise RuntimeError(
                f"canonical Table-3 {hash_key} does not match preserved evidence"
            )

    audited_identity = _validated_identity(
        provenance.get("audited_source_identity"),
        algorithm=TABLE3_PROMOTION_IDENTITY_ALGORITHM,
        description="canonical Table-3 audited source identity",
    )
    current_identity = promotion_source_identity(source_code)
    if current_identity != audited_identity:
        raise RuntimeError(
            "current Source_Code does not match the independently audited "
            "pre-promotion identity"
        )
    audit_identity = {
        "algorithm": "sha256-path-hash-size-notebook-source-v3",
        "sha256": audited_identity["sha256"],
        "file_count": audited_identity["file_count"],
    }

    report = load_json_object(evidence["audit_report_path"])
    status = load_json_object(evidence["bootstrap_status_path"])
    checks = report.get("checks")
    check_names = (
        [item.get("name") for item in checks if isinstance(item, dict)]
        if isinstance(checks, list)
        else []
    )
    if (
        int(report.get("schema_version", -1)) != 1
        or report.get("audit_kind") != "TABLE3_PRE_PROMOTION_READ_ONLY"
        or report.get("status") != "PASS"
        or report.get("candidate_expected_status") != TABLE3_CANDIDATE_STATUS
        or report.get("candidate_promoted") is not False
        or report.get("provider_imported") is not False
        or report.get("qpu_contacted") is not False
        or report.get("candidate_mutated") is not False
        or report.get("reference_mutated") is not False
        or report.get("invocation_mutated") is not False
        or not isinstance(checks, list)
        or not checks
        or any(
            not isinstance(item, dict)
            or item.get("status") != "PASS"
            or not isinstance(item.get("name"), str)
            or not item["name"]
            for item in checks
        )
        or len(check_names) != len(set(check_names))
        or not TABLE3_REQUIRED_AUDIT_CHECKS.issubset(check_names)
        or type(report.get("passed")) is not int
        or report["passed"] != len(checks)
        or type(report.get("failed")) is not int
        or report["failed"] != 0
        or "error" in report
        or report.get("source_identity_before") != audit_identity
        or report.get("source_identity_after") != audit_identity
        or report.get("bootstrap_status_sha256")
        != provenance["bootstrap_status_sha256"]
        or report.get("bootstrap_helper_sha256")
        != provenance["bootstrap_helper_sha256"]
        or not isinstance(report.get("observed_cases"), dict)
        or set(report["observed_cases"]) != set(TABLE3_LABELS)
    ):
        raise RuntimeError(
            "preserved Table-3 audit report is not an authoritative PASS"
        )

    candidate_before = _validated_snapshot(
        report.get("candidate_tree_before"),
        description="preserved candidate snapshot before audit",
    )
    candidate_after = _validated_snapshot(
        report.get("candidate_tree_after"),
        description="preserved candidate snapshot after audit",
    )
    if (
        candidate_before != candidate_after
        or candidate_before["aggregate_sha256"]
        != provenance["candidate_tree_aggregate_sha256"]
    ):
        raise RuntimeError(
            "preserved Table-3 audit report is not bound to the candidate tree"
        )
    reference_before = _validated_snapshot(
        report.get("reference_tree_before"),
        description="preserved reference snapshot before audit",
    )
    reference_after = _validated_snapshot(
        report.get("reference_tree_after"),
        description="preserved reference snapshot after audit",
    )
    if (
        reference_before != reference_after
        or any(
            item["path"] == "table3"
            or item["path"].startswith("table3/")
            for item in reference_before["entries"]
        )
    ):
        raise RuntimeError(
            "preserved Table-3 audit did not observe an absent promotion target"
        )
    invocation_before = _validated_snapshot(
        report.get("invocation_tree_before"),
        description="preserved invocation snapshot before audit",
    )
    invocation_after = _validated_snapshot(
        report.get("invocation_tree_after"),
        description="preserved invocation snapshot after audit",
    )
    if invocation_before != invocation_after:
        raise RuntimeError(
            "preserved bootstrap invocation changed during the Table-3 audit"
        )

    candidate_records = _validated_candidate_artifacts(
        status.get("candidate_artifacts"),
        expected_scientific_artifacts=scientific_artifacts,
        candidate_manifest_sha256=provenance["candidate_manifest_sha256"],
    )
    candidate_snapshot_records = {
        item["path"]: item
        for item in candidate_before["entries"]
        if item.get("kind") == "file"
    }
    if (
        len(candidate_snapshot_records) != len(candidate_before["entries"])
        or set(candidate_snapshot_records) != set(candidate_records)
        or any(
            candidate_snapshot_records[relative]
            != {"path": relative, "kind": "file", **record}
            for relative, record in candidate_records.items()
        )
    ):
        raise RuntimeError(
            "preserved audit snapshot and bootstrap candidate manifest differ"
        )

    helper = status.get("bootstrap_helper")
    stages = status.get("stages")
    if (
        int(status.get("schema_version", -1)) != 1
        or status.get("status") != "CANDIDATE_READY_NOT_CERTIFIED"
        or status.get("provider_imported") is not False
        or status.get("qpu_contacted") is not False
        or status.get("source_identity_before") != audit_identity
        or status.get("source_identity_after") != audit_identity
        or not isinstance(helper, dict)
        or set(helper) != {"path", "sha256"}
        or not isinstance(helper.get("path"), str)
        or helper.get("sha256") != provenance["bootstrap_helper_sha256"]
        or report.get("bootstrap_helper") != helper["path"]
        or report.get("bootstrap_status") != status.get("bootstrap_status")
        or report.get("candidate_directory") != status.get("candidate_directory")
        or report.get("source_code") != status.get("source_code")
        or not isinstance(stages, list)
        or [
            stage.get("name") if isinstance(stage, dict) else None
            for stage in stages
        ]
        != list(TABLE3_BOOTSTRAP_STAGE_NAMES)
    ):
        raise RuntimeError(
            "preserved Table-3 bootstrap status is not audit-bound"
        )
    for stage in stages:
        if (
            not isinstance(stage, dict)
            or stage.get("status") != "PASS"
            or type(stage.get("returncode")) is not int
            or stage["returncode"] != 0
            or not isinstance(stage.get("argv"), list)
            or not stage["argv"]
            or not all(
                isinstance(argument, str) and argument
                for argument in stage["argv"]
            )
            or not isinstance(stage.get("log"), str)
            or not stage["log"]
            or not _valid_sha256(stage.get("log_sha256"))
            or type(stage.get("log_size_bytes")) is not int
            or stage["log_size_bytes"] <= 0
        ):
            raise RuntimeError(
                "preserved Table-3 bootstrap stage evidence is invalid"
            )


def validate_promoted_table3_reference(
    manifest_path: Path | None = None,
    *,
    source_code: Path | None = None,
) -> dict[str, Any]:
    """Fail unless the exact promoted Table-3 payload is present and bound.

    This standard-library-only preflight runs before either quick or full
    certification.  It deliberately does not promote a candidate or infer
    missing files.
    """

    raw_path = TABLE3_MANIFEST if manifest_path is None else manifest_path
    raw_path = raw_path.expanduser()
    if (
        raw_path.is_symlink()
        or raw_path.parent.is_symlink()
        or not raw_path.is_file()
    ):
        raise FileNotFoundError(
            "promoted canonical Table-3 reference is missing; generate, "
            "independently audit, and explicitly promote the candidate first: "
            f"{raw_path}"
        )
    path = raw_path.resolve()
    raw_source_code = ROOT if source_code is None else source_code.expanduser()
    if raw_source_code.is_symlink() or not raw_source_code.is_dir():
        raise RuntimeError(
            f"Source_Code provenance root is invalid: {raw_source_code}"
        )
    source_code = raw_source_code.resolve()
    value = load_json_object(path)
    if int(value.get("schema_version", -1)) != 1:
        raise RuntimeError(f"unsupported canonical Table-3 schema: {path}")
    if value.get("status") != "CANONICAL_PROMOTED":
        raise RuntimeError(
            "Table-3 evidence is not explicitly promoted; certification is "
            f"blocked: {path}"
        )
    if int(value.get("seed", -1)) != TABLE3_SEED:
        raise RuntimeError(f"canonical Table-3 seed changed: {path}")
    if value.get("compilation_protocol") != TABLE3_PROTOCOL:
        raise RuntimeError(f"canonical Table-3 compilation protocol changed: {path}")

    cases = value.get("cases")
    if not isinstance(cases, dict) or set(cases) != set(TABLE3_LABELS):
        raise RuntimeError(
            "canonical Table-3 manifest must contain exactly "
            + ", ".join(TABLE3_LABELS)
        )

    root = path.parent.resolve()
    expected_files = {
        "canonical_table3.json",
        *TABLE3_PROVENANCE_PATHS.values(),
    }
    artifact_hashes: list[tuple[str, str, int]] = []
    for label in TABLE3_LABELS:
        case = cases[label]
        molecule, norb, expected_qubits = TABLE3_CASES[label]
        if (
            not isinstance(case, dict)
            or case.get("label") != label
            or case.get("molecule") != molecule
            or int(case.get("norb", -1)) != norb
            or int(case.get("n_qubits", -1)) != expected_qubits
            or int(case.get("seed", -1)) != TABLE3_SEED
            or case.get("compilation_protocol") != TABLE3_PROTOCOL
        ):
            raise RuntimeError(f"canonical Table-3 case identity changed: {label}")
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
        for description, declaration, expected_relative, path_key, hash_key in declarations:
            if not isinstance(declaration, dict):
                raise RuntimeError(
                    f"canonical {description} declaration is missing: {label}"
                )
            relative = declaration.get(path_key)
            expected_hash = declaration.get(hash_key)
            if relative != expected_relative:
                raise RuntimeError(
                    f"canonical {description} path changed for {label}: {relative!r}"
                )
            if not isinstance(expected_hash, str) or re.fullmatch(
                r"[0-9a-f]{64}", expected_hash
            ) is None:
                raise RuntimeError(
                    f"canonical {description} hash is invalid for {label}"
                )
            raw_artifact = root / expected_relative
            if raw_artifact.is_symlink():
                raise RuntimeError(
                    f"canonical {description} must not be a symlink: {raw_artifact}"
                )
            artifact = raw_artifact.resolve(strict=False)
            if root not in artifact.parents:
                raise RuntimeError(
                    f"canonical {description} escapes the manifest directory: "
                    f"{artifact}"
                )
            if not artifact.is_file():
                raise RuntimeError(
                    f"canonical {description} is missing or a symlink: {artifact}"
                )
            size = artifact.stat().st_size
            if size <= 0:
                raise RuntimeError(f"canonical artifact is empty: {artifact}")
            actual_hash = sha256_file(artifact)
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"canonical {description} checksum mismatch for {label}: "
                    f"{actual_hash} != {expected_hash}"
                )
            expected_files.add(expected_relative)
            artifact_hashes.append((expected_relative, actual_hash, size))
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
                    expected_qubits=expected_qubits,
                )
        if (
            case.get("legacy_generic_unitary") != case.get("generic_qasm")
            or case.get("structured_exact_pauli_network")
            != case.get("structured_qasm")
            or case.get("qasm_file")
            != case["structured_qasm"].get("qasm_file")
            or case.get("qasm_sha256")
            != case["structured_qasm"].get("qasm_sha256")
            or case.get("device_native") is not False
        ):
            raise RuntimeError(
                f"canonical Table-3 compatibility/resource aliases changed: {label}"
            )

    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            raise RuntimeError(
                f"canonical Table-3 directory must not contain symlinks: {candidate}"
            )
        if candidate.is_file():
            if candidate.stat().st_size <= 0:
                raise RuntimeError(f"canonical Table-3 file is empty: {candidate}")
            actual_files.add(candidate.relative_to(root).as_posix())
        elif candidate.is_dir():
            actual_directories.add(candidate.relative_to(root).as_posix())
        else:
            raise RuntimeError(
                "canonical Table-3 directory must not contain special "
                f"filesystem nodes: {candidate}"
            )
    if actual_files != expected_files:
        raise RuntimeError(
            "canonical Table-3 directory has an unexpected file set: "
            f"actual={sorted(actual_files)}, expected={sorted(expected_files)}"
        )
    if actual_directories != {"circuits", "parameters", "provenance"}:
        raise RuntimeError(
            "canonical Table-3 directory has an unexpected directory set: "
            f"{sorted(actual_directories)}"
        )

    aggregate = hashlib.sha256()
    for relative, digest, size in sorted(artifact_hashes):
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\n")
    artifact_aggregate = aggregate.hexdigest()
    provenance = value.get("promotion_provenance")
    provenance_keys = {
        "schema_version",
        "promotion_kind",
        "candidate_status",
        "candidate_manifest_sha256",
        "candidate_tree_aggregate_sha256",
        "audit_report_path",
        "audit_report_sha256",
        "bootstrap_status_path",
        "bootstrap_status_sha256",
        "bootstrap_helper_path",
        "bootstrap_helper_sha256",
        "audited_source_identity",
        "promotion_tool_path",
        "promotion_tool_sha256",
        "audit_tool_path",
        "audit_tool_sha256",
        "artifact_count",
        "artifact_aggregate_sha256",
    }
    if (
        not isinstance(provenance, dict)
        or set(provenance) != provenance_keys
        or int(provenance.get("schema_version", -1)) != 2
        or provenance.get("promotion_kind") != "TABLE3_EXPLICIT_PROMOTION"
        or provenance.get("candidate_status") != TABLE3_CANDIDATE_STATUS
        or any(
            not _valid_sha256(provenance.get(key))
            for key in (
                "candidate_manifest_sha256",
                "candidate_tree_aggregate_sha256",
                "audit_report_sha256",
                "bootstrap_status_sha256",
                "bootstrap_helper_sha256",
                "promotion_tool_sha256",
                "audit_tool_sha256",
            )
        )
        or int(provenance.get("artifact_count", -1)) != 9
        or provenance.get("artifact_aggregate_sha256") != artifact_aggregate
    ):
        raise RuntimeError("canonical Table-3 promotion provenance is invalid")
    _validate_promotion_evidence(
        root=root,
        source_code=source_code,
        provenance=provenance,
        scientific_artifacts={
            relative: (digest, size)
            for relative, digest, size in artifact_hashes
        },
    )
    return {
        "manifest": str(path),
        "manifest_sha256": sha256_file(path),
        "artifact_count": len(artifact_hashes),
        "artifact_aggregate_sha256": artifact_aggregate,
        "promotion_provenance": provenance,
        "status": "PASS",
    }


def artifact_manifest(run_directory: Path) -> dict[str, Any]:
    """Hash every regular artifact emitted by one reproduction invocation."""

    run_directory = run_directory.resolve()
    files: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(run_directory.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise RuntimeError(f"Generated artifact must not be a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError(
                f"Generated artifact must be a regular file: {path}"
            )
        relative = path.relative_to(run_directory).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        files.append({"path": relative, "sha256": digest, "size_bytes": size})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\n")
    if not files:
        raise RuntimeError(f"No generated artifacts found in {run_directory}")
    return {
        "algorithm": "sha256-path-hash-size-v1",
        "aggregate_sha256": aggregate.hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return value


def validate_reproduction_latest(
    *,
    results_root: Path,
    invocation_id: str,
    mode: str,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Accept only the PASS state emitted by this exact child invocation."""

    latest_path = results_root / "latest_run.json"
    if not latest_path.is_file():
        raise RuntimeError("Current reproduction did not write latest_run.json")
    latest = load_json_object(latest_path)
    if latest.get("invocation_id") != invocation_id:
        raise RuntimeError(
            "Refusing stale latest_run.json: "
            f"{latest.get('invocation_id')!r} != {invocation_id!r}"
        )
    if latest.get("status") != "PASS" or latest.get("mode") != mode:
        raise RuntimeError(f"Current reproduction did not report PASS/{mode}: {latest}")

    run_directory = Path(str(latest.get("run_directory", ""))).resolve()
    results_resolved = results_root.resolve()
    if not is_within(run_directory, results_resolved) or run_directory == results_resolved:
        raise RuntimeError(f"Run directory escapes judge results root: {run_directory}")
    if not run_directory.is_dir():
        raise RuntimeError(f"Run directory is missing: {run_directory}")

    summary_path = Path(str(latest.get("summary", ""))).resolve()
    execution_path = Path(str(latest.get("execution_record", ""))).resolve()
    for label, path in (("summary", summary_path), ("execution record", execution_path)):
        if not is_within(path, run_directory) or not path.is_file():
            raise RuntimeError(f"Current {label} is missing or outside the run: {path}")

    summary = load_json_object(summary_path)
    execution = load_json_object(execution_path)
    if summary.get("status") != "PASS":
        raise RuntimeError("Current validation summary is not PASS")
    if execution.get("status") != "PASS":
        raise RuntimeError("Current execution record is not PASS")
    if execution.get("invocation_id") != invocation_id:
        raise RuntimeError("Execution record is not bound to the current invocation")
    if execution.get("mode") != mode:
        raise RuntimeError("Execution record mode does not match the certificate")
    if execution.get("qpu_contacted") is not False:
        raise RuntimeError("Execution record does not prove qpu_contacted=false")
    if execution.get("provider_imported") is not False:
        raise RuntimeError("Execution record does not prove provider_imported=false")
    return latest, run_directory, execution


def terminate_process_group(
    process: subprocess.Popen[str],
    *,
    grace_seconds: float = 5.0,
) -> int:
    """Terminate, escalate if necessary, and reap a child process group."""

    return_code = process.poll()
    if return_code is not None:
        return return_code
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait(timeout=grace_seconds)


def run_managed_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_path: Path,
    on_started: Callable[[int], None] | None = None,
) -> ManagedProcessResult:
    """Stream one child, isolate its process group, and cleanly handle interrupts."""

    process: subprocess.Popen[str] | None = None

    def handle_signal(signum: int, _frame: object) -> None:
        raise ReleaseInterrupted(signum)

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as archive:
            process = subprocess.Popen(
                list(command),
                cwd=cwd,
                env=dict(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            if on_started is not None:
                on_started(process.pid)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                archive.write(line)
            return ManagedProcessResult(
                status="COMPLETED",
                returncode=process.wait(),
                child_pid=process.pid,
            )
    except (KeyboardInterrupt, ReleaseInterrupted) as error:
        return_code = (
            terminate_process_group(process) if process is not None else None
        )
        return ManagedProcessResult(
            status="INTERRUPTED",
            returncode=return_code,
            child_pid=process.pid if process is not None else None,
            error_type=type(error).__name__,
            error_message=str(error),
        )
    except BaseException as error:
        return_code = (
            terminate_process_group(process) if process is not None else None
        )
        return ManagedProcessResult(
            status="FAILED",
            returncode=return_code,
            child_pid=process.pid if process is not None else None,
            error_type=type(error).__name__,
            error_message=str(error),
        )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if process is not None and process.stdout is not None:
            process.stdout.close()


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def refuse_concurrent_certificate(path: Path | None = None) -> None:
    """Reject a genuinely live prior RUNNING record, but permit stale recovery."""

    if path is None:
        path = LATEST_CERTIFICATE
    if not path.is_file():
        return
    try:
        record = load_json_object(path)
        pid = int(record.get("wrapper_pid", -1))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return
    if record.get("status") == "RUNNING" and pid > 0 and process_exists(pid):
        raise RuntimeError(
            f"Certification invocation {record.get('invocation_id')} is still running "
            f"under PID {pid}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="run the quick certificate")
    mode.add_argument("--full", action="store_true", help="run the full certificate")
    parser.add_argument("--invocation-id", help="optional safe identifier for audit systems")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    mode = "full" if args.full else "quick"
    invocation_id = args.invocation_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex
    )
    if INVOCATION.fullmatch(invocation_id) is None:
        raise SystemExit("Invalid --invocation-id; use 8-128 safe identifier characters")

    try:
        refuse_concurrent_certificate()
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    certificate_dir = CERTIFICATES / invocation_id
    certificate_path = certificate_dir / "certificate.json"
    child_log = certificate_dir / "reproduce_console.log"
    source_before = source_identity()
    started_monotonic = time.monotonic()
    record: dict[str, Any] = {
        "schema_version": 1,
        "invocation_id": invocation_id,
        "status": "RUNNING",
        "mode": mode,
        "started_utc": utc_now(),
        "completed_utc": None,
        "wall_seconds": None,
        "wrapper_pid": os.getpid(),
        "child_pid": None,
        "child_returncode": None,
        "environment": None,
        "promoted_table3_reference": None,
        "source_identity_before": source_before,
        "source_identity_after": None,
        "generated_artifacts": None,
        "qpu_contacted": False,
        "provider_imported": False,
        "child_log": str(child_log),
    }
    write_certificate(record, invocation_path=certificate_path)

    def handle_signal(signum: int, _frame: object) -> None:
        raise ReleaseInterrupted(signum)

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    exit_code = 1
    process_result: ManagedProcessResult | None = None
    try:
        if mode == "quick":
            record["promoted_table3_reference"] = (
                validate_promoted_table3_reference(
                    ROOT / "reference" / "table3" / "canonical_table3.json"
                )
            )
        else:
            record["promoted_table3_reference"] = {
                "status": "NOT_REQUIRED_FOR_FRESH_FULL_REPRODUCTION",
                "reason": (
                    "full mode generates and independently validates a fresh "
                    "noncanonical Table-3 candidate"
                ),
            }
        environment = resolve_environment(ROOT)
        record["environment"] = str(environment)
        check_disk_space(environment, mode="reuse")
        environment_python = environment / "bin" / "python"
        if not environment_python.is_file():
            raise FileNotFoundError(
                "pinned environment missing; run `bash setup.sh` first"
            )

        command = [
            str(environment_python),
            "-I",
            "-B",
            str(ROOT / "reproduce.py"),
            f"--{mode}",
            "--invocation-id",
            invocation_id,
        ]
        child_env = sanitized_environment(os.environ, env_dir=environment)

        def child_started(pid: int) -> None:
            record["child_pid"] = pid
            write_certificate(record, invocation_path=certificate_path)

        process_result = run_managed_process(
            command,
            cwd=ROOT,
            env=child_env,
            log_path=child_log,
            on_started=child_started,
        )
        record["child_pid"] = process_result.child_pid
        record["child_returncode"] = process_result.returncode
        if process_result.status == "INTERRUPTED":
            record["status"] = "INTERRUPTED"
            record["error"] = {
                "type": process_result.error_type or "ReleaseInterrupted",
                "message": process_result.error_message or "interrupted",
            }
            exit_code = 130
        elif process_result.status != "COMPLETED":
            raise RuntimeError(
                f"managed child failed: {process_result.error_type}: "
                f"{process_result.error_message}"
            )
        elif process_result.returncode != 0:
            raise ChildProcessError(
                f"reproduce.py exited {process_result.returncode}"
            )

        if record["status"] == "INTERRUPTED":
            raise ReleaseInterrupted(signal.SIGINT)

        source_after = source_identity()
        record["source_identity_after"] = source_after
        if source_after != source_before:
            raise RuntimeError("Source_Code identity changed during certification")
        latest, run_directory, execution = validate_reproduction_latest(
            results_root=RESULTS,
            invocation_id=invocation_id,
            mode=mode,
        )
        record["reproduction_latest"] = latest
        record["execution_record"] = execution
        record["generated_artifacts"] = artifact_manifest(run_directory)
        record["status"] = "PASS"
        exit_code = 0
    except (KeyboardInterrupt, ReleaseInterrupted) as caught:
        record["status"] = "INTERRUPTED"
        if "error" not in record:
            record["error"] = {"type": type(caught).__name__, "message": str(caught)}
        exit_code = 130
    except BaseException as caught:
        record["status"] = "FAILED"
        record["error"] = {"type": type(caught).__name__, "message": str(caught)}
        exit_code = 1
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if record["source_identity_after"] is None:
            try:
                record["source_identity_after"] = source_identity()
            except OSError as error:
                record["source_identity_error"] = str(error)
        record["completed_utc"] = utc_now()
        record["wall_seconds"] = time.monotonic() - started_monotonic
        if child_log.is_file():
            record["child_log_sha256"] = sha256_file(child_log)
        write_certificate(record, invocation_path=certificate_path)

    if record["status"] == "PASS":
        print(f"PASS: invocation-bound certificate written to {certificate_path}")
    else:
        print(
            f"{record['status']}: certificate written to {certificate_path}: "
            f"{record.get('error', {}).get('message', '')}",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
