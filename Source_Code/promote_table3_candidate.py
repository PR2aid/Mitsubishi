#!/usr/bin/env python3
"""Atomically promote one independently audited Table-3 candidate.

Promotion is intentionally separate from generation and audit.  This command
accepts only the exact noncanonical ten-file candidate that a PASS read-only
audit observed, preserves all nine binary/QASM artifacts byte-for-byte, and
publishes into an absent ``reference/table3`` directory with one atomic rename.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat as stat_module
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import uuid


ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_table3_candidate import (  # noqa: E402
    Audit,
    CANDIDATE_STATUS,
    EXPECTED_STAGE_NAMES,
    EXPECTED_CANDIDATE_DIRECTORIES,
    EXPECTED_CANDIDATE_FILES,
    READY_STATUS,
    candidate_tree_manifest,
    run_audit,
    tree_snapshot,
    validate_output_path,
)
from certify_release import (  # noqa: E402
    TABLE3_LABELS,
    TABLE3_SEED,
    promotion_source_identity,
    sha256_file,
    source_identity,
    validate_promoted_table3_reference,
)


AUDIT_PASS_FIELDS = frozenset(
    {
        "schema_version",
        "audit_kind",
        "status",
        "started_utc",
        "completed_utc",
        "wall_seconds",
        "source_code",
        "bootstrap_helper",
        "bootstrap_helper_sha256",
        "bootstrap_status",
        "bootstrap_status_sha256",
        "candidate_directory",
        "output",
        "candidate_expected_status",
        "candidate_promoted",
        "provider_imported",
        "qpu_contacted",
        "source_identity_before",
        "source_identity_after",
        "reference_tree_before",
        "reference_tree_after",
        "invocation_tree_before",
        "invocation_tree_after",
        "candidate_tree_before",
        "candidate_tree_after",
        "candidate_mutated",
        "reference_mutated",
        "invocation_mutated",
        "observed_cases",
        "checks",
        "passed",
        "failed",
    }
)
AUDIT_COMPARISON_EXCLUDED_FIELDS = frozenset(
    {
        "started_utc",
        "completed_utc",
        "wall_seconds",
        "output",
        # The first report itself may have been written inside the bootstrap
        # invocation directory.  Each audit must still prove its own before/
        # after snapshot is unchanged, but the two snapshots may legitimately
        # differ by that already-completed report.
        "invocation_tree_before",
        "invocation_tree_after",
    }
)
REQUIRED_AUDIT_CHECK_NAMES = frozenset(
    {
        "Source_Code is hardened and reference/table3 is absent",
        "bootstrap helper path and bytes are explicitly bound",
        "bootstrap invocation and execution paths are exact",
        "bootstrap candidate-ready status remains noncertifying",
        "bootstrap provider/QPU boundary",
        "bootstrap exact four-stage sequence is declared",
        *(
            f"bootstrap {name} argv and final log bind"
            for name in EXPECTED_STAGE_NAMES
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
PROVENANCE_PATHS = {
    "audit_report_path": "provenance/audit_report.json",
    "bootstrap_status_path": "provenance/bootstrap_status.json",
    "bootstrap_helper_path": "provenance/bootstrap_helper.py",
}
TOOL_PATHS = {
    "promotion_tool_path": "promote_table3_candidate.py",
    "audit_tool_path": "audit_table3_candidate.py",
}
EXECUTING_PROMOTION_TOOL = Path(__file__).resolve()
EXECUTING_AUDIT_TOOL = Path(
    sys.modules[run_audit.__module__].__file__  # type: ignore[arg-type]
).resolve()


class CommittedPromotionIntegrityError(RuntimeError):
    """The kernel committed bytes, but their public path cannot be proven."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json_object_bytes(payload: bytes, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{description} is not a UTF-8 JSON object") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not a JSON object")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve_without_leaf_symlink(path: Path, *, description: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise RuntimeError(f"{description} must not be a symlink: {expanded}")
    return expanded.resolve()


def _read_regular_file_bytes(path: Path, *, description: str) -> bytes:
    """Capture one nonempty regular file once without following a symlink."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for promotion evidence capture")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat_module.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise RuntimeError(
                f"{description} must be a nonempty regular file: {path}"
            )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if any(
            getattr(before, field) != getattr(after, field)
            for field in stable_fields
        ):
            raise RuntimeError(f"{description} changed while it was captured: {path}")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise RuntimeError(
                f"{description} size changed while it was captured: {path}"
            )
        return payload
    finally:
        os.close(descriptor)


def _write_regular_file_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_all_fd(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("short write while publishing external report")
        written += count


def _write_regular_file_at(
    directory_fd: int,
    name: str,
    payload: bytes,
) -> tuple[int, int]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for report publication")
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory_fd,
    )
    created_identity = _stat_identity(os.fstat(descriptor))
    try:
        _write_all_fd(descriptor, payload)
        os.fsync(descriptor)
        observed = os.fstat(descriptor)
        if (
            not stat_module.S_ISREG(observed.st_mode)
            or observed.st_size != len(payload)
        ):
            raise RuntimeError(
                "prepared external report file identity/size is invalid"
            )
        return _stat_identity(observed)
    except BaseException:
        try:
            _unlink_owned_at(
                directory_fd,
                name,
                expected_identity=created_identity,
            )
        except BaseException:
            pass
        raise
    finally:
        os.close(descriptor)


def _read_regular_file_at(
    directory_fd: int,
    name: str,
    *,
    description: str,
) -> tuple[bytes, tuple[int, int]]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("O_NOFOLLOW is required for report verification")
    descriptor = os.open(
        name,
        os.O_RDONLY | os.O_NOFOLLOW,
        dir_fd=directory_fd,
    )
    try:
        before = os.fstat(descriptor)
        if not stat_module.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise RuntimeError(f"{description} is not a nonempty regular file")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        if (
            _stat_identity(before) != _stat_identity(after)
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
        ):
            raise RuntimeError(f"{description} changed while being verified")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise RuntimeError(f"{description} size changed while being verified")
        return payload, _stat_identity(before)
    finally:
        os.close(descriptor)


def _unlink_owned_at(
    directory_fd: int,
    name: str,
    *,
    expected_identity: tuple[int, int] | None,
) -> bool:
    if expected_identity is None:
        return False
    try:
        observed = os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return True
    if (
        not stat_module.S_ISREG(observed.st_mode)
        or _stat_identity(observed) != expected_identity
    ):
        return False
    os.unlink(name, dir_fd=directory_fd)
    return True


def _entry_absent_at(directory_fd: int, name: str) -> bool:
    try:
        os.stat(
            name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return True
    return False


def _cleanup_external_report_paths(
    directory_fd: int,
    *,
    output_name: str,
    prepared_name: str,
    expected_identity: tuple[int, int] | None,
) -> tuple[bool, bool, bool, list[str]]:
    """Remove owned report names and prove their durable absence."""

    removed_output = False
    removed_prepared = False
    cleanup_errors: list[str] = []
    try:
        removed_output = _unlink_owned_at(
            directory_fd,
            output_name,
            expected_identity=expected_identity,
        )
    except BaseException as cleanup_error:
        cleanup_errors.append(
            f"output cleanup {type(cleanup_error).__name__}: "
            f"{cleanup_error}"
        )
    try:
        removed_prepared = _unlink_owned_at(
            directory_fd,
            prepared_name,
            expected_identity=expected_identity,
        )
    except BaseException as cleanup_error:
        cleanup_errors.append(
            f"prepared cleanup {type(cleanup_error).__name__}: "
            f"{cleanup_error}"
        )

    cleanup_proven = False
    try:
        names_absent = _entry_absent_at(
            directory_fd,
            output_name,
        ) and _entry_absent_at(
            directory_fd,
            prepared_name,
        )
        if names_absent:
            _fsync_directory_fd(directory_fd)
            cleanup_proven = _entry_absent_at(
                directory_fd,
                output_name,
            ) and _entry_absent_at(
                directory_fd,
                prepared_name,
            )
    except BaseException as cleanup_error:
        cleanup_errors.append(
            "output-cleanup proof "
            f"{type(cleanup_error).__name__}: {cleanup_error}"
        )
    return (
        removed_output,
        removed_prepared,
        cleanup_proven,
        cleanup_errors,
    )


def _validate_committed_public_reference(
    *,
    reference_parent: Path,
    reference_parent_identity: tuple[int, int],
    destination: Path,
    destination_identity: tuple[int, int],
    expected_validation: Mapping[str, Any],
    source_code: Path,
) -> None:
    """Prove the public path still names the exact validated committed tree."""

    current_reference_parent_stat = os.stat(
        reference_parent,
        follow_symlinks=False,
    )
    current_destination_stat = os.stat(
        destination,
        follow_symlinks=False,
    )
    if (
        not stat_module.S_ISDIR(current_reference_parent_stat.st_mode)
        or _stat_identity(current_reference_parent_stat)
        != reference_parent_identity
        or not stat_module.S_ISDIR(current_destination_stat.st_mode)
        or _stat_identity(current_destination_stat) != destination_identity
    ):
        raise RuntimeError(
            "public reference/table3 path is not the exact committed directory"
        )
    installed_validation = validate_promoted_table3_reference(
        destination / "canonical_table3.json",
        source_code=source_code,
    )
    if installed_validation != expected_validation:
        raise RuntimeError("installed Table-3 validation changed after commit")
    verified_reference_parent_stat = os.stat(
        reference_parent,
        follow_symlinks=False,
    )
    verified_destination_stat = os.stat(
        destination,
        follow_symlinks=False,
    )
    if (
        _stat_identity(verified_reference_parent_stat)
        != reference_parent_identity
        or _stat_identity(verified_destination_stat) != destination_identity
    ):
        raise RuntimeError(
            "public Table-3 path identity changed during postcommit validation"
        )


def _candidate_file_bytes(
    candidate: Path,
    observed_manifest: Mapping[str, Any],
) -> dict[str, bytes]:
    records = observed_manifest.get("files")
    if not isinstance(records, list):
        raise RuntimeError("candidate transport manifest has no file records")
    by_path = {
        record.get("path"): record
        for record in records
        if isinstance(record, dict)
    }
    if (
        len(by_path) != len(records)
        or set(by_path) != set(EXPECTED_CANDIDATE_FILES)
    ):
        raise RuntimeError("candidate transport manifest file records are not exact")
    captured: dict[str, bytes] = {}
    for relative in sorted(EXPECTED_CANDIDATE_FILES):
        record = by_path[relative]
        payload = _read_regular_file_bytes(
            candidate / relative,
            description=f"candidate file {relative}",
        )
        if (
            record.get("sha256") != _sha256_bytes(payload)
            or record.get("size_bytes") != len(payload)
        ):
            raise RuntimeError(
                f"candidate file changed from its audited transport record: "
                f"{relative}"
            )
        captured[relative] = payload
    return captured


def artifact_aggregate(
    candidate_files: Mapping[str, bytes],
) -> tuple[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for relative in sorted(
        set(EXPECTED_CANDIDATE_FILES) - {"canonical_table3.json"}
    ):
        payload = candidate_files[relative]
        digest = _sha256_bytes(payload)
        size = len(payload)
        records.append(
            {"path": relative, "sha256": digest, "size_bytes": size}
        )
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(str(size).encode("ascii"))
        aggregate.update(b"\n")
    return aggregate.hexdigest(), records


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_directory_fd(path: Path, *, description: str) -> int:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError(
            "O_NOFOLLOW and O_DIRECTORY are required for atomic promotion"
        )
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
    )
    if not stat_module.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise RuntimeError(f"{description} is not a directory: {path}")
    return descriptor


def _fsync_directory_fd(descriptor: int) -> None:
    os.fsync(descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _rename_no_replace_at(
    source_directory_fd: int,
    source_name: str,
    destination_directory_fd: int,
    destination_name: str,
    *,
    expected_source_identity: tuple[int, int] | None = None,
    expected_source_is_directory: bool | None = None,
    expected_source_sha256: str | None = None,
    expected_source_size: int | None = None,
) -> None:
    """Linux atomic directory publication that cannot replace any target."""

    if sys.platform != "linux":
        raise RuntimeError(
            "atomic no-replace promotion requires Linux renameat2"
        )
    if expected_source_identity is not None:
        source_stat = os.stat(
            source_name,
            dir_fd=source_directory_fd,
            follow_symlinks=False,
        )
        source_is_directory = stat_module.S_ISDIR(source_stat.st_mode)
        if (
            _stat_identity(source_stat) != expected_source_identity
            or expected_source_is_directory is not None
            and source_is_directory != expected_source_is_directory
        ):
            raise RuntimeError(
                "atomic publication source name no longer identifies the "
                "validated object"
            )
    if expected_source_sha256 is not None or expected_source_size is not None:
        source_payload, source_identity = _read_regular_file_at(
            source_directory_fd,
            source_name,
            description="atomic publication source",
        )
        if (
            source_identity != expected_source_identity
            or expected_source_sha256 is not None
            and _sha256_bytes(source_payload) != expected_source_sha256
            or expected_source_size is not None
            and len(source_payload) != expected_source_size
        ):
            raise RuntimeError(
                "atomic publication source bytes no longer match the "
                "validated payload"
            )
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError(
            "atomic no-replace promotion requires libc renameat2"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_directory_fd,
        os.fsencode(source_name),
        destination_directory_fd,
        os.fsencode(destination_name),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            "promotion destination appeared concurrently",
            destination_name,
        )
    raise OSError(
        error_number,
        os.strerror(error_number),
        f"{source_name} -> {destination_name}",
    )


def _remove_owned_path(
    path: Path,
    *,
    expected_identity: tuple[int, int] | None,
    directory: bool,
) -> bool:
    """Remove only the exact inode created by this promotion."""

    try:
        observed = os.stat(path, follow_symlinks=False)
    except FileNotFoundError:
        return True
    if (
        expected_identity is None
        or _stat_identity(observed) != expected_identity
        or stat_module.S_ISDIR(observed.st_mode) != directory
    ):
        return False
    if directory:
        shutil.rmtree(path)
    else:
        path.unlink()
    return True


def _audit_comparison_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key not in AUDIT_COMPARISON_EXCLUDED_FIELDS
    }


def _verify_complete_pass_audit(
    report: Mapping[str, Any],
    *,
    description: str,
    current_source_identity: Mapping[str, Any],
    current_candidate_snapshot: Mapping[str, Any],
    current_reference_snapshot: Mapping[str, Any],
    bootstrap_helper_path: Path,
    bootstrap_helper_sha256: str,
    bootstrap_status_path: Path,
    bootstrap_sha256: str,
    source_code: Path,
    candidate_directory: Path,
) -> None:
    checks = report.get("checks")
    check_names = (
        [
            item.get("name")
            for item in checks
            if isinstance(item, dict)
        ]
        if isinstance(checks, list)
        else []
    )
    all_checks_pass = (
        isinstance(checks, list)
        and bool(checks)
        and all(
            isinstance(item, dict)
            and item.get("status") == "PASS"
            and isinstance(item.get("name"), str)
            and bool(item["name"])
            for item in checks
        )
    )
    conditions = {
        "complete_schema": set(report) == AUDIT_PASS_FIELDS,
        "schema_version": int(report.get("schema_version", -1)) == 1,
        "audit_kind": report.get("audit_kind")
        == "TABLE3_PRE_PROMOTION_READ_ONLY",
        "status": report.get("status") == "PASS",
        "candidate_boundary": report.get("candidate_expected_status")
        == CANDIDATE_STATUS
        and report.get("candidate_promoted") is False,
        "provider_boundary": report.get("provider_imported") is False
        and report.get("qpu_contacted") is False,
        "all_checks": all_checks_pass
        and len(check_names) == len(set(check_names))
        and REQUIRED_AUDIT_CHECK_NAMES.issubset(check_names)
        and int(report.get("failed", -1)) == 0
        and int(report.get("passed", -1)) == len(checks or []),
        "observed_cases": isinstance(report.get("observed_cases"), dict)
        and set(report["observed_cases"]) == set(TABLE3_LABELS),
        "exact_inputs": report.get("source_code") == str(source_code)
        and report.get("bootstrap_helper") == str(bootstrap_helper_path)
        and report.get("bootstrap_status") == str(bootstrap_status_path)
        and report.get("candidate_directory") == str(candidate_directory),
        "source_identity": report.get("source_identity_before")
        == report.get("source_identity_after")
        == current_source_identity,
        "candidate_snapshot": report.get("candidate_tree_before")
        == report.get("candidate_tree_after")
        == current_candidate_snapshot
        and report.get("candidate_mutated") is False,
        "reference_snapshot": report.get("reference_tree_before")
        == report.get("reference_tree_after")
        == current_reference_snapshot
        and report.get("reference_mutated") is False,
        "invocation_snapshot": report.get("invocation_tree_before")
        == report.get("invocation_tree_after")
        and report.get("invocation_mutated") is False,
        "bootstrap_hash": report.get("bootstrap_status_sha256")
        == bootstrap_sha256,
        "bootstrap_helper_hash": report.get("bootstrap_helper_sha256")
        == bootstrap_helper_sha256,
    }
    failed = sorted(name for name, passed in conditions.items() if not passed)
    if failed:
        raise RuntimeError(
            f"{description} is not a complete promotion-safe audit: "
            + ", ".join(failed)
        )


def _authenticate_audit_report(
    supplied_report: Mapping[str, Any],
    authoritative_report: Mapping[str, Any],
    **verification: Any,
) -> None:
    _verify_complete_pass_audit(
        authoritative_report,
        description="fresh authoritative candidate audit",
        **verification,
    )
    _verify_complete_pass_audit(
        supplied_report,
        description="supplied candidate audit report",
        **verification,
    )
    if _audit_comparison_payload(supplied_report) != _audit_comparison_payload(
        authoritative_report
    ):
        raise RuntimeError(
            "supplied candidate audit report does not match the fresh "
            "authoritative audit"
        )


def _verify_bootstrap_status(
    status: Mapping[str, Any],
    *,
    current_source_identity: Mapping[str, Any],
    observed_candidate_manifest: Mapping[str, Any],
) -> None:
    if (
        int(status.get("schema_version", -1)) != 1
        or status.get("status") != READY_STATUS
        or status.get("provider_imported") is not False
        or status.get("qpu_contacted") is not False
        or status.get("source_identity_before")
        != status.get("source_identity_after")
        or status.get("source_identity_after") != current_source_identity
        or status.get("candidate_artifacts") != observed_candidate_manifest
    ):
        raise RuntimeError("bootstrap status is not bound to this audited candidate")


def promote(
    *,
    source_code: Path,
    candidate_directory: Path,
    audit_report_path: Path,
    bootstrap_helper_path: Path,
    bootstrap_status_path: Path,
    output: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    source_code = _resolve_without_leaf_symlink(
        source_code,
        description="Source_Code",
    )
    candidate_directory = _resolve_without_leaf_symlink(
        candidate_directory,
        description="candidate directory",
    )
    audit_report_path = _resolve_without_leaf_symlink(
        audit_report_path,
        description="audit report",
    )
    bootstrap_helper_path = _resolve_without_leaf_symlink(
        bootstrap_helper_path,
        description="bootstrap helper",
    )
    bootstrap_status_path = _resolve_without_leaf_symlink(
        bootstrap_status_path,
        description="bootstrap status",
    )
    candidate_directory_stat = os.stat(
        candidate_directory,
        follow_symlinks=False,
    )
    if not stat_module.S_ISDIR(candidate_directory_stat.st_mode):
        raise RuntimeError(
            f"candidate boundary is not a directory: {candidate_directory}"
        )
    candidate_directory_identity = (
        candidate_directory_stat.st_dev,
        candidate_directory_stat.st_ino,
    )
    destination = source_code / "reference" / "table3"
    reference_parent = destination.parent
    if (
        destination.exists()
        or destination.is_symlink()
        or not reference_parent.is_dir()
        or reference_parent.is_symlink()
    ):
        raise RuntimeError(
            "promotion destination must be an absent reference/table3 below "
            f"a regular existing reference directory: {destination}"
        )
    reference_parent_stat = os.stat(reference_parent, follow_symlinks=False)
    reference_parent_identity = (
        reference_parent_stat.st_dev,
        reference_parent_stat.st_ino,
    )
    output = validate_output_path(
        output,
        source_code=source_code,
        candidate_directory=candidate_directory,
    )
    for description, path in (
        ("audit report", audit_report_path),
        ("bootstrap helper", bootstrap_helper_path),
        ("bootstrap status", bootstrap_status_path),
    ):
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"{description} must be a nonempty regular file: {path}")

    source_before = source_identity(source_code)
    promotion_source_before = promotion_source_identity(source_code)
    reference_before = tree_snapshot(reference_parent)
    candidate_audit = Audit()
    observed_candidate_manifest = candidate_tree_manifest(
        candidate_audit, candidate_directory
    )
    if candidate_audit.status != "PASS":
        raise RuntimeError("candidate transport boundary did not pass")
    candidate_before = tree_snapshot(candidate_directory)
    candidate_files = _candidate_file_bytes(
        candidate_directory,
        observed_candidate_manifest,
    )
    candidate_manifest_bytes = candidate_files["canonical_table3.json"]
    candidate_manifest = load_json_object_bytes(
        candidate_manifest_bytes,
        description="candidate canonical_table3.json",
    )
    if (
        candidate_manifest.get("status") != CANDIDATE_STATUS
        or "promotion_provenance" in candidate_manifest
        or int(candidate_manifest.get("schema_version", -1)) != 1
        or int(candidate_manifest.get("seed", -1)) != TABLE3_SEED
        or not isinstance(candidate_manifest.get("cases"), dict)
        or set(candidate_manifest["cases"]) != set(TABLE3_LABELS)
    ):
        raise RuntimeError("candidate manifest is not an exact noncanonical payload")
    artifact_digest, artifact_records = artifact_aggregate(candidate_files)

    bootstrap_helper_bytes = _read_regular_file_bytes(
        bootstrap_helper_path,
        description="bootstrap helper",
    )
    bootstrap_status_bytes = _read_regular_file_bytes(
        bootstrap_status_path,
        description="bootstrap status",
    )
    supplied_audit_bytes = _read_regular_file_bytes(
        audit_report_path,
        description="supplied candidate audit report",
    )
    bootstrap_helper_sha = _sha256_bytes(bootstrap_helper_bytes)
    bootstrap_sha = _sha256_bytes(bootstrap_status_bytes)
    supplied_audit_report = load_json_object_bytes(
        supplied_audit_bytes,
        description="supplied candidate audit report",
    )
    bootstrap_status = load_json_object_bytes(
        bootstrap_status_bytes,
        description="bootstrap status",
    )
    with tempfile.TemporaryDirectory(
        prefix="table3-promotion-authoritative-audit-"
    ) as audit_temporary:
        authoritative_path = (
            Path(audit_temporary) / "authoritative_audit.json"
        )
        returned_audit_report = run_audit(
            source_code=source_code,
            bootstrap_helper_path=bootstrap_helper_path,
            bootstrap_status_path=bootstrap_status_path,
            candidate_directory=candidate_directory,
            output=authoritative_path,
        )
        if (
            authoritative_path.is_symlink()
            or not authoritative_path.is_file()
            or authoritative_path.stat().st_size <= 0
        ):
            raise RuntimeError(
                "fresh authoritative candidate audit did not emit a regular "
                "nonempty report"
            )
        authoritative_audit_bytes = _read_regular_file_bytes(
            authoritative_path,
            description="fresh authoritative candidate audit report",
        )
        authoritative_audit_report = load_json_object_bytes(
            authoritative_audit_bytes,
            description="fresh authoritative candidate audit report",
        )
        if authoritative_audit_report != returned_audit_report:
            raise RuntimeError(
                "fresh authoritative candidate audit return value does not "
                "match its persisted report"
            )
        _authenticate_audit_report(
            supplied_audit_report,
            authoritative_audit_report,
            current_source_identity=source_before,
            current_candidate_snapshot=candidate_before,
            current_reference_snapshot=reference_before,
            bootstrap_helper_path=bootstrap_helper_path,
            bootstrap_helper_sha256=bootstrap_helper_sha,
            bootstrap_status_path=bootstrap_status_path,
            bootstrap_sha256=bootstrap_sha,
            source_code=source_code,
            candidate_directory=candidate_directory,
        )
    if (
        _read_regular_file_bytes(
            bootstrap_helper_path,
            description="bootstrap helper after authoritative audit",
        )
        != bootstrap_helper_bytes
        or _read_regular_file_bytes(
            bootstrap_status_path,
            description="bootstrap status after authoritative audit",
        )
        != bootstrap_status_bytes
        or _candidate_file_bytes(
            candidate_directory,
            observed_candidate_manifest,
        )
        != candidate_files
    ):
        raise RuntimeError(
            "candidate/bootstrap evidence changed during the authoritative audit"
        )
    _verify_bootstrap_status(
        bootstrap_status,
        current_source_identity=source_before,
        observed_candidate_manifest=observed_candidate_manifest,
    )

    promotion_tool = source_code / "promote_table3_candidate.py"
    audit_tool = source_code / "audit_table3_candidate.py"
    for description, path in (
        ("promotion tool", promotion_tool),
        ("audit tool", audit_tool),
    ):
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"{description} is not source-bound: {path}")
    promotion_tool_bytes = _read_regular_file_bytes(
        promotion_tool,
        description="selected Source_Code promotion tool",
    )
    audit_tool_bytes = _read_regular_file_bytes(
        audit_tool,
        description="selected Source_Code audit tool",
    )
    if promotion_tool_bytes != _read_regular_file_bytes(
        EXECUTING_PROMOTION_TOOL,
        description="executing promotion implementation",
    ):
        raise RuntimeError(
            "selected Source_Code promotion tool does not match the executing "
            "promotion implementation"
        )
    if audit_tool_bytes != _read_regular_file_bytes(
        EXECUTING_AUDIT_TOOL,
        description="executing authoritative audit implementation",
    ):
        raise RuntimeError(
            "selected Source_Code audit tool does not match the executing "
            "authoritative audit implementation"
        )
    authoritative_audit_sha = hashlib.sha256(
        authoritative_audit_bytes
    ).hexdigest()
    provenance = {
        "schema_version": 2,
        "promotion_kind": "TABLE3_EXPLICIT_PROMOTION",
        "candidate_status": CANDIDATE_STATUS,
        "candidate_manifest_sha256": _sha256_bytes(candidate_manifest_bytes),
        "candidate_tree_aggregate_sha256": candidate_before[
            "aggregate_sha256"
        ],
        **PROVENANCE_PATHS,
        **TOOL_PATHS,
        "audit_report_sha256": authoritative_audit_sha,
        "bootstrap_status_sha256": bootstrap_sha,
        "bootstrap_helper_sha256": bootstrap_helper_sha,
        "audited_source_identity": promotion_source_before,
        "promotion_tool_sha256": _sha256_bytes(promotion_tool_bytes),
        "audit_tool_sha256": _sha256_bytes(audit_tool_bytes),
        "artifact_count": len(artifact_records),
        "artifact_aggregate_sha256": artifact_digest,
    }
    promoted_manifest = json.loads(json.dumps(candidate_manifest))
    promoted_manifest["status"] = "CANONICAL_PROMOTED"
    promoted_manifest["promotion_provenance"] = provenance

    results_parent = source_code / "results"
    if results_parent.exists() or results_parent.is_symlink():
        if results_parent.is_symlink() or not results_parent.is_dir():
            raise RuntimeError(
                f"promotion results directory is not a regular directory: "
                f"{results_parent}"
            )
    else:
        results_parent.mkdir()
    staging_parent = results_parent / "table3_promotion_staging"
    staging_parent.mkdir(exist_ok=True)
    if staging_parent.is_symlink() or not staging_parent.is_dir():
        raise RuntimeError(f"promotion staging directory is a symlink: {staging_parent}")
    if os.stat(staging_parent).st_dev != os.stat(reference_parent).st_dev:
        raise RuntimeError("promotion staging and reference directories are not co-located")
    staging_parent_stat = os.stat(staging_parent, follow_symlinks=False)
    staging_parent_identity = (
        staging_parent_stat.st_dev,
        staging_parent_stat.st_ino,
    )

    output_parent = output.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    if output_parent.is_symlink() or not output_parent.is_dir():
        raise RuntimeError(
            f"promotion output parent is not a regular directory: {output_parent}"
        )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite promotion output: {output}")
    output_parent_identity = _stat_identity(
        os.stat(output_parent, follow_symlinks=False)
    )

    temporary = Path(
        tempfile.mkdtemp(prefix=".table3-promotion-", dir=staging_parent)
    )
    temporary_stat = os.stat(temporary, follow_symlinks=False)
    temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
    prepared_output = output_parent / (
        f".{output.name}.{uuid.uuid4().hex}.promotion-prepared"
    )
    prepared_output_identity: tuple[int, int] | None = None
    commit_observed = False
    temporary_fd: int | None = None
    reference_parent_fd: int | None = None
    staging_parent_fd: int | None = None
    output_parent_fd: int | None = None
    try:
        for relative in EXPECTED_CANDIDATE_DIRECTORIES:
            (temporary / relative).mkdir()
        (temporary / "provenance").mkdir()
        for record in artifact_records:
            relative = record["path"]
            target = temporary / relative
            payload = candidate_files[relative]
            _write_regular_file_bytes(target, payload)
            if _sha256_bytes(payload) != record["sha256"]:
                raise RuntimeError(f"artifact copy changed bytes: {relative}")
        evidence_payloads = {
            PROVENANCE_PATHS["audit_report_path"]: authoritative_audit_bytes,
            PROVENANCE_PATHS["bootstrap_status_path"]: bootstrap_status_bytes,
            PROVENANCE_PATHS["bootstrap_helper_path"]: bootstrap_helper_bytes,
        }
        for relative, payload in evidence_payloads.items():
            target = temporary / relative
            _write_regular_file_bytes(target, payload)
            expected_hash = {
                PROVENANCE_PATHS["audit_report_path"]: authoritative_audit_sha,
                PROVENANCE_PATHS["bootstrap_status_path"]: bootstrap_sha,
                PROVENANCE_PATHS["bootstrap_helper_path"]: bootstrap_helper_sha,
            }[relative]
            if sha256_file(target) != expected_hash:
                raise RuntimeError(
                    f"promotion evidence copy changed hash: {relative}"
                )
        manifest_target = temporary / "canonical_table3.json"
        with manifest_target.open("x", encoding="utf-8") as stream:
            json.dump(promoted_manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        for directory in ("circuits", "parameters", "provenance"):
            _fsync_directory(temporary / directory)
        _fsync_directory(temporary)
        temporary_fd = _open_directory_fd(
            temporary,
            description="validated promotion staging directory",
        )
        if _stat_identity(os.fstat(temporary_fd)) != temporary_identity:
            raise RuntimeError(
                "promotion staging directory identity changed before validation"
            )

        accepted = validate_promoted_table3_reference(
            manifest_target,
            source_code=source_code,
        )
        current_candidate_stat = os.stat(
            candidate_directory,
            follow_symlinks=False,
        )
        current_candidate_identity = (
            current_candidate_stat.st_dev,
            current_candidate_stat.st_ino,
        )
        if (
            candidate_before != tree_snapshot(candidate_directory)
            or current_candidate_identity != candidate_directory_identity
            or _candidate_file_bytes(
                candidate_directory,
                observed_candidate_manifest,
            )
            != candidate_files
        ):
            raise RuntimeError("candidate changed while promotion was prepared")
        if source_before != source_identity(source_code):
            raise RuntimeError("Source_Code changed while promotion was prepared")
        if promotion_source_before != promotion_source_identity(source_code):
            raise RuntimeError(
                "promotion Source_Code identity changed while promotion was prepared"
            )
        if reference_before != tree_snapshot(reference_parent):
            raise RuntimeError("reference tree changed while promotion was prepared")
        if destination.exists() or destination.is_symlink():
            raise RuntimeError("promotion destination appeared concurrently")

        reference_parent_fd = _open_directory_fd(
            reference_parent,
            description="reference parent",
        )
        staging_parent_fd = _open_directory_fd(
            staging_parent,
            description="promotion staging parent",
        )
        output_parent_fd = _open_directory_fd(
            output_parent,
            description="promotion output parent",
        )
        if (
            (
                os.fstat(reference_parent_fd).st_dev,
                os.fstat(reference_parent_fd).st_ino,
            )
            != reference_parent_identity
            or (
                os.fstat(staging_parent_fd).st_dev,
                os.fstat(staging_parent_fd).st_ino,
            )
            != staging_parent_identity
            or _stat_identity(os.fstat(output_parent_fd))
            != output_parent_identity
        ):
            raise RuntimeError(
                "promotion parent directory identity changed before publication"
            )

        final_validation = {
            **accepted,
            "manifest": str(destination / "canonical_table3.json"),
        }
        report = {
            "schema_version": 1,
            "promotion_kind": "TABLE3_EXPLICIT_PROMOTION",
            "status": "PASS",
            "completed_utc": utc_now(),
            "wall_seconds": time.monotonic() - started,
            "destination": str(destination),
            "candidate_directory": str(candidate_directory),
            "candidate_unchanged": True,
            "candidate_artifacts": artifact_records,
            "promotion_provenance": provenance,
            "promoted_reference": final_validation,
            "qpu_contacted": False,
            "provider_imported": False,
            "external_report_valid": True,
            "post_commit_warnings": [],
        }

        try:
            _rename_no_replace_at(
                staging_parent_fd,
                temporary.name,
                reference_parent_fd,
                destination.name,
                expected_source_identity=temporary_identity,
                expected_source_is_directory=True,
            )
            commit_observed = True
        except BaseException as error:
            # Detect the narrow case where renameat2 committed before Python
            # delivered a signal/exception.  Continue only after the same
            # postcommit integrity checks as an ordinary return.
            try:
                committed_stat = os.stat(
                    destination.name,
                    dir_fd=reference_parent_fd,
                    follow_symlinks=False,
                )
                source_missing = False
                try:
                    os.stat(
                        temporary.name,
                        dir_fd=staging_parent_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    source_missing = True
                commit_observed = (
                    source_missing
                    and stat_module.S_ISDIR(committed_stat.st_mode)
                    and _stat_identity(committed_stat) == temporary_identity
                )
            except OSError:
                commit_observed = False
            if not commit_observed:
                raise
            report["post_commit_warnings"].append(
                "commit-boundary warning: "
                f"{type(error).__name__}: {error}"
            )

        try:
            _validate_committed_public_reference(
                reference_parent=reference_parent,
                reference_parent_identity=reference_parent_identity,
                destination=destination,
                destination_identity=temporary_identity,
                expected_validation=final_validation,
                source_code=source_code,
            )
        except BaseException as error:
            raise CommittedPromotionIntegrityError(
                "Table-3 bytes were committed but the exact public "
                "destination/path/content could not be proven; no PASS report "
                "was published"
            ) from error
    except BaseException as error:
        if not commit_observed:
            _remove_owned_path(
                temporary,
                expected_identity=temporary_identity,
                directory=True,
            )
            _remove_owned_path(
                prepared_output,
                expected_identity=prepared_output_identity,
                directory=False,
            )
            for descriptor in (
                output_parent_fd,
                staging_parent_fd,
                reference_parent_fd,
                temporary_fd,
            ):
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
        else:
            # Preserve committed bytes for diagnosis.  Closing descriptors is
            # non-material and must not turn this distinct integrity state
            # into a PASS or trigger destructive cleanup.
            for descriptor in (
                output_parent_fd,
                staging_parent_fd,
                reference_parent_fd,
                temporary_fd,
            ):
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
        if commit_observed and not isinstance(
            error, CommittedPromotionIntegrityError
        ):
            raise CommittedPromotionIntegrityError(
                "Table-3 bytes were committed but postcommit processing was "
                "interrupted; no PASS report was published"
            ) from error
        raise

    # The no-replace directory rename above is the sole commit point.  Every
    # validation and every data write completed before it.  From here onward,
    # diagnostics are best-effort and must never turn a committed promotion
    # into a reported failure.
    post_commit_warnings = report["post_commit_warnings"]
    assert isinstance(post_commit_warnings, list)
    try:
        assert reference_parent_fd is not None
        _fsync_directory_fd(reference_parent_fd)
    except BaseException as error:
        post_commit_warnings.append(
            "reference-parent fsync warning: "
            f"{type(error).__name__}: {error}"
        )
    try:
        assert staging_parent_fd is not None
        _fsync_directory_fd(staging_parent_fd)
    except BaseException as error:
        post_commit_warnings.append(
            "staging-parent fsync warning: "
            f"{type(error).__name__}: {error}"
        )
    # Descriptor-close errors do not change any material committed bytes and
    # are intentionally not added after the report contract is finalized.
    for descriptor in (
        temporary_fd,
        staging_parent_fd,
        reference_parent_fd,
    ):
        if descriptor is None:
            continue
        try:
            os.close(descriptor)
        except OSError:
            pass

    # The optional external report is an all-or-absent diagnostic.  Serialize
    # only after all reference-commit warnings are known.  Bind its prepared
    # inode and exact bytes immediately before publication, then verify the
    # requested namespace/inode/bytes both before and after parent fsync.
    report_published = False
    report_bytes: bytes | None = None
    try:
        assert output_parent_fd is not None
        current_output_parent = os.stat(
            output_parent,
            follow_symlinks=False,
        )
        if (
            not stat_module.S_ISDIR(current_output_parent.st_mode)
            or _stat_identity(current_output_parent)
            != output_parent_identity
        ):
            raise RuntimeError(
                "external report parent path no longer identifies the pinned "
                "directory"
            )
        report_bytes = (
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        prepared_output_identity = _write_regular_file_at(
            output_parent_fd,
            prepared_output.name,
            report_bytes,
        )
        captured_prepared, captured_identity = _read_regular_file_at(
            output_parent_fd,
            prepared_output.name,
            description="prepared external promotion report",
        )
        if (
            captured_identity != prepared_output_identity
            or captured_prepared != report_bytes
        ):
            raise RuntimeError(
                "prepared external promotion report bytes changed"
            )
        _rename_no_replace_at(
            output_parent_fd,
            prepared_output.name,
            output_parent_fd,
            output.name,
            expected_source_identity=prepared_output_identity,
            expected_source_is_directory=False,
            expected_source_sha256=_sha256_bytes(report_bytes),
            expected_source_size=len(report_bytes),
        )
        published_bytes, published_identity = _read_regular_file_at(
            output_parent_fd,
            output.name,
            description="published external promotion report",
        )
        if (
            published_identity != prepared_output_identity
            or published_bytes != report_bytes
            or _stat_identity(
                os.stat(output_parent, follow_symlinks=False)
            )
            != output_parent_identity
            or _stat_identity(
                os.stat(output, follow_symlinks=False)
            )
            != prepared_output_identity
        ):
            raise RuntimeError(
                "published external promotion report path/inode/bytes changed"
            )
        _fsync_directory_fd(output_parent_fd)
        durable_bytes, durable_identity = _read_regular_file_at(
            output_parent_fd,
            output.name,
            description="durable external promotion report",
        )
        if (
            durable_identity != prepared_output_identity
            or durable_bytes != report_bytes
            or _stat_identity(
                os.stat(output_parent, follow_symlinks=False)
            )
            != output_parent_identity
            or _stat_identity(
                os.stat(output, follow_symlinks=False)
            )
            != prepared_output_identity
        ):
            raise RuntimeError(
                "external promotion report changed after parent fsync"
            )
        report_published = True
    except BaseException as error:
        report["external_report_valid"] = False
        removed_output = False
        removed_prepared = False
        cleanup_proven = False
        cleanup_errors: list[str] = []
        if output_parent_fd is not None:
            (
                removed_output,
                removed_prepared,
                cleanup_proven,
                cleanup_errors,
            ) = _cleanup_external_report_paths(
                output_parent_fd,
                output_name=output.name,
                prepared_name=prepared_output.name,
                expected_identity=prepared_output_identity,
            )
        warning = (
            "external promotion report invalid/withheld: "
            f"{type(error).__name__}: {error}; "
            f"owned_output_removed={removed_output}; "
            f"owned_prepared_removed={removed_prepared}"
        )
        if cleanup_errors:
            warning += "; cleanup diagnostics: " + " | ".join(cleanup_errors)
        post_commit_warnings.append(warning)
        if not cleanup_proven:
            if output_parent_fd is not None:
                try:
                    os.close(output_parent_fd)
                except OSError:
                    pass
                output_parent_fd = None
            raise CommittedPromotionIntegrityError(
                "Table-3 bytes were committed, but invalid external report "
                "cleanup could not prove both report paths absent; no PASS "
                "result was returned"
            ) from error
    if post_commit_warnings:
        print(
            "TABLE-3 PROMOTION COMMITTED WITH POST-COMMIT WARNING(S): "
            + " | ".join(post_commit_warnings),
            file=sys.stderr,
        )

    # This is the last process-local integrity check before returning PASS.
    # Active mutation by another actor after this check is outside the
    # achievable guarantee of a process that does not control that actor.
    try:
        _validate_committed_public_reference(
            reference_parent=reference_parent,
            reference_parent_identity=reference_parent_identity,
            destination=destination,
            destination_identity=temporary_identity,
            expected_validation=final_validation,
            source_code=source_code,
        )
    except BaseException as error:
        report["external_report_valid"] = False
        cleanup_proven = False
        cleanup_errors: list[str] = []
        if output_parent_fd is not None:
            (
                _removed_output,
                _removed_prepared,
                cleanup_proven,
                cleanup_errors,
            ) = _cleanup_external_report_paths(
                output_parent_fd,
                output_name=output.name,
                prepared_name=prepared_output.name,
                expected_identity=prepared_output_identity,
            )
        if output_parent_fd is not None:
            try:
                os.close(output_parent_fd)
            except OSError:
                pass
            output_parent_fd = None
        cleanup_detail = (
            "external PASS report paths were removed"
            if cleanup_proven
            else "external PASS report cleanup could not be proven"
        )
        if cleanup_errors:
            cleanup_detail += ": " + " | ".join(cleanup_errors)
        raise CommittedPromotionIntegrityError(
            "Table-3 bytes were committed, but the final public "
            "destination/path/content check failed; "
            f"{cleanup_detail}; no PASS result was returned"
        ) from error

    if output_parent_fd is not None:
        try:
            os.close(output_parent_fd)
        except OSError:
            pass
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-code", type=Path, default=ROOT)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--bootstrap-helper", type=Path, required=True)
    parser.add_argument("--bootstrap-status", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_code = args.source_code.expanduser().resolve()
    output = (
        args.output.expanduser()
        if args.output is not None
        else source_code
        / "results"
        / "table3_promotion"
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex
            + ".json"
        )
    )
    try:
        report = promote(
            source_code=source_code,
            candidate_directory=args.candidate,
            audit_report_path=args.audit_report,
            bootstrap_helper_path=args.bootstrap_helper,
            bootstrap_status_path=args.bootstrap_status,
            output=output,
        )
    except BaseException as error:
        print(f"TABLE-3 PROMOTION FAILED: {type(error).__name__}: {error}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
