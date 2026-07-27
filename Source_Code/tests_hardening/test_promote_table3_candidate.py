from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from audit_table3_candidate import (  # noqa: E402
    Audit,
    CANDIDATE_STATUS,
    CandidateAuditFailure,
    candidate_tree_manifest,
    tree_snapshot,
    validate_bootstrap_status,
)
from certify_release import (  # noqa: E402
    TABLE3_LABELS,
    sha256_file,
    source_identity,
    validate_promoted_table3_reference,
)
import promote_table3_candidate as promotion  # noqa: E402
from tests_hardening import test_certify_release as certificate_tests  # noqa: E402


def _snapshot(entries: list[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(
        entries,
        key=lambda item: (str(item["path"]), str(item["kind"])),
    )
    aggregate = hashlib.sha256(
        (
            json.dumps(ordered, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    return {
        "exists": True,
        "aggregate_sha256": aggregate,
        "entries": ordered,
    }


def _descriptor_identifies_path(descriptor: int, path: Path) -> bool:
    descriptor_stat = os.fstat(descriptor)
    path_stat = os.stat(path, follow_symlinks=False)
    return (
        descriptor_stat.st_dev,
        descriptor_stat.st_ino,
    ) == (
        path_stat.st_dev,
        path_stat.st_ino,
    )


class Table3PromotionTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        source_code = root / "Source_Code"
        source_code.mkdir()
        (source_code / "reference").mkdir()
        for name in (
            "promote_table3_candidate.py",
            "audit_table3_candidate.py",
        ):
            shutil.copyfile(SOURCE_ROOT / name, source_code / name)

        candidate = root / "candidate"
        candidate.mkdir()
        manifest_path = (
            certificate_tests.CertificateInfrastructureTests()
            ._write_promoted_table3(candidate)
        )
        # The shared certificate fixture emits a fully promoted tree.  This
        # fixture deliberately returns it to the exact ten-file candidate
        # transport boundary that promotion accepts.
        shutil.rmtree(candidate / "provenance")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = CANDIDATE_STATUS
        manifest.pop("promotion_provenance")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        helper = root / "bootstrap_helper.py"
        helper.write_text(
            "# exact bootstrap helper fixture\n",
            encoding="utf-8",
        )
        identity = source_identity(source_code)
        candidate_audit = Audit()
        artifacts = candidate_tree_manifest(candidate_audit, candidate)
        self.assertEqual(candidate_audit.status, "PASS")
        bootstrap = root / "bootstrap_status.json"
        bootstrap.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "CANDIDATE_READY_NOT_CERTIFIED",
                    "bootstrap_helper": {
                        "path": str(helper.resolve()),
                        "sha256": sha256_file(helper),
                    },
                    "provider_imported": False,
                    "qpu_contacted": False,
                    "source_identity_before": identity,
                    "source_identity_after": identity,
                    "candidate_artifacts": artifacts,
                    "invocation_id": "fixture-12345678",
                    "invocation_directory": str(root.resolve()),
                    "output_directory": str(root.resolve()),
                    "bootstrap_status": str(bootstrap.resolve()),
                    "source_code": str(source_code.resolve()),
                    "environment": str((root / "environment").resolve()),
                    "environment_python": str(
                        (root / "environment" / "bin" / "python").resolve()
                    ),
                    "candidate_directory": str(candidate.resolve()),
                    "stages": [
                        {
                            "name": name,
                            "status": "PASS",
                            "returncode": 0,
                            "argv": ["python", "-I", "-B", name],
                            "log": str((root / f"{name}.log").resolve()),
                            "log_sha256": hashlib.sha256(
                                name.encode("utf-8")
                            ).hexdigest(),
                            "log_size_bytes": len(name),
                        }
                        for name in promotion.EXPECTED_STAGE_NAMES
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        candidate_snapshot = tree_snapshot(candidate)
        reference_snapshot = tree_snapshot(source_code / "reference")
        checks = [
            {"name": name, "status": "PASS"}
            for name in sorted(promotion.REQUIRED_AUDIT_CHECK_NAMES)
        ]
        checks.append(
            {
                "name": "all three candidate cases independently replayed",
                "status": "PASS",
            }
        )
        audit_report = root / "candidate_audit.json"
        report = {
            "schema_version": 1,
            "audit_kind": "TABLE3_PRE_PROMOTION_READ_ONLY",
            "status": "PASS",
            "started_utc": "2026-07-24T00:00:00Z",
            "completed_utc": "2026-07-24T00:00:01Z",
            "wall_seconds": 1.0,
            "source_code": str(source_code.resolve()),
            "bootstrap_helper": str(helper.resolve()),
            "bootstrap_helper_sha256": sha256_file(helper),
            "bootstrap_status": str(bootstrap.resolve()),
            "bootstrap_status_sha256": sha256_file(bootstrap),
            "candidate_directory": str(candidate.resolve()),
            "output": str((root / "first_audit_output.json").resolve()),
            "candidate_expected_status": CANDIDATE_STATUS,
            "candidate_promoted": False,
            "provider_imported": False,
            "qpu_contacted": False,
            "source_identity_before": identity,
            "source_identity_after": identity,
            "reference_tree_before": reference_snapshot,
            "reference_tree_after": reference_snapshot,
            "invocation_tree_before": _snapshot([]),
            "invocation_tree_after": _snapshot([]),
            "candidate_tree_before": candidate_snapshot,
            "candidate_tree_after": candidate_snapshot,
            "candidate_mutated": False,
            "reference_mutated": False,
            "invocation_mutated": False,
            "observed_cases": {label: {} for label in TABLE3_LABELS},
            "checks": checks,
            "passed": len(checks),
            "failed": 0,
        }
        audit_report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "source_code": source_code,
            "candidate": candidate,
            "helper": helper,
            "bootstrap": bootstrap,
            "audit_report": audit_report,
            "output": source_code / "results" / "promotion.json",
        }

    @staticmethod
    def _authoritative_pass(
        paths: dict[str, Path],
        captured: dict[str, object],
    ):
        supplied = json.loads(
            paths["audit_report"].read_text(encoding="utf-8")
        )

        def run_audit(**arguments):
            report = copy.deepcopy(supplied)
            report["started_utc"] = "2026-07-24T00:01:00Z"
            report["completed_utc"] = "2026-07-24T00:01:02Z"
            report["wall_seconds"] = 2.0
            report["output"] = str(arguments["output"])
            # A completed first audit may itself now be part of the invocation
            # directory; promotion authenticates each audit's immutability
            # separately and does not require these two historical snapshots
            # to be byte-identical.
            report["invocation_tree_before"] = _snapshot(
                [
                    {
                        "path": "prior_audit_report.json",
                        "kind": "file",
                        "sha256": "d" * 64,
                        "size_bytes": 1,
                    }
                ]
            )
            report["invocation_tree_after"] = copy.deepcopy(
                report["invocation_tree_before"]
            )
            arguments["output"].write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            captured["report"] = report
            captured["bytes"] = arguments["output"].read_bytes()
            captured["arguments"] = arguments
            return report

        return run_audit

    def _promote_with_authoritative_pass(
        self,
        paths: dict[str, Path],
        captured: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if captured is None:
            captured = {}
        with mock.patch.object(
            promotion,
            "run_audit",
            side_effect=self._authoritative_pass(paths, captured),
        ):
            return promotion.promote(
                source_code=paths["source_code"],
                candidate_directory=paths["candidate"],
                audit_report_path=paths["audit_report"],
                bootstrap_helper_path=paths["helper"],
                bootstrap_status_path=paths["bootstrap"],
                output=paths["output"],
            )

    def test_atomic_promotion_preserves_artifacts_and_authoritative_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            candidate_before = {
                relative.relative_to(paths["candidate"]).as_posix(): relative.read_bytes()
                for relative in paths["candidate"].rglob("*")
                if relative.is_file()
            }
            captured: dict[str, object] = {}
            report = self._promote_with_authoritative_pass(paths, captured)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(
                json.loads(paths["output"].read_text(encoding="utf-8")),
                report,
            )
            destination = paths["source_code"] / "reference" / "table3"
            validation = validate_promoted_table3_reference(
                destination / "canonical_table3.json",
                source_code=paths["source_code"],
            )
            self.assertEqual(validation["artifact_count"], 9)
            for relative, payload in candidate_before.items():
                if relative == "canonical_table3.json":
                    continue
                self.assertEqual((destination / relative).read_bytes(), payload)

            evidence = destination / "provenance"
            self.assertEqual(
                (evidence / "audit_report.json").read_bytes(),
                captured["bytes"],
            )
            self.assertEqual(
                (evidence / "bootstrap_status.json").read_bytes(),
                paths["bootstrap"].read_bytes(),
            )
            self.assertEqual(
                (evidence / "bootstrap_helper.py").read_bytes(),
                paths["helper"].read_bytes(),
            )
            self.assertEqual(
                captured["arguments"]["bootstrap_helper_path"],
                paths["helper"].resolve(),
            )

            candidate_manifest = json.loads(
                candidate_before["canonical_table3.json"].decode("utf-8")
            )
            promoted_manifest = json.loads(
                (destination / "canonical_table3.json").read_text(
                    encoding="utf-8"
                )
            )
            provenance = promoted_manifest.pop("promotion_provenance")
            promoted_manifest["status"] = CANDIDATE_STATUS
            self.assertEqual(promoted_manifest, candidate_manifest)
            self.assertEqual(provenance["schema_version"], 2)
            self.assertEqual(provenance["artifact_count"], 9)
            for key, expected in promotion.PROVENANCE_PATHS.items():
                self.assertEqual(provenance[key], expected)
            for key, expected in promotion.TOOL_PATHS.items():
                self.assertEqual(provenance[key], expected)
            self.assertEqual(
                provenance["bootstrap_status_sha256"],
                sha256_file(evidence / "bootstrap_status.json"),
            )
            self.assertEqual(
                provenance["bootstrap_helper_sha256"],
                sha256_file(evidence / "bootstrap_helper.py"),
            )
            self.assertEqual(
                provenance["audit_report_sha256"],
                sha256_file(evidence / "audit_report.json"),
            )

    def test_forged_one_check_pass_report_fails_before_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            audit = json.loads(paths["audit_report"].read_text(encoding="utf-8"))
            audit["checks"] = [{"name": "forged shortcut", "status": "PASS"}]
            audit["passed"] = 1
            paths["audit_report"].write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            # The authoritative replay remains the complete report from before
            # the forged caller file was substituted.
            full_report = copy.deepcopy(audit)
            original_checks = [
                {"name": name, "status": "PASS"}
                for name in sorted(promotion.REQUIRED_AUDIT_CHECK_NAMES)
            ]
            original_checks.append(
                {
                    "name": "all three candidate cases independently replayed",
                    "status": "PASS",
                }
            )
            full_report["checks"] = original_checks
            full_report["passed"] = len(original_checks)

            def authoritative(**arguments):
                report = copy.deepcopy(full_report)
                report["output"] = str(arguments["output"])
                arguments["output"].write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                return report

            with mock.patch.object(
                promotion, "run_audit", side_effect=authoritative
            ), self.assertRaisesRegex(
                RuntimeError, "supplied candidate audit report"
            ):
                promotion.promote(
                    source_code=paths["source_code"],
                    candidate_directory=paths["candidate"],
                    audit_report_path=paths["audit_report"],
                    bootstrap_helper_path=paths["helper"],
                    bootstrap_status_path=paths["bootstrap"],
                    output=paths["output"],
                )
            self.assertFalse(paths["output"].exists())
            self.assertFalse(
                (paths["source_code"] / "reference" / "table3").exists()
            )

    def test_minimal_bootstrap_authoritative_failure_is_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            bootstrap = json.loads(
                paths["bootstrap"].read_text(encoding="utf-8")
            )
            for field in ("invocation_id", "stages", "bootstrap_helper"):
                bootstrap.pop(field)
            paths["bootstrap"].write_text(
                json.dumps(bootstrap, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            supplied = json.loads(
                paths["audit_report"].read_text(encoding="utf-8")
            )
            supplied["bootstrap_status_sha256"] = sha256_file(
                paths["bootstrap"]
            )
            paths["audit_report"].write_text(
                json.dumps(supplied, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            def authoritative_failure(**arguments):
                failed = copy.deepcopy(supplied)
                failed["status"] = "FAIL"
                failed["checks"] = [
                    {
                        "name": "bootstrap exact four-stage sequence is declared",
                        "status": "FAIL",
                    }
                ]
                failed["passed"] = 0
                failed["failed"] = 1
                failed["error"] = {
                    "type": "CandidateAuditFailure",
                    "message": "bootstrap exact four-stage sequence is declared",
                }
                failed["output"] = str(arguments["output"])
                arguments["output"].write_text(
                    json.dumps(failed, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                return failed

            with mock.patch.object(
                promotion,
                "run_audit",
                side_effect=authoritative_failure,
            ), self.assertRaisesRegex(
                RuntimeError, "fresh authoritative candidate audit"
            ):
                promotion.promote(
                    source_code=paths["source_code"],
                    candidate_directory=paths["candidate"],
                    audit_report_path=paths["audit_report"],
                    bootstrap_helper_path=paths["helper"],
                    bootstrap_status_path=paths["bootstrap"],
                    output=paths["output"],
                )
            self.assertFalse(paths["output"].exists())
            self.assertFalse(
                (paths["source_code"] / "reference" / "table3").exists()
            )

    def test_real_hardened_bootstrap_validator_rejects_minimal_status(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            status = json.loads(
                paths["bootstrap"].read_text(encoding="utf-8")
            )
            for field in (
                "bootstrap_helper",
                "invocation_directory",
                "output_directory",
                "bootstrap_status",
                "source_code",
                "environment",
                "environment_python",
                "candidate_directory",
                "stages",
            ):
                status.pop(field, None)
            candidate_audit = Audit()
            artifacts = candidate_tree_manifest(
                candidate_audit,
                paths["candidate"],
            )
            self.assertEqual(candidate_audit.status, "PASS")
            validation = Audit()
            with self.assertRaises(CandidateAuditFailure):
                validate_bootstrap_status(
                    validation,
                    status,
                    bootstrap_helper_path=paths["helper"],
                    bootstrap_status_path=paths["bootstrap"],
                    source_code=paths["source_code"],
                    environment=Path(temporary) / "environment",
                    candidate_directory=paths["candidate"],
                    current_source_identity=source_identity(
                        paths["source_code"]
                    ),
                    observed_candidate_manifest=artifacts,
                )
            self.assertEqual(validation.status, "FAIL")
            self.assertIn(
                "bootstrap helper path and bytes are explicitly bound",
                [
                    item["name"]
                    for item in validation.checks
                    if item["status"] == "FAIL"
                ],
            )

    def test_authoritative_report_substitution_fails_before_publish(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            supplied = json.loads(
                paths["audit_report"].read_text(encoding="utf-8")
            )
            substituted: dict[str, bytes] = {}

            def authoritative(**arguments):
                report = copy.deepcopy(supplied)
                report["output"] = str(arguments["output"])
                arguments["output"].write_text(
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                forged = copy.deepcopy(report)
                forged["checks"] = [
                    {"name": "forged shortcut", "status": "PASS"}
                ]
                forged["passed"] = 1
                substituted["bytes"] = (
                    json.dumps(forged, indent=2, sort_keys=True) + "\n"
                ).encode("utf-8")
                return report

            real_capture = promotion._read_regular_file_bytes

            def capture(path, *, description):
                if (
                    Path(path).name == "authoritative_audit.json"
                    and "bytes" in substituted
                ):
                    return substituted["bytes"]
                return real_capture(path, description=description)

            with mock.patch.object(
                promotion,
                "run_audit",
                side_effect=authoritative,
            ), mock.patch.object(
                promotion,
                "_read_regular_file_bytes",
                side_effect=capture,
            ), self.assertRaisesRegex(
                RuntimeError, "return value does not match"
            ):
                promotion.promote(
                    source_code=paths["source_code"],
                    candidate_directory=paths["candidate"],
                    audit_report_path=paths["audit_report"],
                    bootstrap_helper_path=paths["helper"],
                    bootstrap_status_path=paths["bootstrap"],
                    output=paths["output"],
                )
            self.assertFalse(paths["output"].exists())
            self.assertFalse(
                (paths["source_code"] / "reference" / "table3").exists()
            )

    @unittest.skipUnless(
        sys.platform == "linux",
        "atomic no-replace promotion requires Linux",
    )
    def test_concurrent_destination_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            real_rename = promotion._rename_no_replace_at

            def race(
                source_directory_fd,
                source_name,
                destination_directory_fd,
                destination_name,
                **rename_contract,
            ):
                os.mkdir(
                    destination_name,
                    dir_fd=destination_directory_fd,
                )
                concurrent_fd = os.open(
                    destination_name,
                    os.O_RDONLY | os.O_DIRECTORY,
                    dir_fd=destination_directory_fd,
                )
                try:
                    marker_fd = os.open(
                        "concurrent-owner.txt",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=concurrent_fd,
                    )
                    os.write(marker_fd, b"do not replace\n")
                    os.close(marker_fd)
                finally:
                    os.close(concurrent_fd)
                return real_rename(
                    source_directory_fd,
                    source_name,
                    destination_directory_fd,
                    destination_name,
                    **rename_contract,
                )

            with mock.patch.object(
                promotion,
                "_rename_no_replace_at",
                side_effect=race,
            ), self.assertRaisesRegex(
                FileExistsError, "appeared concurrently"
            ):
                self._promote_with_authoritative_pass(paths)
            destination = paths["source_code"] / "reference" / "table3"
            self.assertEqual(
                (destination / "concurrent-owner.txt").read_text(
                    encoding="utf-8"
                ),
                "do not replace\n",
            )
            self.assertFalse(
                (destination / "canonical_table3.json").exists()
            )
            self.assertFalse(paths["output"].exists())

    @unittest.skipUnless(
        sys.platform == "linux",
        "atomic no-replace promotion requires Linux",
    )
    def test_external_report_failure_does_not_reverse_committed_promotion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            real_rename = promotion._rename_no_replace_at

            def fail_external_report(
                source_directory_fd,
                source_name,
                destination_directory_fd,
                destination_name,
                **rename_contract,
            ):
                if destination_name == "table3":
                    return real_rename(
                        source_directory_fd,
                        source_name,
                        destination_directory_fd,
                        destination_name,
                        **rename_contract,
                    )
                raise OSError("injected external-report publication failure")

            with mock.patch.object(
                promotion,
                "_rename_no_replace_at",
                side_effect=fail_external_report,
            ):
                report = self._promote_with_authoritative_pass(paths)
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(report["post_commit_warnings"])
            self.assertIn(
                "external promotion report invalid/withheld",
                report["post_commit_warnings"][0],
            )
            self.assertIs(report["external_report_valid"], False)
            self.assertFalse(paths["output"].exists())
            destination = paths["source_code"] / "reference" / "table3"
            validation = validate_promoted_table3_reference(
                destination / "canonical_table3.json",
                source_code=paths["source_code"],
            )
            self.assertEqual(validation["status"], "PASS")

    def test_reference_fsync_warning_is_in_exact_published_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            real_fsync = promotion._fsync_directory_fd
            reference_parent = paths["source_code"] / "reference"

            def fail_reference_once(descriptor):
                if _descriptor_identifies_path(
                    descriptor,
                    reference_parent,
                ):
                    raise OSError("injected reference-parent fsync failure")
                return real_fsync(descriptor)

            with mock.patch.object(
                promotion,
                "_fsync_directory_fd",
                side_effect=fail_reference_once,
            ):
                report = self._promote_with_authoritative_pass(paths)
            self.assertIs(report["external_report_valid"], True)
            self.assertIn(
                "reference-parent fsync warning",
                report["post_commit_warnings"][0],
            )
            self.assertEqual(
                json.loads(paths["output"].read_text(encoding="utf-8")),
                report,
            )

    def test_staging_parent_fsync_warning_is_in_exact_published_report(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            real_fsync = promotion._fsync_directory_fd
            staging_parent = (
                paths["source_code"]
                / "results"
                / "table3_promotion_staging"
            )

            def fail_staging_once(descriptor):
                if _descriptor_identifies_path(
                    descriptor,
                    staging_parent,
                ):
                    raise OSError("injected staging-parent fsync failure")
                return real_fsync(descriptor)

            with mock.patch.object(
                promotion,
                "_fsync_directory_fd",
                side_effect=fail_staging_once,
            ):
                report = self._promote_with_authoritative_pass(paths)
            self.assertIs(report["external_report_valid"], True)
            self.assertTrue(
                any(
                    "staging-parent fsync warning" in warning
                    for warning in report["post_commit_warnings"]
                )
            )
            self.assertEqual(
                json.loads(paths["output"].read_text(encoding="utf-8")),
                report,
            )

    def test_output_parent_fsync_failure_removes_owned_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            real_fsync = promotion._fsync_directory_fd
            output_parent = paths["output"].parent
            injected = False

            def fail_output_once(descriptor):
                nonlocal injected
                if (
                    not injected
                    and _descriptor_identifies_path(descriptor, output_parent)
                ):
                    injected = True
                    raise OSError("injected output-parent fsync failure")
                return real_fsync(descriptor)

            with mock.patch.object(
                promotion,
                "_fsync_directory_fd",
                side_effect=fail_output_once,
            ):
                report = self._promote_with_authoritative_pass(paths)
            self.assertIs(report["external_report_valid"], False)
            self.assertTrue(
                any(
                    "injected output-parent fsync failure" in warning
                    for warning in report["post_commit_warnings"]
                )
            )
            self.assertFalse(paths["output"].exists())
            self.assertEqual(
                list(paths["output"].parent.glob("*.promotion-prepared")),
                [],
            )

    @unittest.skipUnless(
        sys.platform == "linux",
        "atomic no-replace promotion requires Linux",
    )
    def test_staging_name_substitution_fails_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            real_rename = promotion._rename_no_replace_at

            def substitute_source(
                source_directory_fd,
                source_name,
                destination_directory_fd,
                destination_name,
                **rename_contract,
            ):
                if destination_name == "table3":
                    os.rename(
                        source_name,
                        source_name + ".validated-aside",
                        src_dir_fd=source_directory_fd,
                        dst_dir_fd=source_directory_fd,
                    )
                    os.mkdir(source_name, dir_fd=source_directory_fd)
                return real_rename(
                    source_directory_fd,
                    source_name,
                    destination_directory_fd,
                    destination_name,
                    **rename_contract,
                )

            with mock.patch.object(
                promotion,
                "_rename_no_replace_at",
                side_effect=substitute_source,
            ), self.assertRaisesRegex(
                RuntimeError,
                "source name no longer identifies",
            ):
                self._promote_with_authoritative_pass(paths)
            self.assertFalse(
                (paths["source_code"] / "reference" / "table3").exists()
            )
            self.assertFalse(paths["output"].exists())

    @unittest.skipUnless(
        sys.platform == "linux",
        "atomic no-replace promotion requires Linux",
    )
    def test_reference_parent_replacement_is_nonpass_and_preserves_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            real_rename = promotion._rename_no_replace_at
            reference = paths["source_code"] / "reference"
            moved_reference = paths["source_code"] / "reference-moved"

            def replace_reference_parent(
                source_directory_fd,
                source_name,
                destination_directory_fd,
                destination_name,
                **rename_contract,
            ):
                if destination_name == "table3":
                    os.rename(reference, moved_reference)
                    reference.mkdir()
                return real_rename(
                    source_directory_fd,
                    source_name,
                    destination_directory_fd,
                    destination_name,
                    **rename_contract,
                )

            with mock.patch.object(
                promotion,
                "_rename_no_replace_at",
                side_effect=replace_reference_parent,
            ), self.assertRaises(
                promotion.CommittedPromotionIntegrityError
            ):
                self._promote_with_authoritative_pass(paths)
            self.assertFalse((reference / "table3").exists())
            self.assertTrue(
                (moved_reference / "table3" / "canonical_table3.json").is_file()
            )
            self.assertFalse(paths["output"].exists())

    @unittest.skipUnless(
        sys.platform == "linux",
        "atomic no-replace promotion requires Linux",
    )
    def test_same_inode_prepared_report_mutation_is_withheld(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            real_rename = promotion._rename_no_replace_at

            def mutate_prepared_report(
                source_directory_fd,
                source_name,
                destination_directory_fd,
                destination_name,
                **rename_contract,
            ):
                if destination_name != "table3":
                    descriptor = os.open(
                        source_name,
                        os.O_WRONLY | os.O_TRUNC,
                        dir_fd=source_directory_fd,
                    )
                    try:
                        os.write(descriptor, b'{"status":"FORGED"}\n')
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                return real_rename(
                    source_directory_fd,
                    source_name,
                    destination_directory_fd,
                    destination_name,
                    **rename_contract,
                )

            with mock.patch.object(
                promotion,
                "_rename_no_replace_at",
                side_effect=mutate_prepared_report,
            ):
                report = self._promote_with_authoritative_pass(paths)
            self.assertIs(report["external_report_valid"], False)
            self.assertFalse(paths["output"].exists())
            self.assertTrue(
                any(
                    "source bytes no longer match" in warning
                    for warning in report["post_commit_warnings"]
                )
            )

    @unittest.skipUnless(
        sys.platform == "linux",
        "atomic no-replace promotion requires Linux",
    )
    def test_output_parent_namespace_replacement_withholds_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            real_rename = promotion._rename_no_replace_at
            output_parent = paths["output"].parent
            moved_output_parent = paths["source_code"] / "results-moved"

            def replace_output_parent(
                source_directory_fd,
                source_name,
                destination_directory_fd,
                destination_name,
                **rename_contract,
            ):
                if destination_name != "table3":
                    os.rename(output_parent, moved_output_parent)
                    output_parent.mkdir()
                return real_rename(
                    source_directory_fd,
                    source_name,
                    destination_directory_fd,
                    destination_name,
                    **rename_contract,
                )

            with mock.patch.object(
                promotion,
                "_rename_no_replace_at",
                side_effect=replace_output_parent,
            ):
                report = self._promote_with_authoritative_pass(paths)
            self.assertIs(report["external_report_valid"], False)
            self.assertFalse(paths["output"].exists())
            self.assertFalse(
                (moved_output_parent / paths["output"].name).exists()
            )
            self.assertTrue(
                any(
                    "report parent path" in warning
                    or "path/inode/bytes changed" in warning
                    for warning in report["post_commit_warnings"]
                )
            )

    @unittest.skipUnless(
        sys.platform == "linux",
        "atomic no-replace promotion requires Linux",
    )
    def test_reference_namespace_replacement_during_report_is_nonpass(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            real_rename = promotion._rename_no_replace_at
            reference = paths["source_code"] / "reference"
            moved_reference = paths["source_code"] / "reference-moved"

            def replace_reference_during_report(
                source_directory_fd,
                source_name,
                destination_directory_fd,
                destination_name,
                **rename_contract,
            ):
                result = real_rename(
                    source_directory_fd,
                    source_name,
                    destination_directory_fd,
                    destination_name,
                    **rename_contract,
                )
                if destination_name != "table3":
                    os.rename(reference, moved_reference)
                    reference.mkdir()
                return result

            with mock.patch.object(
                promotion,
                "_rename_no_replace_at",
                side_effect=replace_reference_during_report,
            ), self.assertRaisesRegex(
                promotion.CommittedPromotionIntegrityError,
                "final public destination/path/content check failed",
            ):
                self._promote_with_authoritative_pass(paths)
            self.assertFalse(paths["output"].exists())
            self.assertFalse((reference / "table3").exists())
            self.assertTrue(
                (moved_reference / "table3" / "canonical_table3.json").is_file()
            )

    def test_output_fsync_and_unlink_failure_is_nonpass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            real_fsync = promotion._fsync_directory_fd
            real_unlink = promotion._unlink_owned_at
            output_parent = paths["output"].parent
            injected = False

            def fail_output_fsync_once(descriptor):
                nonlocal injected
                if (
                    not injected
                    and _descriptor_identifies_path(descriptor, output_parent)
                ):
                    injected = True
                    raise OSError("injected output-parent fsync failure")
                return real_fsync(descriptor)

            def fail_published_output_unlink(
                directory_fd,
                name,
                *,
                expected_identity,
            ):
                if name == paths["output"].name:
                    raise OSError("injected owned-output unlink failure")
                return real_unlink(
                    directory_fd,
                    name,
                    expected_identity=expected_identity,
                )

            with mock.patch.object(
                promotion,
                "_fsync_directory_fd",
                side_effect=fail_output_fsync_once,
            ), mock.patch.object(
                promotion,
                "_unlink_owned_at",
                side_effect=fail_published_output_unlink,
            ), self.assertRaisesRegex(
                promotion.CommittedPromotionIntegrityError,
                "cleanup could not prove both report paths absent",
            ):
                self._promote_with_authoritative_pass(paths)
            self.assertTrue(paths["output"].is_file())
            self.assertEqual(
                list(paths["output"].parent.glob("*.promotion-prepared")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
