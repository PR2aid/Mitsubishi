from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import certify_release
from certify_release import (
    artifact_manifest,
    atomic_write_json,
    source_identity,
    terminate_process_group,
    validate_promoted_table3_reference,
    validate_reproduction_latest,
)


class CertificateInfrastructureTests(unittest.TestCase):
    def _write_promoted_table3(
        self,
        root: Path,
        *,
        source_code: Path | None = None,
    ) -> Path:
        source_code = SOURCE_ROOT if source_code is None else source_code
        manifest_path = root / "canonical_table3.json"
        cases: dict[str, object] = {}
        artifact_records: list[tuple[str, str, int]] = []
        for label in certify_release.TABLE3_LABELS:
            molecule, norb, n_qubits = certify_release.TABLE3_CASES[label]
            parameter_relative = (
                f"parameters/{label}_seed-{certify_release.TABLE3_SEED}.npz"
            )
            generic_relative = f"circuits/{label}_generic.qasm"
            structured_relative = f"circuits/{label}_structured.qasm"
            parameter = root / parameter_relative
            parameter.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                parameter,
                singles=np.asarray([[0.1, -0.2]], dtype=np.float64),
                doubles=np.asarray([[0.3]], dtype=np.float64),
            )
            parameter_declaration = {
                "file": parameter_relative,
                "sha256": certify_release.sha256_file(parameter),
                "array_names": ["doubles", "singles"],
                "array_shapes": {"doubles": [1, 1], "singles": [1, 2]},
                "array_replay_exact": True,
            }
            qasm_text = (
                "OPENQASM 2.0;\n"
                'include "qelib1.inc";\n'
                f"qreg q[{n_qubits}];\n"
                "rz(pi/2) q[0];\n"
                "sx q[0];\n"
                "x q[1];\n"
                "cx q[0],q[1];\n"
            )
            qasm_declarations: dict[str, dict[str, object]] = {}
            for key, relative in (
                ("generic_qasm", generic_relative),
                ("structured_qasm", structured_relative),
            ):
                artifact = root / relative
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text(qasm_text, encoding="utf-8", newline="\n")
                qasm_declarations[key] = {
                    **certify_release.TABLE3_PROTOCOL,
                    "qasm_file": relative,
                    "qasm_sha256": certify_release.sha256_file(artifact),
                    "logical_qubits": n_qubits,
                    "depth": 3,
                    "size": 4,
                    "cx": 1,
                    "operations": {"rz": 1, "sx": 1, "x": 1, "cx": 1},
                    "derived_from_reloaded_qasm": True,
                    "compiled_once_before_serialization": True,
                }
            generic = qasm_declarations["generic_qasm"]
            structured = qasm_declarations["structured_qasm"]
            cases[label] = {
                "label": label,
                "seed": certify_release.TABLE3_SEED,
                "molecule": molecule,
                "norb": norb,
                "n_qubits": n_qubits,
                "compilation_protocol": certify_release.TABLE3_PROTOCOL,
                "parameter_artifact": parameter_declaration,
                **qasm_declarations,
                "legacy_generic_unitary": generic,
                "structured_exact_pauli_network": structured,
                "qasm_file": structured_relative,
                "qasm_sha256": structured["qasm_sha256"],
                "device_native": False,
            }
            for relative in (
                parameter_relative,
                generic_relative,
                structured_relative,
            ):
                artifact = root / relative
                artifact_records.append(
                    (
                        relative,
                        certify_release.sha256_file(artifact),
                        artifact.stat().st_size,
                    )
                )
        aggregate = certify_release.hashlib.sha256()
        for relative, digest, size in sorted(artifact_records):
            aggregate.update(relative.encode("utf-8"))
            aggregate.update(b"\0")
            aggregate.update(digest.encode("ascii"))
            aggregate.update(b"\0")
            aggregate.update(str(size).encode("ascii"))
            aggregate.update(b"\n")
        candidate_manifest = {
            "schema_version": 1,
            "status": certify_release.TABLE3_CANDIDATE_STATUS,
            "seed": certify_release.TABLE3_SEED,
            "compilation_protocol": certify_release.TABLE3_PROTOCOL,
            "cases": cases,
        }
        candidate_manifest_bytes = (
            json.dumps(candidate_manifest, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        candidate_records = [
            {
                "path": relative,
                "sha256": digest,
                "size_bytes": size,
            }
            for relative, digest, size in artifact_records
        ]
        candidate_records.append(
            {
                "path": "canonical_table3.json",
                "sha256": certify_release.hashlib.sha256(
                    candidate_manifest_bytes
                ).hexdigest(),
                "size_bytes": len(candidate_manifest_bytes),
            }
        )
        candidate_records.sort(key=lambda item: item["path"])
        candidate_aggregate = certify_release.hashlib.sha256()
        for record in candidate_records:
            candidate_aggregate.update(record["path"].encode("utf-8"))
            candidate_aggregate.update(b"\0")
            candidate_aggregate.update(record["sha256"].encode("ascii"))
            candidate_aggregate.update(b"\0")
            candidate_aggregate.update(
                str(record["size_bytes"]).encode("ascii")
            )
            candidate_aggregate.update(b"\n")
        candidate_artifacts = {
            "algorithm": "sha256-path-hash-size-v1",
            "aggregate_sha256": candidate_aggregate.hexdigest(),
            "file_count": len(candidate_records),
            "files": candidate_records,
        }

        def snapshot(entries: list[dict[str, object]]) -> dict[str, object]:
            ordered = sorted(
                entries,
                key=lambda item: (str(item["path"]), str(item["kind"])),
            )
            digest = certify_release.hashlib.sha256(
                (
                    json.dumps(
                        ordered,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
            ).hexdigest()
            return {
                "exists": True,
                "aggregate_sha256": digest,
                "entries": ordered,
            }

        candidate_snapshot = snapshot(
            [
                {"kind": "file", **record}
                for record in candidate_records
            ]
        )
        empty_snapshot = snapshot([])
        audited_identity = certify_release.promotion_source_identity(source_code)
        audit_identity = {
            "algorithm": "sha256-path-hash-size-notebook-source-v3",
            "sha256": audited_identity["sha256"],
            "file_count": audited_identity["file_count"],
        }

        provenance_directory = root / "provenance"
        provenance_directory.mkdir(exist_ok=True)
        helper_path = provenance_directory / "bootstrap_helper.py"
        helper_path.write_text(
            "# exact preserved bootstrap helper\n",
            encoding="utf-8",
        )
        helper_sha256 = certify_release.sha256_file(helper_path)
        original_helper = "/qbraid/bootstrap_table3_candidate.py"
        original_status = "/qbraid/run-12345678/bootstrap_status.json"
        original_candidate = (
            "/qbraid/run-12345678/advanced_method/"
            "canonical_table3_candidate"
        )
        stages = []
        for name in certify_release.TABLE3_BOOTSTRAP_STAGE_NAMES:
            stages.append(
                {
                    "name": name,
                    "status": "PASS",
                    "returncode": 0,
                    "argv": ["/qbraid/python", "-I", "-B", name],
                    "log": f"/qbraid/run-12345678/{name}.log",
                    "log_sha256": certify_release.hashlib.sha256(
                        name.encode("utf-8")
                    ).hexdigest(),
                    "log_size_bytes": len(name),
                }
            )
        bootstrap_status = {
            "schema_version": 1,
            "invocation_id": "run-12345678",
            "invocation_directory": "/qbraid/run-12345678",
            "output_directory": "/qbraid/run-12345678",
            "bootstrap_status": original_status,
            "bootstrap_helper": {
                "path": original_helper,
                "sha256": helper_sha256,
            },
            "status": "CANDIDATE_READY_NOT_CERTIFIED",
            "source_code": "/qbraid/Source_Code",
            "environment": "/qbraid/environment",
            "environment_python": "/qbraid/environment/bin/python",
            "candidate_directory": original_candidate,
            "provider_imported": False,
            "qpu_contacted": False,
            "stages": stages,
            "source_identity_before": audit_identity,
            "source_identity_after": audit_identity,
            "candidate_artifacts": candidate_artifacts,
        }
        bootstrap_path = provenance_directory / "bootstrap_status.json"
        bootstrap_path.write_text(
            json.dumps(bootstrap_status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        bootstrap_sha256 = certify_release.sha256_file(bootstrap_path)
        checks = [
            {"name": name, "status": "PASS"}
            for name in sorted(certify_release.TABLE3_REQUIRED_AUDIT_CHECKS)
        ]
        audit_report = {
            "schema_version": 1,
            "audit_kind": "TABLE3_PRE_PROMOTION_READ_ONLY",
            "status": "PASS",
            "source_code": bootstrap_status["source_code"],
            "bootstrap_helper": original_helper,
            "bootstrap_helper_sha256": helper_sha256,
            "bootstrap_status": original_status,
            "bootstrap_status_sha256": bootstrap_sha256,
            "candidate_directory": original_candidate,
            "candidate_expected_status": certify_release.TABLE3_CANDIDATE_STATUS,
            "candidate_promoted": False,
            "provider_imported": False,
            "qpu_contacted": False,
            "source_identity_before": audit_identity,
            "source_identity_after": audit_identity,
            "reference_tree_before": empty_snapshot,
            "reference_tree_after": empty_snapshot,
            "invocation_tree_before": empty_snapshot,
            "invocation_tree_after": empty_snapshot,
            "candidate_tree_before": candidate_snapshot,
            "candidate_tree_after": candidate_snapshot,
            "candidate_mutated": False,
            "reference_mutated": False,
            "invocation_mutated": False,
            "observed_cases": {
                label: {} for label in certify_release.TABLE3_LABELS
            },
            "checks": checks,
            "passed": len(checks),
            "failed": 0,
        }
        audit_path = provenance_directory / "audit_report.json"
        audit_path.write_text(
            json.dumps(audit_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        provenance = {
            "schema_version": 2,
            "promotion_kind": "TABLE3_EXPLICIT_PROMOTION",
            "candidate_status": certify_release.TABLE3_CANDIDATE_STATUS,
            "candidate_manifest_sha256": candidate_records[
                next(
                    index
                    for index, record in enumerate(candidate_records)
                    if record["path"] == "canonical_table3.json"
                )
            ]["sha256"],
            "candidate_tree_aggregate_sha256": candidate_snapshot[
                "aggregate_sha256"
            ],
            "audit_report_path": "provenance/audit_report.json",
            "audit_report_sha256": certify_release.sha256_file(audit_path),
            "bootstrap_status_path": "provenance/bootstrap_status.json",
            "bootstrap_status_sha256": bootstrap_sha256,
            "bootstrap_helper_path": "provenance/bootstrap_helper.py",
            "bootstrap_helper_sha256": helper_sha256,
            "audited_source_identity": audited_identity,
            "promotion_tool_path": "promote_table3_candidate.py",
            "promotion_tool_sha256": certify_release.sha256_file(
                source_code / "promote_table3_candidate.py"
            ),
            "audit_tool_path": "audit_table3_candidate.py",
            "audit_tool_sha256": certify_release.sha256_file(
                source_code / "audit_table3_candidate.py"
            ),
            "artifact_count": 9,
            "artifact_aggregate_sha256": aggregate.hexdigest(),
        }
        manifest = {
            **candidate_manifest,
            "status": "CANONICAL_PROMOTED",
            "promotion_provenance": {
                **provenance,
            },
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return manifest_path

    def test_atomic_json_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "status.json"
            atomic_write_json(path, {"status": "RUNNING", "value": 1})
            atomic_write_json(path, {"status": "PASS", "value": 2})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "PASS", "value": 2},
            )
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_source_identity_excludes_generated_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source.py").write_text("value = 1\n", encoding="utf-8")
            before = source_identity(root)
            results = root / "results"
            results.mkdir()
            (results / "generated.json").write_text("{}\n", encoding="utf-8")
            self.assertEqual(source_identity(root), before)
            (root / "source.py").write_text("value = 2\n", encoding="utf-8")
            self.assertNotEqual(source_identity(root), before)

    def test_source_identity_binds_notebook_source_not_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notebook_path = root / "RUNME.ipynb"
            notebook = {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {"kernelspec": {"name": "python3"}},
                "cells": [
                    {
                        "cell_type": "code",
                        "metadata": {},
                        "source": ["value = 1\n"],
                        "execution_count": None,
                        "outputs": [],
                    }
                ],
            }
            notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
            before = source_identity(root)

            notebook["metadata"]["kernelspec"]["name"] = "qbraid-python312"
            notebook["cells"][0]["execution_count"] = 7
            notebook["cells"][0]["outputs"] = [
                {"output_type": "stream", "name": "stdout", "text": ["running\n"]}
            ]
            notebook["cells"][0]["metadata"]["trusted"] = True
            notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
            self.assertEqual(source_identity(root), before)

            checkpoint = root / ".ipynb_checkpoints"
            checkpoint.mkdir()
            (checkpoint / "RUNME-checkpoint.ipynb").write_text(
                json.dumps(notebook),
                encoding="utf-8",
            )
            self.assertEqual(source_identity(root), before)

            notebook["cells"][0]["source"] = ["value = 2\n"]
            notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
            self.assertNotEqual(source_identity(root), before)

    def test_source_identity_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.py"
            link = root / "linked.py"
            target.write_text("value = 1\n", encoding="utf-8")
            try:
                link.symlink_to(target.name)
            except (NotImplementedError, OSError) as error:
                self.skipTest(f"symlinks unavailable: {error}")
            with self.assertRaisesRegex(RuntimeError, "must not contain symlinks"):
                source_identity(root)

    def test_source_and_promoted_reference_reject_special_nodes(self) -> None:
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO creation is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fifo = root / "payload.pipe"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(RuntimeError, "special filesystem"):
                source_identity(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._write_promoted_table3(root)
            os.mkfifo(root / "transport.pipe")
            with self.assertRaisesRegex(RuntimeError, "special filesystem"):
                validate_promoted_table3_reference(manifest)

    def test_promoted_table3_preflight_binds_exact_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._write_promoted_table3(root)
            result = validate_promoted_table3_reference(manifest)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["artifact_count"], 9)

            extra = root / "unexpected.txt"
            extra.write_text("unexpected\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unexpected file set"):
                validate_promoted_table3_reference(manifest)
            extra.unlink()

            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["status"] = "CANDIDATE_NOT_CANONICAL_UNTIL_EXPLICITLY_PROMOTED"
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "not explicitly promoted"):
                validate_promoted_table3_reference(manifest)

    def test_promoted_table3_rejects_arbitrary_provenance_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._write_promoted_table3(root)
            original = json.loads(manifest.read_text(encoding="utf-8"))
            for key in (
                "audit_report_sha256",
                "bootstrap_status_sha256",
                "bootstrap_helper_sha256",
                "promotion_tool_sha256",
                "audit_tool_sha256",
            ):
                with self.subTest(key=key):
                    value = json.loads(json.dumps(original))
                    value["promotion_provenance"][key] = "0" * 64
                    manifest.write_text(
                        json.dumps(value, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "does not match",
                    ):
                        validate_promoted_table3_reference(manifest)

    def test_promoted_table3_rejects_rehashed_forged_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._write_promoted_table3(root)
            audit_path = root / "provenance" / "audit_report.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["checks"] = [{"name": "fabricated audit", "status": "PASS"}]
            audit["passed"] = 1
            audit_path.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["promotion_provenance"][
                "audit_report_sha256"
            ] = certify_release.sha256_file(audit_path)
            manifest.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "not an authoritative PASS",
            ):
                validate_promoted_table3_reference(manifest)

    def test_promoted_table3_rejects_rehashed_minimal_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._write_promoted_table3(root)
            bootstrap_path = root / "provenance" / "bootstrap_status.json"
            bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
            bootstrap["stages"] = []
            bootstrap_path.write_text(
                json.dumps(bootstrap, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            bootstrap_sha256 = certify_release.sha256_file(bootstrap_path)

            audit_path = root / "provenance" / "audit_report.json"
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["bootstrap_status_sha256"] = bootstrap_sha256
            audit_path.write_text(
                json.dumps(audit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["promotion_provenance"].update(
                {
                    "bootstrap_status_sha256": bootstrap_sha256,
                    "audit_report_sha256": certify_release.sha256_file(
                        audit_path
                    ),
                }
            )
            manifest.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "bootstrap status is not audit-bound",
            ):
                validate_promoted_table3_reference(manifest)

    def test_promoted_table3_rejects_post_promotion_source_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_code = Path(temporary) / "Source_Code"
            table3 = source_code / "reference" / "table3"
            table3.mkdir(parents=True)
            for name in (
                "promote_table3_candidate.py",
                "audit_table3_candidate.py",
            ):
                (source_code / name).write_bytes((SOURCE_ROOT / name).read_bytes())
            module = source_code / "scientific_method.py"
            module.write_text("METHOD = 'audited'\n", encoding="utf-8")
            manifest = self._write_promoted_table3(
                table3,
                source_code=source_code,
            )
            result = validate_promoted_table3_reference(
                manifest,
                source_code=source_code,
            )
            self.assertEqual(result["status"], "PASS")

            module.write_text("METHOD = 'changed later'\n", encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError,
                "does not match the independently audited",
            ):
                validate_promoted_table3_reference(
                    manifest,
                    source_code=source_code,
                )

    def test_promoted_table3_rejects_text_npz_and_unconstrained_qasm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._write_promoted_table3(root)
            value = json.loads(manifest_path.read_text(encoding="utf-8"))

            parameter = root / "parameters" / "BeH2-6_seed-3047.npz"
            parameter.write_text("not an npz\n", encoding="utf-8")
            value["cases"]["BeH2-6"]["parameter_artifact"][
                "sha256"
            ] = certify_release.sha256_file(parameter)
            manifest_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "safe NPZ"):
                validate_promoted_table3_reference(manifest_path)

            manifest_path = self._write_promoted_table3(root)
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            qasm = root / "circuits" / "BeH2-6_generic.qasm"
            qasm.write_text(
                qasm.read_text(encoding="utf-8") + "h q[0];\n",
                encoding="utf-8",
            )
            value["cases"]["BeH2-6"]["generic_qasm"][
                "qasm_sha256"
            ] = certify_release.sha256_file(qasm)
            value["cases"]["BeH2-6"]["legacy_generic_unitary"] = value[
                "cases"
            ]["BeH2-6"]["generic_qasm"]
            manifest_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "forbidden statement"):
                validate_promoted_table3_reference(manifest_path)

    def test_promoted_table3_rejects_nonfinite_npz_and_false_resources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._write_promoted_table3(root)
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            parameter = root / "parameters" / "BeH2-12_seed-3047.npz"
            np.savez_compressed(
                parameter,
                singles=np.asarray([[np.nan, -0.2]], dtype=np.float64),
                doubles=np.asarray([[0.3]], dtype=np.float64),
            )
            value["cases"]["BeH2-12"]["parameter_artifact"][
                "sha256"
            ] = certify_release.sha256_file(parameter)
            manifest_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                validate_promoted_table3_reference(manifest_path)

            manifest_path = self._write_promoted_table3(root)
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            value["cases"]["LiH-40"]["structured_qasm"]["depth"] = 99
            value["cases"]["LiH-40"][
                "structured_exact_pauli_network"
            ] = value["cases"]["LiH-40"]["structured_qasm"]
            manifest_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "do not match QASM"):
                validate_promoted_table3_reference(manifest_path)

    def test_artifact_manifest_binds_paths_hashes_and_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            (run / "a.txt").write_text("alpha\n", encoding="utf-8")
            (run / "nested").mkdir()
            (run / "nested" / "b.txt").write_text("beta\n", encoding="utf-8")
            manifest = artifact_manifest(run)
            self.assertEqual(manifest["file_count"], 2)
            self.assertEqual(
                [item["path"] for item in manifest["files"]],
                ["a.txt", "nested/b.txt"],
            )
            self.assertRegex(manifest["aggregate_sha256"], r"^[0-9a-f]{64}$")

    def test_stale_latest_run_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            run = results / "quick_run"
            run.mkdir()
            summary = run / "summary.json"
            execution = run / "execution.json"
            summary.write_text('{"status":"PASS"}\n', encoding="utf-8")
            execution.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "mode": "quick",
                        "invocation_id": "current-1234",
                        "qpu_contacted": False,
                        "provider_imported": False,
                    }
                ),
                encoding="utf-8",
            )
            latest = {
                "status": "PASS",
                "mode": "quick",
                "invocation_id": "stale-12345",
                "run_directory": str(run),
                "summary": str(summary),
                "execution_record": str(execution),
            }
            (results / "latest_run.json").write_text(
                json.dumps(latest),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "stale"):
                validate_reproduction_latest(
                    results_root=results,
                    invocation_id="current-1234",
                    mode="quick",
                )

            latest["invocation_id"] = "current-1234"
            (results / "latest_run.json").write_text(
                json.dumps(latest),
                encoding="utf-8",
            )
            accepted, accepted_run, accepted_execution = validate_reproduction_latest(
                results_root=results,
                invocation_id="current-1234",
                mode="quick",
            )
            self.assertEqual(accepted["invocation_id"], "current-1234")
            self.assertEqual(accepted_run, run.resolve())
            self.assertEqual(accepted_execution["status"], "PASS")

    def test_process_group_is_terminated_and_reaped(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
            text=True,
        )
        return_code = terminate_process_group(process, grace_seconds=1.0)
        self.assertIsNotNone(process.poll())
        self.assertEqual(return_code, process.returncode)
        self.assertLess(return_code, 0)

    def test_process_level_sigint_returns_130_and_records_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            status_path = temporary_path / "status.json"
            marker_path = temporary_path / "child.pid"
            log_path = temporary_path / "child.log"
            harness = """
import json
from pathlib import Path
import sys
sys.path.insert(0, sys.argv[1])
from certify_release import atomic_write_json, run_managed_process
status_path = Path(sys.argv[2])
marker_path = Path(sys.argv[3])
log_path = Path(sys.argv[4])
def started(pid):
    marker_path.write_text(str(pid), encoding="utf-8")
result = run_managed_process(
    [sys.executable, "-c", "import time; time.sleep(60)"],
    cwd=Path.cwd(),
    env={**dict(__import__("os").environ), "PYTHONDONTWRITEBYTECODE": "1"},
    log_path=log_path,
    on_started=started,
)
atomic_write_json(status_path, {
    "status": result.status,
    "returncode": result.returncode,
    "child_pid": result.child_pid,
})
raise SystemExit(result.exit_code)
"""
            wrapper = subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    "-c",
                    harness,
                    str(SOURCE_ROOT),
                    str(status_path),
                    str(marker_path),
                    str(log_path),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            try:
                deadline = time.monotonic() + 5.0
                while not marker_path.is_file() and time.monotonic() < deadline:
                    if wrapper.poll() is not None:
                        break
                    time.sleep(0.02)
                self.assertTrue(marker_path.is_file(), "managed child did not start")
                child_pid = int(marker_path.read_text(encoding="utf-8"))
                os.kill(wrapper.pid, signal.SIGINT)
                self.assertEqual(wrapper.wait(timeout=8.0), 130)
                status = json.loads(status_path.read_text(encoding="utf-8"))
                self.assertEqual(status["status"], "INTERRUPTED")
                self.assertEqual(status["child_pid"], child_pid)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
            finally:
                if wrapper.poll() is None:
                    wrapper.kill()
                    wrapper.wait(timeout=2.0)

    def test_preflight_failure_writes_invocation_bound_failed_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "Source_Code"
            root.mkdir()
            results = root / "results" / "judge_reproduction"
            certificates = results / "certificates"
            latest = results / "latest_certificate.json"
            invocation = "preflight-failure-1234"
            with mock.patch.multiple(
                certify_release,
                ROOT=root,
                SOURCE=root / "source",
                RESULTS=results,
                CERTIFICATES=certificates,
                LATEST_CERTIFICATE=latest,
            ), mock.patch.object(
                certify_release,
                "validate_promoted_table3_reference",
                return_value={"status": "PASS"},
            ), mock.patch.dict(
                os.environ,
                {"QBRAID_GQE_ENV": str(root / "forbidden-environment")},
                clear=False,
            ):
                return_code = certify_release.main(
                    ["--quick", "--invocation-id", invocation]
                )
            self.assertEqual(return_code, 1)
            record = json.loads(latest.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "FAILED")
            self.assertEqual(record["invocation_id"], invocation)
            self.assertIsNone(record["child_pid"])
            self.assertIn("outside", record["error"]["message"])

    def test_quick_fails_before_execution_without_promoted_table3(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "Source_Code"
            root.mkdir()
            results = root / "results" / "judge_reproduction"
            certificates = results / "certificates"
            latest = results / "latest_certificate.json"
            with mock.patch.multiple(
                certify_release,
                ROOT=root,
                SOURCE=root / "source",
                RESULTS=results,
                CERTIFICATES=certificates,
                LATEST_CERTIFICATE=latest,
            ):
                invocation = "missing-table3-quick-1234"
                self.assertEqual(
                    certify_release.main(
                        ["--quick", "--invocation-id", invocation]
                    ),
                    1,
                )
                record = json.loads(latest.read_text(encoding="utf-8"))
                self.assertEqual(record["status"], "FAILED")
                self.assertIsNone(record["child_pid"])
                self.assertIn(
                    "promoted canonical Table-3 reference is missing",
                    record["error"]["message"],
                )

    def test_full_reaches_environment_without_promoted_table3(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "Source_Code"
            root.mkdir()
            results = root / "results" / "judge_reproduction"
            certificates = results / "certificates"
            latest = results / "latest_certificate.json"
            with mock.patch.multiple(
                certify_release,
                ROOT=root,
                SOURCE=root / "source",
                RESULTS=results,
                CERTIFICATES=certificates,
                LATEST_CERTIFICATE=latest,
            ), mock.patch.object(
                certify_release,
                "validate_promoted_table3_reference",
            ) as promoted_preflight, mock.patch.dict(
                os.environ,
                {
                    "QBRAID_GQE_ENV": str(
                        Path(temporary) / "missing-pinned-environment"
                    )
                },
            ):
                invocation = "missing-table3-full-1234"
                self.assertEqual(
                    certify_release.main(
                        ["--full", "--invocation-id", invocation]
                    ),
                    1,
                )
            promoted_preflight.assert_not_called()
            record = json.loads(latest.read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "FAILED")
            self.assertIsNone(record["child_pid"])
            self.assertIn("pinned environment missing", record["error"]["message"])
            self.assertEqual(
                record["promoted_table3_reference"]["status"],
                "NOT_REQUIRED_FOR_FRESH_FULL_REPRODUCTION",
            )


if __name__ == "__main__":
    unittest.main()
