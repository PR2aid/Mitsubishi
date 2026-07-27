from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import validate_submission_results as validation
from certify_release import TABLE3_CANDIDATE_STATUS
from tests_hardening import test_certify_release as certify_tests


class ValidationModeContractTests(unittest.TestCase):
    def test_quick_mode_binds_semantics_and_expected_table3_resources(self) -> None:
        audit = validation.Audit()
        advanced_root = Path("/tmp/advanced")
        advanced = {"structured_resources": {}}
        expected = {"table3_structured_resources": {}}

        with (
            mock.patch.object(validation, "validate_table3_semantics") as semantics,
            mock.patch.object(
                validation, "validate_expected_table3_resources"
            ) as resources,
        ):
            validation.validate_table3_claims(
                audit,
                advanced_root,
                advanced,
                expected,
                "quick",
            )

        semantics.assert_called_once_with(
            audit,
            advanced_root,
            advanced,
            "quick",
        )
        resources.assert_called_once_with(audit, advanced, expected)

    def test_full_mode_additionally_binds_fresh_candidate(
        self,
    ) -> None:
        audit = validation.Audit()
        advanced_root = Path("/tmp/advanced")
        advanced = {"structured_resources": {}}
        expected = {"structured_resources": {}}
        with (
            mock.patch.object(validation, "validate_table3_semantics"),
            mock.patch.object(
                validation, "validate_expected_table3_resources"
            ),
            mock.patch.object(
                validation, "validate_full_table3_candidate_binding"
            ) as binding,
        ):
            validation.validate_table3_claims(
                audit, advanced_root, advanced, expected, "full"
            )
        binding.assert_called_once_with(audit, advanced_root, advanced)

    def test_full_candidate_binding_accepts_exact_bytes_and_rejects_tamper(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            advanced_root = root / "advanced"
            candidate_root = advanced_root / "canonical_table3_candidate"
            candidate_root.mkdir(parents=True)
            certify_tests.CertificateInfrastructureTests()._write_promoted_table3(
                candidate_root
            )
            shutil.rmtree(candidate_root / "provenance")
            candidate_manifest_path = (
                candidate_root / "canonical_table3.json"
            )
            candidate_manifest = json.loads(
                candidate_manifest_path.read_text(encoding="utf-8")
            )
            candidate_manifest["status"] = TABLE3_CANDIDATE_STATUS
            candidate_manifest.pop("promotion_provenance")
            candidate_manifest_path.write_text(
                json.dumps(candidate_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            resources = {
                label: {
                    key: value
                    for key, value in case.items()
                    if key not in {"ansatz", "parameter_artifact"}
                }
                for label, case in candidate_manifest["cases"].items()
            }
            advanced = {"structured_resources": resources}
            audit = validation.Audit()
            validation.validate_full_table3_candidate_binding(
                audit,
                advanced_root,
                advanced,
            )
            self.assertEqual(audit.close(), "PASS")

            qasm = candidate_root / "circuits" / "BeH2-6_generic.qasm"
            original_qasm = qasm.read_bytes()
            qasm.write_text(
                qasm.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            tampered = validation.Audit()
            validation.validate_full_table3_candidate_binding(
                tampered,
                advanced_root,
                advanced,
            )
            self.assertEqual(tampered.close(), "FAIL")

            qasm.write_bytes(original_qasm)
            parameter = (
                candidate_root
                / "parameters"
                / "BeH2-6_seed-3047.npz"
            )
            parameter.write_bytes(parameter.read_bytes() + b"tamper")
            parameter_tampered = validation.Audit()
            validation.validate_full_table3_candidate_binding(
                parameter_tampered,
                advanced_root,
                advanced,
            )
            self.assertEqual(parameter_tampered.close(), "FAIL")

            parameter.write_bytes(parameter.read_bytes()[:-6])
            real_candidate_root = advanced_root / "real_table3_candidate"
            candidate_root.rename(real_candidate_root)
            candidate_root.symlink_to(
                real_candidate_root,
                target_is_directory=True,
            )
            symlinked_root = validation.Audit()
            validation.validate_full_table3_candidate_binding(
                symlinked_root,
                advanced_root,
                advanced,
            )
            self.assertEqual(symlinked_root.close(), "FAIL")


if __name__ == "__main__":
    unittest.main()
