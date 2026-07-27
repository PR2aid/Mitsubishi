from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from environment_contract import (
    FRESH_MIN_FREE_BYTES,
    GIB,
    MINIMUM_VCPU,
    REUSE_MIN_FREE_BYTES,
    check_disk_space,
    resolve_environment,
    sanitized_environment,
)


class EnvironmentContractTests(unittest.TestCase):
    def test_default_is_sibling_of_source_code(self) -> None:
        with mock.patch.dict(os.environ):
            os.environ.pop("QBRAID_GQE_ENV", None)
            with tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "submission" / "Source_Code"
                source.mkdir(parents=True)
                self.assertEqual(
                    resolve_environment(source),
                    source.parent / ".qbraid_gqe_env",
                )

    def test_environment_inside_source_code_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "Source_Code"
            source.mkdir()
            with self.assertRaisesRegex(ValueError, "outside"):
                resolve_environment(source, override=source / ".venv")

    def test_declared_disk_thresholds(self) -> None:
        self.assertEqual(MINIMUM_VCPU, 4)
        self.assertEqual(FRESH_MIN_FREE_BYTES, int(12.5 * GIB))
        self.assertEqual(REUSE_MIN_FREE_BYTES, int(2.0 * GIB))
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "future-environment"
            fresh = check_disk_space(
                target,
                mode="fresh",
                free_bytes=int(14.37 * GIB),
            )
            self.assertEqual(fresh["required_gib"], 12.5)
            with self.assertRaisesRegex(OSError, "Insufficient"):
                check_disk_space(
                    target,
                    mode="fresh",
                    free_bytes=FRESH_MIN_FREE_BYTES - 1,
                )
            check_disk_space(
                target,
                mode="reuse",
                free_bytes=REUSE_MIN_FREE_BYTES,
            )

    def test_sanitization_removes_ambient_python_and_pip_hooks(self) -> None:
        base = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": "/untrusted",
            "PYTHONSTARTUP": "/untrusted/start.py",
            "PYTHONUSERBASE": "/untrusted/user",
            "PIP_INDEX_URL": "https://untrusted.invalid",
            "PIP_EXTRA_INDEX_URL": "https://untrusted.invalid/extra",
            "PIP_UNKNOWN_FUTURE_OPTION": "unsafe",
            "MPLBACKEND": "'module://matplotlib_inline.backend_inline'",
        }
        cleaned = sanitized_environment(base)
        for key in (
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
            "PIP_INDEX_URL",
            "PIP_EXTRA_INDEX_URL",
            "PIP_UNKNOWN_FUTURE_OPTION",
        ):
            self.assertNotIn(key, cleaned)
        self.assertEqual(cleaned["PIP_CONFIG_FILE"], os.devnull)
        self.assertEqual(cleaned["PYTHONNOUSERSITE"], "1")
        self.assertEqual(cleaned["PYTHONSAFEPATH"], "1")
        self.assertEqual(cleaned["UCX_TLS"], "self")
        self.assertEqual(cleaned["OMP_NUM_THREADS"], "1")
        self.assertEqual(cleaned["MPLBACKEND"], "Agg")
        self.assertEqual(cleaned["HF_HUB_OFFLINE"], "1")
        self.assertEqual(cleaned["TRANSFORMERS_OFFLINE"], "1")


if __name__ == "__main__":
    unittest.main()
