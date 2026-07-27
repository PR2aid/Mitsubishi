from __future__ import annotations

from collections import Counter
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from install_locked_requirements import (
    CONTROLLED_PIP_ENVIRONMENT,
    DEFAULT_BATCH_SIZE,
    EXPECTED_LOCKED_DISTRIBUTIONS,
    MAX_BATCH_SIZE,
    check_environment,
    install_commands,
    install_lock,
    parse_lock_file,
    parse_lock_text,
    pip_environment,
    plan_batches,
)


class LowDiskInstallerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lock = SOURCE_ROOT / "requirements.lock"
        cls.requirements = parse_lock_file(cls.lock)

    def test_complete_lock_coverage_exactly_once(self) -> None:
        self.assertEqual(len(self.requirements), EXPECTED_LOCKED_DISTRIBUTIONS)
        batches = plan_batches(self.requirements)
        flattened = [
            requirement.specification
            for batch in batches
            for requirement in batch
        ]
        expected = [requirement.specification for requirement in self.requirements]
        self.assertEqual(Counter(flattened), Counter(expected))
        self.assertTrue(all(count == 1 for count in Counter(flattened).values()))
        self.assertEqual(batches, plan_batches(self.requirements))
        self.assertEqual(batches[0][0].canonical_name, "pip")

    def test_batch_bounds_are_small_and_enforced(self) -> None:
        batches = plan_batches(self.requirements)
        self.assertEqual(DEFAULT_BATCH_SIZE, 4)
        self.assertEqual(MAX_BATCH_SIZE, 8)
        self.assertTrue(all(1 <= len(batch) <= DEFAULT_BATCH_SIZE for batch in batches))
        self.assertEqual(len(batches), 33)
        with self.assertRaisesRegex(ValueError, "between"):
            plan_batches(self.requirements, batch_size=0)
        with self.assertRaisesRegex(ValueError, "between"):
            plan_batches(self.requirements, batch_size=MAX_BATCH_SIZE + 1)

    def test_every_install_command_is_isolated_no_deps_and_no_cache(self) -> None:
        batches = plan_batches(self.requirements)
        commands = install_commands(Path("/dedicated/bin/python"), batches)
        self.assertEqual(len(commands), len(batches))
        for command, batch in zip(commands, batches):
            self.assertIn("--isolated", command)
            self.assertIn("--no-deps", command)
            self.assertIn("--no-cache-dir", command)
            self.assertEqual(command.count("install"), 1)
            self.assertEqual(
                list(command[-len(batch) :]),
                [requirement.specification for requirement in batch],
            )

    def test_malformed_or_ambiguous_lock_lines_are_rejected(self) -> None:
        malformed = (
            "numpy>=2.0",
            "--extra-index-url https://example.invalid",
            "numpy==2.5.1; python_version >= '3.12'",
            "numpy @ https://example.invalid/numpy.whl",
            "numpy==2.5.1 # inline comments are not lock entries",
        )
        for line in malformed:
            with self.subTest(line=line):
                with self.assertRaisesRegex(ValueError, "name==version"):
                    parse_lock_text(line + "\n", expected_count=None)
        with self.assertRaisesRegex(ValueError, "Duplicate canonical"):
            parse_lock_text(
                "my_package==1.0\nmy-package==1.0\n",
                expected_count=None,
            )

    def test_every_mocked_pip_child_gets_only_controlled_pip_environment(self) -> None:
        calls: list[tuple[list[str], dict[str, str]]] = []

        def runner(
            command: list[str],
            *,
            check: bool,
            env: dict[str, str],
        ) -> None:
            self.assertTrue(check)
            calls.append((command, dict(env)))

        ambient = {
            "PATH": os.environ.get("PATH", ""),
            "PIP_INDEX_URL": "https://untrusted.invalid",
            "PIP_ARBITRARY_FUTURE_OPTION": "must-not-survive",
            "PIP_CONFIG_FILE": "/untrusted/pip.conf",
        }
        with redirect_stdout(io.StringIO()):
            batches = install_lock(
                python=Path(sys.executable),
                lock=self.lock,
                base_environment=ambient,
                runner=runner,
            )
        self.assertEqual(len(calls), len(batches))
        for command, environment in calls:
            self.assertIn("--isolated", command)
            self.assertIn("--no-deps", command)
            self.assertIn("--no-cache-dir", command)
            pip_keys = {key for key in environment if key.startswith("PIP_")}
            self.assertEqual(pip_keys, CONTROLLED_PIP_ENVIRONMENT)
            self.assertEqual(environment["PIP_CONFIG_FILE"], os.devnull)
            self.assertNotIn("PIP_ARBITRARY_FUTURE_OPTION", environment)

        check_calls: list[tuple[list[str], dict[str, str]]] = []

        def check_runner(
            command: list[str],
            *,
            check: bool,
            env: dict[str, str],
        ) -> None:
            self.assertTrue(check)
            check_calls.append((command, dict(env)))

        check_environment(
            python=Path(sys.executable),
            base_environment=ambient,
            runner=check_runner,
        )
        self.assertEqual(len(check_calls), 1)
        check_command, check_env = check_calls[0]
        self.assertIn("--isolated", check_command)
        self.assertEqual(check_command[-1], "check")
        self.assertEqual(
            {key for key in check_env if key.startswith("PIP_")},
            CONTROLLED_PIP_ENVIRONMENT,
        )

    def test_pip_environment_removes_unknown_future_pip_keys(self) -> None:
        environment = pip_environment(
            {
                "PIP_SOMETHING_NOT_YET_INVENTED": "unsafe",
                "PIP_CONFIG_FILE": "/unsafe",
            }
        )
        self.assertNotIn("PIP_SOMETHING_NOT_YET_INVENTED", environment)
        self.assertEqual(environment["PIP_CONFIG_FILE"], os.devnull)


if __name__ == "__main__":
    unittest.main()
