#!/usr/bin/env python3
"""Focused provider-free tests for the restart-safe optimizer cache."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SOURCE_CODE = Path(__file__).resolve().parents[2]
SCRIPT = SOURCE_CODE / "source" / "scripts" / "run_enhanced_release.py"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "gqe_restart_cache_test_target",
        SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RestartCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_script()
        cls.frozen = SOURCE_CODE / "frozen_inputs"

    def tearDown(self):
        cache = self.module.RESTART_CACHE
        if cache is not None:
            cache.close()
        self.module.RESTART_CACHE = None

    @staticmethod
    def without_runtime(result):
        return {
            key: value
            for key, value in result.items()
            if key not in {"best_params", "wall_seconds"}
        }

    def test_completed_cold_call_replays_exactly_after_reopen(self):
        with tempfile.TemporaryDirectory(prefix="gqe-cache-test-") as temporary:
            root = Path(temporary)
            output = root / "attempt"
            cache_path = root / "cache"
            problem = self.module.load_named_problem(
                self.frozen,
                "BeH2",
                3,
            )
            config = self.module.ansatz_partitioned()

            self.module.RESTART_CACHE = None
            raw = self.module.run_one(
                problem,
                config,
                2,
                17,
            )
            first_cache = self.module.RestartCache(
                cache_path,
                self.frozen,
                output,
            )
            self.module.RESTART_CACHE = first_cache
            first = self.module.run_one(
                problem,
                config,
                2,
                17,
                cache_role="unit:cold",
            )
            first_parameters = {
                key: value.detach().cpu().numpy().copy()
                for key, value in first["best_params"].items()
            }
            first_payload = {
                key: value
                for key, value in first.items()
                if key != "best_params"
            }
            self.assertEqual(
                json.loads(
                    self.module._canonical_json_bytes(
                        self.without_runtime(raw)
                    )
                ),
                json.loads(
                    self.module._canonical_json_bytes(
                        self.without_runtime(first)
                    )
                ),
            )
            for key, expected in raw["best_params"].items():
                self.assertTrue(
                    np.array_equal(
                        expected.detach().cpu().numpy(),
                        first_parameters[key],
                    )
                )
            self.assertEqual(first_cache.stats, {
                "hits": 0,
                "misses": 1,
                "commits": 1,
            })
            first_cache.close()

            second_cache = self.module.RestartCache(
                cache_path,
                self.frozen,
                output,
            )
            self.module.RESTART_CACHE = second_cache
            second = self.module.run_one(
                problem,
                config,
                2,
                17,
                cache_role="unit:cold",
            )
            second_payload = {
                key: value
                for key, value in second.items()
                if key != "best_params"
            }
            self.assertEqual(
                json.loads(
                    self.module._canonical_json_bytes(first_payload)
                ),
                json.loads(
                    self.module._canonical_json_bytes(second_payload)
                ),
            )
            self.assertEqual(set(first_parameters), set(second["best_params"]))
            for key, expected in first_parameters.items():
                self.assertTrue(
                    np.array_equal(
                        expected,
                        second["best_params"][key]
                        .detach()
                        .cpu()
                        .numpy(),
                    )
                )
            self.assertEqual(second_cache.stats, {
                "hits": 1,
                "misses": 0,
                "commits": 0,
            })
            snapshot = second_cache.export_snapshot(
                output / "restart_cache_snapshot"
            )
            self.assertEqual(snapshot["entry_count"], 1)
            self.assertTrue(
                (output / "restart_cache_snapshot" / "SNAPSHOT_READY.json")
                .is_file()
            )

    def test_role_and_initial_parameters_separate_cache_keys(self):
        with tempfile.TemporaryDirectory(prefix="gqe-cache-key-") as temporary:
            root = Path(temporary)
            cache = self.module.RestartCache(
                root / "cache",
                self.frozen,
                root / "attempt",
            )
            base = {
                "kind": "run_one",
                "role": "role-a",
                "problem": {"name": "problem"},
                "ansatz": {"layers": 2},
                "optimizer": {"steps": 2, "seed": 17},
                "initial_parameters_sha256": None,
            }
            role_changed = {**base, "role": "role-b"}
            init_changed = {
                **base,
                "initial_parameters_sha256": "0" * 64,
            }
            self.assertNotEqual(
                cache._entry_key(base),
                cache._entry_key(role_changed),
            )
            self.assertNotEqual(
                cache._entry_key(base),
                cache._entry_key(init_changed),
            )

    def test_corruption_fails_closed_instead_of_recomputing(self):
        with tempfile.TemporaryDirectory(prefix="gqe-cache-corrupt-") as temporary:
            root = Path(temporary)
            output = root / "attempt"
            cache_path = root / "cache"
            problem = self.module.load_named_problem(
                self.frozen,
                "BeH2",
                3,
            )
            config = self.module.ansatz_partitioned()
            cache = self.module.RestartCache(
                cache_path,
                self.frozen,
                output,
            )
            self.module.RESTART_CACHE = cache
            self.module.run_one(
                problem,
                config,
                2,
                17,
                cache_role="unit:corruption",
            )
            cache.close()
            entry = next(
                path
                for path in (cache_path / "entries").iterdir()
                if path.is_dir() and not path.name.startswith(".")
            )
            parameter_path = entry / "parameters.npz"
            data = bytearray(parameter_path.read_bytes())
            data[len(data) // 2] ^= 0x01
            parameter_path.write_bytes(data)

            reopened = self.module.RestartCache(
                cache_path,
                self.frozen,
                output,
            )
            self.module.RESTART_CACHE = reopened
            with self.assertRaises(ValueError):
                self.module.run_one(
                    problem,
                    config,
                    2,
                    17,
                    cache_role="unit:corruption",
                )
            self.assertEqual(reopened.stats["misses"], 0)
            self.assertEqual(reopened.stats["commits"], 0)

    def test_warm_initialization_and_adaptive_depth_reopen_exactly(self):
        with tempfile.TemporaryDirectory(prefix="gqe-cache-warm-") as temporary:
            root = Path(temporary)
            output = root / "attempt"
            cache_path = root / "cache"
            problem = self.module.load_named_problem(
                self.frozen,
                "BeH2",
                3,
            )
            config = self.module.ansatz_partitioned()
            self.module.RESTART_CACHE = None
            cold = self.module.run_one(problem, config, 2, 17)

            cache = self.module.RestartCache(
                cache_path,
                self.frozen,
                output,
            )
            self.module.RESTART_CACHE = cache
            warm = self.module.run_one(
                problem,
                config,
                2,
                42,
                init_params=cold["best_params"],
                cache_role="unit:warm",
            )
            adaptive_optimizer = self.module.OptConfig(
                steps=2,
                seed=17,
                lr=0.05,
            )
            adaptive = self.module.run_adaptive_depth(
                problem,
                self.module.ansatz_all(),
                adaptive_optimizer,
                l_max=3,
                accept_mha=1.0e9,
            )
            self.assertEqual(len(adaptive["depth_stages"]), 2)
            self.assertFalse(adaptive["depth_stages"][-1]["accepted"])
            self.assertGreater(
                adaptive["complete_cascade_wall_seconds"],
                adaptive["wall_seconds"],
            )
            warm_parameters = {
                key: value.detach().cpu().numpy().copy()
                for key, value in warm["best_params"].items()
            }
            adaptive_parameters = {
                key: value.detach().cpu().numpy().copy()
                for key, value in adaptive["best_params"].items()
            }
            cache.close()

            reopened = self.module.RestartCache(
                cache_path,
                self.frozen,
                output,
            )
            self.module.RESTART_CACHE = reopened
            warm_replay = self.module.run_one(
                problem,
                config,
                2,
                42,
                init_params=cold["best_params"],
                cache_role="unit:warm",
            )
            adaptive_replay = self.module.run_adaptive_depth(
                problem,
                self.module.ansatz_all(),
                adaptive_optimizer,
                l_max=3,
                accept_mha=1.0e9,
            )
            for key, expected in warm_parameters.items():
                self.assertTrue(
                    np.array_equal(
                        expected,
                        warm_replay["best_params"][key]
                        .detach()
                        .cpu()
                        .numpy(),
                    )
                )
            for key, expected in adaptive_parameters.items():
                self.assertTrue(
                    np.array_equal(
                        expected,
                        adaptive_replay["best_params"][key]
                        .detach()
                        .cpu()
                        .numpy(),
                    )
                )
            self.assertEqual(reopened.stats["hits"], 2)
            adaptive_usage = [
                item
                for item in reopened.used_entries
                if item["kind"] == "run_adaptive_depth"
            ]
            self.assertEqual(len(adaptive_usage), 1)
            self.assertAlmostEqual(
                adaptive_usage[0]["optimizer_wall_seconds"],
                adaptive_replay["complete_cascade_wall_seconds"],
                places=9,
            )

    def test_partial_temporary_entry_is_ignored_and_excluded_from_snapshot(self):
        with tempfile.TemporaryDirectory(prefix="gqe-cache-residue-") as temporary:
            root = Path(temporary)
            output = root / "attempt"
            cache_path = root / "cache"
            initial = self.module.RestartCache(
                cache_path,
                self.frozen,
                output,
            )
            self.module.RESTART_CACHE = initial
            initial.close()

            residue_name = f".{'f' * 64}.tmp-999-deadbeef"
            residue = cache_path / "entries" / residue_name
            residue.mkdir()
            (residue / "parameters.npz").write_bytes(
                b"interrupted-before-atomic-rename"
            )

            reopened = self.module.RestartCache(
                cache_path,
                self.frozen,
                output,
            )
            self.module.RESTART_CACHE = reopened
            problem = self.module.load_named_problem(
                self.frozen,
                "BeH2",
                3,
            )
            self.module.run_one(
                problem,
                self.module.ansatz_partitioned(),
                2,
                17,
                cache_role="unit:after-crash-residue",
            )

            provenance = reopened.provenance()
            self.assertEqual(
                provenance["ignored_incomplete_entries"],
                [residue_name],
            )
            self.assertEqual(
                {
                    key: provenance[key]
                    for key in ("hits", "misses", "commits")
                },
                {"hits": 0, "misses": 1, "commits": 1},
            )
            self.assertEqual(provenance["completed_entry_count"], 1)
            self.assertEqual(
                provenance["cache_total_completed_entry_count"],
                1,
            )
            self.assertTrue(residue.is_dir())

            snapshot_root = output / "restart_cache_snapshot"
            snapshot = reopened.export_snapshot(snapshot_root)
            used_key = provenance["used_entries"][0]["key"]
            self.assertEqual(snapshot["entry_count"], 1)
            self.assertEqual(
                {
                    path.name
                    for path in (snapshot_root / "entries").iterdir()
                },
                {used_key},
            )
            snapshot_manifest = json.loads(
                (snapshot_root / "SNAPSHOT_MANIFEST.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(snapshot_manifest["entry_keys"], [used_key])
            self.assertNotIn(
                residue_name,
                json.dumps(snapshot_manifest, sort_keys=True),
            )

    def test_lock_and_symlink_cache_roots_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="gqe-cache-lock-") as temporary:
            root = Path(temporary)
            cache_path = root / "cache"
            output = root / "attempt"
            first = self.module.RestartCache(
                cache_path,
                self.frozen,
                output,
            )
            self.module.RESTART_CACHE = first
            with self.assertRaises(RuntimeError):
                self.module.RestartCache(
                    cache_path,
                    self.frozen,
                    output,
                )
            first.close()

            lock_path = cache_path / ".cache.lock"
            self.assertTrue(lock_path.is_file())
            reopened = self.module.RestartCache(
                cache_path,
                self.frozen,
                output,
            )
            self.module.RESTART_CACHE = reopened
            self.assertFalse(reopened._lock.closed)
            reopened.close()
            self.assertTrue(lock_path.is_file())

            real = root / "real-cache"
            real.mkdir()
            linked = root / "linked-cache"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaises(ValueError):
                self.module.RestartCache(
                    linked,
                    self.frozen,
                    output,
                )

    def test_isolated_workflow_denies_python_network_connections(self):
        import reproduce

        code = (
            "import socket,sys; "
            + reproduce._NETWORK_AUDIT_GUARD
            + "socket.socket().connect(('127.0.0.1',9))"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "outbound network disabled by frozen CPU workflow: "
            "socket.connect",
            result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
