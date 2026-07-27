#!/usr/bin/env python3
"""Run the deterministic baseline and advanced GQE submission controls.

This program is simulator/CPU only.  It never imports a provider, reads
credentials or submits a paid job.  All rows are generated from checksum-
verified frozen molecular inputs and are written to a new immutable directory.
"""

from __future__ import annotations

import argparse
import atexit
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable
import uuid

import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[1]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
if str(SOURCE.parent) not in sys.path:
    sys.path.insert(0, str(SOURCE.parent))

from givens40 import overhead as oh
from givens40.adaptive_topology import (
    candidate_partitions,
    pareto_front,
    pareto_knee,
    partition_diagnostics,
    transfer_parameters,
)
from givens40.budgeted_gqe import (
    apply_token,
    build_residual_pool,
    remaining_u,
    validate_sequence_budget,
)
from givens40.canonical_resources import (
    TABLE3_SEED,
    bind_table3_circuits,
    bound_record_differences,
    candidate_manifest,
    load_canonical_manifest,
    load_parameter_artifact,
    reference_case_paths,
    save_parameter_artifact,
)
from givens40.energy import energy_and_variance, make_energy_fn
from givens40.frozen_problem import load_manifest, load_named_problem, sha256_file
from givens40.qsci import QSCISolver
from givens40.runner import (
    AnsatzConfig,
    OptConfig,
    SectorCircuit,
    run_adaptive_depth as _run_adaptive_depth,
    run_vqe,
)


PRIMARY_SEEDS = (17, 42, 3047)
CHEMICAL_ACCURACY_MHA = 1.6
ENERGY_NONINFERIOR_MHA = 0.1
PHI_MAX = 15.0
RESTART_TIMING_CLAIM_BOUNDARY = (
    "Energy-evaluation counts remain comparable. Optimizer wall times retain "
    "the conditions of the attempt that produced each atomic cache entry; "
    "mixed-origin wall-time sums and ratios are provenance only and must not "
    "be interpreted as same-attempt speedup measurements."
)
TABLE3_CASES = {
    "BeH2-6": ("BeH2", 3),
    "BeH2-12": ("BeH2", 6),
    "LiH-40": ("LiH", 20),
}
CANONICAL_TABLE3_MANIFEST = (
    SOURCE.parent / "reference" / "table3" / "canonical_table3.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def json_default(value: Any):
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        default=json_default,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _write_fsynced_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            default=json_default,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _copy_fsynced(source: Path, destination: Path) -> None:
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_within(path: Path, parent: Path) -> bool:
    candidate = path.resolve(strict=False)
    container = parent.resolve(strict=False)
    return candidate == container or container in candidate.parents


def _reject_symlink_components(path: Path, *, label: str) -> None:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    current = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")


def _parameter_digest(
    parameters: dict[str, torch.Tensor] | None,
) -> str | None:
    if parameters is None:
        return None
    digest = hashlib.sha256()
    for key in sorted(parameters):
        array = np.ascontiguousarray(
            parameters[key].detach().cpu().numpy().astype(np.float64)
        )
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(
            ",".join(str(int(value)) for value in array.shape).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _problem_identity(problem: Any) -> dict[str, Any]:
    fingerprint = problem.meta.get("scientific_fingerprint_sha256")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError(
            f"frozen problem lacks a scientific fingerprint: {problem.name}"
        )
    return {
        "name": str(problem.name),
        "norb": int(problem.norb),
        "n_qubits": int(problem.n_qubits),
        "nelec": [int(value) for value in problem.nelec],
        "scientific_fingerprint_sha256": fingerprint,
    }


class RestartCache:
    """Atomic, replay-validated cache for completed optimizer calls.

    The cache is outside ``Source_Code`` and is activated only through the
    explicit ``GQE_RESTART_CACHE`` environment variable.  Every entry binds
    the exact source identity, locked runtime, frozen inputs, optimizer
    arguments, initialization parameters and full result history.  A cache
    hit is returned only after parameter SHA-256 verification and independent
    energy/Phi replay.
    """

    SCHEMA_VERSION = 1
    KIND = "GQE_RESTART_SAFE_OPTIMIZER_CACHE"

    def __init__(self, root: Path, frozen: Path, output: Path):
        from certify_release import source_identity

        raw_root = root.expanduser()
        _reject_symlink_components(raw_root, label="GQE_RESTART_CACHE")
        self.root = raw_root.resolve(strict=False)
        self.source_root = SOURCE.parent.resolve()
        self.output = output.resolve()
        if _is_within(self.root, self.source_root) or _is_within(
            self.source_root, self.root
        ):
            raise ValueError("restart cache must be disjoint from Source_Code")
        if _is_within(self.root, self.output) or _is_within(
            self.output, self.root
        ):
            raise ValueError("restart cache must be disjoint from attempt output")
        if os.path.lexists(self.root) and self.root.is_symlink():
            raise ValueError(f"restart cache cannot be a symlink: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir() or self.root.is_symlink():
            raise ValueError(f"restart cache is not a regular directory: {self.root}")

        self.entries = self.root / "entries"
        if os.path.lexists(self.entries) and self.entries.is_symlink():
            raise ValueError(f"restart cache entries cannot be a symlink: {self.entries}")
        self.entries.mkdir(exist_ok=True)

        self.lock_path = self.root / ".cache.lock"
        if os.path.lexists(self.lock_path) and self.lock_path.is_symlink():
            raise ValueError(f"restart cache lock cannot be a symlink: {self.lock_path}")
        self._lock = self.lock_path.open("a+b")
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock.close()
            raise RuntimeError(
                f"another process owns the restart cache: {self.root}"
            ) from error
        atexit.register(self.close)

        identity = {
            "schema_version": self.SCHEMA_VERSION,
            "kind": self.KIND,
            "source_identity": source_identity(self.source_root),
            "frozen_manifest_sha256": sha256_file(frozen / "MANIFEST.json"),
            "python": sys.version,
            "packages": {
                name: version(name)
                for name in (
                    "numpy",
                    "scipy",
                    "pyscf",
                    "torch",
                    "qiskit",
                    "qiskit-aer",
                )
            },
            "threads": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "provider_imported": False,
            "qpu_contacted": False,
        }
        self.identity_path = self.root / "CACHE_IDENTITY.json"
        if self.identity_path.exists():
            if not self.identity_path.is_file() or self.identity_path.is_symlink():
                raise ValueError(
                    f"restart cache identity is not a regular file: {self.identity_path}"
                )
            observed = json.loads(self.identity_path.read_text(encoding="utf-8"))
            if observed != identity:
                raise ValueError(
                    "restart cache identity differs from the current source, "
                    "environment or frozen inputs"
                )
        else:
            temporary = self.root / (
                f".CACHE_IDENTITY.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            _write_fsynced_json(temporary, identity)
            os.replace(temporary, self.identity_path)
            _fsync_directory(self.root)
        self.identity_sha256 = sha256_file(self.identity_path)
        self.stats = {"hits": 0, "misses": 0, "commits": 0}
        self.hit_wall_seconds = 0.0
        self.committed_wall_seconds = 0.0
        self.used_entries: list[dict[str, Any]] = []
        self.snapshot_record: dict[str, Any] | None = None

    def close(self) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None or lock.closed:
            return
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        finally:
            lock.close()

    def _entry_key(self, spec: dict[str, Any]) -> str:
        return hashlib.sha256(
            _canonical_json_bytes(
                {
                    "cache_identity_sha256": self.identity_sha256,
                    "spec": spec,
                }
            )
        ).hexdigest()

    def _validate_result(
        self,
        result: dict[str, Any],
        *,
        spec: dict[str, Any],
        problem: Any,
        expected_seed: int,
    ) -> None:
        if not isinstance(result.get("best_params"), dict):
            raise ValueError("cached optimizer result has no parameter mapping")
        ansatz = result.get("ansatz")
        if not isinstance(ansatz, dict):
            raise ValueError("cached optimizer result has no ansatz record")
        config = AnsatzConfig(**ansatz)
        observed_ansatz = json.loads(_canonical_json_bytes(ansatz))
        expected_ansatz = dict(spec["ansatz"])
        if spec["kind"] == "run_adaptive_depth":
            expected_ansatz["layers"] = observed_ansatz.get("layers")
        if observed_ansatz != expected_ansatz:
            raise ValueError("cached optimizer ansatz differs from its call specification")
        circuit = SectorCircuit(problem, config)
        expected_parameters = circuit.init_params(int(expected_seed))
        parameters = result["best_params"]
        if set(parameters) != set(expected_parameters):
            raise ValueError("cached optimizer parameter names changed")
        for key, expected in expected_parameters.items():
            observed = parameters[key]
            if (
                observed.dtype != torch.float64
                or tuple(observed.shape) != tuple(expected.shape)
                or not bool(torch.isfinite(observed).all())
            ):
                raise ValueError(
                    f"cached optimizer parameter tensor is invalid: {key}"
                )

        hdiag = problem.hdiag() if config.init_state == "diag" else None
        energy_fn, _ = make_energy_fn(problem)
        with torch.no_grad():
            state = circuit.forward(parameters, hdiag)
            replay_energy = float(energy_fn(state))
        _, replay_variance_value = energy_and_variance(state, problem)
        replay_variance = float(replay_variance_value)
        recorded_energy = float(result["e_vqe"])
        if (
            not math.isfinite(replay_energy)
            or not math.isfinite(recorded_energy)
            or abs(replay_energy - recorded_energy) > 1e-12
        ):
            raise ValueError(
                "cached optimizer energy replay failed: "
                f"{replay_energy} != {recorded_energy}"
            )
        singles, doubles = circuit.cross_angle_arrays(parameters)
        replay_phi = float(oh.circuit_phi(singles, doubles))
        recorded_phi = float(result["phi"])
        if (
            not math.isfinite(replay_phi)
            or not math.isfinite(recorded_phi)
            or abs(replay_phi - recorded_phi) > 1e-12
        ):
            raise ValueError(
                f"cached optimizer Phi replay failed: {replay_phi} != {recorded_phi}"
            )
        history = result.get("history")
        history_values = (
            [float(value) for value in history]
            if isinstance(history, list)
            else []
        )
        if (
            not isinstance(history, list)
            or not history
            or not all(math.isfinite(value) for value in history_values)
            or abs(min(history_values) - recorded_energy) > 1e-12
        ):
            raise ValueError("cached optimizer history does not bind its best energy")
        if (
            result.get("problem") != problem.name
            or int(result.get("norb", -1)) != int(problem.norb)
            or int(result.get("n_qubits", -1)) != int(problem.n_qubits)
            or list(result.get("nelec", ()))
            != [int(value) for value in problem.nelec]
            or int(result.get("sector_dim", -1))
            != int(np.prod(problem.dims))
            or int(result.get("seed", -1)) != int(expected_seed)
        ):
            raise ValueError("cached optimizer problem or seed identity changed")
        optimizer_spec = spec["optimizer"]
        scheduled_steps = int(optimizer_spec["steps"])
        extension_chunks = int(result.get("extension_chunks_used", -1))
        completed_steps = int(result.get("steps", -1))
        maximum_chunks = int(optimizer_spec["extend_max_chunks"])
        expected_completed_steps = scheduled_steps + extension_chunks * max(
            1,
            scheduled_steps // 2,
        )
        polish_steps = int(optimizer_spec["polish_steps"])
        expected_minimum_evaluations = completed_steps + 2
        evaluations = int(result.get("energy_evaluations", -1))
        if (
            int(result.get("steps_scheduled", -1)) != scheduled_steps
            or extension_chunks < 0
            or extension_chunks > maximum_chunks
            or completed_steps != expected_completed_steps
            or len(history_values)
            not in (
                {completed_steps + 1}
                if polish_steps == 0
                else {completed_steps + 1, completed_steps + 2}
            )
            or evaluations < expected_minimum_evaluations
            or (polish_steps == 0 and evaluations != expected_minimum_evaluations)
            or not math.isclose(
                float(result.get("history_first", math.nan)),
                history_values[0],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("cached optimizer step/evaluation contract changed")
        expected_e_hf = float(problem.e_hf)
        expected_e_casci = (
            None
            if problem.e_casci is None
            else float(problem.e_casci)
        )
        expected_error = (
            None
            if expected_e_casci is None
            else 1000.0 * (replay_energy - expected_e_casci)
        )
        expected_corr = (
            (expected_e_hf - replay_energy)
            / (expected_e_hf - expected_e_casci)
            if expected_e_casci is not None
            and abs(expected_e_hf - expected_e_casci) > 1e-12
            else None
        )

        def scalar_matches(observed: Any, expected: float | None) -> bool:
            if expected is None:
                return observed is None
            try:
                value = float(observed)
            except (TypeError, ValueError):
                return False
            return math.isfinite(value) and math.isclose(
                value,
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )

        if (
            not scalar_matches(result.get("e_hf"), expected_e_hf)
            or not scalar_matches(result.get("e_casci"), expected_e_casci)
            or not scalar_matches(result.get("error_mha"), expected_error)
            or not scalar_matches(result.get("corr_fraction"), expected_corr)
            or not math.isfinite(replay_variance)
            or replay_variance < -1e-14
            or not scalar_matches(
                result.get("variance_ha2"),
                replay_variance,
            )
            or not scalar_matches(
                result.get("weinstein_radius_mha"),
                1000.0 * math.sqrt(max(0.0, replay_variance)),
            )
            or not scalar_matches(
                result.get("u_budget"),
                (
                    None
                    if circuit.topo.u_budget is None
                    else float(circuit.topo.u_budget)
                ),
            )
            or not scalar_matches(
                result.get("u_committed"),
                (
                    None
                    if circuit.topo.u_committed is None
                    else float(circuit.topo.u_committed)
                ),
            )
        ):
            raise ValueError(
                "cached optimizer molecular, variance or budget record changed"
            )
        expected_pairs = [tuple(pair) for pair in circuit.topo.pairs]
        observed_pairs = [tuple(pair) for pair in result.get("pairs", [])]
        if (
            observed_pairs != expected_pairs
            or int(result.get("n_pairs", -1)) != len(expected_pairs)
            or int(result.get("n_cross_pairs", -1))
            != len(circuit.cross_idx)
        ):
            raise ValueError("cached optimizer topology record changed")
        threshold = float(spec["optimizer"]["active_threshold"])
        active = {
            key: int((value.abs() > threshold).sum())
            for key, value in parameters.items()
        }
        if result.get("active_gates") != active:
            raise ValueError("cached optimizer active-gate record changed")
        if (
            not math.isfinite(float(result.get("wall_seconds", math.nan)))
            or float(result["wall_seconds"]) < 0.0
            or evaluations <= 0
        ):
            raise ValueError("cached optimizer accounting record is invalid")
        if spec["kind"] == "run_adaptive_depth":
            stages = result.get("depth_stages")
            trace = result.get("depth_trace")
            initial_layers = int(spec["ansatz"]["layers"])
            l_max = int(spec["l_max"])
            accept_mha = float(spec["accept_mha"])
            if (
                not isinstance(stages, list)
                or not stages
                or not isinstance(trace, list)
                or len(trace) != len(stages)
            ):
                raise ValueError("cached adaptive-depth trace is incomplete")
            accepted_energy = None
            accepted_layer = None
            accepted_evaluations = None
            accepted_wall = None
            evaluation_sum = 0
            wall_sum = 0.0
            rejected = False
            allowed_stage_evaluations = {
                scheduled_steps
                + chunk_index * max(1, scheduled_steps // 2)
                + 2
                for chunk_index in range(maximum_chunks + 1)
            }
            for index, stage in enumerate(stages):
                if not isinstance(stage, dict):
                    raise ValueError("cached adaptive-depth stage is invalid")
                layer = int(stage.get("layers", -1))
                energy = float(stage.get("energy_hartree", math.nan))
                evaluations = int(stage.get("energy_evaluations", 0))
                wall = float(stage.get("wall_seconds", math.nan))
                if (
                    layer != initial_layers + index
                    or layer > l_max
                    or not math.isfinite(energy)
                    or evaluations <= 0
                    or (
                        polish_steps == 0
                        and evaluations not in allowed_stage_evaluations
                    )
                    or not math.isfinite(wall)
                    or wall < 0.0
                    or rejected
                ):
                    raise ValueError(
                        "cached adaptive-depth stage sequence is invalid"
                    )
                if (
                    not isinstance(trace[index], (list, tuple))
                    or len(trace[index]) != 2
                    or int(trace[index][0]) != layer
                    or not math.isfinite(float(trace[index][1]))
                    or abs(float(trace[index][1]) - energy) > 1e-12
                ):
                    raise ValueError(
                        "cached adaptive-depth trace and stages disagree"
                    )
                if index == 0:
                    if stage.get("accepted") is not True:
                        raise ValueError(
                            "cached adaptive-depth root was not accepted"
                        )
                    accepted_energy = energy
                    accepted_layer = layer
                    accepted_evaluations = evaluations
                    accepted_wall = wall
                else:
                    gain = 1000.0 * (float(accepted_energy) - energy)
                    recorded_gain = float(stage.get("gain_mha", math.nan))
                    accepted = bool(gain > accept_mha)
                    if (
                        not math.isfinite(recorded_gain)
                        or abs(recorded_gain - gain) > 1e-10
                        or stage.get("accepted") is not accepted
                    ):
                        raise ValueError(
                            "cached adaptive-depth acceptance rule changed"
                        )
                    if accepted:
                        accepted_energy = energy
                        accepted_layer = layer
                        accepted_evaluations = evaluations
                        accepted_wall = wall
                    else:
                        rejected = True
                        if index != len(stages) - 1:
                            raise ValueError(
                                "cached adaptive-depth continued after rejection"
                            )
                evaluation_sum += evaluations
                wall_sum += wall
            if (
                not (rejected or int(stages[-1]["layers"]) == l_max)
                or accepted_layer != int(result["ansatz"]["layers"])
                or abs(float(accepted_energy) - recorded_energy) > 1e-12
                or int(result.get("energy_evaluations", -1))
                != int(accepted_evaluations)
                or not math.isclose(
                    float(result.get("wall_seconds", math.nan)),
                    float(accepted_wall),
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or int(
                    result.get(
                        "complete_cascade_energy_evaluations",
                        -1,
                    )
                )
                != evaluation_sum
                or not math.isclose(
                    float(
                        result.get(
                            "complete_cascade_wall_seconds",
                            math.nan,
                        )
                    ),
                    wall_sum,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    "cached adaptive-depth final state or cascade accounting changed"
                )

    @staticmethod
    def _accounted_optimizer_wall_seconds(
        result: dict[str, Any],
        spec: dict[str, Any],
    ) -> float:
        if spec["kind"] == "run_adaptive_depth":
            return float(result["complete_cascade_wall_seconds"])
        return float(result["wall_seconds"])

    def _load(
        self,
        entry: Path,
        *,
        key: str,
        spec: dict[str, Any],
        problem: Any,
        expected_seed: int,
    ) -> dict[str, Any]:
        if not entry.is_dir() or entry.is_symlink():
            raise ValueError(f"restart cache entry is not a regular directory: {entry}")
        children = {path.name for path in entry.iterdir()}
        expected_children = {"DONE.json", "manifest.json", "parameters.npz"}
        if children != expected_children:
            raise ValueError(
                f"restart cache entry is incomplete: {entry}; files={sorted(children)}"
            )
        done_path = entry / "DONE.json"
        manifest_path = entry / "manifest.json"
        parameters_path = entry / "parameters.npz"
        for path in (done_path, manifest_path, parameters_path):
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"restart cache artifact is not regular: {path}")

        done = json.loads(done_path.read_text(encoding="utf-8"))
        manifest_sha = sha256_file(manifest_path)
        parameters_sha = sha256_file(parameters_path)
        if done != {
            "schema_version": self.SCHEMA_VERSION,
            "kind": "GQE_RESTART_CACHE_ENTRY_DONE",
            "key": key,
            "manifest_sha256": manifest_sha,
            "parameters_sha256": parameters_sha,
        }:
            raise ValueError(f"restart cache completion record is invalid: {entry}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != self.SCHEMA_VERSION
            or manifest.get("kind") != "GQE_RESTART_CACHE_ENTRY"
            or manifest.get("key") != key
            or manifest.get("cache_identity_sha256") != self.identity_sha256
            or manifest.get("spec") != spec
            or manifest.get("parameters_sha256") != parameters_sha
            or not isinstance(manifest.get("result"), dict)
        ):
            raise ValueError(f"restart cache manifest is invalid: {entry}")

        with np.load(parameters_path, allow_pickle=False) as replay:
            if not replay.files:
                raise ValueError(f"restart cache has no parameters: {entry}")
            parameters = {}
            for key_name in replay.files:
                array = np.asarray(replay[key_name])
                if array.dtype != np.float64 or not np.isfinite(array).all():
                    raise ValueError(
                        f"restart cache parameter array is invalid: {key_name}"
                    )
                parameters[key_name] = torch.from_numpy(array.copy())
        result = dict(manifest["result"])
        if "pairs" in result:
            result["pairs"] = [
                tuple(int(value) for value in pair)
                for pair in result["pairs"]
            ]
        result["best_params"] = parameters
        self._validate_result(
            result,
            spec=spec,
            problem=problem,
            expected_seed=expected_seed,
        )
        return result

    def _commit(
        self,
        entry: Path,
        *,
        key: str,
        spec: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{key}.tmp-{os.getpid()}-",
                dir=self.entries,
            )
        )
        try:
            parameters_path = temporary / "parameters.npz"
            with parameters_path.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    **{
                        name: value.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                        for name, value in sorted(result["best_params"].items())
                    },
                )
                stream.flush()
                os.fsync(stream.fileno())
            parameters_sha = sha256_file(parameters_path)
            payload = {
                name: value
                for name, value in result.items()
                if name != "best_params"
            }
            manifest_path = temporary / "manifest.json"
            _write_fsynced_json(
                manifest_path,
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "kind": "GQE_RESTART_CACHE_ENTRY",
                    "key": key,
                    "cache_identity_sha256": self.identity_sha256,
                    "spec": spec,
                    "parameters_sha256": parameters_sha,
                    "result": payload,
                },
            )
            done_path = temporary / "DONE.json"
            _write_fsynced_json(
                done_path,
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "kind": "GQE_RESTART_CACHE_ENTRY_DONE",
                    "key": key,
                    "manifest_sha256": sha256_file(manifest_path),
                    "parameters_sha256": parameters_sha,
                },
            )
            _fsync_directory(temporary)
            if entry.exists():
                raise FileExistsError(
                    f"restart cache entry appeared during commit: {entry}"
                )
            os.replace(temporary, entry)
            _fsync_directory(self.entries)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def get_or_compute(
        self,
        spec: dict[str, Any],
        *,
        problem: Any,
        expected_seed: int,
        compute,
    ) -> dict[str, Any]:
        spec = json.loads(_canonical_json_bytes(spec))
        key = self._entry_key(spec)
        entry = self.entries / key
        label = (
            f"{spec['kind']}:{spec['problem']['name']}:"
            f"seed-{expected_seed}"
        )
        if os.path.lexists(entry):
            print(f"[restart cache hit] {label} key={key[:12]}", flush=True)
            result = self._load(
                entry,
                key=key,
                spec=spec,
                problem=problem,
                expected_seed=expected_seed,
            )
            self.stats["hits"] += 1
            wall_seconds = self._accounted_optimizer_wall_seconds(
                result,
                spec,
            )
            self.hit_wall_seconds += wall_seconds
            self.used_entries.append(
                {
                    "key": key,
                    "kind": spec["kind"],
                    "role": spec.get("role"),
                    "disposition": "REPLAYED_VALID_CACHE_HIT",
                    "optimizer_wall_seconds": wall_seconds,
                    "wall_time_semantics": (
                        "COMPLETE_ADAPTIVE_DEPTH_CASCADE"
                        if spec["kind"] == "run_adaptive_depth"
                        else "COMPLETED_RUN_ONE_CALL"
                    ),
                }
            )
            return result

        print(f"[restart cache miss] {label} key={key[:12]}", flush=True)
        self.stats["misses"] += 1
        result = compute()
        self._validate_result(
            result,
            spec=spec,
            problem=problem,
            expected_seed=expected_seed,
        )
        self._commit(
            entry,
            key=key,
            spec=spec,
            result=result,
        )
        replay = self._load(
            entry,
            key=key,
            spec=spec,
            problem=problem,
            expected_seed=expected_seed,
        )
        self.stats["commits"] += 1
        wall_seconds = self._accounted_optimizer_wall_seconds(
            replay,
            spec,
        )
        self.committed_wall_seconds += wall_seconds
        self.used_entries.append(
            {
                "key": key,
                "kind": spec["kind"],
                "role": spec.get("role"),
                "disposition": "COMPUTED_AND_ATOMICALLY_COMMITTED",
                "optimizer_wall_seconds": wall_seconds,
                "wall_time_semantics": (
                    "COMPLETE_ADAPTIVE_DEPTH_CASCADE"
                    if spec["kind"] == "run_adaptive_depth"
                    else "COMPLETED_RUN_ONE_CALL"
                ),
            }
        )
        print(f"[restart cache commit] {label} key={key[:12]}", flush=True)
        return replay

    def provenance(self) -> dict[str, Any]:
        cache_completed = sum(
            1
            for path in self.entries.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and (path / "DONE.json").is_file()
        )
        incomplete = sorted(
            path.name
            for path in self.entries.iterdir()
            if path.name.startswith(".") and ".tmp-" in path.name
        )
        return {
            "kind": self.KIND,
            "schema_version": self.SCHEMA_VERSION,
            "path": str(self.root),
            "identity_sha256": self.identity_sha256,
            "completed_entry_count": len(self.used_entries),
            "cache_total_completed_entry_count": cache_completed,
            "ignored_incomplete_entries": incomplete,
            **self.stats,
            "replayed_optimizer_wall_seconds": self.hit_wall_seconds,
            "committed_optimizer_wall_seconds": self.committed_wall_seconds,
            "used_entries": list(self.used_entries),
            "snapshot": self.snapshot_record,
            "provider_imported": False,
            "qpu_contacted": False,
        }

    def export_snapshot(self, destination: Path) -> dict[str, Any]:
        """Copy the exact used cache entries into the immutable attempt tree."""

        destination = destination.resolve(strict=False)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"restart cache snapshot already exists: {destination}"
            )
        if _is_within(destination, self.source_root):
            raise ValueError("restart cache snapshot cannot enter Source_Code")
        used_keys = [str(item["key"]) for item in self.used_entries]
        if len(used_keys) != len(set(used_keys)):
            raise ValueError("restart cache used-entry keys are not unique")

        parent = destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.tmp-{os.getpid()}-",
                dir=parent,
            )
        )
        try:
            files: list[dict[str, Any]] = []

            def copy_and_record(source: Path, relative: Path) -> None:
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not source.is_file() or source.is_symlink():
                    raise ValueError(
                        f"restart cache snapshot source is not regular: {source}"
                    )
                _copy_fsynced(source, target)
                source_hash = sha256_file(source)
                target_hash = sha256_file(target)
                if source_hash != target_hash:
                    raise RuntimeError(
                        f"restart cache snapshot copy changed bytes: {source}"
                    )
                files.append(
                    {
                        "path": relative.as_posix(),
                        "sha256": target_hash,
                        "size_bytes": target.stat().st_size,
                    }
                )

            copy_and_record(
                self.identity_path,
                Path("CACHE_IDENTITY.json"),
            )
            for key in sorted(used_keys):
                entry = self.entries / key
                expected = {"DONE.json", "manifest.json", "parameters.npz"}
                if (
                    not entry.is_dir()
                    or entry.is_symlink()
                    or {path.name for path in entry.iterdir()} != expected
                ):
                    raise ValueError(
                        f"restart cache entry changed before snapshot: {entry}"
                    )
                for name in sorted(expected):
                    copy_and_record(
                        entry / name,
                        Path("entries") / key / name,
                    )

            snapshot_manifest = {
                "schema_version": self.SCHEMA_VERSION,
                "kind": "GQE_RESTART_CACHE_SNAPSHOT",
                "cache_identity_sha256": self.identity_sha256,
                "entry_keys": sorted(used_keys),
                "files": sorted(files, key=lambda item: item["path"]),
            }
            manifest_path = temporary / "SNAPSHOT_MANIFEST.json"
            _write_fsynced_json(manifest_path, snapshot_manifest)
            ready_path = temporary / "SNAPSHOT_READY.json"
            _write_fsynced_json(
                ready_path,
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "kind": "GQE_RESTART_CACHE_SNAPSHOT_READY",
                    "manifest_sha256": sha256_file(manifest_path),
                    "entry_count": len(used_keys),
                    "file_count": len(files),
                },
            )
            for directory in sorted(
                (
                    path
                    for path in temporary.rglob("*")
                    if path.is_dir()
                ),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                _fsync_directory(directory)
            _fsync_directory(temporary)
            os.rename(temporary, destination)
            _fsync_directory(parent)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

        self.snapshot_record = {
            "relative_path": destination.name,
            "manifest_sha256": sha256_file(
                destination / "SNAPSHOT_MANIFEST.json"
            ),
            "ready_sha256": sha256_file(
                destination / "SNAPSHOT_READY.json"
            ),
            "entry_count": len(used_keys),
            "file_count": len(files),
        }
        return dict(self.snapshot_record)


RESTART_CACHE: RestartCache | None = None


def compact_result(result: dict[str, Any], include_history: bool = True) -> dict[str, Any]:
    excluded = {"best_params", "pairs"}
    if not include_history:
        excluded.add("history")
    return {key: value for key, value in result.items() if key not in excluded}


def save_parameters(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    params = result["best_params"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            key: value.detach().cpu().numpy().astype(np.float64)
            for key, value in params.items()
        },
    )
    with np.load(path, allow_pickle=False) as replay:
        replay_equal = all(
            key in replay
            and np.array_equal(
                np.asarray(replay[key]),
                value.detach().cpu().numpy().astype(np.float64),
            )
            for key, value in params.items()
        ) and set(replay.files) == set(params)
    if not replay_equal:
        raise RuntimeError(f"parameter artifact replay failed: {path}")
    parts = path.parts
    portable = (
        Path(*parts[parts.index("parameters"):])
        if "parameters" in parts
        else Path(path.name)
    )
    return {
        "file": str(portable),
        "sha256": sha256_file(path),
        "array_replay_exact": True,
    }


def ansatz_partitioned(left: Iterable[int] | None = None) -> AnsatzConfig:
    return AnsatzConfig(
        layers=2,
        topology="partitioned",
        partition_override=None if left is None else list(map(int, left)),
        phi_max=PHI_MAX,
        beta_cap=0.05,
        beta_cap_double=0.25,
        engine="adjoint",
    )


def ansatz_all(engine: str = "adjoint") -> AnsatzConfig:
    return AnsatzConfig(layers=2, topology="all", engine=engine)


def state_for_result(problem: Any, config: AnsatzConfig, result: dict[str, Any]):
    circuit = SectorCircuit(problem, config)
    hdiag = problem.hdiag() if config.init_state == "diag" else None
    with torch.no_grad():
        state = circuit.forward(result["best_params"], hdiag)
    return circuit, state


def run_one(
    problem: Any,
    config: AnsatzConfig,
    steps: int,
    seed: int,
    *,
    init_params: dict[str, torch.Tensor] | None = None,
    polish_steps: int = 0,
    cache_role: str | None = None,
) -> dict[str, Any]:
    optimizer = OptConfig(
        steps=int(steps),
        lr=0.05,
        seed=int(seed),
        polish_steps=int(polish_steps),
    )

    def compute() -> dict[str, Any]:
        return run_vqe(
            problem,
            config,
            optimizer,
            init_params=init_params,
            return_params=True,
        )

    if RESTART_CACHE is None:
        return compute()
    if not cache_role:
        raise ValueError("restart-cached optimizer calls require a semantic role")
    spec = {
        "kind": "run_one",
        "role": str(cache_role),
        "problem": _problem_identity(problem),
        "ansatz": asdict(config),
        "optimizer": asdict(optimizer),
        "initial_parameters_sha256": _parameter_digest(init_params),
    }
    return RESTART_CACHE.get_or_compute(
        spec,
        problem=problem,
        expected_seed=int(seed),
        compute=compute,
    )


def run_adaptive_depth(
    problem: Any,
    config: AnsatzConfig,
    optimizer: OptConfig,
    *,
    l_max: int = 4,
    accept_mha: float = 0.1,
    init_params: dict[str, torch.Tensor] | None = None,
    log=lambda _message: None,
) -> dict[str, Any]:
    def compute() -> dict[str, Any]:
        return _run_adaptive_depth(
            problem,
            config,
            optimizer,
            l_max=int(l_max),
            accept_mha=float(accept_mha),
            init_params=init_params,
            log=log,
        )

    if RESTART_CACHE is None:
        return compute()
    spec = {
        "kind": "run_adaptive_depth",
        "problem": _problem_identity(problem),
        "ansatz": asdict(config),
        "optimizer": asdict(optimizer),
        "l_max": int(l_max),
        "accept_mha": float(accept_mha),
        "initial_parameters_sha256": _parameter_digest(init_params),
    }
    return RESTART_CACHE.get_or_compute(
        spec,
        problem=problem,
        expected_seed=int(optimizer.seed),
        compute=compute,
    )


def baseline_matrix(
    frozen: Path,
    output: Path,
    seeds: tuple[int, ...],
    quick: bool,
) -> tuple[dict[str, Any], dict[tuple[str, int], tuple[Any, AnsatzConfig, dict]]]:
    """Regenerate the submission-era numerical ladder from frozen inputs."""

    scale = 0.25 if quick else 1.0
    cases = [
        ("BeH2-6-partitioned", "BeH2", 3, ansatz_partitioned(), 150, seeds),
        ("BeH2-12-partitioned", "BeH2", 6, ansatz_partitioned(), 200, seeds),
        ("BeH2-16-all", "BeH2", 8, ansatz_all("autograd"), 300, (17,)),
        ("LiH-40-all", "LiH", 20, ansatz_all(), 300, (17,)),
        ("LiH-40-partitioned", "LiH", 20, ansatz_partitioned(), 300, seeds),
        ("LiH-44-all", "LiH", 22, ansatz_all(), 300, (17,)),
        ("LiH-48-all", "LiH", 24, ansatz_all(), 300, (17,)),
        ("H2O-12-partitioned-heldout", "H2O", 6, ansatz_partitioned(), 200, seeds),
    ]
    records: dict[str, Any] = {}
    live: dict[tuple[str, int], tuple[Any, AnsatzConfig, dict]] = {}
    params_dir = output / "parameters" / "baseline"
    for label, molecule, norb, config, steps, case_seeds in cases:
        problem = load_named_problem(frozen, molecule, norb)
        records[label] = {"molecule": molecule, "norb": norb, "runs": {}}
        for seed in case_seeds:
            actual_steps = max(20, int(round(steps * scale)))
            print(f"[baseline] {label} seed={seed} steps={actual_steps}", flush=True)
            result = run_one(
                problem,
                config,
                actual_steps,
                seed,
                cache_role=f"baseline:{label}:seed-{seed}",
            )
            parameter = save_parameters(params_dir / f"{label}_s{seed}.npz", result)
            row = compact_result(result)
            row["parameter_artifact"] = parameter
            records[label]["runs"][str(seed)] = row
            live[(label, int(seed))] = (problem, config, result)

    # Run the declared N2 adaptive-depth correlation stress control.
    problem = load_named_problem(frozen, "N2", 8)
    config = AnsatzConfig(
        layers=2,
        topology="all",
        engine="autograd",
        beta_cap=0.05,
        beta_cap_double=0.25,
    )
    steps = max(40, int(round(300 * scale)))
    print(f"[baseline] N2-16-adaptive-depth seed=17 steps={steps}", flush=True)
    result = run_adaptive_depth(
        problem,
        config,
        OptConfig(
            steps=steps,
            seed=17,
            lr=0.05,
            extend_max_chunks=1 if not quick else 0,
            extend_eps=1e-7,
            window=min(50, max(10, steps // 3)),
        ),
        l_max=3 if quick else 4,
        accept_mha=0.1,
    )
    parameter = save_parameters(params_dir / "N2-16-adaptive-depth_s17.npz", result)
    records["N2-16-adaptive-depth"] = {
        "molecule": "N2",
        "norb": 8,
        "runs": {"17": {**compact_result(result), "parameter_artifact": parameter}},
    }
    live[("N2-16-adaptive-depth", 17)] = (
        problem,
        replace(config, layers=int(result["ansatz"]["layers"])),
        result,
    )
    return records, live


def normalized_pareto_distance(
    item: dict[str, Any], front: list[dict[str, Any]]
) -> float:
    values = {}
    for key in ("schmidt_entropy_nats", "hamiltonian_cross_weight"):
        column = [float(x[key]) for x in front]
        lo, hi = min(column), max(column)
        values[key] = 0.0 if hi <= lo else (float(item[key]) - lo) / (hi - lo)
    return math.hypot(values["schmidt_entropy_nats"], values["hamiltonian_cross_weight"])


def adaptive_target(
    label: str,
    static_label: str,
    frozen: Path,
    output: Path,
    live: dict[tuple[str, int], tuple[Any, AnsatzConfig, dict]],
    molecule: str,
    norb: int,
    steps: int,
    seeds: tuple[int, ...],
    quick: bool,
) -> tuple[dict[str, Any], dict[int, tuple[Any, AnsatzConfig, dict]]]:
    """Search topology on a neutral pilot and run matched warm/cold controls."""

    problem = load_named_problem(frozen, molecule, norb)
    actual_steps = max(20, int(round(steps * (0.25 if quick else 1.0))))
    pilot_config = ansatz_all("adjoint" if problem.meta["sector_dim"] > 2000 else "autograd")
    print(f"[topology pilot] {label} all-pair seed=17", flush=True)
    pilot = run_one(
        problem,
        pilot_config,
        actual_steps,
        17,
        cache_role=f"adaptive:{label}:pilot",
    )
    pilot_circuit, pilot_state = state_for_result(problem, pilot_config, pilot)

    static_problem, static_config, static_seed17 = live[(static_label, 17)]
    static_circuit, static_state = state_for_result(
        static_problem, static_config, static_seed17
    )
    incumbent_left = tuple(static_circuit.topo.left_block)

    def evaluate(left):
        return partition_diagnostics(
            problem, pilot_circuit.sector, pilot_state, left
        )

    static_diagnostic = evaluate(incumbent_left)

    candidates = candidate_partitions(
        norb,
        incumbent_left,
        exhaustive_max_norb=8,
        local_rounds=1 if quick else 3,
        frontier_width=4 if quick else 8,
        evaluator=evaluate,
    )
    diagnostics = [evaluate(left) for left in candidates]
    front = pareto_front(diagnostics)
    knee = pareto_knee(front)
    min_entropy = min(
        front,
        key=lambda x: (
            x["schmidt_entropy_nats"],
            x["hamiltonian_cross_weight"],
            tuple(x["left_block"]),
        ),
    )
    min_cross = min(
        front,
        key=lambda x: (
            x["hamiltonian_cross_weight"],
            x["schmidt_entropy_nats"],
            tuple(x["left_block"]),
        ),
    )
    priority = []
    for item in (knee, min_entropy, min_cross):
        if tuple(item["left_block"]) not in {
            tuple(x["left_block"]) for x in priority
        }:
            priority.append(item)
    for item in sorted(
        front,
        key=lambda x: (
            normalized_pareto_distance(x, front),
            tuple(x["left_block"]),
        ),
    ):
        if tuple(item["left_block"]) not in {
            tuple(x["left_block"]) for x in priority
        }:
            priority.append(item)
    max_screen = 2 if quick else min(5, len(priority))
    screened = []
    accepted = None
    params_dir = output / "parameters" / "adaptive"
    for diagnostic in priority[:max_screen]:
        left = diagnostic["left_block"]
        candidate_config = ansatz_partitioned(left)
        candidate_circuit = SectorCircuit(problem, candidate_config)
        warm, transfer = transfer_parameters(
            static_circuit,
            candidate_circuit,
            static_seed17["best_params"],
            seed=17,
            new_pair_scale=0.0,
        )
        print(f"[topology screen] {label} left={left}", flush=True)
        result = run_one(
            problem,
            candidate_config,
            actual_steps,
            17,
            init_params=warm,
            cache_role=(
                f"adaptive:{label}:screen:"
                + "-".join(map(str, left))
            ),
        )
        final_circuit, final_state = state_for_result(problem, candidate_config, result)
        final_diag = partition_diagnostics(
            problem, final_circuit.sector, final_state, left
        )
        entropy_delta = float(diagnostic["schmidt_entropy_nats"]) - float(
            static_diagnostic["schmidt_entropy_nats"]
        )
        cross_delta = float(diagnostic["hamiltonian_cross_weight"]) - float(
            static_diagnostic["hamiltonian_cross_weight"]
        )
        improves_entropy = entropy_delta < -1e-14
        improves_cross = cross_delta < -1e-14
        nonworse_entropy = entropy_delta <= 1e-14
        nonworse_cross = cross_delta <= 1e-14
        passes = {
            "energy_noninferior_within_0p1_mha": bool(
                1000.0 * (result["e_vqe"] - static_seed17["e_vqe"])
                <= ENERGY_NONINFERIOR_MHA + 1e-9
            ),
            "within_phi_budget": bool(result["phi"] <= PHI_MAX + 1e-10),
            "differs_from_static_partition": tuple(left) != tuple(incumbent_left),
            "candidate_on_screened_pareto_front": True,
            "improves_at_least_one_declared_pre_screen_metric": bool(
                improves_entropy or improves_cross
            ),
            "strict_pareto_dominance_vs_static": bool(
                (improves_entropy or improves_cross)
                and nonworse_entropy
                and nonworse_cross
            ),
        }
        parameter = save_parameters(
            params_dir / f"{label}_screen_left-{'-'.join(map(str, left))}_s17.npz",
            result,
        )
        screen = {
            "pre_screen": diagnostic,
            "post_optimization": final_diag,
            "transfer": transfer,
            "result": compact_result(result),
            "parameter_artifact": parameter,
            "acceptance": passes,
            "tradeoff_vs_static_pre_screen": {
                "schmidt_entropy_delta_nats": entropy_delta,
                "hamiltonian_cross_weight_delta": cross_delta,
                "schmidt_entropy_ratio": float(
                    diagnostic["schmidt_entropy_nats"]
                    / max(1e-30, static_diagnostic["schmidt_entropy_nats"])
                ),
                "hamiltonian_cross_weight_ratio": float(
                    diagnostic["hamiltonian_cross_weight"]
                    / max(1e-30, static_diagnostic["hamiltonian_cross_weight"])
                ),
                "interpretation": (
                    "strict_pareto_dominance"
                    if passes["strict_pareto_dominance_vs_static"]
                    else "pareto_front_tradeoff_not_dominance"
                ),
            },
        }
        screened.append(screen)
        required_gates = (
            "energy_noninferior_within_0p1_mha",
            "within_phi_budget",
            "differs_from_static_partition",
            "candidate_on_screened_pareto_front",
            "improves_at_least_one_declared_pre_screen_metric",
        )
        if all(passes[name] for name in required_gates) and accepted is None:
            accepted = screen
            break

    if accepted is None:
        selected_left = list(incumbent_left)
        decision = "RETAIN_STATIC_NO_ADAPTIVE_CANDIDATE_PASSED_ALL_GATES"
    else:
        selected_left = list(accepted["pre_screen"]["left_block"])
        decision = "ACCEPT_CONSTRAINED_PARETO_FRONT_TRADEOFF"

    selected_config = ansatz_partitioned(selected_left)
    selected_live: dict[int, tuple[Any, AnsatzConfig, dict]] = {}
    matched = {}
    for seed in seeds:
        static_problem, static_config_seed, static_result = live[(static_label, seed)]
        source_circuit = SectorCircuit(static_problem, static_config_seed)
        target_circuit = SectorCircuit(problem, selected_config)
        warm_init, transfer = transfer_parameters(
            source_circuit,
            target_circuit,
            static_result["best_params"],
            seed=seed,
            new_pair_scale=0.0,
        )
        print(f"[topology matched] {label} seed={seed} warm/cold", flush=True)
        warm_result = run_one(
            problem,
            selected_config,
            actual_steps,
            seed,
            init_params=warm_init,
            cache_role=f"adaptive:{label}:matched-warm:seed-{seed}",
        )
        cold_result = run_one(
            problem,
            selected_config,
            actual_steps,
            seed,
            cache_role=f"adaptive:{label}:matched-cold:seed-{seed}",
        )
        warm_parameter = save_parameters(
            params_dir / f"{label}_warm_s{seed}.npz", warm_result
        )
        cold_parameter = save_parameters(
            params_dir / f"{label}_cold_s{seed}.npz", cold_result
        )
        matched[str(seed)] = {
            "static": compact_result(static_result),
            "adaptive_warm": compact_result(warm_result),
            "adaptive_cold": compact_result(cold_result),
            "transfer": transfer,
            "parameter_artifacts": {
                "adaptive_warm": warm_parameter,
                "adaptive_cold": cold_parameter,
            },
            "cost_accounting": {
                "static_direct_energy_evaluations": int(
                    static_result["energy_evaluations"]
                ),
                "static_direct_wall_seconds": float(static_result["wall_seconds"]),
                "adaptive_warm_final_rung_energy_evaluations": int(
                    warm_result["energy_evaluations"]
                ),
                "adaptive_warm_final_rung_wall_seconds": float(
                    warm_result["wall_seconds"]
                ),
                "source_plus_adaptive_warm_energy_evaluations": int(
                    static_result["energy_evaluations"]
                    + warm_result["energy_evaluations"]
                ),
                "source_plus_adaptive_warm_wall_seconds": float(
                    static_result["wall_seconds"] + warm_result["wall_seconds"]
                ),
                "adaptive_cold_energy_evaluations": int(
                    cold_result["energy_evaluations"]
                ),
                "adaptive_cold_wall_seconds": float(cold_result["wall_seconds"]),
                "transfer_exact_without_cap_clipping": bool(
                    transfer["transfer_exact_on_shared_values"]
                ),
                "wall_time_speedup_claim_permitted": RESTART_CACHE is None,
                "wall_time_interpretation": (
                    RESTART_TIMING_CLAIM_BOUNDARY
                    if RESTART_CACHE is not None
                    else "All compared optimizer arms ran in this attempt."
                ),
            },
        }
        selected_live[int(seed)] = (problem, selected_config, warm_result)

    record = {
        "label": label,
        "molecule": molecule,
        "norb": norb,
        "n_qubits": 2 * norb,
        "pilot": compact_result(pilot),
        "static_left_block": list(incumbent_left),
        "static_pilot_diagnostics": static_diagnostic,
        "candidate_count": len(diagnostics),
        "pareto_front": front,
        "screened_candidates": screened,
        "search_cost": {
            "all_pair_pilot_energy_evaluations": int(pilot["energy_evaluations"]),
            "screened_candidate_energy_evaluations": int(
                sum(item["result"]["energy_evaluations"] for item in screened)
            ),
            "complete_search_energy_evaluations": int(
                pilot["energy_evaluations"]
                + sum(item["result"]["energy_evaluations"] for item in screened)
            ),
            "all_pair_pilot_wall_seconds": float(pilot["wall_seconds"]),
            "screened_candidate_wall_seconds": float(
                sum(item["result"]["wall_seconds"] for item in screened)
            ),
            "complete_search_wall_seconds": float(
                pilot["wall_seconds"]
                + sum(item["result"]["wall_seconds"] for item in screened)
            ),
            "note": (
                "one-time topology search cost; matched warm arms additionally "
                "report source-plus-final-rung cost per seed"
            ),
            "wall_time_speedup_claim_permitted": RESTART_CACHE is None,
            "wall_time_interpretation": (
                RESTART_TIMING_CLAIM_BOUNDARY
                if RESTART_CACHE is not None
                else "All compared optimizer arms ran in this attempt."
            ),
        },
        "decision": decision,
        "selected_left_block": selected_left,
        "matched_controls": matched,
        "method_label": "Hikihara-inspired entanglement-informed adaptive bipartition",
        "selection_label": "Pareto-front-screened constrained tradeoff",
        "claim_boundary": "single root split; not automatic TTN optimization",
    }
    return record, selected_live


def resource_audit(
    label: str,
    problem: Any,
    config: AnsatzConfig,
    result: dict[str, Any],
    artifact_directory: Path,
    *,
    relative_root: Path,
) -> dict[str, Any]:
    """Bind resource claims to the exact generic/structured QASM artifacts."""

    circuit = SectorCircuit(problem, config)
    record = bind_table3_circuits(
        label,
        circuit,
        result["best_params"],
        float(result["e_vqe"]),
        artifact_directory,
        relative_root=relative_root,
    )
    reported_phi = float(result["phi"])
    recomputed_phi = float(record["cutting_accounting"]["recomputed_phi"])
    record["cutting_accounting"].update(
        {
            "reported_phi": reported_phi,
            "absolute_difference": abs(reported_phi - recomputed_phi),
        }
    )
    if abs(reported_phi - recomputed_phi) > 1e-12:
        raise RuntimeError(f"cross-angle Phi replay failed for {label}")
    return record


def _canonical_quick_case(
    label: str,
    frozen: Path,
    canonical_manifest_path: Path,
    canonical: dict[str, Any],
) -> tuple[Any, AnsatzConfig, dict[str, Any], dict[str, Any]]:
    """Load one explicitly promoted canonical seed-3047 parameter snapshot."""

    if label not in canonical["cases"]:
        raise KeyError(f"canonical Table-3 manifest omits {label}")
    case = canonical["cases"][label]
    molecule, norb = TABLE3_CASES[label]
    if case.get("molecule") != molecule or int(case.get("norb", -1)) != norb:
        raise ValueError(f"canonical Table-3 molecular identity changed for {label}")
    if int(case.get("seed", -1)) != TABLE3_SEED:
        raise ValueError(f"canonical Table-3 seed changed for {label}")
    ansatz = case.get("ansatz")
    if not isinstance(ansatz, dict):
        raise ValueError(f"canonical Table-3 ansatz is missing for {label}")
    config = AnsatzConfig(**ansatz)
    selected_left = [int(value) for value in case["selected_left_block"]]
    if config.topology != "partitioned" or list(config.partition_override or []) != selected_left:
        raise ValueError(f"canonical Table-3 topology is inconsistent for {label}")

    paths = reference_case_paths(canonical_manifest_path, case)
    for arm in ("generic_qasm", "structured_qasm"):
        path = paths[arm]
        if not path.is_file():
            raise FileNotFoundError(f"canonical {arm} artifact is missing: {path}")
        actual_hash = sha256_file(path)
        expected_hash = str(case[arm]["qasm_sha256"])
        if actual_hash != expected_hash:
            raise ValueError(
                f"canonical {arm} checksum mismatch for {label}: "
                f"{actual_hash} != {expected_hash}"
            )
    params = load_parameter_artifact(
        paths["parameter"], case["parameter_artifact"]["sha256"]
    )
    problem = load_named_problem(frozen, molecule, norb)
    circuit = SectorCircuit(problem, config)
    hdiag = problem.hdiag() if config.init_state == "diag" else None
    energy_fn, _ = make_energy_fn(problem)
    with torch.no_grad():
        state = circuit.forward(params, hdiag)
        energy = float(energy_fn(state))
    expected_energy = float(case["sector_replay_energy_hartree"])
    if abs(energy - expected_energy) > 1e-10:
        raise RuntimeError(
            f"canonical parameter energy replay failed for {label}: "
            f"{energy} != {expected_energy}"
        )
    result = {
        "best_params": params,
        "e_vqe": energy,
        "phi": float(case["cutting_accounting"]["reported_phi"]),
        "seed": TABLE3_SEED,
    }
    return problem, config, result, case


def embedded_state(
    source_circuit: SectorCircuit,
    source_state: torch.Tensor,
    target_circuit: SectorCircuit,
) -> np.ndarray:
    """Embed a nested active-space sector state into the larger sector."""

    out = np.zeros((target_circuit.sector.dimA, target_circuit.sector.dimB))
    source = source_state.detach().cpu().numpy()
    for ia, sa in enumerate(source_circuit.sector.alpha.strs):
        ta = int(np.searchsorted(target_circuit.sector.alpha.strs, int(sa)))
        for ib, sb in enumerate(source_circuit.sector.beta.strs):
            tb = int(np.searchsorted(target_circuit.sector.beta.strs, int(sb)))
            out[ta, tb] = source[ia, ib]
    return out


def first_call_to_threshold(history: list[float], threshold: float) -> int | None:
    best = np.inf
    for index, energy in enumerate(history, start=1):
        best = min(best, float(energy))
        if best <= threshold:
            return index
    return None


def warm_ladder(
    molecule: str,
    rungs: list[tuple[int, int]],
    frozen: Path,
    output: Path,
    seeds: tuple[int, ...],
    quick: bool,
) -> dict[str, Any]:
    """Matched identity-initialized active-space warm-start controls."""

    scale = 0.25 if quick else 1.0
    record = {"molecule": molecule, "rungs": rungs, "seeds": {}}
    params_dir = output / "parameters" / "warm_ladders"
    for seed in seeds:
        previous = None
        cascade = []
        final_warm = None
        for norb, declared_steps in rungs:
            problem = load_named_problem(frozen, molecule, norb)
            config = ansatz_all("adjoint" if problem.meta["sector_dim"] > 2000 else "autograd")
            steps = max(20, int(round(declared_steps * scale)))
            init = None
            transfer = None
            fidelity = None
            if previous is not None:
                old_problem, old_config, old_result, old_circuit, old_state = previous
                target_circuit = SectorCircuit(problem, config)
                init, transfer = transfer_parameters(
                    old_circuit,
                    target_circuit,
                    old_result["best_params"],
                    seed=seed,
                    new_pair_scale=0.0,
                )
                with torch.no_grad():
                    initialized = target_circuit.forward(init, problem.hdiag()).numpy()
                embedded = embedded_state(old_circuit, old_state, target_circuit)
                fidelity = float(
                    abs(np.vdot(embedded.reshape(-1), initialized.reshape(-1))) ** 2
                    / (np.vdot(embedded.reshape(-1), embedded.reshape(-1)).real
                       * np.vdot(initialized.reshape(-1), initialized.reshape(-1)).real)
                )
            print(
                f"[warm ladder] {molecule} no={norb} seed={seed} "
                f"{'warm' if init is not None else 'cold root'}",
                flush=True,
            )
            result = run_one(
                problem,
                config,
                steps,
                seed,
                init_params=init,
                cache_role=(
                    f"warm-ladder:{molecule}:no-{norb}:"
                    f"seed-{seed}:rung"
                ),
            )
            circuit, state = state_for_result(problem, config, result)
            parameter = save_parameters(
                params_dir / f"{molecule}_no{norb}_warm_s{seed}.npz", result
            )
            cascade.append(
                {
                    "norb": norb,
                    "n_qubits": 2 * norb,
                    "warm_started": init is not None,
                    "transfer": transfer,
                    "initial_embedded_state_fidelity": fidelity,
                    "result": compact_result(result),
                    "parameter_artifact": parameter,
                }
            )
            previous = (problem, config, result, circuit, state)
            final_warm = result

        final_norb, final_steps_declared = rungs[-1]
        final_problem = load_named_problem(frozen, molecule, final_norb)
        final_config = ansatz_all(
            "adjoint" if final_problem.meta["sector_dim"] > 2000 else "autograd"
        )
        final_steps = max(20, int(round(final_steps_declared * scale)))
        print(f"[warm ladder] {molecule} final cold seed={seed}", flush=True)
        cold = run_one(
            final_problem,
            final_config,
            final_steps,
            seed,
            cache_role=(
                f"warm-ladder:{molecule}:no-{final_norb}:"
                f"seed-{seed}:direct-cold"
            ),
        )
        cold_parameter = save_parameters(
            params_dir / f"{molecule}_no{final_norb}_direct-cold_s{seed}.npz", cold
        )
        best_final = min(float(final_warm["e_vqe"]), float(cold["e_vqe"]))
        threshold = best_final + 0.0001
        total_calls = sum(int(item["result"]["energy_evaluations"]) for item in cascade)
        total_time = sum(float(item["result"]["wall_seconds"]) for item in cascade)
        record["seeds"][str(seed)] = {
            "cascade": cascade,
            "warm_final": compact_result(final_warm),
            "cold_final": compact_result(cold),
            "cold_final_parameter_artifact": cold_parameter,
            "warm_final_calls_to_within_0p1_mha": first_call_to_threshold(
                final_warm["history"], threshold
            ),
            "cold_final_calls_to_within_0p1_mha": first_call_to_threshold(
                cold["history"], threshold
            ),
            "full_cascade_energy_evaluations": total_calls,
            "full_cascade_wall_seconds": total_time,
            "cascade_to_direct_call_ratio": total_calls
            / max(1, int(cold["energy_evaluations"])),
            "cascade_to_direct_wall_ratio": total_time
            / max(1e-12, float(cold["wall_seconds"])),
            "wall_time_speedup_claim_permitted": RESTART_CACHE is None,
            "wall_time_interpretation": (
                RESTART_TIMING_CLAIM_BOUNDARY
                if RESTART_CACHE is not None
                else "All compared optimizer arms ran in this attempt."
            ),
            "final_energy_difference_warm_minus_cold_mha": 1000.0
            * (float(final_warm["e_vqe"]) - float(cold["e_vqe"])),
        }
    return record


def residual_qsci_controls(
    problem: Any,
    config: AnsatzConfig,
    result: dict[str, Any],
    *,
    qsci_k: int,
    candidate_budget: int = 126,
    ngates: int = 6,
    seed: int = 3047,
) -> dict[str, Any]:
    """Identity, random-feasible and greedy QSCI residual controls."""

    circuit, backbone_state = state_for_result(problem, config, result)
    energy_fn, _ = make_energy_fn(problem)
    solver = QSCISolver(problem, circuit.sector)
    initial_index = int(
        np.argmax(np.abs(circuit.sector.initial_state(problem.hdiag()).numpy()).reshape(-1))
    )
    pool, chosen_pairs = build_residual_pool(
        problem,
        circuit.topo.left_block,
        max_pairs=min(12, max(2, problem.norb)),
    )
    residual_budget = remaining_u(PHI_MAX, result["phi"])
    pool = [
        token
        for token in pool
        if token.kind == "identity" or token.u_cost <= residual_budget + 1e-12
    ]

    def evaluate(tokens):
        with torch.no_grad():
            state = backbone_state
            for token in tokens:
                state = apply_token(state, circuit.sector, token, config.convention)
            raw = float(energy_fn(state))
        selected = solver.top_k(
            state.detach().cpu().numpy(),
            min(qsci_k, solver.dimension),
            mandatory=(initial_index,),
        )
        return {
            "raw_state_energy_hartree": raw,
            "raw_state_error_mha": 1000.0 * (raw - problem.e_casci),
            "qsci": selected,
            "tokens": [token.record() for token in tokens],
            "state": state,
        }

    identity_tokens = [pool[0]] * ngates
    identity = evaluate(identity_tokens)
    rng = np.random.default_rng(seed)
    random_best = None
    random_calls = 0
    for _ in range(candidate_budget):
        sequence = []
        used = 0.0
        for _gate in range(ngates):
            allowed = [token for token in pool if used + token.u_cost <= residual_budget + 1e-12]
            token = allowed[int(rng.integers(len(allowed)))]
            sequence.append(token)
            used += token.u_cost
        trial = evaluate(sequence)
        random_calls += 1
        if random_best is None or trial["qsci"]["energy_hartree"] < random_best["qsci"]["energy_hartree"]:
            random_best = trial

    greedy = []
    greedy_calls = 0
    used = 0.0
    per_step = max(1, candidate_budget // ngates)
    for _gate in range(ngates):
        allowed = [token for token in pool if used + token.u_cost <= residual_budget + 1e-12]
        # Deterministic score order; include identity and the strongest tokens.
        allowed = sorted(
            allowed,
            key=lambda token: (
                token.kind != "identity",
                -float(token.pair_score),
                float(token.u_cost),
                token.label,
            ),
        )[:per_step]
        trials = [(evaluate(greedy + [token]), token) for token in allowed]
        greedy_calls += len(trials)
        best_trial, best_token = min(
            trials,
            key=lambda pair: (
                pair[0]["qsci"]["energy_hartree"],
                pair[1].label,
            ),
        )
        greedy.append(best_token)
        used += best_token.u_cost
    greedy_result = evaluate(greedy)
    return {
        "qsci_k": int(qsci_k),
        "candidate_budget": int(candidate_budget),
        "sequence_length": int(ngates),
        "pool_size": len(pool),
        "chosen_cross_pairs": [list(pair) for pair in chosen_pairs],
        "identity": {k: v for k, v in identity.items() if k != "state"},
        "random_feasible_best": {
            **{k: v for k, v in random_best.items() if k != "state"},
            "candidate_evaluations": random_calls,
        },
        "greedy_feasible": {
            **{k: v for k, v in greedy_result.items() if k != "state"},
            "candidate_evaluations": greedy_calls,
        },
        "promotion_rule": (
            "Transformer GQE-QSCI must beat identity, random and greedy on "
            "independent finite-shot evaluation at the same declared budget"
        ),
    }


def qsci_analysis(
    label: str,
    problem: Any,
    config: AnsatzConfig,
    result: dict[str, Any],
    *,
    quick: bool,
) -> dict[str, Any]:
    circuit, state = state_for_result(problem, config, result)
    solver = QSCISolver(problem, circuit.sector)
    initial = int(
        np.argmax(np.abs(circuit.sector.initial_state(problem.hdiag()).numpy()).reshape(-1))
    )
    dimension = solver.dimension
    if dimension <= 10:
        k_grid = [2, 4, 6, dimension]
        primary_k = min(4, dimension)
    elif dimension <= 250:
        k_grid = [4, 8, 16, 32, 64]
        primary_k = 16
    else:
        k_grid = [4, 8, 16, 32, 64]
        primary_k = 32
    k_grid = sorted({min(dimension, k) for k in k_grid})
    shots_grid = [250, 350, 1000] if quick else [250, 350, 1000, 2000, 10000, 50000]
    trial_count = 5 if quick else 30
    trial_seeds = [90_000 + i for i in range(trial_count)]
    state_np = state.detach().cpu().numpy()
    return {
        "label": label,
        "raw_state_energy_hartree": float(result["e_vqe"]),
        "raw_state_error_mha": float(result["error_mha"]),
        "sector_dimension": dimension,
        "primary_qsci_k": primary_k,
        "top_k": [
            solver.top_k(state_np, k, mandatory=(initial,)) for k in k_grid
        ],
        "finite_shot_protocol": (
            "uncapped unique-sample subspace; determinant count grows with shots; "
            "not a fixed-K promotion endpoint"
        ),
        "finite_shot": [
            solver.sampled_trials(
                state_np, shots, trial_seeds, mandatory=(initial,)
            )
            for shots in shots_grid
        ],
        "finite_shot_fixed_total_k_protocol": (
            "count-ranked subspace capped at primary_qsci_k including the "
            "mandatory Hartree-Fock determinant"
        ),
        "finite_shot_fixed_total_k": [
            solver.sampled_trials_capped(
                state_np,
                shots,
                trial_seeds,
                max_determinants=primary_k,
                mandatory=(initial,),
                min_count=1,
            )
            for shots in shots_grid
        ],
        "full_sector_check": solver.energy(range(dimension)) if dimension <= 500 else None,
    }


def generic_noise_qsci(
    problem: Any,
    config: AnsatzConfig,
    result: dict[str, Any],
    *,
    qsci_k: int,
    quick: bool,
) -> dict[str, Any]:
    """Non-target-specific 6q error sensitivity; not a backend prediction."""

    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

    from givens40.structured_qiskit_export import (
        build_structured_qiskit_circuit,
    )

    circuit, _ = state_for_result(problem, config, result)
    logical = build_structured_qiskit_circuit(circuit, result["best_params"])
    compiled = transpile(
        logical,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=3,
        seed_transpiler=3047,
    )
    measured = compiled.copy()
    measured.measure_all()
    solver = QSCISolver(problem, circuit.sector)
    initial = int(
        np.argmax(np.abs(circuit.sector.initial_state(problem.hdiag()).numpy()).reshape(-1))
    )
    profiles = [
        ("ideal", 0.0, 0.0, 0.0),
        ("optimistic_generic", 1e-4, 1e-3, 0.002),
        ("moderate_generic", 5e-4, 5e-3, 0.01),
        ("stress_generic", 1e-3, 1e-2, 0.02),
    ]
    shots_grid = [250, 1000] if quick else [250, 1000, 2000]
    replicates = 3 if quick else 20
    output_profiles = []
    for profile, p1, p2, readout in profiles:
        if profile == "ideal":
            noise_model = None
        else:
            noise_model = NoiseModel()
            noise_model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["sx", "x"])
            noise_model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
            ro = ReadoutError([[1 - readout, readout], [readout, 1 - readout]])
            for qubit in range(problem.n_qubits):
                noise_model.add_readout_error(ro, [qubit])
        simulator = AerSimulator(noise_model=noise_model)
        shot_records = []
        for shots in shots_grid:
            trials = []
            for rep in range(replicates):
                counts = simulator.run(
                    measured,
                    shots=int(shots),
                    seed_simulator=700_000 + 1000 * shots + rep,
                ).result().get_counts()
                trial = solver.from_counts(
                    counts,
                    mandatory=(initial,),
                    min_count=2,
                    max_determinants=qsci_k,
                )
                trials.append(trial)
            errors = np.asarray([item["error_mha"] for item in trials])
            valid = np.asarray([item["valid_sector_fraction"] for item in trials])
            shot_records.append(
                {
                    "shots": shots,
                    "trials": trials,
                    "median_error_mha": float(np.median(errors)),
                    "maximum_error_mha": float(np.max(errors)),
                    "median_valid_sector_fraction": float(np.median(valid)),
                    "chemical_accuracy_success_fraction": float(
                        np.mean(errors <= CHEMICAL_ACCURACY_MHA)
                    ),
                }
            )
        output_profiles.append(
            {
                "profile": profile,
                "one_qubit_depolarizing_probability": p1,
                "two_qubit_depolarizing_probability": p2,
                "symmetric_readout_flip_probability": readout,
                "shots": shot_records,
            }
        )
    return {
        "scope": "generic Aer sensitivity envelope; not target-native and not a device prediction",
        "interpretation_boundary": (
            "Noise can augment determinant discovery after symmetry filtering and "
            "thereby lower a selected-CI energy even while state fidelity and valid-"
            "sector yield worsen. Such an apparent improvement is a postselection "
            "pathology, not noise robustness or device readiness."
        ),
        "profiles": output_profiles,
        "qsci_k": qsci_k,
        "selection_min_count": 2,
        "determinant_budget_semantics": (
            "qsci_k is a fixed total budget including the mandatory determinant"
        ),
        "compiled_depth": int(compiled.depth()),
        "compiled_cx": int(compiled.count_ops().get("cx", 0)),
    }


def twenty_plus_twenty_projection(
    problem: Any,
    config: AnsatzConfig,
    result: dict[str, Any],
    topology_record: dict[str, Any],
    resource: dict[str, Any],
) -> dict[str, Any]:
    circuit = SectorCircuit(problem, config)
    singles, doubles = circuit.cross_angle_arrays(result["best_params"])
    left = list(circuit.topo.left_block)
    right = list(circuit.topo.right_block)
    return {
        "status": "RESOURCE_PROJECTION_ONLY_NOT_EXECUTED",
        "logical_problem_qubits": 40,
        "spatial_orbitals_per_fragment": [len(left), len(right)],
        "spin_qubits_per_fragment": [2 * len(left), 2 * len(right)],
        "left_spatial_orbitals": left,
        "right_spatial_orbitals": right,
        "cross_pair_count": int(len(circuit.cross_idx)),
        "cross_single_gate_angles": [float(x) for x in singles],
        "cross_pair_double_angles": [float(x) for x in doubles],
        "theoretical_optimal_phi": float(result["phi"]),
        "theoretical_shot_multiplier_phi_squared": float(result["phi"] ** 2),
        "full_logical_structured_resources": resource[
            "structured_exact_pauli_network"
        ],
        "topology_decision": topology_record["decision"],
        "executable_cross_single_qpd_available": True,
        "executable_cross_pair_double_qpd_available": False,
        "sampled_qpd_estimator_available": False,
        "joint_cut_fragment_qsci_reconstruction_available": False,
        "paid_execution_allowed": False,
        "claim_boundary": (
            "A balanced 10|10 spatial-orbital root defines two 20-spin-qubit "
            "fragments, but global reconstruction has not been implemented; "
            "this is not two independent 20q energy calculations."
        ),
    }


def write_csv_tables(output: Path, summary: dict[str, Any]) -> None:
    table_dir = output / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    timing_boundary = summary.get(
        "timing_claim_boundary",
        {
            "wall_time_speedup_claim_permitted": True,
            "interpretation": "All compared optimizer arms ran in this attempt.",
        },
    )
    baseline_path = table_dir / "deterministic_baseline.csv"
    with baseline_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "case",
            "seed",
            "n_qubits",
            "sector_dim",
            "e_vqe",
            "e_casci",
            "error_mha",
            "phi",
            "energy_evaluations",
            "wall_seconds",
            "complete_cascade_energy_evaluations",
            "complete_cascade_wall_seconds",
            "wall_time_speedup_claim_permitted",
            "wall_time_interpretation",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case, record in summary["baseline"].items():
            for seed, run in record["runs"].items():
                writer.writerow(
                    {
                        "case": case,
                        "seed": seed,
                        **{
                            key: run.get(key)
                            for key in fields[2:-2]
                        },
                        "wall_time_speedup_claim_permitted": (
                            timing_boundary[
                                "wall_time_speedup_claim_permitted"
                            ]
                        ),
                        "wall_time_interpretation": timing_boundary[
                            "interpretation"
                        ],
                    }
                )

    adaptive_path = table_dir / "adaptive_topology_matched.csv"
    with adaptive_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "case",
            "decision",
            "selected_left_block",
            "seed",
            "variant",
            "error_mha",
            "phi",
            "energy_evaluations",
            "wall_seconds",
            "source_plus_warm_energy_evaluations",
            "source_plus_warm_wall_seconds",
            "transfer_exact_without_cap_clipping",
            "wall_time_speedup_claim_permitted",
            "wall_time_interpretation",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case, record in summary["adaptive_topology"].items():
            for seed, controls in record["matched_controls"].items():
                for variant in ("static", "adaptive_warm", "adaptive_cold"):
                    run = controls[variant]
                    writer.writerow(
                        {
                            "case": case,
                            "decision": record["decision"],
                            "selected_left_block": json.dumps(record["selected_left_block"]),
                            "seed": seed,
                            "variant": variant,
                            "error_mha": run["error_mha"],
                            "phi": run["phi"],
                            "energy_evaluations": run["energy_evaluations"],
                            "wall_seconds": run["wall_seconds"],
                            "source_plus_warm_energy_evaluations": (
                                controls["cost_accounting"][
                                    "source_plus_adaptive_warm_energy_evaluations"
                                ]
                                if variant == "adaptive_warm"
                                else None
                            ),
                            "source_plus_warm_wall_seconds": (
                                controls["cost_accounting"][
                                    "source_plus_adaptive_warm_wall_seconds"
                                ]
                                if variant == "adaptive_warm"
                                else None
                            ),
                            "transfer_exact_without_cap_clipping": (
                                controls["cost_accounting"][
                                    "transfer_exact_without_cap_clipping"
                                ]
                                if variant == "adaptive_warm"
                                else None
                            ),
                            "wall_time_speedup_claim_permitted": controls[
                                "cost_accounting"
                            ]["wall_time_speedup_claim_permitted"],
                            "wall_time_interpretation": controls[
                                "cost_accounting"
                            ]["wall_time_interpretation"],
                        }
                    )

    resource_path = table_dir / "structured_export_resources.csv"
    with resource_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "case",
            "n_qubits",
            "legacy_cx",
            "structured_cx",
            "legacy_depth",
            "structured_depth",
            "cx_reduction_fraction",
            "state_fidelity",
            "device_native",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case, resource in summary["structured_resources"].items():
            writer.writerow(
                {
                    "case": case,
                    "n_qubits": resource["n_qubits"],
                    "legacy_cx": resource["legacy_generic_unitary"]["cx"],
                    "structured_cx": resource["structured_exact_pauli_network"]["cx"],
                    "legacy_depth": resource["legacy_generic_unitary"]["depth"],
                    "structured_depth": resource["structured_exact_pauli_network"]["depth"],
                    "cx_reduction_fraction": resource["cx_reduction_fraction"],
                    "state_fidelity": resource["state_fidelity"],
                    "device_native": resource["device_native"],
                }
            )

    warm_path = table_dir / "warm_start_controls.csv"
    with warm_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "ladder",
            "seed",
            "final_n_qubits",
            "minimum_embedded_state_fidelity",
            "warm_minus_cold_final_energy_mha",
            "warm_final_calls_to_within_0p1_mha",
            "cold_final_calls_to_within_0p1_mha",
            "full_cascade_energy_evaluations",
            "direct_final_energy_evaluations",
            "cascade_to_direct_call_ratio",
            "cascade_to_direct_wall_ratio",
            "wall_time_speedup_claim_permitted",
            "wall_time_interpretation",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for ladder, record in summary["warm_start"].items():
            final_n_qubits = 2 * int(record["rungs"][-1][0])
            for seed, run in record["seeds"].items():
                fidelities = [
                    item["initial_embedded_state_fidelity"]
                    for item in run["cascade"]
                    if item["initial_embedded_state_fidelity"] is not None
                ]
                writer.writerow(
                    {
                        "ladder": ladder,
                        "seed": seed,
                        "final_n_qubits": final_n_qubits,
                        "minimum_embedded_state_fidelity": min(fidelities),
                        "warm_minus_cold_final_energy_mha": run[
                            "final_energy_difference_warm_minus_cold_mha"
                        ],
                        "warm_final_calls_to_within_0p1_mha": run[
                            "warm_final_calls_to_within_0p1_mha"
                        ],
                        "cold_final_calls_to_within_0p1_mha": run[
                            "cold_final_calls_to_within_0p1_mha"
                        ],
                        "full_cascade_energy_evaluations": run[
                            "full_cascade_energy_evaluations"
                        ],
                        "direct_final_energy_evaluations": run["cold_final"][
                            "energy_evaluations"
                        ],
                        "cascade_to_direct_call_ratio": run[
                            "cascade_to_direct_call_ratio"
                        ],
                        "cascade_to_direct_wall_ratio": run[
                            "cascade_to_direct_wall_ratio"
                        ],
                        "wall_time_speedup_claim_permitted": run[
                            "wall_time_speedup_claim_permitted"
                        ],
                        "wall_time_interpretation": run[
                            "wall_time_interpretation"
                        ],
                    }
                )

    qsci_path = table_dir / "qsci_fixed_budget.csv"
    with qsci_path.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "case",
            "raw_state_error_mha",
            "primary_k",
            "shots",
            "trial_count",
            "fixed_k_median_error_mha",
            "fixed_k_min_error_mha",
            "fixed_k_max_error_mha",
            "chemical_accuracy_success_fraction",
            "median_determinant_count",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case, record in summary["qsci"].items():
            for shot_record in record["finite_shot_fixed_total_k"]:
                writer.writerow(
                    {
                        "case": case,
                        "raw_state_error_mha": record["raw_state_error_mha"],
                        "primary_k": record["primary_qsci_k"],
                        "shots": shot_record["shots"],
                        "trial_count": shot_record["trial_count"],
                        "fixed_k_median_error_mha": shot_record[
                            "error_mha_median"
                        ],
                        "fixed_k_min_error_mha": shot_record["error_mha_min"],
                        "fixed_k_max_error_mha": shot_record["error_mha_max"],
                        "chemical_accuracy_success_fraction": shot_record[
                            "chemical_accuracy_success_fraction"
                        ],
                        "median_determinant_count": shot_record[
                            "determinant_count_median"
                        ],
                    }
                )


def main() -> int:
    global RESTART_CACHE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PRIMARY_SEEDS))
    args = parser.parse_args()
    frozen = args.frozen_inputs.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    output.mkdir(parents=True)
    cache_value = os.environ.get("GQE_RESTART_CACHE")
    if cache_value:
        cache_path = Path(cache_value).expanduser()
        if not cache_path.is_absolute():
            raise ValueError("GQE_RESTART_CACHE must be an absolute path")
        RESTART_CACHE = RestartCache(cache_path, frozen, output)
    seeds = tuple(dict.fromkeys(int(x) for x in args.seeds))
    if 17 not in seeds or 3047 not in seeds:
        raise ValueError("seeds must include topology seed 17 and GQE seed 3047")
    canonical_reference = None
    if args.quick:
        # Fail before any optimization work.  A quick run is a replay of an
        # explicitly promoted full-run reference, never a shortened optimizer
        # result silently relabeled as the canonical Table-3 circuit.
        canonical_reference = load_canonical_manifest(
            CANONICAL_TABLE3_MANIFEST
        )
        missing_cases = sorted(set(TABLE3_CASES) - set(canonical_reference["cases"]))
        if missing_cases:
            raise ValueError(
                "canonical Table-3 manifest is incomplete: "
                + ", ".join(missing_cases)
            )

    started = time.perf_counter()
    manifest = load_manifest(frozen)
    source_root = Path(__file__).resolve().parents[1]
    implementation_files = [
        Path(__file__).resolve(),
        source_root / "givens40" / "adaptive_topology.py",
        source_root / "givens40" / "canonical_resources.py",
        source_root / "givens40" / "chemistry.py",
        source_root / "givens40" / "frozen_problem.py",
        source_root / "givens40" / "reference_audit.py",
        source_root / "givens40" / "qsci.py",
        source_root / "givens40" / "runner.py",
        source_root / "givens40" / "structured_qiskit_export.py",
        source_root / "scripts" / "qbraid_budgeted_gqe.py",
    ]
    metadata = {
        "schema_version": 1,
        "started_utc": utc_now(),
        "quick": bool(args.quick),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: version(name)
            for name in (
                "numpy",
                "scipy",
                "pyscf",
                "torch",
                "qiskit",
                "qiskit-aer",
            )
        },
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "frozen_manifest_sha256": sha256_file(frozen / "MANIFEST.json"),
        "frozen_repeat_fingerprints_identical": manifest[
            "repeat_scientific_fingerprints_identical"
        ],
        "seeds": list(seeds),
        "qpu_contacted": False,
        "provider_imported": False,
        "implementation_sha256": {
            str(path.relative_to(source_root)): sha256_file(path)
            for path in implementation_files
        },
    }
    if RESTART_CACHE is not None:
        metadata["restart_cache"] = RESTART_CACHE.provenance()
    write_json(output / "run_metadata.json", metadata)

    baseline, live = baseline_matrix(frozen, output, seeds, args.quick)
    adaptive = {}
    adaptive_live = {}
    targets = [
        ("BeH2-6", "BeH2-6-partitioned", "BeH2", 3, 150),
        ("BeH2-12", "BeH2-12-partitioned", "BeH2", 6, 200),
        ("H2O-12-heldout", "H2O-12-partitioned-heldout", "H2O", 6, 200),
        ("LiH-40", "LiH-40-partitioned", "LiH", 20, 300),
    ]
    for label, static_label, molecule, norb, steps in targets:
        record, selected = adaptive_target(
            label,
            static_label,
            frozen,
            output,
            live,
            molecule,
            norb,
            steps,
            seeds,
            args.quick,
        )
        adaptive[label] = record
        adaptive_live[label] = selected

    resources = {}
    if args.quick:
        assert canonical_reference is not None
        artifact_root = output / "canonical_table3_replay"
        for label in TABLE3_CASES:
            problem, config, result, canonical_case = _canonical_quick_case(
                label,
                frozen,
                CANONICAL_TABLE3_MANIFEST,
                canonical_reference,
            )
            print(f"[canonical resource replay] {label}", flush=True)
            record = resource_audit(
                label,
                problem,
                config,
                result,
                artifact_root,
                relative_root=artifact_root,
            )
            differences = bound_record_differences(record, canonical_case)
            if differences:
                raise RuntimeError(
                    f"canonical Table-3 replay differs for {label}:\n"
                    + "\n".join(f"- {item}" for item in differences)
                )
            record["reference_mode"] = "PROMOTED_CANONICAL_REPLAY"
            record["artifact_root"] = str(artifact_root.relative_to(output))
            resources[label] = record
    else:
        artifact_root = output / "canonical_table3_candidate"
        candidate_cases = {}
        for label, (molecule, norb) in TABLE3_CASES.items():
            problem, config, result = adaptive_live[label][TABLE3_SEED]
            print(f"[candidate resource generation] {label}", flush=True)
            parameter = save_parameter_artifact(
                artifact_root
                / "parameters"
                / f"{label}_seed-{TABLE3_SEED}.npz",
                result["best_params"],
            )
            parameter["file"] = str(
                Path(parameter["file"]).resolve().relative_to(
                    artifact_root.resolve()
                )
            )
            record = resource_audit(
                label,
                problem,
                config,
                result,
                artifact_root,
                relative_root=artifact_root,
            )
            record["reference_mode"] = "FULL_RUN_CANDIDATE"
            record["artifact_root"] = str(artifact_root.relative_to(output))
            resources[label] = record
            candidate_cases[label] = {
                **record,
                "molecule": molecule,
                "norb": int(norb),
                "ansatz": asdict(config),
                "parameter_artifact": parameter,
            }
        write_json(
            artifact_root / "canonical_table3.json",
            candidate_manifest(candidate_cases),
        )

    warm = {
        "BeH2-6-12-16": warm_ladder(
            "BeH2", [(3, 150), (6, 200), (8, 300)], frozen, output, seeds, args.quick
        ),
        "LiH-6-12-40": warm_ladder(
            "LiH", [(3, 150), (6, 200), (20, 300)], frozen, output, seeds, args.quick
        ),
    }

    qsci = {}
    residual_controls = {}
    for label in ("BeH2-6", "BeH2-12", "LiH-40"):
        problem, config, result = adaptive_live[label][3047]
        print(f"[QSCI] {label}", flush=True)
        qsci[label] = qsci_analysis(label, problem, config, result, quick=args.quick)
        residual_controls[label] = residual_qsci_controls(
            problem,
            config,
            result,
            qsci_k=int(qsci[label]["primary_qsci_k"]),
            candidate_budget=24 if args.quick else 126,
        )

    problem6, config6, result6 = adaptive_live["BeH2-6"][3047]
    print("[generic-noise QSCI] BeH2-6", flush=True)
    noise = generic_noise_qsci(
        problem6,
        config6,
        result6,
        qsci_k=int(qsci["BeH2-6"]["primary_qsci_k"]),
        quick=args.quick,
    )

    problem40, config40, result40 = adaptive_live["LiH-40"][3047]
    projection = twenty_plus_twenty_projection(
        problem40,
        config40,
        result40,
        adaptive["LiH-40"],
        resources["LiH-40"],
    )
    if RESTART_CACHE is not None:
        RESTART_CACHE.export_snapshot(output / "restart_cache_snapshot")

    summary = {
        "schema_version": 1,
        "status": "COMPLETED",
        "scope": "frozen-input CPU/simulator submission evidence; no physical QPU execution",
        "baseline": baseline,
        "adaptive_topology": adaptive,
        "structured_resources": resources,
        "warm_start": warm,
        "qsci": qsci,
        "qsci_residual_controls": residual_controls,
        "generic_noise_qsci_6q": noise,
        "twenty_plus_twenty_projection": projection,
        "hardware_gate": {
            "decision": "HOLD_PENDING_LIVE_TARGET_NATIVE_PREFLIGHT",
            "reason": (
                "No account-accessible target, calibration/noise snapshot or live "
                "price was available locally. Generic noise sensitivity cannot replace "
                "a target-native compile and power/cost gate."
            ),
            "qpu_contacted": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    if RESTART_CACHE is not None:
        cache_provenance = RESTART_CACHE.provenance()
        summary["restart_cache"] = cache_provenance
        summary["timing_claim_boundary"] = {
            "energy_evaluation_counts_comparable": True,
            "wall_time_speedup_claim_permitted": False,
            "interpretation": RESTART_TIMING_CLAIM_BOUNDARY,
        }
        summary["runtime_accounting"] = {
            "current_attempt_wall_seconds": summary["runtime_seconds"],
            "replayed_completed_optimizer_wall_seconds": cache_provenance[
                "replayed_optimizer_wall_seconds"
            ],
            "current_attempt_completed_optimizer_wall_seconds": (
                cache_provenance["committed_optimizer_wall_seconds"]
            ),
            "effective_completed_work_plus_current_attempt_seconds": (
                summary["runtime_seconds"]
                + cache_provenance["replayed_optimizer_wall_seconds"]
            ),
            "interrupted_incomplete_optimizer_work_excluded": True,
            "interpretation": (
                "Per-result wall_seconds is the measured optimizer time from "
                "the attempt that produced the atomically validated cache entry. "
                "runtime_seconds is this final assembly attempt only; the "
                "effective field adds completed optimizer work replayed from "
                "earlier attempts and excludes work lost inside an interrupted "
                "optimizer call. Mixed-origin wall-time sums and ratios are "
                "not same-attempt speedup measurements."
            ),
        }
    write_json(output / "enhanced_release_summary.json", summary)
    write_csv_tables(output, summary)
    metadata["completed_utc"] = utc_now()
    metadata["runtime_seconds"] = summary["runtime_seconds"]
    if RESTART_CACHE is not None:
        metadata["restart_cache"] = RESTART_CACHE.provenance()
    write_json(output / "run_metadata.json", metadata)
    print(f"SUBMISSION SIMULATOR RELEASE COMPLETED: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
