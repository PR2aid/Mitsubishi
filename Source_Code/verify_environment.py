#!/usr/bin/env python3
"""Verify the complete qBraid dependency lock and report its fingerprint."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import argparse
import json
import os
from pathlib import Path
import platform
import re
import sys


ROOT = Path(__file__).resolve().parent
LOCK = ROOT / "requirements.lock"
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)$")
EXPECTED_LOCKED_DISTRIBUTIONS = 126


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def runtime_smoke() -> dict[str, object]:
    """Import the native stack in its required order and exercise qpp-cpu."""

    if os.environ.get("UCX_TLS") != "self":
        raise RuntimeError("UCX_TLS=self is required for the single-process CPU runtime")

    # Ordering is deliberate: Lightning initializes the compatible runtime
    # layer before CUDA-Q Solvers loads its MPI/UCX-linked native libraries.
    import lightning  # noqa: F401
    import cudaq
    import cudaq_solvers  # noqa: F401

    cudaq.set_target("qpp-cpu")

    @cudaq.kernel
    def one_bit():
        qubit = cudaq.qubit()
        x(qubit)
        mz(qubit)

    counts = cudaq.sample(one_bit, shots_count=8)
    sampled_shots = sum(int(value) for _, value in counts.items())
    target = str(cudaq.get_target().name)
    if target != "qpp-cpu":
        raise RuntimeError(f"CUDA-Q target mismatch: {target} != qpp-cpu")
    if sampled_shots != 8:
        raise RuntimeError(f"qpp-cpu smoke returned {sampled_shots} shots, expected 8")
    return {
        "lightning_before_cudaq_solvers": True,
        "target": target,
        "sampled_shots": sampled_shots,
        "ucx_tls": os.environ["UCX_TLS"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="also import the native stack and execute an 8-shot qpp-cpu circuit",
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(
            f"This release lock is validated for Python 3.12; found "
            f"{platform.python_version()}. Select a qBraid Python 3.12 "
            "kernel or rerun setup with PYTHON_BIN=python3.12."
        )
    if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
        raise SystemExit(
            "This release lock is validated for Linux x86-64; found "
            f"{platform.system()} {platform.machine()}. Launch a Linux x86-64 "
            "qBraid instance and rerun setup."
        )
    cpu_count = os.cpu_count() or 0
    if cpu_count < 4:
        raise SystemExit(
            "This release requires at least 4 visible vCPU to preserve the "
            f"validated thread contract; found {cpu_count}"
        )

    expected: dict[str, tuple[str, str]] = {}
    for line_number, raw in enumerate(LOCK.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN.fullmatch(line)
        if not match:
            raise SystemExit(f"Unsupported non-exact lock entry on line {line_number}: {line}")
        name, version = match.groups()
        key = canonical(name)
        if key in expected:
            raise SystemExit(f"Duplicate canonical lock name on line {line_number}: {name}")
        expected[key] = (name, version)

    if len(expected) != EXPECTED_LOCKED_DISTRIBUTIONS:
        raise SystemExit(
            "Locked distribution count mismatch: "
            f"{len(expected)} != {EXPECTED_LOCKED_DISTRIBUTIONS}"
        )

    mismatches: list[str] = []
    installed: dict[str, tuple[str, str]] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            mismatches.append("installed distribution has no Name metadata")
            continue
        key = canonical(name)
        version = distribution.version
        if key in installed:
            mismatches.append(f"{name}: duplicate installed distribution metadata")
            continue
        installed[key] = (name, version)

    for key, (name, required) in sorted(expected.items()):
        if key not in installed:
            mismatches.append(f"{name}: missing (expected {required})")
            continue
        actual = installed[key][1]
        if actual != required:
            mismatches.append(f"{name}: {actual} != {required}")

    for key in sorted(set(installed) - set(expected)):
        name, version = installed[key]
        mismatches.append(f"{name}: unexpected installed distribution {version}")

    if mismatches:
        raise SystemExit("Locked environment mismatch:\n  " + "\n  ".join(mismatches))

    report = {
        "status": "PASS",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "visible_vcpu": cpu_count,
        "locked_distributions": len(expected),
        "requirements_lock_sha256": hashlib.sha256(LOCK.read_bytes()).hexdigest(),
    }
    if args.smoke:
        report["runtime_smoke"] = runtime_smoke()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
