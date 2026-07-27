#!/usr/bin/env python3
"""Install the exact main lock in deterministic, low-disk pip batches."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment_contract import sanitized_environment  # noqa: E402


EXPECTED_LOCKED_DISTRIBUTIONS = 126
DEFAULT_BATCH_SIZE = 4
MAX_BATCH_SIZE = 8
PIN = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9][A-Za-z0-9_.+!-]*)$"
)
CONTROLLED_PIP_ENVIRONMENT = {
    "PIP_CONFIG_FILE",
    "PIP_DISABLE_PIP_VERSION_CHECK",
    "PIP_NO_INPUT",
}


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class LockedRequirement:
    name: str
    version: str
    line_number: int

    @property
    def specification(self) -> str:
        return f"{self.name}=={self.version}"

    @property
    def canonical_name(self) -> str:
        return canonical(self.name)


def parse_lock_text(
    text: str,
    *,
    expected_count: int | None = EXPECTED_LOCKED_DISTRIBUTIONS,
) -> tuple[LockedRequirement, ...]:
    """Parse only exact ``name==version`` entries and reject ambiguity."""

    requirements: list[LockedRequirement] = []
    seen: dict[str, int] = {}
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = PIN.fullmatch(line)
        if match is None:
            raise ValueError(
                f"Unsupported lock entry on line {line_number}; "
                f"expected only name==version: {line}"
            )
        name, version = match.groups()
        key = canonical(name)
        if key in seen:
            raise ValueError(
                f"Duplicate canonical package {name!r} on lines "
                f"{seen[key]} and {line_number}"
            )
        seen[key] = line_number
        requirements.append(LockedRequirement(name, version, line_number))

    if expected_count is not None and len(requirements) != expected_count:
        raise ValueError(
            f"Lock entry count mismatch: {len(requirements)} != {expected_count}"
        )
    return tuple(requirements)


def parse_lock_file(
    path: Path,
    *,
    expected_count: int | None = EXPECTED_LOCKED_DISTRIBUTIONS,
) -> tuple[LockedRequirement, ...]:
    return parse_lock_text(
        path.read_text(encoding="utf-8"),
        expected_count=expected_count,
    )


def plan_batches(
    requirements: Sequence[LockedRequirement],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[tuple[LockedRequirement, ...], ...]:
    """Return deterministic batches, bootstrapping the one locked pip first."""

    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(
            f"batch_size must be between 1 and {MAX_BATCH_SIZE}; found {batch_size}"
        )
    pip_requirements = [
        requirement
        for requirement in requirements
        if requirement.canonical_name == "pip"
    ]
    if len(pip_requirements) != 1:
        raise ValueError(
            f"Exactly one locked pip distribution is required; found {len(pip_requirements)}"
        )
    pip_requirement = pip_requirements[0]
    remaining = [
        requirement
        for requirement in requirements
        if requirement is not pip_requirement
    ]
    batches: list[tuple[LockedRequirement, ...]] = [(pip_requirement,)]
    batches.extend(
        tuple(remaining[offset : offset + batch_size])
        for offset in range(0, len(remaining), batch_size)
    )
    return tuple(batches)


def pip_environment(
    base: Mapping[str, str] | None = None,
    *,
    env_dir: Path | None = None,
) -> dict[str, str]:
    """Remove every ambient ``PIP_*`` key, then add only controlled values."""

    cleaned = sanitized_environment(base, env_dir=env_dir)
    unexpected = {
        key for key in cleaned if key.startswith("PIP_")
    } - CONTROLLED_PIP_ENVIRONMENT
    if unexpected:
        raise RuntimeError(
            "Unexpected pip environment variables survived sanitization: "
            + ", ".join(sorted(unexpected))
        )
    if cleaned.get("PIP_CONFIG_FILE") != os.devnull:
        raise RuntimeError("PIP_CONFIG_FILE must be disabled with os.devnull")
    return cleaned


def install_commands(
    python: Path,
    batches: Sequence[Sequence[LockedRequirement]],
) -> tuple[tuple[str, ...], ...]:
    prefix = (
        str(python),
        "-I",
        "-B",
        "-m",
        "pip",
        "--isolated",
        "install",
        "--no-deps",
        "--no-cache-dir",
    )
    return tuple(
        (*prefix, *(requirement.specification for requirement in batch))
        for batch in batches
    )


def check_command(python: Path) -> tuple[str, ...]:
    return (
        str(python),
        "-I",
        "-B",
        "-m",
        "pip",
        "--isolated",
        "check",
    )


Runner = Callable[..., subprocess.CompletedProcess[object]]


def install_lock(
    *,
    python: Path,
    lock: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    base_environment: Mapping[str, str] | None = None,
    runner: Runner = subprocess.run,
) -> tuple[tuple[LockedRequirement, ...], ...]:
    """Install every exact lock entry once, with one pip process per batch."""

    requirements = parse_lock_file(lock)
    batches = plan_batches(requirements, batch_size=batch_size)
    environment = pip_environment(
        base_environment,
        env_dir=python.resolve(strict=False).parent.parent,
    )
    commands = install_commands(python, batches)
    for index, (batch, command) in enumerate(zip(batches, commands), 1):
        print(
            f"Installing locked batch {index}/{len(batches)} "
            f"({len(batch)} distributions)"
        )
        runner(list(command), check=True, env=environment)
    return batches


def check_environment(
    *,
    python: Path,
    base_environment: Mapping[str, str] | None = None,
    runner: Runner = subprocess.run,
) -> None:
    environment = pip_environment(
        base_environment,
        env_dir=python.resolve(strict=False).parent.parent,
    )
    runner(list(check_command(python)), check=True, env=environment)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    install = subparsers.add_parser("install", help="install the complete exact lock")
    install.add_argument("--python", type=Path, required=True)
    install.add_argument("--lock", type=Path, required=True)
    install.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        choices=range(1, MAX_BATCH_SIZE + 1),
    )

    check = subparsers.add_parser("check", help="run pip check in the clean environment")
    check.add_argument("--python", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if not args.python.is_file():
            raise FileNotFoundError(f"Environment Python is missing: {args.python}")
        if args.command == "install":
            install_lock(
                python=args.python,
                lock=args.lock,
                batch_size=args.batch_size,
            )
        else:
            check_environment(python=args.python)
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
