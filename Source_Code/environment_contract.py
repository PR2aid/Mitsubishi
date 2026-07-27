#!/usr/bin/env python3
"""Shared, standard-library-only environment contract for qBraid execution."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
from typing import Mapping


DEFAULT_ENV_BASENAME = ".qbraid_gqe_env"
GIB = 1024**3
FRESH_MIN_FREE_BYTES = int(12.5 * GIB)
REUSE_MIN_FREE_BYTES = int(2.0 * GIB)
MINIMUM_VCPU = 4

_PYTHON_ENV_KEYS = {
    "PYTHONBREAKPOINT",
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONPATH",
    "PYTHONPLATLIBDIR",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONWARNINGS",
}


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def is_within(path: Path, parent: Path) -> bool:
    """Return whether *path* is *parent* or one of its descendants."""

    candidate = _resolved(path)
    container = _resolved(parent)
    return candidate == container or container in candidate.parents


def resolve_environment(
    source_root: Path,
    *,
    override: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve and validate the dedicated environment outside ``Source_Code``."""

    root = _resolved(source_root)
    raw = override
    if raw is None:
        raw = os.environ.get("QBRAID_GQE_ENV")
    candidate = _resolved(Path(raw) if raw else root.parent / DEFAULT_ENV_BASENAME)
    if is_within(candidate, root):
        raise ValueError(
            "QBRAID_GQE_ENV must be outside the immutable Source_Code folder; "
            f"received {candidate}"
        )
    return candidate


def nearest_existing_parent(path: Path) -> Path:
    """Return the nearest existing directory at or above *path*."""

    candidate = _resolved(path)
    while not candidate.exists():
        if candidate.parent == candidate:
            raise FileNotFoundError(f"No existing parent for {path}")
        candidate = candidate.parent
    if not candidate.is_dir():
        candidate = candidate.parent
    return candidate


def required_free_bytes(mode: str) -> int:
    if mode == "fresh":
        return FRESH_MIN_FREE_BYTES
    if mode == "reuse":
        return REUSE_MIN_FREE_BYTES
    raise ValueError(f"Unknown disk-preflight mode: {mode}")


def check_disk_space(
    path: Path,
    *,
    mode: str,
    free_bytes: int | None = None,
) -> dict[str, int | float | str]:
    """Fail unless the target filesystem meets the declared mode threshold."""

    target = nearest_existing_parent(path)
    available = shutil.disk_usage(target).free if free_bytes is None else int(free_bytes)
    required = required_free_bytes(mode)
    if available < required:
        raise OSError(
            f"Insufficient free disk for {mode}: "
            f"{available / GIB:.2f} GiB available, "
            f"{required / GIB:.2f} GiB required on {target}"
        )
    return {
        "mode": mode,
        "filesystem_probe": str(target),
        "available_bytes": available,
        "available_gib": available / GIB,
        "required_bytes": required,
        "required_gib": required / GIB,
    }


def sanitized_environment(
    base: Mapping[str, str] | None = None,
    *,
    env_dir: Path | None = None,
    mpl_config_dir: Path | None = None,
) -> dict[str, str]:
    """Return a deterministic child environment without ambient Python/pip hooks."""

    cleaned = dict(os.environ if base is None else base)
    for key in _PYTHON_ENV_KEYS:
        cleaned.pop(key, None)
    for key in tuple(cleaned):
        if key.startswith("PIP_"):
            cleaned.pop(key, None)

    cleaned.update(
        {
            "PIP_CONFIG_FILE": os.devnull,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONSAFEPATH": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
            "UCX_TLS": "self",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    if env_dir is not None:
        cleaned["QBRAID_GQE_ENV"] = str(_resolved(env_dir))
    if mpl_config_dir is not None:
        cleaned["MPLCONFIGDIR"] = str(_resolved(mpl_config_dir))
    return cleaned


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve = subparsers.add_parser("resolve", help="print the validated environment path")
    resolve.add_argument("--root", type=Path, required=True)
    resolve.add_argument("--override")

    disk = subparsers.add_parser("check-disk", help="run a fresh/reuse disk preflight")
    disk.add_argument("--root", type=Path, required=True)
    disk.add_argument("--override")
    disk.add_argument("--mode", choices=("fresh", "reuse"), required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        environment = resolve_environment(args.root, override=args.override)
        if args.command == "resolve":
            print(environment)
            return 0
        report = check_disk_space(environment, mode=args.mode)
        print(
            f"Disk preflight ({args.mode}): "
            f"{report['available_gib']:.2f} GiB available; "
            f"{report['required_gib']:.2f} GiB required"
        )
        return 0
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
