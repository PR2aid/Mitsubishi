#!/usr/bin/env python3
"""Regenerate and validate the results reported in the Phase 3 write-up.

The default quick mode is a short environment and scientific-integrity check.
Use ``--full`` for the complete frozen-input simulator matrix used by judges.
Neither mode imports a provider or submits a quantum-computer job.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Sequence
import uuid


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
FROZEN = ROOT / "frozen_inputs"
RESULTS = ROOT / "results" / "judge_reproduction"
sys.dont_write_bytecode = True
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment_contract import resolve_environment, sanitized_environment


_NETWORK_AUDIT_GUARD = (
    "sys.addaudithook(lambda event,args: "
    "(_ for _ in ()).throw(RuntimeError("
    "'outbound network disabled by frozen CPU workflow: '+event)) "
    "if event in {'socket.connect','socket.getaddrinfo',"
    "'socket.gethostbyname','socket.gethostbyaddr','socket.sendto'} "
    "else None); "
)
_SCRIPT_BOOTSTRAP = (
    "import runpy,sys; "
    + _NETWORK_AUDIT_GUARD
    + "root=sys.argv.pop(1); source=sys.argv.pop(1); script=sys.argv.pop(1); "
    "sys.path[:0]=[source,root]; sys.argv[0]=script; "
    "runpy.run_path(script,run_name='__main__')"
)
_MODULE_BOOTSTRAP = (
    "import runpy,sys; "
    + _NETWORK_AUDIT_GUARD
    + "root=sys.argv.pop(1); source=sys.argv.pop(1); module=sys.argv.pop(1); "
    "sys.path[:0]=[source,root]; sys.argv[0]=module; "
    "runpy.run_module(module,run_name='__main__')"
)
INVOCATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{7,127}$")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def isolated_script_command(
    python: str,
    script: Path,
    *arguments: str,
) -> list[str]:
    """Run a trusted release script after isolated interpreter startup."""

    return [
        python,
        "-I",
        "-B",
        "-c",
        _SCRIPT_BOOTSTRAP,
        str(ROOT),
        str(SOURCE),
        str(script),
        *arguments,
    ]


def isolated_module_command(
    python: str,
    module: str,
    *arguments: str,
) -> list[str]:
    """Run a trusted release module after isolated interpreter startup."""

    return [
        python,
        "-I",
        "-B",
        "-c",
        _MODULE_BOOTSTRAP,
        str(ROOT),
        str(SOURCE),
        module,
        *arguments,
    ]


def run_logged(
    label: str,
    command: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_dir: Path,
) -> Path:
    """Run one fixed command, stream its output, and require success."""

    log_path = log_dir / f"{label}.log"
    print(f"\n--- {label} ---\n$ {' '.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as archive:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            archive.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{label} failed with exit status {return_code}; see {log_path}")
    return log_path


def require_48_tests(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"(?m)^\s*(\d+) passed,\s*(\d+) failed\s*$", text)
    if match is None or tuple(map(int, match.groups())) != (48, 0):
        raise RuntimeError("The core scientific suite did not report exactly 48 passed, 0 failed")


def objective_command(
    system: str,
    objective: str,
    left: list[int],
    k: int,
    shots: int,
    output: Path,
) -> list[str]:
    command = [
        "bash",
        "source/scripts/run_budgeted_gqe.sh",
        "--system",
        system,
        "--full",
        "--partition-left",
        *map(str, left),
        "--objective",
        objective,
        "--qsci-k",
        str(k),
        "--heldout-shots",
        str(shots),
        "--heldout-trials",
        "30",
        "--output",
        str(output),
        "--console-log",
        str(output.with_suffix(".log")),
    ]
    if system in {"beh2-6", "beh2-12"}:
        command.append("--cudaq-state-crosscheck")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="short scientific-integrity check (default)")
    mode.add_argument("--full", action="store_true", help="regenerate every write-up simulator result")
    parser.add_argument(
        "--invocation-id",
        help="certificate invocation identifier; generated for direct runs when omitted",
    )
    args = parser.parse_args(argv)
    selected = "full" if args.full else "quick"
    invocation_id = args.invocation_id or f"direct-{stamp()}-{uuid.uuid4().hex}"
    if INVOCATION.fullmatch(invocation_id) is None:
        raise SystemExit("Invalid --invocation-id; use 8-128 safe identifier characters")
    started_utc = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()

    for required in (SOURCE, FROZEN / "MANIFEST.json", ROOT / "expected_metrics.json"):
        if not required.exists():
            raise SystemExit(f"Missing required submission input: {required}")

    try:
        env_dir = resolve_environment(ROOT)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    env_python = env_dir / "bin" / "python"
    if not env_python.is_file():
        raise SystemExit("Pinned environment missing. Run `bash setup.sh` first.")
    if Path(sys.executable).resolve() != env_python.resolve():
        raise SystemExit(
            f"Activate the pinned environment first: `source {env_dir}/bin/activate`."
        )

    run_stem = f"{selected}_{stamp()}_{invocation_id[:12]}"
    run_dir = RESULTS / run_stem
    suffix = 1
    while run_dir.exists():
        run_dir = RESULTS / f"{run_stem}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True)

    env = sanitized_environment(
        os.environ,
        env_dir=env_dir,
        mpl_config_dir=run_dir / ".matplotlib",
    )
    env.update(
        {
            "QBRAID_GQE_ENV": str(env_dir),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "GQE_FROZEN_INPUT_DIR": str(FROZEN.resolve()),
            "GQE_FROZEN_INPUT_MODE": "required",
            "UCX_TLS": "self",
        }
    )
    python = str(env_python)

    run_logged(
        "locked_environment",
        isolated_script_command(
            python,
            ROOT / "verify_environment.py",
            "--smoke",
            "--output",
            str(run_dir / "environment.json"),
        ),
        cwd=ROOT,
        env=env,
        log_dir=run_dir,
    )
    run_logged(
        "dependency_consistency",
        [python, "-I", "-B", "-m", "pip", "--isolated", "check"],
        cwd=ROOT,
        env=env,
        log_dir=run_dir,
    )
    run_logged(
        "hardening_regressions",
        isolated_module_command(
            python,
            "unittest",
            "tests_hardening.test_environment_contract",
            "tests_hardening.test_low_disk_installer",
            "tests_hardening.test_certify_release",
            "tests_hardening.test_runtime_contract",
            "tests_hardening.test_validation_mode_contract",
            "-v",
        ),
        cwd=ROOT,
        env=env,
        log_dir=run_dir,
    )
    run_logged(
        "scientific_binding_regressions",
        isolated_module_command(
            python,
            "unittest",
            "tests.test_reference_audit",
            "tests.test_canonical_resources",
            "tests.test_qasm2_sx_portability",
            "tests.test_restart_cache",
            "tests.test_validation_semantics",
            "-v",
        ),
        cwd=SOURCE,
        env=env,
        log_dir=run_dir,
    )
    run_logged(
        "independent_frozen_reference_audit",
        isolated_script_command(
            python,
            SOURCE / "scripts" / "audit_frozen_references.py",
            "--frozen-inputs",
            str(FROZEN),
            "--output",
            str(run_dir / "frozen_reference_audit.json"),
        ),
        cwd=ROOT,
        env=env,
        log_dir=run_dir,
    )

    core_log = run_logged(
        "core_scientific_tests",
        isolated_module_command(python, "tests.test_all"),
        cwd=SOURCE,
        env=env,
        log_dir=run_dir,
    )
    require_48_tests(core_log)
    run_logged(
        "advanced_method_tests",
        isolated_module_command(python, "unittest", "tests.test_advanced", "-v"),
        cwd=SOURCE,
        env=env,
        log_dir=run_dir,
    )
    run_logged(
        "structured_export_tests",
        isolated_module_command(
            python, "unittest", "tests.test_structured_export", "-v"
        ),
        cwd=SOURCE,
        env=env,
        log_dir=run_dir,
    )
    run_logged(
        "cutting_budget_tests",
        isolated_module_command(python, "tests.test_budgeted_gqe"),
        cwd=SOURCE,
        env=env,
        log_dir=run_dir,
    )
    run_logged(
        "exact_qpd_6q",
        isolated_module_command(
            python,
            "tests.test_qpd",
            "--output",
            str(run_dir / "qpd_result.json"),
        ),
        cwd=SOURCE,
        env=env,
        log_dir=run_dir,
    )

    advanced_command = isolated_script_command(
        python,
        SOURCE / "scripts" / "run_enhanced_release.py",
        "--frozen-inputs",
        str(FROZEN),
        "--output",
        str(run_dir / "advanced_method"),
    )
    if selected == "quick":
        advanced_command.append("--quick")
    run_logged("advanced_method_matrix", advanced_command, cwd=ROOT, env=env, log_dir=run_dir)

    h2_mode = "--full" if selected == "full" else "--smoke"
    h2_output = run_dir / ("h2_gqe_full.json" if selected == "full" else "h2_gqe_smoke.json")
    run_logged(
        f"h2_transformer_gqe_{selected}",
        ["bash", "source/scripts/run_qbraid_gqe.sh", h2_mode, "--output", str(h2_output)],
        cwd=ROOT,
        env=env,
        log_dir=run_dir,
    )

    if selected == "full":
        objectives = run_dir / "objectives"
        objectives.mkdir()
        systems = {
            "beh2-6": {"left": [1], "k": 4, "shots": 2000},
            "beh2-12": {"left": [0, 3, 4], "k": 16, "shots": 10000},
            "lih-40": {"left": [0, 1, 4, 5, 8, 9, 10, 13, 17, 18], "k": 32, "shots": 10000},
        }
        for system, config in systems.items():
            stem = system.replace("-", "_")
            for objective, suffix_name in (("exact-energy", "exact_energy"), ("qsci-topk", "qsci_topk")):
                output = objectives / f"{stem}_{suffix_name}.json"
                run_logged(
                    f"{stem}_{suffix_name}",
                    objective_command(system, objective, config["left"], config["k"], config["shots"], output),
                    cwd=ROOT,
                    env=env,
                    log_dir=run_dir,
                )
            run_logged(
                f"{stem}_objective_comparison",
                [
                    *isolated_script_command(
                        python,
                        SOURCE / "scripts" / "compare_gqe_objectives.py",
                    ),
                    "--exact",
                    str(objectives / f"{stem}_exact_energy.json"),
                    "--qsci",
                    str(objectives / f"{stem}_qsci_topk.json"),
                    "--output",
                    str(objectives / f"{stem}_objective_comparison.json"),
                ],
                cwd=ROOT,
                env=env,
                log_dir=run_dir,
            )

        run_logged(
            "finite_shot_beh2_6q",
            [
                *isolated_script_command(
                    python,
                    SOURCE / "scripts" / "run_finite_shot_release.py",
                ),
                "--shots",
                "20000",
                "--seed",
                "7",
                "--output",
                str(run_dir / "finite_shot_beh2_6q.json"),
            ],
            cwd=ROOT,
            env=env,
            log_dir=run_dir,
        )

    summary = run_dir / "reproduction_summary.json"
    run_logged(
        "validate_submission_results",
        isolated_script_command(
            python,
            ROOT / "validate_submission_results.py",
            "--mode",
            selected,
            "--run-dir",
            str(run_dir),
            "--output",
            str(summary),
        ),
        cwd=ROOT,
        env=env,
        log_dir=run_dir,
    )
    completed_utc = datetime.now(timezone.utc)
    lock_sha256 = hashlib.sha256((ROOT / "requirements.lock").read_bytes()).hexdigest()
    execution_record = {
        "schema_version": 1,
        "invocation_id": invocation_id,
        "status": "PASS",
        "mode": selected,
        "started_utc": started_utc.isoformat(),
        "completed_utc": completed_utc.isoformat(),
        "wall_seconds": time.monotonic() - started_monotonic,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "requirements_lock_sha256": lock_sha256,
        "qpu_contacted": False,
        "provider_imported": False,
        "validation_summary": str(summary),
    }
    execution_record_path = run_dir / "execution_record.json"
    execution_record_path.write_text(
        json.dumps(execution_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    latest = {
        "invocation_id": invocation_id,
        "mode": selected,
        "run_directory": str(run_dir),
        "summary": str(summary),
        "execution_record": str(execution_record_path),
        "status": "PASS",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "latest_run.json").write_text(json.dumps(latest, indent=2) + "\n", encoding="utf-8")
    print("\nALL REQUESTED PHASE 3 RESULTS REPRODUCED WITHIN DECLARED TOLERANCES")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
