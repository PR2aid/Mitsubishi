#!/usr/bin/env python3
"""Guarded terminal entry point for the optional IonQ Forte-1 experiment.

With no action flag, this script performs only local NumPy/Qiskit circuit
reconstruction. ``--device-dry-run`` additionally asks qBraid to transform and
validate the circuit but does not submit it. A paid task requires both
``--submit`` and the exact confirmation phrase printed by the preflight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import forte_hardware as fh


ROOT = Path(__file__).resolve().parent


def local_preflight() -> tuple[dict, dict]:
    """Rebuild the frozen circuit and require independent state fidelity."""
    reference = fh.load_reference(ROOT)
    preflight = fh.qiskit_preflight(reference)
    print("NumPy/Qiskit fidelity:", f"{preflight['fidelity_numpy_vs_qiskit']:.16f}")
    print("Diagnostic transpilation fidelity:", f"{preflight['diagnostic_fidelity']:.16f}")
    print("Diagnostic depth:", preflight["diagnostic_depth"])
    print("Diagnostic operations:", preflight["diagnostic_operations"])
    print("Circuit SHA-256:", preflight["hardware_circuit_qasm2_sha256"])
    print("Ideal <W>, W=X0 tensor X1 tensor Y2 tensor Y3:", f"{preflight['ideal_xxyy_expectation']:.12f}")
    print("Required confirmation phrase:")
    print(fh.expected_confirmation())
    return reference, preflight


def resolve_and_dry_run(preflight: dict):
    """Resolve exactly Forte-1 and execute qBraid's no-submission transform."""
    from qbraid.runtime import QbraidProvider

    provider = QbraidProvider()
    device = provider.get_device(fh.TARGET_DEVICE_ID)
    snapshot = fh.assert_forte_ready(device)
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    device.set_options(prepare=False)
    try:
        transformed = device.apply_runtime_profile(preflight["hardware_circuit"])
    finally:
        device.set_options(prepare=True)
    print("Forte transpile/transform/strict-validation dry-run: PASS")
    print("Transformed program type:", type(transformed).__name__)
    return provider, device


def submit(reference: dict, preflight: dict, confirmation: str) -> None:
    """Submit one locked 350-shot task after all explicit safeguards pass."""
    fh.assert_submission_guard(True, confirmation)

    provider, device = resolve_and_dry_run(preflight)
    fh.create_attempt_lock(ROOT, preflight)
    job = device.run(preflight["hardware_circuit"], shots=fh.SHOTS)
    record = fh.write_submission_record(ROOT, job, device, preflight, reference)
    print("FORTE JOB SUBMITTED")
    print("Job ID:", job.id)
    print("Submission record:", record)


def retrieve(preflight: dict) -> None:
    """Retrieve the task named in the locally generated submission record."""
    from qbraid.runtime import QbraidJob, QbraidProvider

    submission_path = ROOT / fh.RESULTS_DIR / fh.SUBMISSION_RECORD
    if not submission_path.is_file():
        raise FileNotFoundError("No local submission record; no job can be retrieved")
    submission_record = json.loads(submission_path.read_text(encoding="utf-8"))
    provider = QbraidProvider()
    device = provider.get_device(fh.TARGET_DEVICE_ID)
    job = QbraidJob(submission_record["job_id"], device=device, client=provider.client)
    status = job.status()
    status_name = str(getattr(status, "name", status)).upper()
    print("Job ID:", job.id)
    print("Status:", status_name)
    if "COMPLETED" not in status_name:
        print("Result not requested because the job is not completed.")
        return
    result = job.result()
    result_path = fh.write_result_record(
        ROOT, job, result, result.data.get_counts(), preflight
    )
    print("RESULT SAVED:", result_path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run local preflight, optional device dry-run, submit, or retrieval."""

    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--device-dry-run", action="store_true")
    actions.add_argument("--submit", action="store_true")
    actions.add_argument("--retrieve", action="store_true")
    parser.add_argument(
        "--confirmation",
        default="",
        help="exact phrase printed by the preflight; required with --submit",
    )
    args = parser.parse_args(argv)

    reference, preflight = local_preflight()
    if args.device_dry_run:
        resolve_and_dry_run(preflight)
    elif args.submit:
        submit(reference, preflight, args.confirmation)
    elif args.retrieve:
        retrieve(preflight)
    else:
        print("LOCAL PREFLIGHT PASSED; no qBraid provider contacted and no QPU task submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
