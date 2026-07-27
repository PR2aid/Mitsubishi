"""Guarded IonQ Forte-1 runner for the frozen 4-qubit H2 GQE state.

The module keeps circuit construction, independent ideal simulation, budget
checks, result reduction, and record serialization outside notebook state so
the hardware experiment can be reviewed and reproduced before submission.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
from typing import Any, Mapping

import numpy as np


N_QUBITS = 4
N_ELECTRONS = 2
SHOTS = 350
TARGET_DEVICE_ID = "aws:ionq:qpu:forte-1"
MEASUREMENT_WORD = "XXYY"  # q0-first coherence witness

PER_TASK_CREDITS = 30
PER_SHOT_CREDITS = 8
EXPECTED_QPU_CREDITS = PER_TASK_CREDITS + PER_SHOT_CREDITS * SHOTS
MAX_TOTAL_CREDITS = 3000

REFERENCE_FILE = "H2_GQE_REFERENCE.json"
RESULTS_DIR = "results"
SUBMISSION_RECORD = "forte_submission.json"
RESULT_RECORD = "forte_result.json"


def utc_now() -> str:
    """Return the current UTC timestamp in a portable ISO-8601 form."""

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    """Compute a streaming SHA-256 digest for one local record."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    """Return an installed package version or ``not-installed``."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def load_reference(root: Path) -> dict[str, Any]:
    """Load and structurally validate the frozen H2 GQE definition."""

    path = root / REFERENCE_FILE
    reference = json.loads(path.read_text(encoding="utf-8"))
    assert reference["system"]["n_qubits"] == N_QUBITS
    assert reference["system"]["n_electrons"] == N_ELECTRONS
    assert len(reference["selected_operators"]) == 10
    assert reference["hardware_plan"]["shots"] == SHOTS
    assert reference["hardware_plan"]["target_device_id"] == TARGET_DEVICE_ID
    assert EXPECTED_QPU_CREDITS <= MAX_TOTAL_CREDITS
    return reference


_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}


def _pauli_matrix_q0_first(word: str) -> np.ndarray:
    if len(word) != N_QUBITS or any(char not in _PAULI for char in word):
        raise ValueError(f"Invalid {N_QUBITS}-qubit Pauli word: {word!r}")
    matrix = np.array([[1.0 + 0.0j]])
    # NumPy basis indices and Qiskit counts are q_(n-1)...q_0.
    for char in reversed(word):
        matrix = np.kron(matrix, _PAULI[char])
    return matrix


def numpy_reference_state(reference: Mapping[str, Any]) -> np.ndarray:
    """Independent simulation of CUDA-Q exp_pauli(theta, P)=exp(-i theta P)."""
    state = np.zeros(1 << N_QUBITS, dtype=complex)
    state[(1 << N_ELECTRONS) - 1] = 1.0  # X on q0 and q1
    identity = np.eye(1 << N_QUBITS, dtype=complex)
    for item in reference["selected_operators"]:
        theta = float(item["coefficient"])
        pauli = _pauli_matrix_q0_first(item["pauli_word"])
        state = (math.cos(theta) * identity - 1j * math.sin(theta) * pauli) @ state
    return state


def ideal_probabilities(state: np.ndarray, threshold: float = 1e-12) -> dict[str, float]:
    """Return non-negligible computational-basis probabilities."""

    probs = np.abs(state) ** 2
    return {
        format(index, f"0{N_QUBITS}b"): float(value)
        for index, value in enumerate(probs)
        if value > threshold
    }


def pauli_expectation(state: np.ndarray, word: str) -> float:
    """Evaluate one q0-first Pauli expectation on a statevector."""

    return float(np.vdot(state, _pauli_matrix_q0_first(word) @ state).real)


def _apply_pauli_rotation(qc: Any, word: str, theta: float) -> None:
    """Append exp(-i theta P) using standard basis changes and a parity ladder."""
    support = [qubit for qubit, char in enumerate(word) if char != "I"]
    if not support:
        return

    for qubit in support:
        char = word[qubit]
        if char == "X":
            qc.h(qubit)
        elif char == "Y":
            qc.sdg(qubit)
            qc.h(qubit)

    target = support[-1]
    for qubit in support[:-1]:
        qc.cx(qubit, target)
    qc.rz(2.0 * theta, target)
    for qubit in reversed(support[:-1]):
        qc.cx(qubit, target)

    for qubit in reversed(support):
        char = word[qubit]
        if char == "X":
            qc.h(qubit)
        elif char == "Y":
            qc.h(qubit)
            qc.s(qubit)


def build_state_preparation(reference: Mapping[str, Any]) -> Any:
    """Build the frozen ten-rotation H2 state-preparation circuit."""

    from qiskit import QuantumCircuit

    qc = QuantumCircuit(N_QUBITS, name="h2_gqe_state")
    for qubit in range(N_ELECTRONS):
        qc.x(qubit)
    for item in reference["selected_operators"]:
        _apply_pauli_rotation(qc, item["pauli_word"], float(item["coefficient"]))
    return qc


def build_hardware_circuit(reference: Mapping[str, Any]) -> Any:
    """Prepare the GQE state and measure the XXYY coherence witness once per shot."""
    qc = build_state_preparation(reference)
    qc.name = "gic2026_h2_gqe_xxyy"
    for qubit, char in enumerate(MEASUREMENT_WORD):
        if char == "X":
            qc.h(qubit)
        elif char == "Y":
            qc.sdg(qubit)
            qc.h(qubit)
        elif char not in {"I", "Z"}:
            raise ValueError(f"Unsupported measurement Pauli {char!r}")
    qc.measure_all()
    return qc


def qiskit_preflight(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Cross-check state fidelity, transpilation, hash, and ideal witness."""

    from qiskit import qasm2, transpile
    from qiskit.quantum_info import Statevector, state_fidelity

    state_circuit = build_state_preparation(reference)
    hardware_circuit = build_hardware_circuit(reference)
    numpy_state = numpy_reference_state(reference)
    qiskit_state = np.asarray(Statevector.from_instruction(state_circuit).data)
    fidelity = float(state_fidelity(numpy_state, qiskit_state, validate=True))
    if fidelity < 1.0 - 1e-12:
        raise RuntimeError(f"Circuit/reference fidelity check failed: {fidelity:.16g}")

    # This is a diagnostic basis only; the qBraid runtime performs the final
    # AWS/IonQ-aware transform and strict device validation before submission.
    diagnostic = transpile(
        state_circuit,
        basis_gates=["rx", "ry", "rz", "cx"],
        optimization_level=3,
        seed_transpiler=3047,
    )
    diagnostic_fidelity = float(
        state_fidelity(numpy_state, Statevector.from_instruction(diagnostic).data)
    )
    if diagnostic_fidelity < 1.0 - 1e-12:
        raise RuntimeError("Diagnostic transpilation changed the state")

    qasm = qasm2.dumps(hardware_circuit)
    circuit_sha256 = hashlib.sha256(qasm.encode("utf-8")).hexdigest()
    expected_witness = pauli_expectation(numpy_state, MEASUREMENT_WORD)
    return {
        "state_circuit": state_circuit,
        "hardware_circuit": hardware_circuit,
        "fidelity_numpy_vs_qiskit": fidelity,
        "diagnostic_fidelity": diagnostic_fidelity,
        "diagnostic_depth": int(diagnostic.depth()),
        "diagnostic_operations": {str(k): int(v) for k, v in diagnostic.count_ops().items()},
        "hardware_circuit_qasm2_sha256": circuit_sha256,
        "ideal_probabilities_z_basis": ideal_probabilities(numpy_state),
        "ideal_xxyy_expectation": expected_witness,
        "qasm2": qasm,
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return str(value)


def device_snapshot(device: Any) -> dict[str, Any]:
    """Normalize the qBraid device identity, status, and metadata."""

    metadata = device.metadata()
    try:
        status = device.status()
    except Exception:
        status = metadata.get("status", "UNKNOWN") if isinstance(metadata, Mapping) else "UNKNOWN"
    return {
        "id": str(device.id),
        "status": str(getattr(status, "name", status)),
        "metadata": _jsonable(metadata),
    }


def assert_forte_ready(device: Any) -> dict[str, Any]:
    """Require exactly the online non-simulator Forte-1 route."""

    snapshot = device_snapshot(device)
    if snapshot["id"] != TARGET_DEVICE_ID:
        raise RuntimeError(f"Wrong device: {snapshot['id']!r}")
    if snapshot["id"].startswith("openquantum:"):
        raise RuntimeError("Open Quantum route is not authorized by this runner")
    status = snapshot["status"].upper()
    if status != "ONLINE":
        raise RuntimeError(f"Forte-1 is not online: {snapshot['status']}")
    metadata = snapshot["metadata"]
    if isinstance(metadata, Mapping):
        simulator = metadata.get("simulator", metadata.get("isSimulator"))
        if isinstance(simulator, str):
            simulator = simulator.strip().lower() == "true"
        if simulator is True:
            raise RuntimeError("Resolved route identifies itself as a simulator")
        n_qubits = metadata.get("num_qubits", metadata.get("numQubits"))
        if n_qubits is not None and int(n_qubits) < N_QUBITS:
            raise RuntimeError(f"Device reports only {n_qubits} qubits")
    return snapshot


def expected_confirmation() -> str:
    """Return the exact human confirmation required for paid submission."""

    return (
        f"SUBMIT {SHOTS} SHOTS TO {TARGET_DEVICE_ID} "
        f"FOR {EXPECTED_QPU_CREDITS} CREDITS"
    )


def assert_submission_guard(authorize: bool, confirmation: str) -> None:
    """Enforce device, shot, budget, Boolean, and phrase locks."""

    if SHOTS != 350:
        raise RuntimeError("Shot lock changed")
    if EXPECTED_QPU_CREDITS != 2830:
        raise RuntimeError("Expected-credit lock changed")
    if EXPECTED_QPU_CREDITS > MAX_TOTAL_CREDITS:
        raise RuntimeError("Current plan exceeds the 3,000-credit ceiling")
    if not authorize:
        raise RuntimeError("Submission remains locked: set AUTHORIZE_QPU_SUBMISSION=True")
    if confirmation != expected_confirmation():
        raise RuntimeError("Confirmation phrase does not exactly match the displayed phrase")


def create_attempt_lock(root: Path, preflight: Mapping[str, Any]) -> Path:
    """Atomically create the local fail-closed duplicate-submission lock.

    The lock protects only this extracted copy. Judges must still avoid running
    concurrent copies and should confirm account-level job history in qBraid.
    """

    results = root / RESULTS_DIR
    results.mkdir(exist_ok=True)
    path = results / "FORTE_SUBMISSION_ATTEMPT_LOCK.json"
    payload = {
        "created_utc": utc_now(),
        "device_id": TARGET_DEVICE_ID,
        "shots": SHOTS,
        "expected_qpu_credits": EXPECTED_QPU_CREDITS,
        "circuit_sha256": preflight["hardware_circuit_qasm2_sha256"],
    }
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except FileExistsError as error:
        raise RuntimeError(
            "Attempt lock exists; refusing a possible duplicate QPU charge"
        ) from error
    return path


def write_submission_record(
    root: Path,
    job: Any,
    device: Any,
    preflight: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> Path:
    """Persist the immutable job identity and circuit/budget metadata."""

    results = root / RESULTS_DIR
    results.mkdir(exist_ok=True)
    record = {
        "record_type": "gic2026_h2_gqe_forte_submission",
        "submitted_utc": utc_now(),
        "job_id": str(job.id),
        "device": device_snapshot(device),
        "shots": SHOTS,
        "budget": {
            "max_total_credits": MAX_TOTAL_CREDITS,
            "advertised_per_task_credits": PER_TASK_CREDITS,
            "advertised_per_shot_credits": PER_SHOT_CREDITS,
            "expected_qpu_credits": EXPECTED_QPU_CREDITS,
            "note": "Estimate from qBraid pricing checked 2026-07-21; dashboard is authoritative.",
        },
        "circuit": {
            "n_qubits": N_QUBITS,
            "measurement": MEASUREMENT_WORD,
            "qasm2_sha256": preflight["hardware_circuit_qasm2_sha256"],
            "ideal_xxyy_expectation": preflight["ideal_xxyy_expectation"],
        },
        "source": reference["source_definition"],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "qbraid": package_version("qbraid"),
            "qbraid-core": package_version("qbraid-core"),
            "qiskit": package_version("qiskit"),
            "numpy": package_version("numpy"),
        },
    }
    path = results / SUBMISSION_RECORD
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def normalize_counts(raw_counts: Mapping[Any, Any]) -> dict[str, int]:
    """Normalize provider count keys to sorted four-bit strings."""

    normalized: dict[str, int] = {}
    for raw_key, raw_value in raw_counts.items():
        if isinstance(raw_key, int):
            bitstring = format(raw_key, f"0{N_QUBITS}b")
        else:
            key = str(raw_key).replace(" ", "").replace("_", "")
            if key.lower().startswith("0x"):
                bitstring = format(int(key, 16), f"0{N_QUBITS}b")
            elif set(key) <= {"0", "1"}:
                bitstring = key.zfill(N_QUBITS)[-N_QUBITS:]
            else:
                raise ValueError(f"Unrecognized count key: {raw_key!r}")
        normalized[bitstring] = normalized.get(bitstring, 0) + int(round(float(raw_value)))
    return dict(sorted(normalized.items()))


def witness_from_counts(counts: Mapping[str, int], word: str = MEASUREMENT_WORD) -> tuple[float, float]:
    """Compute the Pauli witness and its binomial shot-noise standard error."""

    total = sum(int(v) for v in counts.values())
    if total <= 0:
        raise ValueError("Counts are empty")
    support = [q for q, char in enumerate(word) if char != "I"]
    weighted = 0
    for bitstring, count in counts.items():
        index = int(bitstring, 2)
        parity = -1 if sum((index >> q) & 1 for q in support) % 2 else 1
        weighted += parity * int(count)
    expectation = weighted / total
    standard_error = math.sqrt(max(0.0, 1.0 - expectation**2) / total)
    return float(expectation), float(standard_error)


def write_result_record(
    root: Path,
    job: Any,
    result: Any,
    raw_counts: Mapping[Any, Any],
    preflight: Mapping[str, Any],
) -> Path:
    """Reduce completed counts and persist a machine-readable result."""

    results = root / RESULTS_DIR
    submission = json.loads((results / SUBMISSION_RECORD).read_text(encoding="utf-8"))
    counts = normalize_counts(raw_counts)
    measured, standard_error = witness_from_counts(counts)
    ideal = float(preflight["ideal_xxyy_expectation"])
    status = job.status()
    record = {
        "record_type": "gic2026_h2_gqe_forte_result",
        "retrieved_utc": utc_now(),
        "job_id": str(job.id),
        "status": str(getattr(status, "name", status)),
        "device_id": TARGET_DEVICE_ID,
        "shots_requested": SHOTS,
        "shots_returned": int(sum(counts.values())),
        "measurement": MEASUREMENT_WORD,
        "counts": counts,
        "ideal_expectation": ideal,
        "measured_expectation": measured,
        "shot_noise_standard_error": standard_error,
        "absolute_deviation": abs(measured - ideal),
        "submission_record_sha256": sha256_file(results / SUBMISSION_RECORD),
        "result_details": _jsonable(getattr(result, "details", None)),
    }
    path = results / RESULT_RECORD
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
