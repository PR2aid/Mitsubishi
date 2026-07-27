"""Canonical, artifact-bound Table-3 circuit compilation.

Each generic/structured logical circuit is transpiled exactly once under one
declared protocol.  The exact compiled object is serialized to QASM, reloaded,
and only that reloaded artifact is used for gate counts, depth, size, hashes,
and (where dense validation is feasible) state, leakage, and energy checks.

This deliberately separates *candidate generation* from *canonical replay*.
The module can emit a candidate reference after a full optimization, but it
will never manufacture or silently bless a missing canonical reference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


TABLE3_SEED = 3047
TABLE3_BASIS_GATES = ("rz", "sx", "x", "cx")
TABLE3_OPTIMIZATION_LEVEL = 3
TABLE3_SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compilation_protocol() -> dict[str, Any]:
    """Return the one allowed diagnostic compilation protocol."""

    return {
        "basis_gates": list(TABLE3_BASIS_GATES),
        "optimization_level": TABLE3_OPTIMIZATION_LEVEL,
        "seed_transpiler": TABLE3_SEED,
        "connectivity": "all-to-all",
        "scope": "diagnostic logical compile; not device-native",
    }


def save_parameter_artifact(
    path: str | Path, params: Mapping[str, Any]
) -> dict[str, Any]:
    """Write and exactly replay one canonical float64 parameter snapshot."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        str(key): np.asarray(value.detach().cpu().numpy(), dtype=np.float64)
        for key, value in params.items()
    }
    np.savez_compressed(path, **arrays)
    with np.load(path, allow_pickle=False) as replay:
        exact = set(replay.files) == set(arrays) and all(
            np.array_equal(np.asarray(replay[key]), arrays[key])
            for key in arrays
        )
    if not exact:
        raise RuntimeError(f"parameter snapshot failed exact replay: {path}")
    return {
        "file": str(path),
        "sha256": sha256_file(path),
        "array_names": sorted(arrays),
        "array_shapes": {
            key: list(value.shape) for key, value in sorted(arrays.items())
        },
        "array_replay_exact": True,
    }


def load_parameter_artifact(
    path: str | Path, expected_sha256: str
) -> dict[str, Any]:
    """Load a checksum-bound parameter snapshot as torch tensors."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"canonical parameter artifact is missing: {path}")
    actual = sha256_file(path)
    if actual != str(expected_sha256):
        raise ValueError(
            f"canonical parameter checksum mismatch for {path}: "
            f"{actual} != {expected_sha256}"
        )
    import torch

    with np.load(path, allow_pickle=False) as archive:
        if not archive.files:
            raise ValueError(f"canonical parameter artifact is empty: {path}")
        return {
            str(key): torch.as_tensor(
                np.asarray(archive[key], dtype=np.float64).copy(),
                dtype=torch.float64,
            )
            for key in archive.files
        }


def _portable_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def strict_load_qasm2_legacy_sx(path: str | Path) -> Any:
    """Strictly load a Qiskit-exported QASM2 file with only ``sx`` enabled.

    Qiskit's legacy OpenQASM 2 exporter can emit ``sx`` without declaring it
    in ``qelib1.inc``.  The matching importer therefore needs one documented
    legacy custom instruction.  Restricting the profile to exactly ``sx``
    avoids enabling the remainder of Qiskit's broader legacy dialect.
    """

    from qiskit import qasm2

    sx_only = tuple(
        item
        for item in qasm2.LEGACY_CUSTOM_INSTRUCTIONS
        if item.name == "sx"
    )
    if len(sx_only) != 1:
        raise RuntimeError(
            "Qiskit legacy OpenQASM 2 profile must contain exactly one "
            f"'sx' instruction; found {len(sx_only)}"
        )
    sx_instruction = sx_only[0]
    if (
        sx_instruction.name != "sx"
        or sx_instruction.num_params != 0
        or sx_instruction.num_qubits != 1
        or sx_instruction.builtin is not True
    ):
        raise RuntimeError(
            "Qiskit legacy 'sx' instruction has an unexpected signature"
        )
    return qasm2.load(
        str(path),
        include_path=(),
        include_input_directory=None,
        custom_instructions=sx_only,
        custom_classical=(),
        strict=True,
    )


def _write_and_reload_compiled_qasm(
    logical_circuit: Any,
    path: Path,
    *,
    relative_root: Path,
) -> tuple[Any, dict[str, Any]]:
    """Compile once, serialize that object, reload it, and inspect the reload."""

    from qiskit import qasm2, transpile

    compiled = transpile(
        logical_circuit,
        basis_gates=list(TABLE3_BASIS_GATES),
        optimization_level=TABLE3_OPTIMIZATION_LEVEL,
        seed_transpiler=TABLE3_SEED,
    )
    qasm_text = qasm2.dumps(compiled)
    if not qasm_text.endswith("\n"):
        qasm_text += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(qasm_text, encoding="utf-8", newline="\n")
    reloaded = strict_load_qasm2_legacy_sx(path)
    operations = {
        str(name): int(count) for name, count in reloaded.count_ops().items()
    }
    unexpected = set(operations) - set(TABLE3_BASIS_GATES)
    if unexpected:
        raise RuntimeError(
            f"reloaded QASM contains gates outside the canonical basis: "
            f"{sorted(unexpected)}"
        )
    record = {
        **compilation_protocol(),
        "qasm_file": _portable_path(path, relative_root),
        "qasm_sha256": sha256_file(path),
        "logical_qubits": int(reloaded.num_qubits),
        "depth": int(reloaded.depth()),
        "size": int(reloaded.size()),
        "cx": int(operations.get("cx", 0)),
        "operations": operations,
        "derived_from_reloaded_qasm": True,
        "compiled_once_before_serialization": True,
    }
    return reloaded, record


def _sector_projection(full_state: np.ndarray, circuit: Any) -> tuple[np.ndarray, float]:
    """Project a blocked-spin full-register state into the declared sector."""

    state = np.asarray(full_state, dtype=np.complex128).reshape(-1)
    sector = circuit.sector
    ci = np.zeros((sector.dimA, sector.dimB), dtype=np.complex128)
    for ia, alpha in enumerate(sector.alpha.strs):
        for ib, beta in enumerate(sector.beta.strs):
            address = int(alpha) | (int(beta) << int(circuit.prob.norb))
            ci[ia, ib] = state[address]
    total_probability = float(np.vdot(state, state).real)
    sector_probability = float(np.vdot(ci, ci).real)
    if total_probability <= 0.0:
        raise ValueError("QASM statevector has zero norm")
    leakage = max(0.0, 1.0 - sector_probability / total_probability)
    return ci, leakage


def _complex_sector_energy(ci: np.ndarray, problem: Any) -> float:
    """Evaluate a potentially complex CI vector with real PySCF contractions."""

    from pyscf.fci import direct_spin1

    vector = np.asarray(ci, dtype=np.complex128)
    norm = float(np.vdot(vector, vector).real)
    if norm <= 0.0:
        raise ValueError("sector projection has zero norm")
    effective = direct_spin1.absorb_h1e(
        problem.h1e,
        problem.eri,
        problem.norb,
        problem.nelec,
        0.5,
    )
    sigma_real = np.asarray(
        direct_spin1.contract_2e(
            effective,
            np.ascontiguousarray(vector.real),
            problem.norb,
            problem.nelec,
        )
    )
    sigma_imag = np.asarray(
        direct_spin1.contract_2e(
            effective,
            np.ascontiguousarray(vector.imag),
            problem.norb,
            problem.nelec,
        )
    )
    sigma = sigma_real + 1j * sigma_imag
    return float(np.vdot(vector, sigma).real / norm + problem.ecore)


def _dense_qasm_audit(
    generic_qasm: Any,
    structured_qasm: Any,
    circuit: Any,
    params: Mapping[str, Any],
    expected_energy_hartree: float,
) -> dict[str, Any]:
    from qiskit.quantum_info import Statevector

    from givens40.structured_qiskit_export import (
        sector_statevector,
        state_fidelity,
    )

    generic_state = np.asarray(
        Statevector.from_instruction(generic_qasm).data, dtype=np.complex128
    )
    structured_state = np.asarray(
        Statevector.from_instruction(structured_qasm).data, dtype=np.complex128
    )
    expected_state = np.asarray(
        sector_statevector(circuit, params), dtype=np.complex128
    )
    generic_ci, generic_leakage = _sector_projection(generic_state, circuit)
    structured_ci, structured_leakage = _sector_projection(
        structured_state, circuit
    )
    generic_energy = _complex_sector_energy(generic_ci, circuit.prob)
    structured_energy = _complex_sector_energy(structured_ci, circuit.prob)
    return {
        "scope": "dense validation of the exact reloaded QASM artifacts",
        "generic_vs_structured_state_fidelity": state_fidelity(
            generic_state, structured_state
        ),
        "generic_vs_sector_state_fidelity": state_fidelity(
            expected_state, generic_state
        ),
        "structured_vs_sector_state_fidelity": state_fidelity(
            expected_state, structured_state
        ),
        "generic_sector_leakage_probability": generic_leakage,
        "structured_sector_leakage_probability": structured_leakage,
        "generic_energy_hartree": generic_energy,
        "structured_energy_hartree": structured_energy,
        "sector_replay_energy_hartree": float(expected_energy_hartree),
        "generic_energy_difference_hartree": abs(
            generic_energy - float(expected_energy_hartree)
        ),
        "structured_energy_difference_hartree": abs(
            structured_energy - float(expected_energy_hartree)
        ),
    }


def bind_table3_circuits(
    label: str,
    circuit: Any,
    params: Mapping[str, Any],
    expected_energy_hartree: float,
    artifact_directory: str | Path,
    *,
    relative_root: str | Path,
) -> dict[str, Any]:
    """Create one fully artifact-bound generic/structured Table-3 record."""

    from givens40 import overhead as oh
    from givens40.qiskit_export import build_qiskit_circuit
    from givens40.structured_qiskit_export import (
        build_structured_qiskit_circuit,
    )

    artifact_directory = Path(artifact_directory)
    relative_root = Path(relative_root)
    generic_logical = build_qiskit_circuit(circuit, params)
    structured_logical = build_structured_qiskit_circuit(circuit, params)
    generic_qasm, generic_record = _write_and_reload_compiled_qasm(
        generic_logical,
        artifact_directory / "circuits" / f"{label}_generic.qasm",
        relative_root=relative_root,
    )
    structured_qasm, structured_record = _write_and_reload_compiled_qasm(
        structured_logical,
        artifact_directory / "circuits" / f"{label}_structured.qasm",
        relative_root=relative_root,
    )
    single_angles, double_angles = circuit.cross_angle_arrays(params)
    recomputed_phi = oh.circuit_phi(single_angles, double_angles)
    cutting = {
        "single_cross_angles": np.asarray(
            single_angles, dtype=np.float64
        ).tolist(),
        "double_cross_angles": np.asarray(
            double_angles, dtype=np.float64
        ).tolist(),
        "recomputed_phi": float(recomputed_phi),
    }

    dense = None
    if int(circuit.prob.n_qubits) <= 12:
        dense = _dense_qasm_audit(
            generic_qasm,
            structured_qasm,
            circuit,
            params,
            expected_energy_hartree,
        )
        if (
            dense["generic_vs_structured_state_fidelity"] < 1.0 - 1e-10
            or dense["generic_vs_sector_state_fidelity"] < 1.0 - 1e-10
            or dense["structured_vs_sector_state_fidelity"] < 1.0 - 1e-10
            or dense["generic_sector_leakage_probability"] > 1e-10
            or dense["structured_sector_leakage_probability"] > 1e-10
            or dense["generic_energy_difference_hartree"] > 1e-9
            or dense["structured_energy_difference_hartree"] > 1e-9
        ):
            raise RuntimeError(f"exact-QASM scientific binding failed for {label}")

    fidelity = (
        None
        if dense is None
        else dense["generic_vs_structured_state_fidelity"]
    )
    return {
        "label": label,
        "seed": TABLE3_SEED,
        "molecule": circuit.prob.meta.get("molecule"),
        "norb": int(circuit.prob.norb),
        "n_qubits": int(circuit.prob.n_qubits),
        "nelec_alpha_beta": [int(value) for value in circuit.prob.nelec],
        "selected_left_block": (
            None
            if circuit.topo.left_block is None
            else [int(value) for value in circuit.topo.left_block]
        ),
        "compilation_protocol": compilation_protocol(),
        "generic_qasm": generic_record,
        "structured_qasm": structured_record,
        # Compatibility aliases used by the submission table/validator.
        "legacy_generic_unitary": generic_record,
        "structured_exact_pauli_network": structured_record,
        "cx_reduction_fraction": 1.0
        - structured_record["cx"] / max(1, generic_record["cx"]),
        "depth_reduction_fraction": 1.0
        - structured_record["depth"] / max(1, generic_record["depth"]),
        "state_fidelity": fidelity,
        "equivalence_scope": (
            dense["scope"]
            if dense is not None
            else "compositional exact primitive tests; no dense 40q state allocated"
        ),
        "qasm_validation_scope": (
            dense["scope"]
            if dense is not None
            else (
                "40-qubit dense state/leakage/energy allocation intentionally "
                "omitted; exact primitive tests plus checksum-bound parameter "
                "and QASM artifacts define this scope"
            )
        ),
        "dense_qasm_audit": dense,
        "sector_replay_energy_hartree": float(expected_energy_hartree),
        "cutting_accounting": cutting,
        "qasm_file": structured_record["qasm_file"],
        "qasm_sha256": structured_record["qasm_sha256"],
        "device_native": False,
    }


def load_canonical_manifest(path: str | Path) -> dict[str, Any]:
    """Load a canonical manifest; never create a default or inferred one."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            "canonical Table-3 reference is missing; run the unchanged full "
            f"workflow to generate a candidate, audit it, and explicitly promote "
            f"it before quick replay: {path}"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    if int(value.get("schema_version", -1)) != TABLE3_SCHEMA_VERSION:
        raise ValueError(f"unsupported canonical Table-3 schema: {path}")
    if value.get("status") != "CANONICAL_PROMOTED":
        raise ValueError(
            "Table-3 reference has not been explicitly promoted after audit; "
            f"refusing to use candidate evidence as a trusted input: {path}"
        )
    if int(value.get("seed", -1)) != TABLE3_SEED:
        raise ValueError(f"canonical Table-3 seed is not {TABLE3_SEED}: {path}")
    if value.get("compilation_protocol") != compilation_protocol():
        raise ValueError(f"canonical Table-3 compilation protocol changed: {path}")
    if not isinstance(value.get("cases"), dict):
        raise ValueError(f"canonical Table-3 manifest has no cases: {path}")
    return value


def reference_case_paths(
    manifest_path: str | Path, case: dict[str, Any]
) -> dict[str, Path]:
    """Resolve files declared by a canonical case within its manifest directory."""

    root = Path(manifest_path).resolve().parent
    parameter = case.get("parameter_artifact") or {}
    generic = case.get("generic_qasm") or {}
    structured = case.get("structured_qasm") or {}
    declarations = {
        "parameter": parameter.get("file"),
        "generic_qasm": generic.get("qasm_file"),
        "structured_qasm": structured.get("qasm_file"),
    }
    missing = [name for name, value in declarations.items() if not value]
    if missing:
        raise ValueError(
            f"canonical case omits required paths: {', '.join(sorted(missing))}"
        )
    resolved: dict[str, Path] = {}
    for name, relative in declarations.items():
        path = (root / str(relative)).resolve()
        if root not in path.parents:
            raise ValueError(
                f"canonical {name} path must remain strictly below the "
                f"manifest directory: {path}"
            )
        resolved[name] = path
    return resolved


def bound_record_differences(
    actual: dict[str, Any], expected: dict[str, Any]
) -> list[str]:
    """Return exact canonical binding differences without accepting tolerances."""

    differences: list[str] = []
    for field in ("seed", "n_qubits", "selected_left_block", "compilation_protocol"):
        if actual.get(field) != expected.get(field):
            differences.append(
                f"{field}: actual={actual.get(field)!r}, "
                f"expected={expected.get(field)!r}"
            )
    qasm_fields = (
        "qasm_sha256",
        "basis_gates",
        "optimization_level",
        "seed_transpiler",
        "logical_qubits",
        "depth",
        "size",
        "cx",
        "operations",
    )
    for arm in ("generic_qasm", "structured_qasm"):
        actual_arm = actual.get(arm) or {}
        expected_arm = expected.get(arm) or {}
        for field in qasm_fields:
            if actual_arm.get(field) != expected_arm.get(field):
                differences.append(
                    f"{arm}.{field}: actual={actual_arm.get(field)!r}, "
                    f"expected={expected_arm.get(field)!r}"
                )
    return differences


def candidate_manifest(cases: dict[str, Any]) -> dict[str, Any]:
    """Build an unhashed candidate manifest from freshly generated artifacts."""

    return {
        "schema_version": TABLE3_SCHEMA_VERSION,
        "status": "CANDIDATE_NOT_CANONICAL_UNTIL_EXPLICITLY_PROMOTED",
        "seed": TABLE3_SEED,
        "compilation_protocol": compilation_protocol(),
        "cases": cases,
        "claim_boundary": (
            "This file is emitted evidence, not a trusted input. Quick mode "
            "requires a separately promoted checksum-bound canonical manifest."
        ),
    }
