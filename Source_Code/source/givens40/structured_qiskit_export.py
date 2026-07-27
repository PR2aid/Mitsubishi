"""Exact, hardware-friendlier export of the Givens-sector state circuit.

This module replaces the generic
four-qubit ``UnitaryGate`` used for a pair-double exchange with an exact
commuting-Pauli network synthesized by Qiskit's Rustiq implementation.

Conventions match ``Source_Code/source/givens40``:

* qubit ``k`` is blocked spin-orbital ``k``;
* full-register amplitudes are indexed LSB-first;
* pair-double gate order is ``(p_alpha, q_alpha, p_beta, q_beta)``;
* matrix angle ``delta`` rotates ``|p_alpha p_beta>`` into
  ``|q_alpha q_beta>`` with the sign used by ``pairdouble16``.

The eight Pauli generators commute, so the synthesized product is exact; no
Trotter approximation is introduced.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

import numpy as np


# H = i (|q-pair><p-pair| - |p-pair><q-pair|), expressed q0-first.
# pairdouble16(delta) = exp(-i * delta * H).
_PAIR_DOUBLE_HAMILTONIAN: tuple[tuple[float, str], ...] = (
    (+0.125, "XXXY"),
    (-0.125, "XXYX"),
    (+0.125, "XYXX"),
    (+0.125, "XYYY"),
    (-0.125, "YXXX"),
    (-0.125, "YXYY"),
    (+0.125, "YYXY"),
    (-0.125, "YYYX"),
)


def append_single_exchange(qc: Any, beta: float, qubits: Sequence[int]) -> None:
    """Append the exact two-qubit Givens exchange used by the sector engine."""

    if len(qubits) != 2 or len(set(map(int, qubits))) != 2:
        raise ValueError("single exchange requires two distinct qubits")
    i, j = map(int, qubits)
    qc.cx(j, i)
    qc.cry(2.0 * float(beta), i, j)
    qc.cx(j, i)


def structured_pair_double_circuit(delta: float) -> Any:
    """Return an exact four-qubit pair-double circuit.

    Qiskit's sparse Pauli-network angle is the Pauli-rotation angle ``theta``
    in ``exp(-i theta P / 2)``.  Each term therefore receives
    ``theta = 2 * delta * coefficient``.
    """

    from qiskit.synthesis import synth_pauli_network_rustiq

    network = [
        (word, [0, 1, 2, 3], 2.0 * float(delta) * coefficient)
        for coefficient, word in _PAIR_DOUBLE_HAMILTONIAN
    ]
    return synth_pauli_network_rustiq(
        4,
        network,
        optimize_count=True,
        preserve_order=True,
        upto_clifford=False,
        upto_phase=False,
        resynth_clifford_method=0,
    )


def append_pair_double(qc: Any, delta: float, qubits: Sequence[int]) -> None:
    """Append the structured pair-double on ``(pa, qa, pb, qb)``."""

    mapped = list(map(int, qubits))
    if len(mapped) != 4 or len(set(mapped)) != 4:
        raise ValueError("pair double requires four distinct qubits")
    qc.compose(structured_pair_double_circuit(float(delta)), qubits=mapped, inplace=True)


def initial_occupation_bits(circ: Any) -> int:
    """Return the blocked-register initial determinant as an integer."""

    no = int(circ.prob.norb)
    init = circ.sector.initial_state(
        circ.prob.hdiag() if circ.acfg.init_state == "diag" else None
    )
    ia, ib = np.unravel_index(int(np.argmax(np.abs(init.detach().cpu().numpy()))), init.shape)
    return int(circ.sector.alpha.strs[ia]) | (int(circ.sector.beta.strs[ib]) << no)


def build_structured_qiskit_circuit(circ: Any, params: Mapping[str, Any]) -> Any:
    """Build the exact logical state-preparation circuit with no opaque unitary."""

    from qiskit import QuantumCircuit
    from givens40.export import export_gate_list

    n = int(circ.prob.n_qubits)
    no = int(circ.prob.norb)
    qc = QuantumCircuit(n, name=f"gqe_backbone_{n}q")
    occupied = initial_occupation_bits(circ)
    for qubit in range(n):
        if (occupied >> qubit) & 1:
            qc.x(qubit)

    for gate in export_gate_list(circ, params):
        theta = float(gate["theta"])
        if abs(theta) < 1e-14:
            continue
        if gate["op"] == "SingleExcitation":
            # export theta = -2 beta; invert to the matrix angle.
            beta = -0.5 * theta
            append_single_exchange(qc, beta, gate["wires"])
        elif gate["op"] == "DoubleExcitation":
            # export order is (pa, pb, qa, qb); structured order is
            # (pa, qa, pb, qb), matching pairdouble16 and Qiskit LSB order.
            delta = -0.5 * theta
            pa, pb, qa, qb = map(int, gate["wires"])
            append_pair_double(qc, delta, (pa, qa, pb, qb))
        else:
            raise ValueError(f"unsupported exported operation: {gate['op']!r}")

    # Defensive check against accidentally reintroducing an opaque unitary.
    forbidden = {"unitary", "hamiltonian"} & set(map(str, qc.count_ops()))
    if forbidden:
        raise RuntimeError(f"opaque operation remains in logical circuit: {sorted(forbidden)}")
    if no * 2 != n:
        raise RuntimeError("unexpected active-space qubit count")
    return qc


def sector_statevector(circ: Any, params: Mapping[str, Any]) -> np.ndarray:
    """Evaluate the same parameterized circuit with the fixed-sector engine."""

    import torch

    hdiag = circ.prob.hdiag() if circ.acfg.init_state == "diag" else None
    with torch.no_grad():
        sector = circ.forward(params, hdiag).detach().cpu().numpy()
    return np.asarray(circ.sector.embed(sector), dtype=np.complex128)


def state_fidelity(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Pure-state fidelity, insensitive to global phase."""

    a = np.asarray(reference, dtype=np.complex128).reshape(-1)
    b = np.asarray(candidate, dtype=np.complex128).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("statevectors have different dimensions")
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(abs(np.vdot(a, b)) ** 2)


_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1, -1]).astype(complex),
}


def pauli_expectation_q0_first(state: np.ndarray, word: str) -> float:
    """Evaluate one q0-first Pauli word without constructing its dense matrix."""

    psi = np.asarray(state, dtype=np.complex128).reshape(-1)
    n = int(round(math.log2(psi.size)))
    if psi.size != 1 << n or len(word) != n or any(ch not in _PAULI for ch in word):
        raise ValueError("state dimension and Pauli word are inconsistent")

    # Apply P by bit operations.  Y|b> = i*(-1)^b |1-b>.
    out = np.zeros_like(psi)
    for index, amplitude in enumerate(psi):
        target = index
        phase = 1.0 + 0.0j
        for q, char in enumerate(word):
            bit = (index >> q) & 1
            if char == "X":
                target ^= 1 << q
            elif char == "Y":
                target ^= 1 << q
                phase *= 1j if bit == 0 else -1j
            elif char == "Z" and bit:
                phase *= -1.0
        out[target] += phase * amplitude
    return float(np.vdot(psi, out).real)


def coherence_witnesses(circ: Any, state: np.ndarray) -> list[dict[str, Any]]:
    """Rank predeclared low-weight coherence witnesses for the circuit family.

    Candidates are generated only from orbital pairs already present in the
    frozen topology.  They contain X/Y on the excitation support and identity
    elsewhere, so their expectation on the initial computational-basis
    determinant is exactly zero.  Even-Y candidates are used because the
    frozen sector state is real.
    """

    n = int(circ.prob.n_qubits)
    no = int(circ.prob.norb)
    candidates: set[str] = set()
    for p, q in map(tuple, circ.topo.pairs):
        for offset in (0, no):
            for chars in (("X", "X"), ("Y", "Y")):
                word = ["I"] * n
                word[p + offset], word[q + offset] = chars
                candidates.add("".join(word))

        support = (p, q, p + no, q + no)
        for mask in range(16):
            chars = tuple("Y" if (mask >> k) & 1 else "X" for k in range(4))
            if chars.count("Y") % 2:
                continue
            word = ["I"] * n
            for wire, char in zip(support, chars):
                word[wire] = char
            candidates.add("".join(word))

    ranked = []
    for word in candidates:
        value = pauli_expectation_q0_first(state, word)
        ranked.append(
            {
                "word_q0_first": word,
                "expectation": value,
                "absolute_expectation": abs(value),
                "pauli_weight": sum(ch != "I" for ch in word),
                "initial_determinant_expectation": 0.0,
            }
        )
    ranked.sort(
        key=lambda item: (
            -item["absolute_expectation"],
            item["pauli_weight"],
            item["word_q0_first"],
        )
    )
    return ranked


def measurement_circuit(state_circuit: Any, word: str) -> Any:
    """Append q0-first Pauli basis rotations and all-qubit measurement."""

    if len(word) != state_circuit.num_qubits:
        raise ValueError("Pauli word width does not match circuit")
    qc = state_circuit.copy(name=f"{state_circuit.name}_{word}")
    for qubit, char in enumerate(word):
        if char == "X":
            qc.h(qubit)
        elif char == "Y":
            qc.sdg(qubit)
            qc.h(qubit)
        elif char not in {"I", "Z"}:
            raise ValueError(f"unsupported Pauli character: {char!r}")
    qc.measure_all()
    return qc


def normalize_counts(raw_counts: Mapping[Any, Any], n_qubits: int) -> dict[str, int]:
    """Normalize Qiskit/provider count keys to fixed-width binary strings."""

    normalized: dict[str, int] = {}
    for raw_key, raw_value in raw_counts.items():
        if isinstance(raw_key, int):
            bitstring = format(raw_key, f"0{n_qubits}b")
        else:
            key = str(raw_key).replace(" ", "").replace("_", "")
            if key.lower().startswith("0x"):
                bitstring = format(int(key, 16), f"0{n_qubits}b")
            elif set(key) <= {"0", "1"}:
                bitstring = key.zfill(n_qubits)[-n_qubits:]
            else:
                raise ValueError(f"unrecognized count key: {raw_key!r}")
        normalized[bitstring] = normalized.get(bitstring, 0) + int(round(float(raw_value)))
    return dict(sorted(normalized.items()))


def witness_from_counts(counts: Mapping[str, int], word: str) -> tuple[float, int, int]:
    """Return the Pauli expectation and positive/negative parity totals."""

    n = len(word)
    normalized = normalize_counts(counts, n)
    support = [q for q, char in enumerate(word) if char != "I"]
    plus = minus = 0
    for bitstring, count in normalized.items():
        index = int(bitstring, 2)
        odd = sum((index >> q) & 1 for q in support) % 2
        if odd:
            minus += int(count)
        else:
            plus += int(count)
    total = plus + minus
    if total <= 0:
        raise ValueError("counts are empty")
    return float((plus - minus) / total), plus, minus


def clopper_pearson_expectation(plus: int, minus: int, alpha: float = 0.05) -> list[float]:
    """Exact two-sided binomial interval transformed from p to mu=2p-1."""

    from scipy.stats import beta

    plus, minus = int(plus), int(minus)
    if plus < 0 or minus < 0 or plus + minus <= 0:
        raise ValueError("invalid parity totals")
    lo = 0.0 if plus == 0 else float(beta.ppf(alpha / 2, plus, minus + 1))
    hi = 1.0 if minus == 0 else float(beta.ppf(1 - alpha / 2, plus + 1, minus))
    return [2.0 * lo - 1.0, 2.0 * hi - 1.0]


def transpiled_resources(
    circuit: Any,
    *,
    basis_gates: Iterable[str] = ("rz", "sx", "x", "cx"),
    optimization_level: int = 3,
    seed_transpiler: int = 3047,
) -> tuple[Any, dict[str, Any]]:
    """Compile to a diagnostic all-to-all basis and return auditable resources."""

    from qiskit import transpile

    compiled = transpile(
        circuit,
        basis_gates=list(basis_gates),
        optimization_level=int(optimization_level),
        seed_transpiler=int(seed_transpiler),
    )
    ops = {str(key): int(value) for key, value in compiled.count_ops().items()}
    record = {
        "basis_gates": list(basis_gates),
        "optimization_level": int(optimization_level),
        "seed_transpiler": int(seed_transpiler),
        "logical_qubits": int(circuit.num_qubits),
        "depth": int(compiled.depth()),
        "size": int(compiled.size()),
        "cx": int(ops.get("cx", 0)),
        "operations": ops,
        "scope": "diagnostic all-to-all basis; not a device-native compile",
    }
    return compiled, record
