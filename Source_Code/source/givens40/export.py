"""Export optimized circuits as standard quantum circuits (PennyLane / QASM).

Alignment proof for competition submission: the ansatz IS a quantum
circuit. Every sector-engine gate maps 1:1 onto PennyLane primitives
(qubit-convention Givens G(beta) == qml.SingleExcitation(2*beta);
pair-double D(delta) == qml.DoubleExcitation(2*delta) on
(p_a, q_a, p_b, q_b) ... wire order fixed by the equivalence test), and
the exported circuit's statevector matches the sector engine exactly.

Wire convention: wire k = spin-orbital k (alpha p -> wire p, beta p ->
wire p + norb), matching sector.py's blocked-JW layout. PennyLane's
statevector indexes wire 0 as the MOST significant bit, so the
equivalence test permutes indices accordingly.
"""
from __future__ import annotations

import numpy as np

from .runner import SectorCircuit


def export_gate_list(circ: SectorCircuit, params: dict) -> list[dict]:
    """Ordered list of standard gates with resolved angles.

    Returns dicts: {op: 'SingleExcitation'|'DoubleExcitation',
                    wires: tuple, theta: float}
    with theta in the PennyLane convention (theta = 2 * matrix angle).
    """
    angles = circ.flat_angles(params).detach().numpy()
    specs = circ.gate_specs()
    no = circ.prob.norb
    out = []
    # Conventions calibrated against the validated dense gates (see tests):
    #   G(beta) on (p,q)          == qml.SingleExcitation(-2*beta, [p, q])
    #   D(delta) pairing (p<->q)  == qml.DoubleExcitation(-2*delta,
    #                                  [p_a, p_b, q_a, q_b])
    for g, a in zip(specs, angles):
        if g.kind == "s":
            off = 0 if g.spin == "a" else no
            out.append(dict(op="SingleExcitation",
                            wires=(g.p + off, g.q + off), theta=-2.0 * float(a)))
        else:
            out.append(dict(op="DoubleExcitation",
                            wires=(g.p, g.p + no, g.q, g.q + no),
                            theta=-2.0 * float(a)))
    return out


def to_pennylane_source(circ: SectorCircuit, params: dict,
                        shots: int | None = None) -> str:
    """Self-contained PennyLane script reproducing the optimized circuit."""
    gates = export_gate_list(circ, params)
    no = circ.prob.norb
    n = 2 * no
    init = circ.sector.initial_state(circ.prob.hdiag()
                                     if circ.acfg.init_state == "diag" else None)
    ia, ib = np.unravel_index(int(np.argmax(np.abs(init.numpy()))), init.shape)
    occ_bits = int(circ.sector.alpha.strs[ia]) | (int(circ.sector.beta.strs[ib]) << no)
    occ = [k for k in range(n) if (occ_bits >> k) & 1]
    lines = [
        "import pennylane as qml",
        "import numpy as np",
        "",
        f"n_qubits = {n}",
        f"dev = qml.device('default.qubit', wires=n_qubits"
        + (f", shots={shots}" if shots else "") + ")",
        "",
        "@qml.qnode(dev)",
        "def circuit():",
        f"    for w in {occ}:  # initial determinant",
        "        qml.PauliX(wires=w)",
    ]
    for g in gates:
        if abs(g["theta"]) < 1e-14:
            continue
        lines.append(f"    qml.{g['op']}({g['theta']!r}, wires={list(g['wires'])})")
    lines += ["    return qml.state()", ""]
    return "\n".join(lines)


def pennylane_statevector(circ: SectorCircuit, params: dict) -> np.ndarray:
    """Execute the exported circuit in PennyLane; return amplitudes indexed
    by our LSB-first convention (bit k of the index = wire k)."""
    import pennylane as qml

    gates = export_gate_list(circ, params)
    no = circ.prob.norb
    n = 2 * no
    init = circ.sector.initial_state(circ.prob.hdiag()
                                     if circ.acfg.init_state == "diag" else None)
    ia, ib = np.unravel_index(int(np.argmax(np.abs(init.numpy()))), init.shape)
    occ_bits = int(circ.sector.alpha.strs[ia]) | (int(circ.sector.beta.strs[ib]) << no)

    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def qnode():
        for w in range(n):
            if (occ_bits >> w) & 1:
                qml.PauliX(wires=w)
        for g in gates:
            getattr(qml, g["op"])(g["theta"], wires=list(g["wires"]))
        return qml.state()

    psi = np.asarray(qnode())
    # PennyLane: wire 0 is the MSB. Reindex to our LSB-first convention.
    out = psi.reshape([2] * n).transpose(list(range(n - 1, -1, -1))).reshape(-1)
    return out
