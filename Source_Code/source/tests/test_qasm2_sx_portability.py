"""Focused tests for the narrow Qiskit legacy-``sx`` QASM2 profile."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "source"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from audit_table3_candidate import (
    strict_load_qasm2_legacy_sx as audit_loader,
)
from givens40.canonical_resources import (
    strict_load_qasm2_legacy_sx as generation_loader,
)
from validate_submission_results import (
    strict_load_qasm2_legacy_sx as release_loader,
)


LOADERS = (generation_loader, audit_loader, release_loader)


@dataclass(frozen=True)
class FakeInstruction:
    name: str
    num_params: int
    num_qubits: int
    builtin: bool


def fake_qiskit(
    instructions: tuple[FakeInstruction, ...],
) -> tuple[ModuleType, list[tuple[tuple[object, ...], dict[str, object]]]]:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def load(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return "parsed-circuit"

    module = ModuleType("qiskit")
    module.qasm2 = SimpleNamespace(
        LEGACY_CUSTOM_INSTRUCTIONS=instructions,
        load=load,
    )
    return module, calls


class NarrowLegacySxProfileTests(unittest.TestCase):
    def test_no_bare_qasm_importer_bypasses_the_three_narrow_loaders(
        self,
    ) -> None:
        allowed = {
            (
                ROOT / "source" / "givens40" / "canonical_resources.py",
                "strict_load_qasm2_legacy_sx",
            ),
            (
                ROOT / "audit_table3_candidate.py",
                "strict_load_qasm2_legacy_sx",
            ),
            (
                ROOT / "validate_submission_results.py",
                "strict_load_qasm2_legacy_sx",
            ),
        }
        observed: set[tuple[Path, str]] = set()
        forbidden: list[str] = []
        for source_path in sorted(ROOT.rglob("*.py")):
            tree = ast.parse(
                source_path.read_text(encoding="utf-8"),
                filename=str(source_path),
            )
            function_stack: list[str] = []

            class Visitor(ast.NodeVisitor):
                def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                    function_stack.append(node.name)
                    self.generic_visit(node)
                    function_stack.pop()

                visit_AsyncFunctionDef = visit_FunctionDef

                def visit_Call(self, node: ast.Call) -> None:
                    function_name = (
                        function_stack[-1] if function_stack else "<module>"
                    )
                    call = node.func
                    if (
                        isinstance(call, ast.Attribute)
                        and isinstance(call.value, ast.Name)
                        and call.value.id == "qasm2"
                        and call.attr in {"load", "loads"}
                    ):
                        location = (source_path, function_name)
                        observed.add(location)
                        if location not in allowed:
                            forbidden.append(
                                f"{source_path}:{node.lineno}: "
                                f"qasm2.{call.attr} in {function_name}"
                            )
                    if (
                        isinstance(call, ast.Attribute)
                        and call.attr
                        in {"from_qasm_file", "from_qasm_str"}
                    ):
                        forbidden.append(
                            f"{source_path}:{node.lineno}: "
                            f"{call.attr} in {function_name}"
                        )
                    self.generic_visit(node)

            Visitor().visit(tree)

        self.assertEqual(forbidden, [])
        self.assertEqual(observed, allowed)

    def test_every_loader_uses_only_sx_and_all_strict_options(self) -> None:
        instructions = (
            FakeInstruction("rxx", 1, 2, True),
            FakeInstruction("sx", 0, 1, True),
            FakeInstruction("u0", 1, 1, True),
        )
        path = Path("/tmp/canonical.qasm")
        for loader in LOADERS:
            with self.subTest(loader=loader.__module__):
                module, calls = fake_qiskit(instructions)
                with mock.patch.dict(sys.modules, {"qiskit": module}):
                    self.assertEqual(loader(path), "parsed-circuit")
                self.assertEqual(len(calls), 1)
                args, kwargs = calls[0]
                self.assertEqual(args, (str(path),))
                self.assertEqual(kwargs["include_path"], ())
                self.assertIsNone(kwargs["include_input_directory"])
                self.assertEqual(kwargs["custom_classical"], ())
                self.assertIs(kwargs["strict"], True)
                selected = kwargs["custom_instructions"]
                self.assertIsInstance(selected, tuple)
                self.assertEqual(selected, (instructions[1],))

    def test_missing_duplicate_or_malformed_sx_fails_before_parse(self) -> None:
        profiles = (
            (FakeInstruction("rxx", 1, 2, True),),
            (
                FakeInstruction("sx", 0, 1, True),
                FakeInstruction("sx", 0, 1, True),
            ),
            (FakeInstruction("sx", 1, 1, True),),
            (FakeInstruction("sx", 0, 2, True),),
            (FakeInstruction("sx", 0, 1, False),),
        )
        for loader in LOADERS:
            for instructions in profiles:
                with self.subTest(
                    loader=loader.__module__,
                    instructions=instructions,
                ):
                    module, calls = fake_qiskit(instructions)
                    with mock.patch.dict(sys.modules, {"qiskit": module}):
                        with self.assertRaises(RuntimeError):
                            loader("/tmp/canonical.qasm")
                    self.assertEqual(calls, [])

    def test_real_qiskit_preserves_resources_and_rejects_broader_dialect(
        self,
    ) -> None:
        try:
            from qiskit import QuantumCircuit, qasm2
            from qiskit.quantum_info import Statevector, state_fidelity
        except ImportError:
            self.skipTest("Qiskit is exercised in the locked qBraid environment")

        canonical = (
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "qreg q[2];\n"
            "sx q[0];\n"
            "cx q[0],q[1];\n"
        )
        broader_legacy = (
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "qreg q[2];\n"
            "rxx(0.25) q[0],q[1];\n"
        )
        strict_error = (
            "OPENQASM 2.0;\n"
            'include "qelib1.inc";\n'
            "qreg q[1]\n"
            "sx q[0];\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical_path = root / "canonical.qasm"
            canonical_path.write_text(
                canonical,
                encoding="utf-8",
                newline="\n",
            )
            digest_before = hashlib.sha256(canonical_path.read_bytes()).hexdigest()

            expected = QuantumCircuit(2)
            expected.sx(0)
            expected.cx(0, 1)
            expected_state = Statevector.from_instruction(expected)

            for loader in LOADERS:
                with self.subTest(loader=loader.__module__):
                    circuit = loader(canonical_path)
                    self.assertEqual(circuit.num_qubits, 2)
                    self.assertEqual(circuit.depth(), 2)
                    self.assertEqual(circuit.size(), 2)
                    self.assertEqual(
                        dict(circuit.count_ops()),
                        {"sx": 1, "cx": 1},
                    )
                    actual_state = Statevector.from_instruction(circuit)
                    self.assertAlmostEqual(
                        state_fidelity(expected_state, actual_state),
                        1.0,
                        places=14,
                    )

            self.assertEqual(
                hashlib.sha256(canonical_path.read_bytes()).hexdigest(),
                digest_before,
            )

            broader_path = root / "broader.qasm"
            broader_path.write_text(
                broader_legacy,
                encoding="utf-8",
                newline="\n",
            )
            malformed_path = root / "malformed.qasm"
            malformed_path.write_text(
                strict_error,
                encoding="utf-8",
                newline="\n",
            )
            include_path = root / "local-include.qasm"
            (root / "unsafe.inc").write_text(
                "gate unsafe a { x a; }\n",
                encoding="utf-8",
                newline="\n",
            )
            include_path.write_text(
                "OPENQASM 2.0;\n"
                'include "unsafe.inc";\n'
                "qreg q[1];\n"
                "unsafe q[0];\n",
                encoding="utf-8",
                newline="\n",
            )

            for loader in LOADERS:
                with self.subTest(loader=loader.__module__, case="rxx"):
                    with self.assertRaises(qasm2.QASM2ParseError):
                        loader(broader_path)
                with self.subTest(loader=loader.__module__, case="strict"):
                    with self.assertRaises(qasm2.QASM2ParseError):
                        loader(malformed_path)
                with self.subTest(loader=loader.__module__, case="include"):
                    with self.assertRaises(qasm2.QASM2ParseError):
                        loader(include_path)


if __name__ == "__main__":
    unittest.main()
