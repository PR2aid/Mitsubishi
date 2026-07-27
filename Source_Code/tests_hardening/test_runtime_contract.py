from __future__ import annotations

import ast
import builtins
import json
from pathlib import Path
import symtable
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RuntimeContractTests(unittest.TestCase):
    def text(self, relative: str) -> str:
        return (ROOT / relative).read_text(encoding="utf-8")

    def test_no_main_runner_defaults_to_source_code_dot_venv(self) -> None:
        for relative in (
            "setup.sh",
            "source/scripts/setup_qbraid_gqe.sh",
            "source/scripts/run_qbraid_gqe.sh",
            "source/scripts/run_budgeted_gqe.sh",
            "reproduce.py",
        ):
            self.assertNotIn("PACKAGE_ROOT}/.venv", self.text(relative), relative)
            self.assertNotIn("ROOT / \".venv\"", self.text(relative), relative)

    def test_runtime_contract_is_applied_before_cudaq_solvers(self) -> None:
        for relative in (
            "source/scripts/qbraid_gqe.py",
            "source/scripts/qbraid_budgeted_gqe.py",
        ):
            source = self.text(relative)
            self.assertLess(
                source.index("import lightning", source.index("def run(")),
                source.index("import cudaq_solvers", source.index("def run(")),
                relative,
            )
            self.assertIn('cudaq.set_target("qpp-cpu")', source)

        verifier = self.text("verify_environment.py")
        self.assertLess(
            verifier.index("import lightning"),
            verifier.index("import cudaq_solvers"),
        )
        self.assertIn('cudaq.set_target("qpp-cpu")', verifier)
        self.assertIn('os.environ.get("UCX_TLS") != "self"', verifier)

    def test_enhanced_release_binds_structured_qiskit_builder(self) -> None:
        relative = "source/scripts/run_enhanced_release.py"
        source = self.text(relative)
        tree = ast.parse(source, filename=relative)
        generic_noise = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "generic_noise_qsci"
        )
        imported = {
            alias.asname or alias.name
            for node in ast.walk(generic_noise)
            if isinstance(node, ast.ImportFrom)
            and node.module == "givens40.structured_qiskit_export"
            for alias in node.names
        }
        self.assertIn("build_structured_qiskit_circuit", imported)

        loaded = {
            node.id
            for node in ast.walk(generic_noise)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        self.assertIn("build_structured_qiskit_circuit", loaded)

        module_table = symtable.symtable(source, relative, "exec")
        module_bound = {
            symbol.get_name()
            for symbol in module_table.get_symbols()
            if symbol.is_assigned()
            or symbol.is_imported()
            or symbol.is_namespace()
        }
        generic_table = next(
            child
            for child in module_table.get_children()
            if child.get_name() == "generic_noise_qsci"
        )
        unresolved = sorted(
            symbol.get_name()
            for symbol in generic_table.get_symbols()
            if symbol.is_referenced()
            and symbol.is_global()
            and symbol.get_name() not in module_bound
            and symbol.get_name() not in vars(builtins)
        )
        self.assertEqual([], unresolved)

    def test_shell_runners_are_isolated_and_ucx_self_only(self) -> None:
        for relative in (
            "source/scripts/run_qbraid_gqe.sh",
            "source/scripts/run_budgeted_gqe.sh",
        ):
            source = self.text(relative)
            self.assertIn("-u PYTHONPATH", source)
            self.assertIn("UCX_TLS=self", source)
            self.assertIn("MPLBACKEND=Agg", source)
            self.assertIn("HF_HUB_OFFLINE=1", source)
            self.assertIn("TRANSFORMERS_OFFLINE=1", source)
            self.assertIn('"${PYTHON}" -I -B', source)

    def test_notebook_uses_external_environment_and_certificate(self) -> None:
        notebook = json.loads(self.text("QBRAID_RUNME.ipynb"))
        source = "\n".join(
            line
            for cell in notebook["cells"]
            for line in cell.get("source", [])
        )
        self.assertIn("resolve_environment(ROOT)", source)
        self.assertIn("certify_release.py", source)
        self.assertIn("latest_certificate.json", source)
        self.assertIn("setup_env['PYTHON_BIN'] = sys.executable", source)
        self.assertNotIn("ROOT / '.venv'", source)

    def test_reuse_branch_precedes_any_install(self) -> None:
        setup = self.text("source/scripts/setup_qbraid_gqe.sh")
        reuse = setup.index('if [[ -e "${ENV_DIR}" ]]')
        verify = setup.index("Verifying existing environment", reuse)
        fresh_install = setup.index(
            '"${PACKAGE_ROOT}/install_locked_requirements.py" install',
            verify,
        )
        self.assertLess(reuse, verify)
        self.assertLess(verify, fresh_install)
        self.assertIn("will not be repaired in place", setup)
        self.assertIn("cpu_count < 4", setup)
        self.assertNotIn("-m pip", setup)
        self.assertIn("done < <(compgen -e)", setup)
        self.assertIn("MPLBACKEND=Agg", setup)
        self.assertIn('if [[ "${PIP_VARIABLE}" == PIP_* ]]', setup)


if __name__ == "__main__":
    unittest.main()
