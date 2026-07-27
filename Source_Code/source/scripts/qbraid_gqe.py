# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# Copyright (c) 2026 the challenge submission authors.                        #
#                                                                              #
# Portions of this file are adapted from NVIDIA CUDA-QX's ``gqe_h2.py`` and   #
# PySCF molecular-Hamiltonian generator. Those sources and this adaptation are #
# made available under the Apache License 2.0.                                 #
#                                                                              #
# Upstream example:                                                            #
# https://github.com/NVIDIA/cudaqx/blob/main/docs/sphinx/examples/solvers/python/gqe_h2.py
# ============================================================================ #
"""Credential-free CUDA-QX GQE verification for qBraid Lab.

This deliberately small bridge runs NVIDIA's Transformer-based Generative
Quantum Eigensolver (GQE) on H2/STO-3G using CUDA-Q's local ``qpp-cpu`` target.
It requires no qBraid API key, remote service, GPU, or QPU credits.

This is a standalone official-API closure benchmark. It does not use the
submission's Givens backbone, residual gate pool, or cutting-budget mask; those
remain separate large-system contributions and are not claimed by this file.

Examples
--------
Run the judge-friendly verification (the default)::

    python scripts/qbraid_gqe.py --smoke

Run the longer reference configuration::

    python scripts/qbraid_gqe.py --full

Both modes print a human-readable summary and write a machine-readable JSON
record. ``--output`` can override the safe default location under
``results/judge_reproduction/``.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

# Must be set before importing torch. It is harmless on the CPU path and keeps
# the same deterministic contract as NVIDIA's official example on a GPU host.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SEED = 3047
N_QUBITS = 4
N_ELECTRONS = 2
H2_BOND_ANGSTROM = 0.7474
CHEMICAL_ACCURACY_HARTREE = 1.6e-3

MODE_CONFIGS: dict[str, dict[str, int]] = {
    # The official API and model are exercised, but the Transformer width is
    # reduced so a CPU-only judge can complete this verification promptly.
    "smoke": {
        "max_iters": 2,
        "num_samples": 2,
        "ngates": 3,
        "n_embd": 96,
    },
    # Retains the official example's 25 iterations, 5 samples and 10 gates.
    # A narrower embedding makes this practical on a qBraid CPU instance.
    "full": {
        "max_iters": 25,
        "num_samples": 5,
        "ngates": 10,
        "n_embd": 96,
    },
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="run the short, CPU-only judge verification (default)",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="run 25 GQE training iterations and a 10-operator ansatz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "JSON output path (default: "
            "results/judge_reproduction/qbraid_gqe_<mode>.json)"
        ),
    )
    parser.add_argument(
        "--console-log",
        type=Path,
        help="console transcript path (default: beside the JSON result)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print per-iteration CUDA-QX GQE diagnostics",
    )
    args = parser.parse_args(argv)
    args.mode = "full" if args.full else "smoke"
    if args.output is None:
        repo_root = Path(__file__).resolve().parents[2]
        args.output = (
            repo_root
            / "results"
            / "judge_reproduction"
            / f"qbraid_gqe_{args.mode}.json"
        )
    if args.console_log is None:
        args.console_log = args.output.with_suffix(".log")
    return args


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _portable_path(path: Path) -> str:
    """Prefer a repository-relative artifact path when one is available."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path.resolve())


class _Tee:
    """Mirror Python console output to the terminal and an archival log."""

    def __init__(self, terminal: Any, archive: Any):
        self.terminal = terminal
        self.archive = archive

    def write(self, text: str) -> int:
        self.terminal.write(text)
        self.archive.write(text)
        return len(text)

    def flush(self) -> None:
        self.terminal.flush()
        self.archive.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())


def _build_h2_problem(solvers: Any, np: Any) -> tuple[Any, float, float]:
    """Build H2/STO-3G with PySCF and transform it through CUDA-Q Solvers.

    The direct integral path avoids the helper HTTP service used internally by
    ``solvers.create_molecule``. That makes the example self-contained in a
    restricted qBraid terminal while retaining the official Jordan-Wigner API.

    Returns
    -------
    spin_hamiltonian, fci_energy, hf_energy
        The CUDA-Q spin Hamiltonian and total electronic+nuclear energies.
    """
    from pyscf import ao2mo, gto, scf

    molecule = gto.M(
        atom=(
            f"H 0.0 0.0 0.0; "
            f"H 0.0 0.0 {H2_BOND_ANGSTROM:.4f}"
        ),
        basis="sto-3g",
        charge=0,
        spin=0,
        unit="Angstrom",
        verbose=0,
    )
    # H2/STO-3G is tiny. Forcing in-core integrals also makes the run portable
    # to containers that intentionally mount a restricted ``/proc`` filesystem.
    molecule.incore_anyway = True
    mean_field = scf.RHF(molecule)
    mean_field.verbose = 0
    hf_energy = float(mean_field.kernel())
    if not mean_field.converged:
        raise RuntimeError("PySCF RHF did not converge for H2/STO-3G")

    coefficients = mean_field.mo_coeff
    h1_ao = molecule.intor("int1e_kin") + molecule.intor("int1e_nuc")
    h1_mo = coefficients.T @ h1_ao @ coefficients
    h2_ao = molecule.intor("int2e_sph", aosym="1")
    h2_mo = ao2mo.incore.full(h2_ao, coefficients)

    # Convert chemist notation to the convention consumed by CUDA-QX, matching
    # its Apache-licensed PySCF generator.
    h2_mo = h2_mo.transpose(0, 2, 3, 1)
    n_spatial = h1_mo.shape[0]
    n_spin = 2 * n_spatial
    one_body = np.zeros((n_spin, n_spin), dtype=np.complex128)
    two_body = np.zeros((n_spin, n_spin, n_spin, n_spin),
                        dtype=np.complex128)

    for p in range(n_spatial):
        for q in range(n_spatial):
            one_body[2 * p, 2 * q] = h1_mo[p, q]
            one_body[2 * p + 1, 2 * q + 1] = h1_mo[p, q]
            for r in range(n_spatial):
                for s in range(n_spatial):
                    value = 0.5 * h2_mo[p, q, r, s]
                    two_body[2 * p, 2 * q, 2 * r, 2 * s] = value
                    two_body[2 * p + 1, 2 * q + 1, 2 * r + 1,
                             2 * s + 1] = value
                    two_body[2 * p, 2 * q + 1, 2 * r + 1,
                             2 * s] = value
                    two_body[2 * p + 1, 2 * q, 2 * r,
                             2 * s + 1] = value

    spin_hamiltonian = solvers.jordan_wigner(
        np.ascontiguousarray(one_body),
        np.ascontiguousarray(two_body),
        float(molecule.energy_nuc()),
    )
    # Exact full-CI reference in the two-electron sector. Diagonalizing the
    # 6-by-6 block is both faster and less environment-sensitive than starting
    # a separate PySCF Davidson solver for this four-qubit verification.
    dense_hamiltonian = np.asarray(spin_hamiltonian.to_matrix())
    sector_indices = [
        basis_index for basis_index in range(1 << n_spin)
        if basis_index.bit_count() == N_ELECTRONS
    ]
    sector_hamiltonian = dense_hamiltonian[np.ix_(sector_indices,
                                                  sector_indices)]
    exact_energy = float(np.linalg.eigvalsh(sector_hamiltonian).min().real)
    return spin_hamiltonian, exact_energy, hf_energy


def _operator_pool(cudaq: Any) -> list[Any]:
    """Return the Pauli-rotation pool from NVIDIA's official H2 example."""
    spin = cudaq.spin
    i = 0
    generators = [
        spin.y(i) * spin.z(i + 1) * spin.x(i + 2) * spin.i(i + 3),
        spin.x(i) * spin.z(i + 1) * spin.y(i + 2) * spin.i(i + 3),
        spin.i(i) * spin.y(i + 1) * spin.z(i + 2) * spin.x(i + 3),
        spin.i(i) * spin.x(i + 1) * spin.z(i + 2) * spin.y(i + 3),
        spin.x(i) * spin.x(i + 1) * spin.x(i + 2) * spin.y(i + 3),
        spin.x(i) * spin.x(i + 1) * spin.y(i + 2) * spin.x(i + 3),
        spin.x(i) * spin.y(i + 1) * spin.y(i + 2) * spin.y(i + 3),
        spin.y(i) * spin.x(i + 1) * spin.y(i + 2) * spin.y(i + 3),
        spin.x(i) * spin.y(i + 1) * spin.x(i + 2) * spin.x(i + 3),
        spin.y(i) * spin.x(i + 1) * spin.x(i + 2) * spin.x(i + 3),
        spin.y(i) * spin.y(i + 1) * spin.x(i + 2) * spin.y(i + 3),
        spin.y(i) * spin.y(i + 1) * spin.y(i + 2) * spin.x(i + 3),
    ]
    angles = [
        0.003125,
        -0.003125,
        0.00625,
        -0.00625,
        0.0125,
        -0.0125,
        0.025,
        -0.025,
        0.05,
        -0.05,
        0.1,
        -0.1,
    ]
    return [angle * generator for angle in angles for generator in generators]


def _selected_operator_records(op_pool: list[Any], indices: list[int]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for position, raw_index in enumerate(indices):
        index = int(raw_index)
        operator = op_pool[index]
        word = str(operator.get_pauli_word(N_QUBITS))
        weight = sum(character.upper() != "I" for character in word)
        records.append({
            "position": position,
            "pool_index": index,
            "coefficient": float(operator.evaluate_coefficient().real),
            "pauli_word": word,
            "pauli_weight": weight,
            # Standard parity-ladder compilation of exp(-i theta P).
            "estimated_two_qubit_gates": max(0, 2 * (weight - 1)),
        })
    return records


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one GQE mode and return its serializable result record."""
    try:
        # Import order is part of the qBraid CPU runtime contract. Lightning
        # initializes before CUDA-Q Solvers loads its MPI/UCX-linked libraries.
        import lightning  # noqa: F401
        import cudaq
        import cudaq_solvers as solvers
        from cudaq_solvers.gqe_algorithm.gqe import get_default_config
        import numpy as np
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Missing qBraid GQE dependencies. Run "
            "`bash scripts/setup_qbraid_gqe.sh --setup-only` first."
        ) from error

    if sys.version_info < (3, 11):
        raise RuntimeError("CUDA-QX Solvers 0.6.0 requires Python 3.11 or newer")

    # Explicitly local and credential-free. Do not silently select a GPU or QPU.
    cudaq.set_target("qpp-cpu")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if hasattr(cudaq, "set_random_seed"):
        cudaq.set_random_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    mode_cfg = MODE_CONFIGS[args.mode]
    cfg = get_default_config()
    cfg.seed = SEED
    cfg.max_iters = mode_cfg["max_iters"]
    cfg.num_samples = mode_cfg["num_samples"]
    cfg.ngates = mode_cfg["ngates"]
    cfg.small = True
    cfg.n_embd = mode_cfg["n_embd"]
    # CUDA-QX uses token 0 as its sequence seed; GPT-2's text-specific defaults
    # (50256) lie outside this 144-operator vocabulary and are unnecessary.
    cfg.bos_token_id = None
    cfg.eos_token_id = None
    cfg.pad_token_id = None
    cfg.use_fabric_logging = False
    cfg.fabric_logger = None
    trajectory_path = args.output.with_name(args.output.stem + "_trajectory.jsonl")
    # FileMonitor warns before overwriting; remove only this mode's own derived
    # trace so repeat judge runs stay clean and deterministic.
    trajectory_path.unlink(missing_ok=True)
    cfg.save_trajectory = True
    cfg.trajectory_file_path = str(trajectory_path)
    cfg.verbose = bool(args.verbose)

    spin_hamiltonian, exact_energy, hf_energy = _build_h2_problem(solvers, np)
    op_pool = _operator_pool(cudaq)

    def term_coefficients(operator: Any) -> list[complex]:
        # Every entry in this official pool is a one-term SpinOperator. Calling
        # the operator methods works with CUDA-Q 0.14.x and newer; iteration in
        # 0.14 returns SpinOperatorElement objects without these methods.
        return [operator.evaluate_coefficient()]

    def term_words(operator: Any) -> list[Any]:
        return [operator.get_pauli_word(N_QUBITS)]

    @cudaq.kernel
    def ansatz_kernel(n_qubits: int, n_electrons: int,
                      coefficients: list[float], words: list[cudaq.pauli_word]):
        qubits = cudaq.qvector(n_qubits)
        for electron in range(n_electrons):
            x(qubits[electron])
        for operator_index in range(len(coefficients)):
            exp_pauli(coefficients[operator_index], qubits,
                      words[operator_index])

    def cost(sampled_operators: list[Any], **_: Any) -> float:
        coefficients: list[float] = []
        words: list[Any] = []
        for operator in sampled_operators:
            coefficients.extend(
                coefficient.real for coefficient in term_coefficients(operator)
            )
            words.extend(term_words(operator))
        observed = cudaq.observe(
            ansatz_kernel,
            spin_hamiltonian,
            N_QUBITS,
            N_ELECTRONS,
            coefficients,
            words,
        )
        return float(observed.expectation())

    start = time.perf_counter()
    # When a ConfigDict is supplied, CUDA-QX reads iteration/gate settings from
    # it; passing duplicate keyword values would be ignored by version 0.6.0.
    minimum_energy, best_indices = solvers.gqe(cost, op_pool, config=cfg)
    runtime_seconds = time.perf_counter() - start
    trajectory = [
        json.loads(line) for line in trajectory_path.read_text(
            encoding="utf-8").splitlines() if line.strip()
    ]

    selected = _selected_operator_records(op_pool,
                                          [int(index) for index in best_indices])
    absolute_error = abs(float(minimum_energy) - exact_energy)
    target = cudaq.get_target()
    result: dict[str, Any] = {
        "status": "completed",
        "mode": args.mode,
        "scope": (
            "standalone NVIDIA CUDA-QX GQE closure benchmark; it does not use "
            "the submission's Givens residual pool or cutting-budget mask"
        ),
        "system": {
            "molecule": "H2",
            "basis": "STO-3G",
            "bond_length_angstrom": H2_BOND_ANGSTROM,
            "n_qubits": N_QUBITS,
            "n_electrons": N_ELECTRONS,
        },
        "backend": {
            "target": str(target.name),
            "num_qpus": int(target.num_qpus()),
            "credential_free": True,
        },
        "reproducibility": {
            "seed": SEED,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                "cuda-quantum-cu12": _package_version("cuda-quantum-cu12"),
                "cudaq-solvers": _package_version("cudaq-solvers"),
                "torch": _package_version("torch"),
                "lightning": _package_version("lightning"),
                "transformers": _package_version("transformers"),
                "numpy": _package_version("numpy"),
                "pyscf": _package_version("pyscf"),
                "mpi4py": _package_version("mpi4py"),
            },
        },
        "gqe_config": {
            **mode_cfg,
            "small_transformer": True,
            "operator_pool_size": len(op_pool),
        },
        "energies_hartree": {
            "hartree_fock": hf_energy,
            "exact_fci": exact_energy,
            "gqe_minimum": float(minimum_energy),
            "absolute_error": absolute_error,
            "chemical_accuracy_threshold": CHEMICAL_ACCURACY_HARTREE,
            "within_chemical_accuracy": absolute_error
            <= CHEMICAL_ACCURACY_HARTREE,
        },
        "runtime_seconds": runtime_seconds,
        "training_trace": {
            "format": "CUDA-QX FileMonitor JSON Lines",
            "path": _portable_path(trajectory_path),
            "epochs": trajectory,
        },
        "selected_pool_indices": [int(index) for index in best_indices],
        "selected_operators": selected,
        "selected_nonlocal_operator_count": sum(
            item["pauli_weight"] > 1 for item in selected
        ),
        "estimated_two_qubit_gate_count": sum(
            item["estimated_two_qubit_gates"] for item in selected
        ),
        "two_qubit_count_definition": (
            "2*(Pauli weight-1) per exp_pauli using the standard parity-ladder "
            "decomposition; this is a deterministic compilation estimate"
        ),
    }
    return result


def _print_summary(result: dict[str, Any], output: Path) -> None:
    energies = result["energies_hartree"]
    packages = result["reproducibility"]["packages"]
    print("\n=== CUDA-QX GQE / qBraid verification ===")
    print(f"Status:                 {result['status']}")
    print(f"Mode:                   {result['mode']}")
    print(f"Target:                 {result['backend']['target']}")
    print(f"Python:                 {result['reproducibility']['python']}")
    print(f"CUDA-Q:                 {packages['cuda-quantum-cu12']}")
    print(f"CUDA-Q Solvers:         {packages['cudaq-solvers']}")
    print(f"PyTorch:                {packages['torch']}")
    print(f"Seed:                   {result['reproducibility']['seed']}")
    print(f"GQE minimum (Ha):       {energies['gqe_minimum']:.12f}")
    print(f"Exact FCI (Ha):         {energies['exact_fci']:.12f}")
    print(f"Absolute error (Ha):    {energies['absolute_error']:.6e}")
    print(f"Runtime (s):            {result['runtime_seconds']:.3f}")
    print(f"Selected pool indices:  {result['selected_pool_indices']}")
    print(
        "Estimated 2q gates:    "
        f"{result['estimated_two_qubit_gate_count']} "
        "(standard Pauli parity ladders)"
    )
    print(f"Training trace:         {result['training_trace']['path']}")
    print(f"Console log:            {result['console_log']}")
    print(f"JSON result:            {_portable_path(output)}")
    print("JSON_RESULT=" + json.dumps(result, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    """Run one configured H2 GQE experiment and write its JSON record."""

    args = _parse_args(argv)
    args.output = args.output.expanduser().resolve()
    args.console_log = args.console_log.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.console_log.parent.mkdir(parents=True, exist_ok=True)
    with args.console_log.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(_Tee(sys.stdout, log_file)), redirect_stderr(
                _Tee(sys.stderr, log_file)):
            try:
                result = run(args)
            except Exception as error:
                print(f"ERROR: {type(error).__name__}: {error}",
                      file=sys.stderr)
                return 1

            result["console_log"] = _portable_path(args.console_log)
            args.output.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _print_summary(result, args.output)
    return 0


if __name__ == "__main__":
    exit_code = main()
    # CUDA-QX Solvers loads several native runtimes (CUDA-Q, MPI and PyTorch).
    # On qBraid's Ubuntu CPU image, their interpreter-shutdown destructors can
    # conflict after an otherwise successful qpp-cpu run and abort with
    # ``free(): invalid pointer``. At this point ``main`` has already left all
    # file context managers, so the JSON, trajectory and console log are closed
    # and durable. Flush the terminal streams and let the OS reclaim the native
    # process state without running the conflicting third-party destructors.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
