#!/usr/bin/env python3
"""Hard-budget CUDA-QX GQE over the submission's residual exchange gates.

This is an additive experiment; it does not replace the verified H2 closure
benchmark in ``qbraid_gqe.py``.  CUDA-QX's Transformer is executed at the
declared molecular width, while candidate energies are evaluated exactly in
the fixed-particle-number sector.  No dense 40-qubit state vector is created.

Recommended staged execution on qBraid::

    python scripts/qbraid_budgeted_gqe.py --system beh2-6 --smoke
    python scripts/qbraid_budgeted_gqe.py --system beh2-12 --smoke
    python scripts/qbraid_budgeted_gqe.py --system lih-40 --smoke
    python scripts/qbraid_budgeted_gqe.py --system lih-40 --full

Every generated sequence is masked *before sampling* so the measured frozen
backbone overhead plus residual-token overhead cannot exceed ``phi_max``.
"""
from __future__ import annotations

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

# This experiment is intentionally CPU-only and credential-free.  Hiding a
# visible GPU before importing PyTorch also avoids accidental paid-resource use
# and incompatible GPU-kernel selection in a qBraid GPU image.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SEED = 3047
CHEMICAL_ACCURACY_HARTREE = 1.6e-3

SYSTEMS: dict[str, dict[str, Any]] = {
    "beh2-6": {
        "molecule": "BeH2",
        "active_orbitals": 3,
        "backbone_steps": {"smoke": 30, "full": 150},
        "max_pairs": 2,
        "heldout_shots": 2000,
    },
    "beh2-12": {
        "molecule": "BeH2",
        "active_orbitals": 6,
        "backbone_steps": {"smoke": 50, "full": 200},
        "max_pairs": 6,
        "heldout_shots": 10000,
    },
    "lih-40": {
        "molecule": "LiH",
        "active_orbitals": 20,
        "backbone_steps": {"smoke": 50, "full": 300},
        "max_pairs": 12,
        "heldout_shots": 10000,
    },
}

GQE_MODES: dict[str, dict[str, int]] = {
    "smoke": {
        "max_iters": 2,
        "num_samples": 2,
        "ngates": 3,
        "n_embd": 96,
    },
    "full": {
        "max_iters": 25,
        "num_samples": 5,
        "ngates": 6,
        "n_embd": 96,
    },
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=sorted(SYSTEMS), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="short staged test")
    mode.add_argument("--full", action="store_true", help="25-iteration run")
    parser.add_argument("--phi-max", type=float, default=15.0)
    parser.add_argument("--backbone-steps", type=int)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument(
        "--partition-left",
        type=int,
        nargs="+",
        default=None,
        help="explicit left spatial-orbital block for a tested adaptive partition",
    )
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--ngates", type=int)
    parser.add_argument(
        "--objective",
        choices=("exact-energy", "qsci-topk"),
        default="exact-energy",
        help=(
            "policy cost: raw variational energy or deterministic top-K QSCI "
            "energy from the candidate state's determinant probabilities"
        ),
    )
    parser.add_argument(
        "--qsci-k",
        type=int,
        default=16,
        help="determinant budget for --objective qsci-topk",
    )
    parser.add_argument(
        "--heldout-shots",
        type=int,
        help="predeclared primary finite-shot QSCI endpoint",
    )
    parser.add_argument(
        "--heldout-trials",
        type=int,
        default=30,
        help="number of unseen finite-shot trials (minimum 30 for release runs)",
    )
    parser.add_argument(
        "--angle-bins",
        type=float,
        nargs="+",
        default=[0.0125, 0.025, 0.05],
        help="positive matrix-angle magnitudes; both signs are included",
    )
    parser.add_argument(
        "--no-pair-doubles",
        action="store_true",
        help="use only alpha/beta single-excitation residual tokens",
    )
    parser.add_argument(
        "--cudaq-state-crosscheck",
        action="store_true",
        help=(
            "for <=12 qubits, embed the selected sector state and independently "
            "evaluate its energy with CUDA-Q qpp-cpu"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--console-log", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    args.mode = "full" if args.full else "smoke"
    if args.phi_max < 1.0:
        parser.error("--phi-max must be at least 1")
    if any(value <= 0.0 for value in args.angle_bins):
        parser.error("--angle-bins values must be positive")
    for name in ("backbone_steps", "max_pairs", "max_iters", "num_samples", "ngates"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.qsci_k <= 0:
        parser.error("--qsci-k must be positive")
    if args.heldout_shots is not None and args.heldout_shots <= 0:
        parser.error("--heldout-shots must be positive")
    if args.heldout_trials <= 0:
        parser.error("--heldout-trials must be positive")

    repo_root = Path(__file__).resolve().parents[2]
    stem = f"budgeted_gqe_{args.system.replace('-', '_')}_{args.mode}"
    if args.output is None:
        args.output = repo_root / "results" / "judge_reproduction" / f"{stem}.json"
    if args.console_log is None:
        args.console_log = args.output.with_suffix(".log")
    return args


class _Tee:
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


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _portable_path(path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return str(path.resolve().relative_to(repo_root))
    except ValueError:
        return str(path.resolve())


def _compact_backbone(result: dict) -> dict:
    keys = (
        "problem",
        "n_qubits",
        "norb",
        "nelec",
        "sector_dim",
        "seed",
        "steps",
        "e_vqe",
        "e_hf",
        "e_casci",
        "error_mha",
        "n_pairs",
        "n_cross_pairs",
        "active_gates",
        "phi",
        "variance_ha2",
        "u_budget",
        "u_committed",
        "wall_seconds",
        "ansatz",
    )
    return {key: result[key] for key in keys}


def _cudaq_spin_hamiltonian(solvers, np, problem):
    """Convert the active-space integrals through CUDA-QX Jordan--Wigner.

    This matches the integral convention used by the verified H2 bridge.
    """
    n_spatial = int(problem.norb)
    n_spin = 2 * n_spatial
    one_body = np.zeros((n_spin, n_spin), dtype=np.complex128)
    two_body = np.zeros((n_spin, n_spin, n_spin, n_spin), dtype=np.complex128)
    two_spatial = np.asarray(problem.eri).transpose(0, 2, 3, 1)
    for p in range(n_spatial):
        for q in range(n_spatial):
            one_body[2 * p, 2 * q] = problem.h1e[p, q]
            one_body[2 * p + 1, 2 * q + 1] = problem.h1e[p, q]
            for r in range(n_spatial):
                for s in range(n_spatial):
                    value = 0.5 * two_spatial[p, q, r, s]
                    two_body[2 * p, 2 * q, 2 * r, 2 * s] = value
                    two_body[2 * p + 1, 2 * q + 1, 2 * r + 1, 2 * s + 1] = value
                    two_body[2 * p, 2 * q + 1, 2 * r + 1, 2 * s] = value
                    two_body[2 * p + 1, 2 * q, 2 * r, 2 * s + 1] = value
    return solvers.jordan_wigner(
        np.ascontiguousarray(one_body),
        np.ascontiguousarray(two_body),
        float(problem.ecore),
    )


def _make_budget_masked_model(
    transformer_class,
    torch,
    functional,
    get_device,
    cfg,
    cost,
    token_costs: list[float],
    residual_u_budget: float,
    num_qpus: int,
):
    """Construct a pinned-CUDA-QX Transformer with pre-sampling masking."""

    class BudgetMaskedTransformer(transformer_class):
        def __init__(self):
            super().__init__(cfg, cost, loss="exp", numQPUs=num_qpus)
            costs = torch.tensor(
                token_costs, dtype=torch.float64, device=get_device()
            )
            self.register_buffer("residual_token_costs", costs)
            self.residual_u_budget = float(residual_u_budget)
            self.generated_sequence_us: list[list[float]] = []

        def generate(self, idx=None, ngates=None):
            if idx is None:
                idx = self._starting_idx.clone()
            condition_length = idx.size(dim=1)
            if ngates is None:
                ngates = self.ngates
            used_u = torch.zeros(
                idx.shape[0],
                dtype=self.residual_token_costs.dtype,
                device=idx.device,
            )
            logits_base = None
            for _ in range(ngates):
                logits_base = self.generate_logits(idx)
                logits = logits_base[:, -1, :]
                remaining = self.residual_u_budget - used_u
                allowed = self.residual_token_costs[None, :] <= (
                    remaining[:, None] + 1e-12
                )
                # Token zero is the zero-cost identity and is therefore always
                # a safe padding action even when no budget remains.
                allowed[:, 0] = True
                masked_logits = logits.masked_fill(
                    ~allowed, torch.finfo(logits.dtype).max
                )
                probabilities = functional.softmax(
                    -self.temperature * masked_logits, dim=-1
                )
                idx_next = torch.multinomial(probabilities, num_samples=1)
                used_u = used_u + self.residual_token_costs[idx_next[:, 0]]
                idx = torch.cat((idx, idx_next), dim=1)
            self.generated_sequence_us.append(
                used_u.detach().cpu().numpy().tolist()
            )
            return idx[:, condition_length:], logits_base

    return BudgetMaskedTransformer()


def run(args: argparse.Namespace) -> dict:
    """Execute one hard-budget-masked policy search and return its record."""

    # Keep CUDA-Q visible in module globals so kernels defined below can be
    # compiled from their annotations and bodies by the CUDA-Q decorator.
    global cudaq
    try:
        # Import order is part of the qBraid CPU runtime contract. Lightning
        # initializes before CUDA-Q Solvers loads its MPI/UCX-linked libraries.
        import lightning  # noqa: F401
        import cudaq
        import cudaq_solvers as solvers
        from cudaq_solvers.gqe_algorithm.gqe import get_default_config
        from cudaq_solvers.gqe_algorithm.transformer import (
            Transformer,
            get_device,
        )
        import numpy as np
        import torch
        from torch.nn import functional as F
    except ImportError as error:
        raise RuntimeError(
            "Missing qBraid GQE dependencies. Run "
            "`bash scripts/setup_qbraid_gqe.sh --setup-only` first."
        ) from error

    if _package_version("cudaq-solvers") != "0.6.0":
        raise RuntimeError(
            "This pre-sampling mask is validated against cudaq-solvers==0.6.0; "
            f"found {_package_version('cudaq-solvers')}"
        )

    from givens40.budgeted_gqe import (
        apply_token,
        blocked_to_interleaved_fock_state,
        build_residual_pool,
        remaining_u,
        sequence_u,
        validate_sequence_budget,
    )
    from givens40.chemistry import build_cas
    from givens40.energy import make_energy_fn
    from givens40.qsci import QSCISolver
    from givens40.runner import (
        AnsatzConfig,
        OptConfig,
        SectorCircuit,
        run_vqe,
    )

    cudaq.set_target("qpp-cpu")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if hasattr(cudaq, "set_random_seed"):
        cudaq.set_random_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    system_cfg = SYSTEMS[args.system]
    mode_cfg = dict(GQE_MODES[args.mode])
    for name in ("max_iters", "num_samples", "ngates"):
        override = getattr(args, name)
        if override is not None:
            mode_cfg[name] = override
    backbone_steps = args.backbone_steps or system_cfg["backbone_steps"][args.mode]
    max_pairs = args.max_pairs or system_cfg["max_pairs"]

    total_start = time.perf_counter()
    problem = build_cas(
        system_cfg["molecule"], system_cfg["active_orbitals"]
    )
    ansatz_cfg = AnsatzConfig(
        layers=2,
        topology="partitioned",
        phi_max=float(args.phi_max),
        beta_cap=0.05,
        beta_cap_double=0.25,
        engine="adjoint",
        partition_override=args.partition_left,
    )
    opt_cfg = OptConfig(steps=int(backbone_steps), lr=0.05, seed=SEED)

    print(
        f"Optimizing frozen {problem.n_qubits}-qubit backbone "
        f"({backbone_steps} Adam steps)...",
        flush=True,
    )
    backbone = run_vqe(
        problem,
        ansatz_cfg,
        opt_cfg,
        return_params=True,
    )
    circuit = SectorCircuit(problem, ansatz_cfg)
    hdiag = problem.hdiag()
    energy_fn, _ = make_energy_fn(problem)
    with torch.no_grad():
        backbone_state = circuit.forward(backbone["best_params"], hdiag)
        backbone_energy = float(energy_fn(backbone_state))
    if abs(backbone_energy - float(backbone["e_vqe"])) > 1e-10:
        raise RuntimeError("frozen-backbone energy failed internal replay check")
    if circuit.topo.left_block is None:
        raise RuntimeError("partitioned topology did not expose a left block")
    qsci_solver = QSCISolver(problem, circuit.sector)
    initial_sector_state = circuit.sector.initial_state(hdiag)
    initial_determinant = int(
        np.argmax(np.abs(initial_sector_state.detach().cpu().numpy()).reshape(-1))
    )

    def evaluate_tokens(sampled_tokens: list[Any]) -> dict[str, Any]:
        with torch.no_grad():
            state = backbone_state
            for token in sampled_tokens:
                state = apply_token(
                    state, circuit.sector, token, convention=ansatz_cfg.convention
                )
            raw_energy = float(energy_fn(state))
        qsci = None
        if args.objective == "qsci-topk":
            qsci = qsci_solver.top_k(
                state.detach().cpu().numpy(),
                min(int(args.qsci_k), qsci_solver.dimension),
                mandatory=(initial_determinant,),
            )
            objective_energy = float(qsci["energy_hartree"])
        else:
            objective_energy = raw_energy
        return {
            "objective_energy_hartree": objective_energy,
            "raw_state_energy_hartree": raw_energy,
            "qsci_topk": qsci,
            "state": state,
        }

    backbone_evaluation = evaluate_tokens([])
    backbone_objective = float(backbone_evaluation["objective_energy_hartree"])

    pool, chosen_pairs = build_residual_pool(
        problem,
        circuit.topo.left_block,
        max_pairs=max_pairs,
        angle_bins=tuple(float(x) for x in args.angle_bins),
        include_pair_doubles=not args.no_pair_doubles,
    )
    residual_budget = remaining_u(args.phi_max, backbone["phi"])
    # Remove tokens that can never fit even as the first action.  Sequence-level
    # masking still handles cumulative cost after every sampled action.
    pool = [
        token for token in pool
        if token.kind == "identity" or token.u_cost <= residual_budget + 1e-12
    ]
    if len(pool) <= 1:
        raise RuntimeError(
            "the optimized backbone leaves no residual budget; reduce backbone "
            "cross caps or increase phi_max"
        )

    cost_stats = {
        "calls": 0,
        "max_residual_u": 0.0,
        "max_total_phi": float(backbone["phi"]),
        "objective": args.objective,
        "qsci_k": int(args.qsci_k) if args.objective == "qsci-topk" else None,
    }

    def cost(sampled_tokens: list[Any], **_: Any) -> float:
        accounting = validate_sequence_budget(
            sampled_tokens,
            base_phi=float(backbone["phi"]),
            phi_max=float(args.phi_max),
        )
        evaluation = evaluate_tokens(sampled_tokens)
        cost_stats["calls"] += 1
        cost_stats["max_residual_u"] = max(
            cost_stats["max_residual_u"], accounting["residual_u"]
        )
        cost_stats["max_total_phi"] = max(
            cost_stats["max_total_phi"], accounting["phi"]
        )
        return float(evaluation["objective_energy_hartree"])

    cfg = get_default_config()
    cfg.seed = SEED
    cfg.max_iters = int(mode_cfg["max_iters"])
    cfg.num_samples = int(mode_cfg["num_samples"])
    cfg.ngates = int(mode_cfg["ngates"])
    cfg.small = True
    cfg.n_embd = int(mode_cfg["n_embd"])
    cfg.energy_offset = -float(backbone_objective)
    cfg.bos_token_id = None
    cfg.eos_token_id = None
    cfg.pad_token_id = None
    cfg.use_fabric_logging = False
    cfg.fabric_logger = None
    cfg.verbose = bool(args.verbose)
    cfg.vocab_size = len(pool)
    trajectory_path = args.output.with_name(
        args.output.stem + "_trajectory.jsonl"
    )
    trajectory_path.unlink(missing_ok=True)
    cfg.save_trajectory = True
    cfg.trajectory_file_path = str(trajectory_path)

    model = _make_budget_masked_model(
        Transformer,
        torch,
        F,
        get_device,
        cfg,
        cost,
        [float(token.u_cost) for token in pool],
        residual_budget,
        int(cudaq.get_target().num_qpus()),
    )

    print(
        f"Running masked CUDA-QX GQE: pool={len(pool)}, "
        f"pairs={len(chosen_pairs)}, ngates={cfg.ngates}, "
        f"iters={cfg.max_iters}, samples={cfg.num_samples}...",
        flush=True,
    )
    gqe_start = time.perf_counter()
    sampled_minimum, sampled_indices = solvers.gqe(
        cost,
        pool,
        config=cfg,
        model=model,
    )
    gqe_seconds = time.perf_counter() - gqe_start
    sampled_indices = [int(index) for index in sampled_indices]
    sampled_tokens = [pool[index] for index in sampled_indices]
    sampled_objective_replayed = cost(sampled_tokens)
    sampled_evaluation = evaluate_tokens(sampled_tokens)
    sampled_minimum = float(sampled_minimum)
    if abs(sampled_minimum - sampled_objective_replayed) > 2e-5:
        raise RuntimeError(
            "CUDA-QX sampled minimum failed double-precision replay: "
            f"{sampled_minimum} vs {sampled_objective_replayed}"
        )

    # The identity sequence is a guaranteed feasible candidate.  Keep the raw
    # Transformer minimum and separately report a monotone guarded best.
    if backbone_objective <= sampled_objective_replayed + 1e-12:
        selected_indices = [0] * int(cfg.ngates)
        selected_tokens = [pool[0]] * int(cfg.ngates)
        selected_evaluation = backbone_evaluation
        selection_source = "identity_guard"
    else:
        selected_indices = sampled_indices
        selected_tokens = sampled_tokens
        selected_evaluation = sampled_evaluation
        selection_source = "cudaqx_sampled_minimum"
    selected_energy = float(selected_evaluation["raw_state_energy_hartree"])
    selected_objective = float(selected_evaluation["objective_energy_hartree"])

    # The policy objective is a training metric.  Evaluate both the guarded
    # state and its frozen-backbone identity control on one predeclared,
    # independent, fixed-K finite-shot endpoint.  The same seeds and protocol
    # are used by the exact-energy and QSCI-topK arms in the paired postprocess.
    heldout_shots = int(
        args.heldout_shots
        if args.heldout_shots is not None
        else system_cfg["heldout_shots"]
    )
    heldout_seeds = [91000 + index for index in range(int(args.heldout_trials))]
    heldout_k = min(int(args.qsci_k), qsci_solver.dimension)

    def compact_qsci(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in record.items()
            if key != "determinant_indices"
        }

    def heldout_state_evaluation(state: Any, raw_energy: float) -> dict[str, Any]:
        state_np = state.detach().cpu().numpy()
        deterministic = compact_qsci(
            qsci_solver.top_k(
                state_np,
                heldout_k,
                mandatory=(initial_determinant,),
            )
        )
        finite = qsci_solver.sampled_trials_capped(
            state_np,
            heldout_shots,
            heldout_seeds,
            max_determinants=heldout_k,
            mandatory=(initial_determinant,),
            min_count=1,
        )
        finite["trials"] = [compact_qsci(trial) for trial in finite["trials"]]
        return {
            "raw_state": {
                "energy_hartree": float(raw_energy),
                "error_mha": (
                    None
                    if problem.e_casci is None
                    else 1000.0 * (float(raw_energy) - float(problem.e_casci))
                ),
            },
            "deterministic_topk": deterministic,
            "finite_shot": finite,
        }

    heldout_evaluation = {
        "protocol": {
            "id": "gqe_qsci_heldout_v1",
            "state_source": "guarded_result_and_frozen_backbone_identity_control",
            "sampling": "ideal_computational_basis_multinomial",
            "primary_shots": heldout_shots,
            "seeds": heldout_seeds,
            "trial_count": len(heldout_seeds),
            "qsci_k": heldout_k,
            "selection_min_count": 1,
            "max_sampled_determinants": heldout_k,
            "mandatory_determinants": [initial_determinant],
            "determinant_budget_semantics": (
                "K is the total selected-subspace budget, including the "
                "mandatory Hartree-Fock determinant"
            ),
            "casci_reference_usage": "post_hoc_error_and_variational_audit_only",
            "promotion_endpoint_predeclared_before_policy_runs": True,
        },
        "guarded_state": heldout_state_evaluation(
            selected_evaluation["state"], selected_energy
        ),
        "frozen_backbone_identity_control": heldout_state_evaluation(
            backbone_state, backbone_energy
        ),
    }

    # Matched classical policy controls use the same residual vocabulary,
    # sequence length, objective and candidate-evaluation ceiling.  Their
    # selection sees only the deterministic training objective; the finite-
    # shot seeds above remain held out.
    control_budget = int(cfg.max_iters * cfg.num_samples + 1)
    identity_tokens = [pool[0]] * int(cfg.ngates)
    identity_evaluation = backbone_evaluation

    control_rng = np.random.default_rng(93000)
    random_best_tokens = identity_tokens
    random_best_evaluation = identity_evaluation
    random_evaluations = 0
    for _ in range(control_budget):
        candidate: list[Any] = []
        for _position in range(int(cfg.ngates)):
            feasible = []
            for token in pool:
                try:
                    validate_sequence_budget(
                        candidate + [token],
                        base_phi=float(backbone["phi"]),
                        phi_max=float(args.phi_max),
                    )
                except ValueError:
                    continue
                feasible.append(token)
            if not feasible:
                candidate.append(pool[0])
            else:
                candidate.append(feasible[int(control_rng.integers(len(feasible)))])
        evaluation = evaluate_tokens(candidate)
        random_evaluations += 1
        if (
            float(evaluation["objective_energy_hartree"])
            < float(random_best_evaluation["objective_energy_hartree"]) - 1e-12
        ):
            random_best_tokens = candidate
            random_best_evaluation = evaluation

    candidates_per_step = max(1, control_budget // int(cfg.ngates))
    stratified_indices = sorted(
        {
            int(round(value))
            for value in np.linspace(
                0,
                len(pool) - 1,
                num=min(candidates_per_step, len(pool)),
            )
        }
    )
    greedy_prefix: list[Any] = []
    greedy_evaluation = identity_evaluation
    greedy_evaluations = 0
    for _position in range(int(cfg.ngates)):
        step_best_token = pool[0]
        step_best_evaluation = greedy_evaluation
        for index in stratified_indices:
            token = pool[index]
            full_candidate = (
                greedy_prefix
                + [token]
                + [pool[0]] * (int(cfg.ngates) - len(greedy_prefix) - 1)
            )
            try:
                validate_sequence_budget(
                    full_candidate,
                    base_phi=float(backbone["phi"]),
                    phi_max=float(args.phi_max),
                )
            except ValueError:
                continue
            evaluation = evaluate_tokens(full_candidate)
            greedy_evaluations += 1
            if (
                float(evaluation["objective_energy_hartree"])
                < float(step_best_evaluation["objective_energy_hartree"]) - 1e-12
            ):
                step_best_token = token
                step_best_evaluation = evaluation
        greedy_prefix.append(step_best_token)
        greedy_evaluation = step_best_evaluation

    def control_record(
        name: str,
        tokens: list[Any],
        evaluation: dict[str, Any],
        candidate_evaluations: int,
    ) -> dict[str, Any]:
        return {
            "name": name,
            "candidate_evaluations": int(candidate_evaluations),
            "objective": args.objective,
            "objective_energy_hartree": float(
                evaluation["objective_energy_hartree"]
            ),
            "raw_state_energy_hartree": float(
                evaluation["raw_state_energy_hartree"]
            ),
            "tokens": [token.record() for token in tokens],
            "heldout": heldout_state_evaluation(
                evaluation["state"],
                float(evaluation["raw_state_energy_hartree"]),
            ),
        }

    heldout_evaluation["matched_policy_controls"] = {
        "identity": control_record(
            "identity", identity_tokens, identity_evaluation, 0
        ),
        "random_feasible": control_record(
            "random_feasible",
            random_best_tokens,
            random_best_evaluation,
            random_evaluations,
        ),
        "stratified_pool_greedy": control_record(
            "stratified_pool_greedy",
            greedy_prefix,
            greedy_evaluation,
            greedy_evaluations,
        ),
        "candidate_budget_ceiling": control_budget,
        "control_seed": 93000,
        "greedy_pool_indices": stratified_indices,
        "selection_uses_heldout_samples": False,
    }

    selected_budget = validate_sequence_budget(
        selected_tokens,
        base_phi=float(backbone["phi"]),
        phi_max=float(args.phi_max),
    )
    cudaq_crosscheck = None
    if args.cudaq_state_crosscheck:
        if problem.n_qubits > 12:
            raise ValueError(
                "--cudaq-state-crosscheck is intentionally limited to <=12 "
                "qubits; do not allocate a dense 40-qubit state"
            )
        crosscheck_start = time.perf_counter()
        with torch.no_grad():
            selected_state = backbone_state
            for token in selected_tokens:
                selected_state = apply_token(
                    selected_state,
                    circuit.sector,
                    token,
                    convention=ansatz_cfg.convention,
                )
        blocked_state = circuit.sector.embed(
            selected_state.detach().cpu().numpy()
        ).astype(np.complex128, copy=False)
        full_state = blocked_to_interleaved_fock_state(
            blocked_state, problem.norb
        )
        spin_hamiltonian = _cudaq_spin_hamiltonian(
            solvers, np, problem
        )
        initial_state = cudaq.State.from_data(
            np.ascontiguousarray(full_state)
        )

        @cudaq.kernel
        def state_preparation(state: cudaq.State):
            qubits = cudaq.qvector(state)

        cudaq_energy = float(
            cudaq.observe(
                state_preparation, spin_hamiltonian, initial_state
            ).expectation()
        )
        difference = abs(cudaq_energy - float(selected_energy))
        if difference > 1e-8:
            raise RuntimeError(
                "CUDA-Q qpp-cpu state-energy cross-check failed: "
                f"sector={selected_energy:.12f}, cudaq={cudaq_energy:.12f}, "
                f"difference={difference:.3e}"
            )
        cudaq_crosscheck = {
            "status": "passed",
            "method": (
                "blocked-spin sector state embedded in the full register, "
                "fermionically permuted (including parity phases) to CUDA-QX "
                "interleaved spin-orbital order, prepared "
                "by cudaq.State.from_data, and observed on qpp-cpu against the "
                "CUDA-QX Jordan-Wigner Hamiltonian"
            ),
            "n_qubits": int(problem.n_qubits),
            "sector_energy_hartree": float(selected_energy),
            "cudaq_energy_hartree": cudaq_energy,
            "absolute_difference_hartree": difference,
            "runtime_seconds": time.perf_counter() - crosscheck_start,
        }
    trajectory = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    mask_violations = 0
    max_trajectory_residual_u = 0.0
    for epoch in trajectory:
        for row in epoch["indices"]:
            tokens = [pool[int(index)] for index in row]
            residual = sequence_u(tokens)
            max_trajectory_residual_u = max(
                max_trajectory_residual_u, residual
            )
            try:
                validate_sequence_budget(
                    tokens,
                    base_phi=float(backbone["phi"]),
                    phi_max=float(args.phi_max),
                )
            except ValueError:
                mask_violations += 1
    if mask_violations:
        raise RuntimeError(
            f"pre-sampling budget mask failed for {mask_violations} sequences"
        )

    exact = problem.e_casci
    absolute_error = (
        abs(float(selected_energy) - float(exact)) if exact is not None else None
    )
    result = {
        "status": "completed",
        "mode": args.mode,
        "scope": (
            "CUDA-QX Transformer-GQE policy at the declared molecular width; "
            "candidate costs use either raw exact-sector energy or a declared "
            "top-K QSCI subspace energy and do not allocate a dense full-register "
            "state vector"
        ),
        "system": {
            "label": args.system,
            "molecule": system_cfg["molecule"],
            "active_orbitals": int(system_cfg["active_orbitals"]),
            "n_qubits": int(problem.n_qubits),
            "n_electrons": [int(x) for x in problem.nelec],
            "sector_dimension": int(backbone["sector_dim"]),
            "basis": problem.meta.get("basis"),
            "scientific_fingerprint_sha256": problem.meta.get(
                "scientific_fingerprint_sha256"
            ),
        },
        "backend": {
            "gqe_implementation": "NVIDIA CUDA-QX Solvers 0.6.0",
            "cudaq_control_target": str(cudaq.get_target().name),
            "cost_evaluator": (
                "deterministic top-K QSCI over exact PySCF sector columns"
                if args.objective == "qsci-topk"
                else "exact PySCF fixed-particle sector contraction"
            ),
            "dense_statevector_used": False,
            "gpu_used": False,
            "qpu_used": False,
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
                "numpy": _package_version("numpy"),
                "pyscf": _package_version("pyscf"),
                "transformers": _package_version("transformers"),
            },
        },
        "backbone": _compact_backbone(backbone),
        "residual_vocabulary": {
            "kind": "pre-sampling hard-budget-masked exchange tokens",
            "pool_size": len(pool),
            "selected_cross_pairs": [list(pair) for pair in chosen_pairs],
            "max_pairs": int(max_pairs),
            "angle_bins": [float(x) for x in args.angle_bins],
            "includes_signed_angles": True,
            "includes_alpha_beta_singles": True,
            "includes_pair_doubles": not args.no_pair_doubles,
            "left_block": [int(x) for x in circuit.topo.left_block],
            "right_block": [int(x) for x in circuit.topo.right_block],
        },
        "gqe_config": {
            **mode_cfg,
            "small_transformer": True,
            "energy_offset": float(cfg.energy_offset),
            "objective": args.objective,
            "qsci_k": int(args.qsci_k) if args.objective == "qsci-topk" else None,
            "candidate_evaluation_budget": int(cfg.max_iters * cfg.num_samples + 1),
        },
        "backbone_objective": {
            "objective": args.objective,
            "objective_energy_hartree": backbone_objective,
            "raw_state_energy_hartree": backbone_energy,
            "qsci_topk": backbone_evaluation["qsci_topk"],
        },
        "sampled_result": {
            "minimum_reported_by_cudaqx_float32": sampled_minimum,
            "minimum_replayed_float64": sampled_objective_replayed,
            "objective_energy_hartree": float(
                sampled_evaluation["objective_energy_hartree"]
            ),
            "raw_state_energy_hartree": float(
                sampled_evaluation["raw_state_energy_hartree"]
            ),
            "qsci_topk": sampled_evaluation["qsci_topk"],
            "pool_indices": sampled_indices,
            "tokens": [
                token.record(index)
                for index, token in zip(sampled_indices, sampled_tokens)
            ],
        },
        "guarded_result": {
            "selection_source": selection_source,
            "energy_hartree": float(selected_energy),
            "raw_state_energy_hartree": float(selected_energy),
            "objective": args.objective,
            "objective_energy_hartree": float(selected_objective),
            "qsci_topk": selected_evaluation["qsci_topk"],
            "pool_indices": selected_indices,
            "tokens": [
                token.record(index)
                for index, token in zip(selected_indices, selected_tokens)
            ],
            "improvement_over_frozen_backbone_mha": 1000.0
            * (float(backbone_energy) - float(selected_energy)),
            "objective_improvement_over_frozen_backbone_mha": 1000.0
            * (float(backbone_objective) - float(selected_objective)),
            "exact_casci_hartree": exact,
            "absolute_error_hartree": absolute_error,
            "error_mha": (
                1000.0 * (float(selected_energy) - float(exact))
                if exact is not None
                else None
            ),
            "within_chemical_accuracy": (
                absolute_error <= CHEMICAL_ACCURACY_HARTREE
                if absolute_error is not None
                else None
            ),
        },
        "heldout_evaluation": heldout_evaluation,
        "cutting_budget": {
            **selected_budget,
            "mask_applied_before_sampling": True,
            "trajectory_mask_violations": mask_violations,
            "max_trajectory_residual_u": max_trajectory_residual_u,
        },
        "cudaq_state_crosscheck": cudaq_crosscheck,
        "runtime_seconds": {
            "backbone": float(backbone["wall_seconds"]),
            "gqe": gqe_seconds,
            "total": time.perf_counter() - total_start,
        },
        "cost_statistics": cost_stats,
        "training_trace": {
            "format": "CUDA-QX FileMonitor JSON Lines",
            "path": _portable_path(trajectory_path),
            "epochs": trajectory,
            "generated_sequence_u_by_epoch": model.generated_sequence_us,
        },
        "limitations": [
            "The large-width result exploits a fixed low-electron sector and "
            "is not evidence of generic 40-qubit classical hardness.",
            "CUDA-QX performs generative policy training; the 40-qubit cost "
            "callback is the sector evaluator, not CUDA-Q state-vector observe.",
            "Residual angles are selected from a finite declared grid; no "
            "post-selection continuous residual-angle optimization is claimed.",
            "The QSCI top-K policy objective, when selected, is a deterministic "
            "simulator training proxy; promotion depends only on the separate "
            "fixed-K, finite-shot held-out endpoint and is not guaranteed.",
        ],
    }
    return result


def _print_summary(result: dict, output: Path) -> None:
    guarded = result["guarded_result"]
    budget = result["cutting_budget"]
    runtime = result["runtime_seconds"]
    print("\n=== Budget-masked CUDA-QX GQE / qBraid ===")
    print(f"Status:                  {result['status']}")
    print(f"System:                  {result['system']['label']}")
    print(f"Qubits / sector dim:     {result['system']['n_qubits']} / "
          f"{result['system']['sector_dimension']}")
    print(f"Residual pool size:      {result['residual_vocabulary']['pool_size']}")
    print(f"Policy objective:        {result['gqe_config']['objective']}")
    print(f"Selection source:        {guarded['selection_source']}")
    print(f"Frozen backbone (Ha):    {result['backbone']['e_vqe']:.12f}")
    print(f"Guarded raw state (Ha):  {guarded['raw_state_energy_hartree']:.12f}")
    print(f"Guarded objective (Ha):  {guarded['objective_energy_hartree']:.12f}")
    if guarded["error_mha"] is not None:
        print(f"Error vs CASCI (mHa):    {guarded['error_mha']:+.6f}")
    print(f"Final phi / phi_max:     {budget['phi']:.6f} / {budget['phi_max']:.6f}")
    print(f"Mask violations:         {budget['trajectory_mask_violations']}")
    if result["cudaq_state_crosscheck"] is not None:
        print(
            "CUDA-Q state check:       PASS "
            f"(diff={result['cudaq_state_crosscheck']['absolute_difference_hartree']:.3e} Ha)"
        )
    print(f"GQE / total time (s):    {runtime['gqe']:.3f} / {runtime['total']:.3f}")
    print(f"Selected pool indices:   {guarded['pool_indices']}")
    print(f"Training trace:          {result['training_trace']['path']}")
    print(f"Console log:             {result['console_log']}")
    print(f"JSON result:             {_portable_path(output)}")
    print("JSON_RESULT=" + json.dumps(result, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    """Run one masked-GQE configuration and persist all provenance fields."""

    args = _parse_args(argv)
    args.output = args.output.expanduser().resolve()
    args.console_log = args.console_log.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.console_log.parent.mkdir(parents=True, exist_ok=True)
    with args.console_log.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(_Tee(sys.stdout, log_file)), redirect_stderr(
            _Tee(sys.stderr, log_file)
        ):
            try:
                result = run(args)
            except Exception as error:
                print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
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
    # Match the verified H2 bridge: qBraid's CPU image can abort during native
    # CUDA-QX/MPI/PyTorch interpreter teardown after all outputs are closed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
