#!/usr/bin/env python3
"""Apply the predeclared held-out gate to exact-energy vs QSCI GQE arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


MINIMUM_MHA = 0.1
RAW_DEGRADATION_LIMIT_MHA = 1.6
BOOTSTRAP_SEED = 2607
BOOTSTRAP_REPLICATES = 20_000


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(path)
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trial_errors(record: dict[str, Any]) -> dict[int, float]:
    return {
        int(item["seed"]): float(item["error_mha"])
        for item in record["finite_shot"]["trials"]
    }


def median_error(record: dict[str, Any]) -> float:
    return float(record["finite_shot"]["error_mha_median"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", type=Path, required=True)
    parser.add_argument("--qsci", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    exact_path = args.exact.expanduser().resolve()
    qsci_path = args.qsci.expanduser().resolve()
    exact = load(exact_path)
    qsci = load(qsci_path)

    if exact["gqe_config"]["objective"] != "exact-energy":
        raise ValueError("--exact does not contain the exact-energy arm")
    if qsci["gqe_config"]["objective"] != "qsci-topk":
        raise ValueError("--qsci does not contain the qsci-topk arm")

    parity_values = {
        "system_label": (
            exact["system"]["label"],
            qsci["system"]["label"],
        ),
        "scientific_fingerprint_sha256": (
            exact["system"]["scientific_fingerprint_sha256"],
            qsci["system"]["scientific_fingerprint_sha256"],
        ),
        "training_seed": (
            exact["reproducibility"]["seed"],
            qsci["reproducibility"]["seed"],
        ),
        "left_block": (
            exact["residual_vocabulary"]["left_block"],
            qsci["residual_vocabulary"]["left_block"],
        ),
        "right_block": (
            exact["residual_vocabulary"]["right_block"],
            qsci["residual_vocabulary"]["right_block"],
        ),
        "pool_size": (
            exact["residual_vocabulary"]["pool_size"],
            qsci["residual_vocabulary"]["pool_size"],
        ),
        "angle_bins": (
            exact["residual_vocabulary"]["angle_bins"],
            qsci["residual_vocabulary"]["angle_bins"],
        ),
        "sequence_length": (
            exact["gqe_config"]["ngates"],
            qsci["gqe_config"]["ngates"],
        ),
        "candidate_evaluation_budget": (
            exact["gqe_config"]["candidate_evaluation_budget"],
            qsci["gqe_config"]["candidate_evaluation_budget"],
        ),
        "phi_max": (
            exact["cutting_budget"]["phi_max"],
            qsci["cutting_budget"]["phi_max"],
        ),
        "backbone_raw_energy": (
            exact["backbone_objective"]["raw_state_energy_hartree"],
            qsci["backbone_objective"]["raw_state_energy_hartree"],
        ),
        "heldout_protocol": (
            exact["heldout_evaluation"]["protocol"],
            qsci["heldout_evaluation"]["protocol"],
        ),
    }
    parity = {
        name: {"match": left == right, "exact": left, "qsci": right}
        for name, (left, right) in parity_values.items()
    }
    parity_passed = all(item["match"] for item in parity.values())

    exact_guarded = exact["heldout_evaluation"]["guarded_state"]
    qsci_guarded = qsci["heldout_evaluation"]["guarded_state"]
    exact_errors = trial_errors(exact_guarded)
    qsci_errors = trial_errors(qsci_guarded)
    if set(exact_errors) != set(qsci_errors):
        raise ValueError("held-out seed sets differ")
    seeds = sorted(exact_errors)
    deltas = np.asarray(
        [qsci_errors[seed] - exact_errors[seed] for seed in seeds],
        dtype=np.float64,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = deltas[
        rng.integers(0, len(deltas), size=(BOOTSTRAP_REPLICATES, len(deltas)))
    ]
    bootstrap_medians = np.median(sampled, axis=1)
    bootstrap_ci = np.quantile(bootstrap_medians, [0.025, 0.975])
    median_delta = float(np.median(deltas))

    qsci_controls = qsci["heldout_evaluation"]["matched_policy_controls"]
    qsci_median = median_error(qsci_guarded)
    identity_median = median_error(qsci_controls["identity"]["heldout"])
    random_median = median_error(qsci_controls["random_feasible"]["heldout"])
    greedy_median = median_error(
        qsci_controls["stratified_pool_greedy"]["heldout"]
    )
    raw_exact = float(exact_guarded["raw_state"]["error_mha"])
    raw_qsci = float(qsci_guarded["raw_state"]["error_mha"])

    criteria = {
        "all_protocol_parity_checks_pass": parity_passed,
        "paired_median_improves_by_at_least_0p1_mha": (
            median_delta <= -MINIMUM_MHA
        ),
        "paired_bootstrap_95pct_upper_bound_below_zero": (
            float(bootstrap_ci[1]) < 0.0
        ),
        "beats_qsci_identity_by_at_least_0p1_mha": (
            qsci_median <= identity_median - MINIMUM_MHA
        ),
        "beats_qsci_random_by_at_least_0p1_mha": (
            qsci_median <= random_median - MINIMUM_MHA
        ),
        "beats_qsci_greedy_by_at_least_0p1_mha": (
            qsci_median <= greedy_median - MINIMUM_MHA
        ),
        "raw_state_degradation_within_chemical_accuracy": (
            raw_qsci - raw_exact <= RAW_DEGRADATION_LIMIT_MHA
        ),
        "zero_budget_mask_violations": (
            int(exact["cutting_budget"]["trajectory_mask_violations"]) == 0
            and int(qsci["cutting_budget"]["trajectory_mask_violations"]) == 0
        ),
    }
    promoted = all(criteria.values())
    output = {
        "schema_version": 1,
        "status": "COMPLETED",
        "system": exact["system"]["label"],
        "input_files": {
            "exact_energy": {
                "file": exact_path.name,
                "sha256": sha256(exact_path),
            },
            "qsci_topk": {
                "file": qsci_path.name,
                "sha256": sha256(qsci_path),
            },
        },
        "protocol_parity": parity,
        "paired_primary": {
            "delta_definition": "QSCI-arm error minus exact-energy-arm error",
            "seeds": seeds,
            "deltas_mha": [float(x) for x in deltas],
            "median_delta_mha": median_delta,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_95pct_ci_mha": [float(x) for x in bootstrap_ci],
            "qsci_win_fraction": float(np.mean(deltas < 0.0)),
        },
        "heldout_median_errors_mha": {
            "exact_energy_transformer": median_error(exact_guarded),
            "qsci_topk_transformer": qsci_median,
            "qsci_identity": identity_median,
            "qsci_random_feasible": random_median,
            "qsci_stratified_pool_greedy": greedy_median,
        },
        "raw_state_errors_mha": {
            "exact_energy_transformer": raw_exact,
            "qsci_topk_transformer": raw_qsci,
            "qsci_minus_exact": raw_qsci - raw_exact,
        },
        "promotion_rule": {
            "minimum_meaningful_improvement_mha": MINIMUM_MHA,
            "raw_state_degradation_limit_mha": RAW_DEGRADATION_LIMIT_MHA,
            "criteria": criteria,
        },
        "decision": (
            "PROMOTE_QSCI_OBJECTIVE"
            if promoted
            else "DO_NOT_PROMOTE_QSCI_OBJECTIVE_RETAIN_EXACT_ENERGY"
        ),
        "promoted": promoted,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output["decision"])
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
