"""Quantum-selected configuration interaction on the existing determinant sector.

The state-preparation circuit supplies computational-basis determinant
samples.  The Hamiltonian is then diagonalized classically in the unique,
symmetry-valid sampled subspace.  Exact CASCI values are never read during
selection; they are attached only as a post-hoc audit.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

import numpy as np

from .energy import _SigmaCache


def canonical_sequence_seed(sequence: Iterable[Any], base_seed: int = 0) -> int:
    """Stable 32-bit seed derived from a token sequence and a declared seed."""

    records = []
    for token in sequence:
        if hasattr(token, "record"):
            records.append(token.record())
        elif isinstance(token, Mapping):
            records.append(dict(token))
        else:
            records.append(str(token))
    payload = json.dumps(
        {"base_seed": int(base_seed), "sequence": records},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


class QSCISolver:
    """Cached selected-subspace Hamiltonian builder for one CAS problem."""

    def __init__(self, problem: Any, sector: Any):
        self.problem = problem
        self.sector = sector
        self.dimension = int(sector.dimA * sector.dimB)
        self._cache = _SigmaCache(problem)
        self._columns: dict[int, np.ndarray] = {}

    def _column(self, index: int) -> np.ndarray:
        index = int(index)
        if not 0 <= index < self.dimension:
            raise IndexError(index)
        if index not in self._columns:
            basis = np.zeros((self.sector.dimA, self.sector.dimB), dtype=np.float64)
            basis.reshape(-1)[index] = 1.0
            self._columns[index] = np.asarray(
                self._cache.sigma(basis), dtype=np.float64
            ).reshape(-1)
        return self._columns[index]

    def normalized_indices(self, indices: Iterable[int]) -> np.ndarray:
        selected = np.asarray(sorted({int(x) for x in indices}), dtype=np.int64)
        if selected.size == 0:
            raise ValueError("QSCI subspace is empty")
        if selected[0] < 0 or selected[-1] >= self.dimension:
            raise ValueError("QSCI determinant index outside sector")
        return selected

    def subspace_hamiltonian(self, indices: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        selected = self.normalized_indices(indices)
        size = len(selected)
        matrix = np.empty((size, size), dtype=np.float64)
        for column, determinant in enumerate(selected):
            matrix[:, column] = self._column(int(determinant))[selected]
        matrix = 0.5 * (matrix + matrix.T)
        matrix.flat[:: size + 1] += float(self.problem.ecore)
        return selected, matrix

    def energy(self, indices: Iterable[int]) -> dict[str, Any]:
        selected, matrix = self.subspace_hamiltonian(indices)
        eigenvalues = np.linalg.eigvalsh(matrix)
        energy = float(eigenvalues[0])
        reference = self.problem.e_casci
        if reference is not None and energy < float(reference) - 1e-9:
            raise RuntimeError(
                "QSCI variational bound violated: "
                f"{energy:.12f} < CASCI {float(reference):.12f}"
            )
        return {
            "energy_hartree": energy,
            "determinant_count": int(len(selected)),
            "determinant_indices": [int(x) for x in selected],
            "error_mha": (
                None if reference is None else 1000.0 * (energy - float(reference))
            ),
        }

    @staticmethod
    def probabilities(state: np.ndarray) -> np.ndarray:
        flat = np.asarray(state).reshape(-1)
        probabilities = np.square(np.abs(flat)).astype(np.float64)
        total = float(probabilities.sum())
        if total <= 0:
            raise ValueError("zero-norm state")
        return probabilities / total

    def top_k(
        self,
        state: np.ndarray,
        k: int,
        *,
        mandatory: Iterable[int] = (),
    ) -> dict[str, Any]:
        probabilities = self.probabilities(state)
        k = max(1, min(int(k), self.dimension))
        mandatory_set = {int(x) for x in mandatory}
        if len(mandatory_set) > k:
            raise ValueError("mandatory determinants exceed the total QSCI budget")
        order = np.lexsort((np.arange(self.dimension), -probabilities))
        selected = set(mandatory_set)
        for index in order:
            if len(selected) >= k:
                break
            selected.add(int(index))
        result = self.energy(selected)
        idx = np.asarray(result["determinant_indices"], dtype=np.int64)
        result.update(
            {
                "selection": "deterministic_top_k_probability",
                "requested_k": int(k),
                "captured_probability_mass": float(probabilities[idx].sum()),
                "maximum_determinant_probability": float(probabilities.max()),
                "mandatory_determinants": sorted(mandatory_set),
                "determinant_budget_semantics": (
                    "requested_k is the total selected-subspace budget, "
                    "including mandatory determinants"
                ),
            }
        )
        return result

    def sample(
        self,
        state: np.ndarray,
        shots: int,
        seed: int,
        *,
        mandatory: Iterable[int] = (),
    ) -> dict[str, Any]:
        if int(shots) <= 0:
            raise ValueError("shots must be positive")
        probabilities = self.probabilities(state)
        rng = np.random.default_rng(int(seed))
        draws = rng.choice(self.dimension, size=int(shots), p=probabilities)
        selected = set(int(x) for x in np.unique(draws))
        selected.update(int(x) for x in mandatory)
        result = self.energy(selected)
        idx = np.asarray(result["determinant_indices"], dtype=np.int64)
        result.update(
            {
                "selection": "seeded_computational_basis_sampling",
                "shots": int(shots),
                "seed": int(seed),
                "observed_unique_determinants": int(len(np.unique(draws))),
                "captured_exact_probability_mass": float(probabilities[idx].sum()),
                "maximum_determinant_probability": float(probabilities.max()),
            }
        )
        return result

    def sample_capped(
        self,
        state: np.ndarray,
        shots: int,
        seed: int,
        *,
        max_determinants: int,
        mandatory: Iterable[int] = (),
        min_count: int = 1,
    ) -> dict[str, Any]:
        """Sample determinants and enforce a fixed total QSCI budget.

        ``max_determinants`` includes every mandatory determinant.  Measured
        determinants are ranked by descending observed count (then index),
        and only the remaining slots are retained.  This prevents a broader
        or noisier state from receiving extra classical diagonalization work
        merely because it produced more unique bit strings.
        """

        if int(shots) <= 0:
            raise ValueError("shots must be positive")
        if int(max_determinants) <= 0:
            raise ValueError("max_determinants must be positive")
        if int(min_count) <= 0:
            raise ValueError("min_count must be positive")

        probabilities = self.probabilities(state)
        rng = np.random.default_rng(int(seed))
        draws = rng.choice(self.dimension, size=int(shots), p=probabilities)
        observed, counts = np.unique(draws, return_counts=True)
        frequencies = {
            int(index): int(count)
            for index, count in zip(observed, counts)
            if int(count) >= int(min_count)
        }
        mandatory_set = {int(x) for x in mandatory}
        if len(mandatory_set) > int(max_determinants):
            raise ValueError("mandatory determinants exceed the total QSCI budget")
        selected = set(mandatory_set)
        ranked = sorted(frequencies, key=lambda x: (-frequencies[x], x))
        for index in ranked:
            if len(selected) >= int(max_determinants):
                break
            selected.add(int(index))

        result = self.energy(selected)
        retained = np.asarray(result["determinant_indices"], dtype=np.int64)
        result.update(
            {
                "selection": "seeded_sampling_count_ranked_fixed_total_budget",
                "shots": int(shots),
                "seed": int(seed),
                "observed_unique_determinants": int(len(observed)),
                "retained_determinants": int(len(retained)),
                "captured_exact_probability_mass": float(
                    probabilities[retained].sum()
                ),
                "maximum_determinant_probability": float(probabilities.max()),
                "selection_min_count": int(min_count),
                "max_determinants": int(max_determinants),
                "mandatory_determinants": sorted(mandatory_set),
                "determinant_budget_semantics": (
                    "max_determinants is the total selected-subspace budget, "
                    "including mandatory determinants"
                ),
            }
        )
        return result

    def sampled_trials_capped(
        self,
        state: np.ndarray,
        shots: int,
        seeds: Iterable[int],
        *,
        max_determinants: int,
        mandatory: Iterable[int] = (),
        min_count: int = 1,
    ) -> dict[str, Any]:
        """Repeated fixed-budget sampling for a predeclared held-out endpoint."""

        trials = [
            self.sample_capped(
                state,
                int(shots),
                int(seed),
                max_determinants=int(max_determinants),
                mandatory=mandatory,
                min_count=int(min_count),
            )
            for seed in seeds
        ]
        errors = np.asarray(
            [x["error_mha"] for x in trials if x["error_mha"] is not None],
            dtype=np.float64,
        )
        dimensions = np.asarray([x["determinant_count"] for x in trials])
        return {
            "shots": int(shots),
            "trials": trials,
            "trial_count": len(trials),
            "error_mha_median": None if not len(errors) else float(np.median(errors)),
            "error_mha_mean": None if not len(errors) else float(np.mean(errors)),
            "error_mha_q25": None if not len(errors) else float(np.quantile(errors, 0.25)),
            "error_mha_q75": None if not len(errors) else float(np.quantile(errors, 0.75)),
            "error_mha_min": None if not len(errors) else float(np.min(errors)),
            "error_mha_max": None if not len(errors) else float(np.max(errors)),
            "chemical_accuracy_success_fraction": (
                None
                if not len(errors)
                else float(np.mean(errors <= 1.6))
            ),
            "determinant_count_median": float(np.median(dimensions)),
            "max_determinants": int(max_determinants),
            "selection_min_count": int(min_count),
        }

    def sampled_trials(
        self,
        state: np.ndarray,
        shots: int,
        seeds: Iterable[int],
        *,
        mandatory: Iterable[int] = (),
    ) -> dict[str, Any]:
        trials = [
            self.sample(state, int(shots), int(seed), mandatory=mandatory)
            for seed in seeds
        ]
        errors = np.asarray(
            [x["error_mha"] for x in trials if x["error_mha"] is not None],
            dtype=np.float64,
        )
        dimensions = np.asarray([x["determinant_count"] for x in trials])
        return {
            "shots": int(shots),
            "trials": trials,
            "trial_count": len(trials),
            "error_mha_median": None if not len(errors) else float(np.median(errors)),
            "error_mha_mean": None if not len(errors) else float(np.mean(errors)),
            "error_mha_min": None if not len(errors) else float(np.min(errors)),
            "error_mha_max": None if not len(errors) else float(np.max(errors)),
            "determinant_count_median": float(np.median(dimensions)),
        }

    def index_from_register_integer(self, value: int) -> int | None:
        """Map one blocked-spin measured register integer into sector order."""

        value = int(value)
        mask = (1 << int(self.sector.norb)) - 1
        alpha = value & mask
        beta = value >> int(self.sector.norb)
        if alpha.bit_count() != int(self.sector.na) or beta.bit_count() != int(
            self.sector.nb
        ):
            return None
        ia = int(np.searchsorted(self.sector.alpha.strs, alpha))
        ib = int(np.searchsorted(self.sector.beta.strs, beta))
        if (
            ia >= self.sector.dimA
            or ib >= self.sector.dimB
            or int(self.sector.alpha.strs[ia]) != alpha
            or int(self.sector.beta.strs[ib]) != beta
        ):
            return None
        return ia * self.sector.dimB + ib

    def from_counts(
        self,
        counts: Mapping[Any, Any],
        *,
        mandatory: Iterable[int] = (),
        min_count: int = 1,
        max_determinants: int | None = None,
    ) -> dict[str, Any]:
        mandatory_set = set(int(x) for x in mandatory)
        if int(min_count) <= 0:
            raise ValueError("min_count must be positive")
        frequencies: dict[int, int] = {}
        valid_shots = 0
        invalid_shots = 0
        for raw, count_raw in counts.items():
            count = int(round(float(count_raw)))
            if isinstance(raw, int):
                value = int(raw)
            else:
                key = str(raw).replace(" ", "").replace("_", "")
                value = int(key, 16) if key.lower().startswith("0x") else int(key, 2)
            index = self.index_from_register_integer(value)
            if index is None:
                invalid_shots += count
            else:
                valid_shots += count
                frequencies[index] = frequencies.get(index, 0) + count
        ranked = sorted(frequencies, key=lambda x: (-frequencies[x], x))
        ranked = [x for x in ranked if frequencies[x] >= int(min_count)]
        if max_determinants is not None:
            if int(max_determinants) <= 0:
                raise ValueError("max_determinants must be positive")
            if len(mandatory_set) > int(max_determinants):
                raise ValueError(
                    "mandatory determinants exceed the total QSCI budget"
                )
            selected = set(mandatory_set)
            for index in ranked:
                if len(selected) >= int(max_determinants):
                    break
                selected.add(int(index))
        else:
            selected = mandatory_set | set(ranked)
        result = self.energy(selected)
        result.update(
            {
                "selection": "measured_counts_with_particle_sector_filter",
                "valid_sector_shots": int(valid_shots),
                "invalid_sector_shots": int(invalid_shots),
                "valid_sector_fraction": (
                    valid_shots / (valid_shots + invalid_shots)
                    if valid_shots + invalid_shots
                    else 0.0
                ),
                "selection_min_count": int(min_count),
                "selection_max_determinants": (
                    None if max_determinants is None else int(max_determinants)
                ),
                "retained_measured_determinants": int(
                    len(set(selected) - mandatory_set)
                ),
                "determinant_budget_semantics": (
                    "selection_max_determinants is the total selected-subspace "
                    "budget, including mandatory determinants"
                    if max_determinants is not None
                    else "uncapped selected subspace"
                ),
            }
        )
        return result
