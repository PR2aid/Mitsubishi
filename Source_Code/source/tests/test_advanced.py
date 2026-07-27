"""Regression tests for the deterministic advanced-method arm."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


SOURCE = Path(__file__).resolve().parents[1]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from givens40.adaptive_topology import (
    partition_diagnostics,
    schmidt_spectrum_spatial_partition,
    transfer_parameters,
)
from givens40.chemistry import build_cas
from givens40.frozen_problem import load_named_problem
from givens40.qsci import QSCISolver, canonical_sequence_seed
from givens40.runner import AnsatzConfig, SectorCircuit


class AdvancedMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("GQE_FROZEN_INPUT_DIR")
        if not root:
            raise unittest.SkipTest("GQE_FROZEN_INPUT_DIR is not set")
        cls.frozen = Path(root)

    def test_frozen_problem_replay_is_bitwise_identical(self):
        a = load_named_problem(self.frozen, "BeH2", 3)
        b = load_named_problem(self.frozen, "BeH2", 3)
        self.assertTrue(np.array_equal(a.h1e, b.h1e))
        self.assertTrue(np.array_equal(a.eri, b.eri))
        self.assertEqual(
            a.meta["scientific_fingerprint_sha256"],
            b.meta["scientific_fingerprint_sha256"],
        )

    def test_nested_orbitals_are_exact_prefixes(self):
        small = load_named_problem(self.frozen, "BeH2", 3)
        large = load_named_problem(self.frozen, "BeH2", 6)
        self.assertTrue(
            np.array_equal(
                small.meta["mo_coeff_active"],
                large.meta["mo_coeff_active"][:, : small.norb],
            )
        )

    def test_intermediate_validation_rung_is_derived_without_scf(self):
        middle = load_named_problem(self.frozen, "BeH2", 5)
        parent = load_named_problem(self.frozen, "BeH2", 6)
        self.assertTrue(middle.meta["derived_without_scf"])
        self.assertEqual(middle.meta["derived_parent_norb"], 6)
        self.assertTrue(np.array_equal(middle.h1e, parent.h1e[:5, :5]))
        self.assertTrue(
            np.array_equal(middle.eri, parent.eri[:5, :5, :5, :5])
        )

    def test_qsci_full_sector_recovers_casci_and_is_variational(self):
        problem = load_named_problem(self.frozen, "BeH2", 3)
        circuit = SectorCircuit(problem, AnsatzConfig())
        solver = QSCISolver(problem, circuit.sector)
        full = solver.energy(range(solver.dimension))
        self.assertAlmostEqual(full["energy_hartree"], problem.e_casci, places=10)
        subset = solver.energy([0])
        self.assertGreaterEqual(subset["energy_hartree"], problem.e_casci - 1e-10)

    def test_qsci_nested_subspaces_are_monotone(self):
        problem = load_named_problem(self.frozen, "BeH2", 3)
        circuit = SectorCircuit(problem, AnsatzConfig())
        solver = QSCISolver(problem, circuit.sector)
        e1 = solver.energy([0])["energy_hartree"]
        e2 = solver.energy([0, 1])["energy_hartree"]
        e3 = solver.energy([0, 1, 2])["energy_hartree"]
        self.assertLessEqual(e2, e1 + 1e-12)
        self.assertLessEqual(e3, e2 + 1e-12)

    def test_qsci_topk_counts_mandatory_determinant_inside_k(self):
        problem = load_named_problem(self.frozen, "BeH2", 3)
        circuit = SectorCircuit(problem, AnsatzConfig())
        solver = QSCISolver(problem, circuit.sector)
        state = np.zeros((circuit.sector.dimA, circuit.sector.dimB))
        state.reshape(-1)[1] = 1.0
        state.reshape(-1)[2] = 0.5
        result = solver.top_k(state, 2, mandatory=(0,))
        self.assertEqual(result["determinant_count"], 2)
        self.assertEqual(result["determinant_indices"], [0, 1])
        self.assertIn("total selected-subspace budget",
                      result["determinant_budget_semantics"])

    def test_qsci_seed_and_sampling_are_reproducible(self):
        problem = load_named_problem(self.frozen, "BeH2", 3)
        circuit = SectorCircuit(problem, AnsatzConfig())
        params = circuit.init_params(17)
        state = circuit.forward(params, problem.hdiag()).detach().numpy()
        solver = QSCISolver(problem, circuit.sector)
        a = solver.sample(state, 100, 7, mandatory=(0,))
        b = solver.sample(state, 100, 7, mandatory=(0,))
        self.assertEqual(a, b)
        self.assertEqual(canonical_sequence_seed([{"a": 1}], 3),
                         canonical_sequence_seed([{"a": 1}], 3))

    def test_qsci_capped_sampling_uses_one_fixed_total_budget(self):
        problem = load_named_problem(self.frozen, "BeH2", 3)
        circuit = SectorCircuit(problem, AnsatzConfig())
        params = circuit.init_params(17)
        state = circuit.forward(params, problem.hdiag()).detach().numpy()
        solver = QSCISolver(problem, circuit.sector)
        a = solver.sample_capped(
            state, 500, 91, max_determinants=3, mandatory=(0,)
        )
        b = solver.sample_capped(
            state, 500, 91, max_determinants=3, mandatory=(0,)
        )
        self.assertEqual(a, b)
        self.assertLessEqual(a["determinant_count"], 3)
        self.assertIn(0, a["determinant_indices"])
        self.assertEqual(a["max_determinants"], 3)
        self.assertIn("total selected-subspace budget",
                      a["determinant_budget_semantics"])

    def test_schmidt_product_state_and_complement_agree(self):
        problem = load_named_problem(self.frozen, "BeH2", 3)
        circuit = SectorCircuit(problem, AnsatzConfig())
        state = circuit.sector.initial_state(problem.hdiag())
        left = schmidt_spectrum_spatial_partition(circuit.sector, state, [0])
        right = schmidt_spectrum_spatial_partition(circuit.sector, state, [1, 2])
        self.assertAlmostEqual(float(left[0]), 1.0, places=14)
        self.assertAlmostEqual(float(right[0]), 1.0, places=14)
        self.assertTrue(np.allclose(left, right, atol=1e-14))

    def test_partition_override_preserves_budget_metadata(self):
        problem = load_named_problem(self.frozen, "BeH2", 6)
        config = AnsatzConfig(
            topology="partitioned",
            partition_override=[0, 2, 3],
            phi_max=15.0,
            beta_cap=0.05,
            beta_cap_double=0.25,
        )
        circuit = SectorCircuit(problem, config)
        self.assertEqual(circuit.topo.left_block, [0, 2, 3])
        self.assertIsNotNone(circuit.topo.u_budget)
        for pair, is_cross in zip(circuit.topo.pairs, circuit.topo.cross_mask):
            expected = (pair[0] in {0, 2, 3}) != (pair[1] in {0, 2, 3})
            self.assertEqual(bool(is_cross), expected)

    def test_topology_transfer_preserves_shared_effective_angles(self):
        problem = load_named_problem(self.frozen, "BeH2", 6)
        source = SectorCircuit(
            problem,
            AnsatzConfig(
                topology="partitioned",
                partition_override=[0, 1, 5],
                beta_cap=0.05,
                beta_cap_double=0.25,
            ),
        )
        target = SectorCircuit(
            problem,
            AnsatzConfig(
                topology="partitioned",
                partition_override=[0, 2, 3],
                beta_cap=0.05,
                beta_cap_double=0.25,
            ),
        )
        source_params = source.init_params(17)
        mapped, audit = transfer_parameters(
            source, target, source_params, seed=17, new_pair_scale=0.0
        )
        source_eff = source._effective_angles(source_params)
        target_eff = target._effective_angles(mapped)
        spos = {tuple(pair): i for i, pair in enumerate(source.topo.pairs)}
        tpos = {tuple(pair): i for i, pair in enumerate(target.topo.pairs)}
        target_cross = set(int(x) for x in target.cross_idx)
        for pair in set(spos) & set(tpos):
            for key in set(source_eff) & set(target_eff):
                expected = source_eff[key].select(1, spos[pair])
                if tpos[pair] in target_cross:
                    cap = 0.25 if key == "doubles" else 0.05
                    expected = torch.clamp(
                        expected, -cap * (1 - 1e-9), cap * (1 - 1e-9)
                    )
                self.assertTrue(
                    torch.allclose(
                        expected,
                        target_eff[key].select(1, tpos[pair]),
                        atol=1e-10,
                        rtol=0,
                    )
                )
        self.assertGreater(audit["copied_pair_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
