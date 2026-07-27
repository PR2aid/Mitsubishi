"""Dependency-light tests for independent reference-audit certificates."""

from __future__ import annotations

import unittest

import numpy as np

from givens40.reference_audit import (
    array_sha256,
    eigen_residual_certificate,
    frozen_basis_certificate,
    manifest_source,
    mo_coefficients_with_frozen_active_basis,
    occupation_order_certificate,
    orbital_order_certificate,
    orbital_spectrum_certificate,
    orbital_subspace_certificate,
    sector_consistency_certificate,
)


class ReferenceAuditTests(unittest.TestCase):
    @staticmethod
    def _generalized_orbital_fixture():
        overlap = np.diag([2.0, 3.0, 4.0])
        canonical = np.diag(1.0 / np.sqrt(np.diag(overlap)))
        active = canonical[:, :2]
        core = canonical[:, 2:]
        active_energies = np.array([-0.8, 0.2])
        all_energies = np.array([-0.8, 0.2, -1.5])
        fock = (
            overlap
            @ canonical
            @ np.diag(all_energies)
            @ canonical.T
            @ overlap
        )
        return active, active_energies, overlap, fock, core

    def test_full_eigen_certificate_has_explicit_small_residual(self):
        matrix = np.array([[1.0, 0.2], [0.2, 2.0]])
        record = eigen_residual_certificate(
            matrix,
            np.array([1, 0]),
            2,
            -3.0,
            residual_atol=1e-12,
        )
        self.assertTrue(record["all_determinant_addresses_present"])
        self.assertTrue(record["converged"])
        self.assertLess(record["eigen_residual_norm_hartree"], 1e-12)
        self.assertAlmostEqual(
            record["total_energy_hartree"],
            float(np.linalg.eigvalsh(matrix)[0] - 3.0),
            places=14,
        )

    def test_partial_pspace_cannot_be_certified_as_full(self):
        with self.assertRaises(ValueError):
            eigen_residual_certificate(
                np.array([[1.0]]),
                np.array([0]),
                2,
                0.0,
                residual_atol=1e-12,
            )

    def test_occupation_order_is_not_only_a_multiset_check(self):
        good = occupation_order_certificate(
            np.array([2.0, 2.0, 0.0]),
            np.array([2.0, 2.0, 0.0]),
            (2, 2),
            atol=1e-12,
        )
        bad = occupation_order_certificate(
            np.array([2.0, 2.0, 0.0]),
            np.array([2.0, 0.0, 2.0]),
            (2, 2),
            atol=1e-12,
        )
        self.assertTrue(good["passed"])
        self.assertFalse(bad["passed"])

    def test_manifest_source_accepts_public_source_field_names(self):
        source = manifest_source(
            {
                "name": "X_no3",
                "source": {
                    "molecule": "X",
                    "geometry": "H 0 0 0",
                    "basis": "sto-3g",
                    "frozen_core_orbitals": 1,
                    "charge": 0,
                    "spin": 0,
                    "unit": "Angstrom",
                },
            }
        )
        self.assertEqual(source["atom"], "H 0 0 0")
        self.assertEqual(source["ncore"], 1)

    def test_sector_certificate_binds_bundle_electrons_and_dimension(self):
        valid = sector_consistency_certificate(
            np.array([2, 2]),
            (2, 2),
            6,
            225,
        )
        self.assertTrue(valid["passed"])
        self.assertTrue(valid["bundle_matches_manifest"])
        self.assertTrue(valid["declared_dimension_matches_combinatorial"])

        wrong_bundle = sector_consistency_certificate(
            np.array([2, 1]),
            (2, 2),
            6,
            225,
        )
        self.assertFalse(wrong_bundle["passed"])
        self.assertFalse(wrong_bundle["bundle_matches_manifest"])

        wrong_dimension = sector_consistency_certificate(
            np.array([2, 2]),
            (2, 2),
            6,
            224,
        )
        self.assertFalse(wrong_dimension["passed"])
        self.assertFalse(
            wrong_dimension["declared_dimension_matches_combinatorial"]
        )

    def test_frozen_basis_certificate_checks_metric_core_and_fock(self):
        active, energies, overlap, fock, core = (
            self._generalized_orbital_fixture()
        )
        record = frozen_basis_certificate(
            active,
            energies,
            overlap,
            fock,
            core,
            orthonormality_atol=1e-12,
            core_orthogonality_atol=1e-12,
            fock_residual_atol=1e-12,
        )
        self.assertTrue(record["passed"])
        self.assertTrue(record["s_orthonormality"]["passed"])
        self.assertTrue(record["core_active_orthogonality"]["passed"])
        self.assertTrue(
            record["generalized_fock_eigen_residual"]["passed"]
        )
        self.assertLess(
            record["generalized_fock_eigen_residual"][
                "max_abs_residual"
            ],
            1e-14,
        )

    def test_frozen_basis_certificate_rejects_each_orbital_invariant(self):
        active, energies, overlap, fock, core = (
            self._generalized_orbital_fixture()
        )
        nonorthogonal = active.copy()
        nonorthogonal[:, 0] *= 1.01
        bad_metric = frozen_basis_certificate(
            nonorthogonal,
            energies,
            overlap,
            fock,
            core,
            orthonormality_atol=1e-12,
            core_orthogonality_atol=1e-12,
            fock_residual_atol=1.0,
        )
        self.assertFalse(bad_metric["s_orthonormality"]["passed"])

        bad_core = frozen_basis_certificate(
            active,
            energies,
            overlap,
            fock,
            active[:, :1],
            orthonormality_atol=1e-12,
            core_orthogonality_atol=1e-12,
            fock_residual_atol=1e-12,
        )
        self.assertFalse(
            bad_core["core_active_orthogonality"]["passed"]
        )

        bad_fock = fock.copy()
        bad_fock[0, 0] += 0.1
        bad_residual = frozen_basis_certificate(
            active,
            energies,
            overlap,
            bad_fock,
            core,
            orthonormality_atol=1e-12,
            core_orthogonality_atol=1e-12,
            fock_residual_atol=1e-12,
        )
        self.assertFalse(
            bad_residual["generalized_fock_eigen_residual"]["passed"]
        )

    def test_fock_residual_rejects_cross_eigenvalue_rotation(self):
        active, energies, overlap, fock, core = (
            self._generalized_orbital_fixture()
        )
        theta = np.pi / 4.0
        rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )
        rotated = active @ rotation
        subspace = orbital_subspace_certificate(
            rotated, active, overlap, atol=1e-12
        )
        certificate = frozen_basis_certificate(
            rotated,
            energies,
            overlap,
            fock,
            core,
            orthonormality_atol=1e-12,
            core_orthogonality_atol=1e-12,
            fock_residual_atol=1e-12,
        )
        self.assertTrue(subspace["passed"])
        self.assertFalse(
            certificate["generalized_fock_eigen_residual"]["passed"]
        )

    def test_order_invariant_subspace_accepts_virtual_permutation(self):
        overlap = np.eye(3)
        frozen = np.eye(3)[:, :2]
        permuted = frozen[:, ::-1]
        order_sensitive, _ = orbital_order_certificate(
            frozen, permuted, overlap, atol=1e-12
        )
        invariant = orbital_subspace_certificate(
            frozen, permuted, overlap, atol=1e-12
        )
        self.assertFalse(order_sensitive["passed"])
        self.assertFalse(order_sensitive["gating"])
        self.assertTrue(invariant["passed"])

    def test_orbital_spectrum_is_invariant_to_virtual_order(self):
        record = orbital_spectrum_certificate(
            np.array([-0.8, 0.2]),
            np.array([1.5, 0.2, -0.8]),
            atol=1e-12,
        )
        self.assertTrue(record["passed"])
        tampered = orbital_spectrum_certificate(
            np.array([-0.8, 0.25]),
            np.array([1.5, 0.2, -0.8]),
            atol=1e-12,
        )
        self.assertFalse(tampered["passed"])

    def test_frozen_active_block_is_inserted_without_sign_alignment(self):
        regenerated = np.arange(20.0).reshape(4, 5)
        frozen = -np.arange(8.0).reshape(4, 2)
        combined = mo_coefficients_with_frozen_active_basis(
            regenerated, frozen, 1
        )
        np.testing.assert_array_equal(combined[:, 1:3], frozen)
        np.testing.assert_array_equal(combined[:, :1], regenerated[:, :1])
        np.testing.assert_array_equal(combined[:, 3:], regenerated[:, 3:])
        with self.assertRaises(ValueError):
            mo_coefficients_with_frozen_active_basis(
                regenerated, np.zeros((3, 2)), 1
            )

    def test_frozen_array_hash_binds_dtype_shape_and_values(self):
        value = np.arange(6, dtype=np.float64).reshape(2, 3)
        self.assertEqual(array_sha256(value), array_sha256(value.copy()))
        changed = value.copy()
        changed[0, 0] = 1.0
        self.assertNotEqual(array_sha256(value), array_sha256(changed))
        self.assertNotEqual(
            array_sha256(value), array_sha256(value.astype(np.float32))
        )


if __name__ == "__main__":
    unittest.main()
