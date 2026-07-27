"""Independent regeneration audit for the frozen molecular references.

The frozen ``.npz`` bundles are the deterministic inputs used by the simulator.
Checksum verification proves that those inputs have not changed, but it does
not independently establish that their RHF and CASCI reference values follow
from the declared geometry, basis, electron sector, and orbital ordering.

This module rebuilds each unique mean-field problem from the manifest and then
audits the checksum-bound frozen active orbitals directly.  It verifies their
S-orthonormality, orthogonality to the regenerated frozen core, and generalized
Fock eigen residual before regenerating the active-space integrals *in that
frozen basis*.  This is invariant to harmless sign, permutation, or rotation
choices inside regenerated near-degenerate virtual subspaces.

The regenerated integrals are finally diagonalized in the *complete*
determinant ``p-space``.  The resulting certificate includes an explicit
eigen-residual; it never treats an iterative FCI energy without a convergence
certificate as an independently verified reference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


# These controls must be in force before NumPy/PySCF load a BLAS runtime.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np


DEFAULT_TOLERANCES: dict[str, float] = {
    "rhf_energy_hartree": 5e-10,
    "casci_energy_hartree": 1e-9,
    "ecore_hartree": 1e-9,
    "integral_max_abs": 2e-9,
    "mo_energy_max_abs": 2e-9,
    "occupation_max_abs": 1e-12,
    "orbital_overlap": 2e-7,
    "orbital_orthonormality": 2e-8,
    "core_active_orthogonality": 2e-8,
    "generalized_fock_residual": 2e-8,
    "eigen_residual_hartree": 1e-10,
}


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash one numerical array using the frozen-input schema."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def manifest_source(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one manifest problem's public molecular specification."""

    source = dict(record.get("source") or {})
    meta = dict(record.get("meta") or {})
    normalized = {
        "molecule": source.get("molecule", meta.get("molecule")),
        "atom": source.get("geometry", meta.get("atom")),
        "basis": source.get("basis", meta.get("basis")),
        "charge": int(source.get("charge", meta.get("charge", 0))),
        "spin": int(source.get("spin", meta.get("spin", 0))),
        "unit": source.get("unit", meta.get("unit", "Angstrom")),
        "ncore": int(
            source.get("frozen_core_orbitals", meta.get("ncore", 0))
        ),
    }
    missing = [key for key in ("molecule", "atom", "basis") if not normalized[key]]
    if missing:
        raise ValueError(
            f"manifest source is missing {', '.join(missing)} for "
            f"{record.get('name', '<unnamed>')}"
        )
    return normalized


def source_key(source: dict[str, Any]) -> tuple[Any, ...]:
    """Return a stable cache key for one unique mean-field calculation."""

    return tuple(
        source[key]
        for key in ("molecule", "atom", "basis", "charge", "spin", "unit", "ncore")
    )


def expected_rhf_occupations(
    norb: int, nelec_alpha_beta: Iterable[int]
) -> np.ndarray:
    """Return the canonical RHF/ROHF occupation prefix for an active space."""

    na, nb = (int(value) for value in nelec_alpha_beta)
    if not (0 <= nb <= na <= int(norb)):
        raise ValueError("invalid alpha/beta electron sector")
    expected = np.zeros(int(norb), dtype=np.float64)
    expected[:nb] = 2.0
    expected[nb:na] = 1.0
    return expected


def sector_consistency_certificate(
    bundle_nelec: np.ndarray,
    manifest_nelec: Iterable[int],
    norb: int,
    declared_sector_dimension: int,
) -> dict[str, Any]:
    """Bind bundle electrons and declared dimension to the manifest sector."""

    manifest = tuple(int(value) for value in manifest_nelec)
    if len(manifest) != 2:
        raise ValueError("manifest electron sector must contain alpha and beta")
    na, nb = manifest
    if not (0 <= nb <= na <= int(norb)):
        raise ValueError("invalid manifest alpha/beta electron sector")
    bundle = np.asarray(bundle_nelec)
    bundle_matches = bundle.shape == (2,) and np.array_equal(
        bundle, np.asarray(manifest)
    )
    combinatorial_dimension = math.comb(int(norb), na) * math.comb(
        int(norb), nb
    )
    declared_matches = (
        int(declared_sector_dimension) == combinatorial_dimension
    )
    return {
        "passed": bool(bundle_matches and declared_matches),
        "manifest_nelec_alpha_beta": list(manifest),
        "bundle_nelec_alpha_beta": bundle.tolist(),
        "bundle_matches_manifest": bool(bundle_matches),
        "declared_sector_dimension": int(declared_sector_dimension),
        "combinatorial_sector_dimension": int(combinatorial_dimension),
        "declared_dimension_matches_combinatorial": bool(declared_matches),
    }


def occupation_order_certificate(
    frozen: np.ndarray,
    regenerated: np.ndarray,
    nelec_alpha_beta: Iterable[int],
    *,
    atol: float,
) -> dict[str, Any]:
    """Certify both frozen/regenerated occupation values and their ordering."""

    frozen = np.asarray(frozen, dtype=np.float64)
    regenerated = np.asarray(regenerated, dtype=np.float64)
    if frozen.shape != regenerated.shape:
        return {
            "passed": False,
            "shape_match": False,
            "frozen_shape": list(frozen.shape),
            "regenerated_shape": list(regenerated.shape),
        }
    expected = expected_rhf_occupations(frozen.size, nelec_alpha_beta)
    difference = float(np.max(np.abs(frozen - regenerated), initial=0.0))
    frozen_expected = float(np.max(np.abs(frozen - expected), initial=0.0))
    regenerated_expected = float(
        np.max(np.abs(regenerated - expected), initial=0.0)
    )
    passed = (
        difference <= atol
        and frozen_expected <= atol
        and regenerated_expected <= atol
    )
    return {
        "passed": bool(passed),
        "shape_match": True,
        "max_abs_frozen_vs_regenerated": difference,
        "max_abs_frozen_vs_expected_order": frozen_expected,
        "max_abs_regenerated_vs_expected_order": regenerated_expected,
        "expected": expected.tolist(),
        "frozen": frozen.tolist(),
        "regenerated": regenerated.tolist(),
        "absolute_tolerance": float(atol),
    }


def orbital_order_certificate(
    frozen_coefficients: np.ndarray,
    regenerated_coefficients: np.ndarray,
    ao_overlap: np.ndarray,
    *,
    atol: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Check sign-insensitive one-to-one orbital order and return sign alignment."""

    frozen = np.asarray(frozen_coefficients, dtype=np.float64)
    regenerated = np.asarray(regenerated_coefficients, dtype=np.float64)
    overlap = np.asarray(ao_overlap, dtype=np.float64)
    if frozen.shape != regenerated.shape:
        raise ValueError(
            "frozen and regenerated active-orbital coefficient shapes differ"
        )
    metric_overlap = frozen.T @ overlap @ regenerated
    diagonal = np.diag(metric_overlap)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    absolute_overlap = np.abs(metric_overlap)
    off_diagonal = absolute_overlap.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    min_diagonal = float(np.min(np.abs(diagonal), initial=1.0))
    max_off_diagonal = float(np.max(off_diagonal, initial=0.0))
    passed = min_diagonal >= 1.0 - atol and max_off_diagonal <= atol
    return (
        {
            "passed": bool(passed),
            "gating": False,
            "scope": (
                "order-sensitive sign-aligned diagnostic only; regenerated "
                "virtual ordering is not a certification requirement"
            ),
            "minimum_absolute_diagonal_overlap": min_diagonal,
            "maximum_absolute_offdiagonal_overlap": max_off_diagonal,
            "absolute_tolerance": float(atol),
            "sign_alignment": signs.astype(int).tolist(),
        },
        signs,
    )


def orbital_subspace_certificate(
    frozen_coefficients: np.ndarray,
    regenerated_coefficients: np.ndarray,
    ao_overlap: np.ndarray,
    *,
    atol: float,
) -> dict[str, Any]:
    """Report an order-invariant principal-overlap subspace diagnostic."""

    frozen = np.asarray(frozen_coefficients, dtype=np.float64)
    regenerated = np.asarray(regenerated_coefficients, dtype=np.float64)
    overlap = np.asarray(ao_overlap, dtype=np.float64)
    shape_valid = (
        frozen.ndim == 2
        and regenerated.ndim == 2
        and overlap.ndim == 2
        and overlap.shape[0] == overlap.shape[1]
        and frozen.shape[0] == overlap.shape[0]
        and regenerated.shape[0] == overlap.shape[0]
        and regenerated.shape[1] >= frozen.shape[1]
    )
    finite = bool(
        shape_valid
        and np.all(np.isfinite(frozen))
        and np.all(np.isfinite(regenerated))
        and np.all(np.isfinite(overlap))
    )
    if not finite:
        return {
            "passed": False,
            "shape_valid": bool(shape_valid),
            "finite": False,
            "frozen_shape": list(frozen.shape),
            "regenerated_shape": list(regenerated.shape),
            "absolute_tolerance": float(atol),
        }
    if frozen.shape[1] == 0:
        singular_values = np.empty(0, dtype=np.float64)
        minimum = 1.0
        maximum = 1.0
    else:
        metric_overlap = frozen.T @ overlap @ regenerated
        singular_values = np.linalg.svd(
            metric_overlap, compute_uv=False
        )
        minimum = float(np.min(singular_values))
        maximum = float(np.max(singular_values))
    passed = minimum >= 1.0 - atol and maximum <= 1.0 + atol
    return {
        "passed": bool(passed),
        "shape_valid": True,
        "finite": True,
        "frozen_dimension": int(frozen.shape[1]),
        "regenerated_dimension": int(regenerated.shape[1]),
        "minimum_principal_overlap": minimum,
        "maximum_principal_overlap": maximum,
        "maximum_principal_angle_radians": float(
            math.acos(np.clip(minimum, -1.0, 1.0))
        ),
        "singular_values": singular_values.tolist(),
        "absolute_tolerance": float(atol),
    }


def frozen_basis_certificate(
    frozen_coefficients: np.ndarray,
    frozen_energies: np.ndarray,
    ao_overlap: np.ndarray,
    ao_fock: np.ndarray,
    core_coefficients: np.ndarray,
    *,
    orthonormality_atol: float,
    core_orthogonality_atol: float,
    fock_residual_atol: float,
) -> dict[str, Any]:
    """Certify frozen active orbitals without comparing their column order."""

    coefficients = np.asarray(frozen_coefficients, dtype=np.float64)
    energies = np.asarray(frozen_energies, dtype=np.float64)
    overlap = np.asarray(ao_overlap, dtype=np.float64)
    fock = np.asarray(ao_fock, dtype=np.float64)
    core = np.asarray(core_coefficients, dtype=np.float64)
    shape_valid = (
        coefficients.ndim == 2
        and energies.shape == (coefficients.shape[1],)
        and overlap.shape == (coefficients.shape[0],) * 2
        and fock.shape == overlap.shape
        and core.ndim == 2
        and core.shape[0] == coefficients.shape[0]
    )
    finite = bool(
        shape_valid
        and np.all(np.isfinite(coefficients))
        and np.all(np.isfinite(energies))
        and np.all(np.isfinite(overlap))
        and np.all(np.isfinite(fock))
        and np.all(np.isfinite(core))
    )
    if not finite:
        return {
            "passed": False,
            "shape_valid": bool(shape_valid),
            "finite": False,
            "coefficient_shape": list(coefficients.shape),
            "energy_shape": list(energies.shape),
            "overlap_shape": list(overlap.shape),
            "fock_shape": list(fock.shape),
            "core_shape": list(core.shape),
        }

    identity = np.eye(coefficients.shape[1], dtype=np.float64)
    gram = coefficients.T @ overlap @ coefficients
    orthonormality_error = float(
        np.max(np.abs(gram - identity), initial=0.0)
    )
    core_overlap = core.T @ overlap @ coefficients
    core_error = float(np.max(np.abs(core_overlap), initial=0.0))
    residual = (
        fock @ coefficients
        - (overlap @ coefficients) * energies[None, :]
    )
    residual_max_abs = float(np.max(np.abs(residual), initial=0.0))
    residual_column_norm_max = float(
        np.max(np.linalg.norm(residual, axis=0), initial=0.0)
    )
    orthonormality_passed = (
        orthonormality_error <= orthonormality_atol
    )
    core_passed = core_error <= core_orthogonality_atol
    residual_passed = residual_max_abs <= fock_residual_atol
    return {
        "passed": bool(
            orthonormality_passed and core_passed and residual_passed
        ),
        "shape_valid": True,
        "finite": True,
        "s_orthonormality": {
            "passed": bool(orthonormality_passed),
            "max_abs_error": orthonormality_error,
            "absolute_tolerance": float(orthonormality_atol),
        },
        "core_active_orthogonality": {
            "passed": bool(core_passed),
            "max_abs_overlap": core_error,
            "absolute_tolerance": float(core_orthogonality_atol),
            "core_orbital_count": int(core.shape[1]),
        },
        "generalized_fock_eigen_residual": {
            "passed": bool(residual_passed),
            "equation": "F C_active = S C_active epsilon_active",
            "max_abs_residual": residual_max_abs,
            "maximum_column_l2_norm": residual_column_norm_max,
            "absolute_tolerance": float(fock_residual_atol),
        },
    }


def orbital_spectrum_certificate(
    frozen_energies: np.ndarray,
    regenerated_noncore_energies: np.ndarray,
    *,
    atol: float,
) -> dict[str, Any]:
    """Compare eigenvalues to the regenerated spectrum without ordering them."""

    frozen = np.asarray(frozen_energies, dtype=np.float64).reshape(-1)
    regenerated = np.asarray(
        regenerated_noncore_energies, dtype=np.float64
    ).reshape(-1)
    finite = bool(
        frozen.size > 0
        and regenerated.size >= frozen.size
        and np.all(np.isfinite(frozen))
        and np.all(np.isfinite(regenerated))
    )
    if not finite:
        return {
            "passed": False,
            "gating": False,
            "finite": False,
            "frozen_count": int(frozen.size),
            "regenerated_count": int(regenerated.size),
            "absolute_tolerance_hartree": float(atol),
        }
    nearest = np.min(
        np.abs(frozen[:, None] - regenerated[None, :]), axis=1
    )
    maximum = float(np.max(nearest, initial=0.0))
    return {
        "passed": bool(maximum <= atol),
        "gating": False,
        "finite": True,
        "comparison": (
            "non-gating nearest regenerated noncore eigenvalue diagnostic; "
            "invariant to virtual permutation and degenerate-subspace rotations"
        ),
        "nearest_absolute_differences_hartree": nearest.tolist(),
        "maximum_nearest_difference_hartree": maximum,
        "absolute_tolerance_hartree": float(atol),
    }


def mo_coefficients_with_frozen_active_basis(
    regenerated_coefficients: np.ndarray,
    frozen_active_coefficients: np.ndarray,
    ncore: int,
) -> np.ndarray:
    """Insert the exact frozen active block into a regenerated MO container."""

    regenerated = np.asarray(regenerated_coefficients, dtype=np.float64)
    frozen = np.asarray(frozen_active_coefficients, dtype=np.float64)
    core_count = int(ncore)
    if (
        regenerated.ndim != 2
        or frozen.ndim != 2
        or regenerated.shape[0] != frozen.shape[0]
        or core_count < 0
        or core_count + frozen.shape[1] > regenerated.shape[1]
    ):
        raise ValueError(
            "frozen active orbitals do not fit the regenerated MO container"
        )
    result = regenerated.copy()
    result[:, core_count : core_count + frozen.shape[1]] = frozen
    return result


def eigen_residual_certificate(
    hamiltonian: np.ndarray,
    addresses: np.ndarray,
    full_dimension: int,
    ecore: float,
    *,
    residual_atol: float,
) -> dict[str, Any]:
    """Diagonalize a full determinant matrix and certify its ground residual."""

    matrix = np.asarray(hamiltonian, dtype=np.float64)
    addresses = np.asarray(addresses, dtype=np.int64).reshape(-1)
    dimension = int(full_dimension)
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"p-space matrix is {matrix.shape}, expected {(dimension, dimension)}"
        )
    full_addresses = (
        addresses.size == dimension
        and np.array_equal(np.sort(addresses), np.arange(dimension))
    )
    hermiticity_error = float(np.max(np.abs(matrix - matrix.T), initial=0.0))
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    electronic_energy = float(eigenvalues[0])
    vector = np.asarray(eigenvectors[:, 0], dtype=np.float64)
    residual = matrix @ vector - electronic_energy * vector
    residual_norm = float(np.linalg.norm(residual))
    relative_residual = residual_norm / max(
        1.0, float(np.linalg.norm(matrix, ord=2))
    )
    converged = (
        full_addresses
        and hermiticity_error <= residual_atol
        and residual_norm <= residual_atol
    )
    return {
        "solver": "numpy.linalg.eigh on PySCF full determinant p-space",
        "full_determinant_dimension": dimension,
        "p_space_dimension": int(addresses.size),
        "all_determinant_addresses_present": bool(full_addresses),
        "electronic_energy_hartree": electronic_energy,
        "total_energy_hartree": electronic_energy + float(ecore),
        "eigen_residual_norm_hartree": residual_norm,
        "relative_eigen_residual": relative_residual,
        "hermiticity_max_abs_hartree": hermiticity_error,
        "residual_absolute_tolerance_hartree": float(residual_atol),
        "converged": bool(converged),
    }


def full_pspace_certificate(
    h1e: np.ndarray,
    eri: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    ecore: float,
    *,
    residual_atol: float,
) -> dict[str, Any]:
    """Build and solve the complete PySCF determinant p-space."""

    from pyscf.fci import cistring, direct_spin1

    dimension = int(
        cistring.num_strings(int(norb), int(nelec[0]))
        * cistring.num_strings(int(norb), int(nelec[1]))
    )
    hdiag = direct_spin1.make_hdiag(h1e, eri, int(norb), nelec)
    addresses, matrix = direct_spin1.pspace(
        h1e,
        eri,
        int(norb),
        nelec,
        hdiag=hdiag,
        np=dimension,
    )
    return eigen_residual_certificate(
        matrix,
        addresses,
        dimension,
        ecore,
        residual_atol=residual_atol,
    )


def _load_frozen_arrays(
    directory: Path, record: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    bundle = directory / record["bundle_file"]
    actual_bundle_sha = sha256_file(bundle)
    expected_bundle_sha = str(record["bundle_sha256"])
    if actual_bundle_sha != expected_bundle_sha:
        raise ValueError(
            f"frozen bundle checksum mismatch for {bundle.name}: "
            f"{actual_bundle_sha}"
        )
    with np.load(bundle, allow_pickle=False) as archive:
        arrays = {
            key: np.asarray(archive[key]).copy()
            for key in (
                "h1e",
                "eri",
                "nelec",
                "mo_coeff_active",
                "mo_energy_active",
                "mo_occ_active",
            )
        }
        scalars = {
            "ecore": float(np.asarray(archive["ecore"]).item()),
            "e_hf": float(np.asarray(archive["e_hf"]).item()),
            "e_casci": float(np.asarray(archive["e_casci"]).item()),
        }
    expected_array_hashes = record.get("array_sha256")
    expected_active_hash = (
        expected_array_hashes.get("mo_coeff_active")
        if isinstance(expected_array_hashes, dict)
        else None
    )
    actual_active_hash = array_sha256(arrays["mo_coeff_active"])
    return arrays, {
        **scalars,
        "bundle_file": bundle.name,
        "bundle_sha256": actual_bundle_sha,
        "bundle_checksum_match": True,
        "frozen_active_orbitals": {
            "array": "mo_coeff_active",
            "sha256": actual_active_hash,
            "expected_sha256": expected_active_hash,
            "checksum_match": bool(
                isinstance(expected_active_hash, str)
                and actual_active_hash == expected_active_hash
            ),
        },
    }


def _build_mean_field(source: dict[str, Any]):
    from pyscf import gto, scf

    molecule = gto.M(
        atom=source["atom"],
        basis=source["basis"],
        charge=int(source["charge"]),
        spin=int(source["spin"]),
        unit=source["unit"],
        verbose=0,
    )
    molecule.incore_anyway = True
    mean_field = scf.RHF(molecule) if int(source["spin"]) == 0 else scf.ROHF(molecule)
    mean_field.conv_tol = 1e-12
    mean_field.max_cycle = 200
    energy = float(mean_field.kernel())
    return molecule, mean_field, energy


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right), initial=0.0))


def audit_frozen_references(
    frozen_directory: str | Path,
    *,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Regenerate and certify all manifest-declared RHF/CASCI references."""

    from pyscf import ao2mo, mcscf

    directory = Path(frozen_directory).expanduser().resolve()
    manifest_path = directory / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = manifest.get("problems")
    if not isinstance(problems, dict) or not problems:
        raise ValueError("frozen manifest contains no problems")
    tolerance = dict(DEFAULT_TOLERANCES)
    if tolerances:
        unknown = set(tolerances) - set(tolerance)
        if unknown:
            raise ValueError(f"unknown audit tolerances: {sorted(unknown)}")
        tolerance.update({key: float(value) for key, value in tolerances.items()})

    groups: dict[tuple[Any, ...], list[tuple[str, dict[str, Any], dict[str, Any]]]] = {}
    for name, record in sorted(problems.items()):
        source = manifest_source(record)
        groups.setdefault(source_key(source), []).append((name, record, source))

    records: dict[str, Any] = {}
    mean_fields: dict[str, Any] = {}
    for group in groups.values():
        source = group[0][2]
        molecule, mean_field, rhf_energy = _build_mean_field(source)
        group_name = str(source["molecule"])
        mean_fields[group_name] = {
            "source": source,
            "converged": bool(mean_field.converged),
            "rhf_energy_hartree": rhf_energy,
            "maximum_active_orbitals": max(int(item[1]["norb"]) for item in group),
        }
        ao_overlap = np.asarray(molecule.intor_symmetric("int1e_ovlp"), dtype=np.float64)
        regenerated_mo = np.asarray(mean_field.mo_coeff, dtype=np.float64)
        regenerated_energies = np.asarray(mean_field.mo_energy, dtype=np.float64)
        regenerated_occupations = np.asarray(mean_field.mo_occ, dtype=np.float64)
        ao_fock = np.asarray(mean_field.get_fock(), dtype=np.float64)
        ncore = int(source["ncore"])

        for name, record, _ in sorted(group):
            arrays, bundle = _load_frozen_arrays(directory, record)
            norb = int(record["norb"])
            nelec = tuple(int(value) for value in record["nelec_alpha_beta"])
            sector = sector_consistency_certificate(
                arrays["nelec"],
                nelec,
                norb,
                int(record["sector_dimension"]),
            )
            active_slice = slice(ncore, ncore + norb)
            regenerated_active = regenerated_mo[:, active_slice].copy()
            frozen_active = np.asarray(
                arrays["mo_coeff_active"], dtype=np.float64
            )
            orbital_order, _ = orbital_order_certificate(
                frozen_active,
                regenerated_active,
                ao_overlap,
                atol=tolerance["orbital_overlap"],
            )
            frozen_basis = frozen_basis_certificate(
                frozen_active,
                arrays["mo_energy_active"],
                ao_overlap,
                ao_fock,
                regenerated_mo[:, :ncore],
                orthonormality_atol=tolerance[
                    "orbital_orthonormality"
                ],
                core_orthogonality_atol=tolerance[
                    "core_active_orthogonality"
                ],
                fock_residual_atol=tolerance[
                    "generalized_fock_residual"
                ],
            )
            spectrum = orbital_spectrum_certificate(
                arrays["mo_energy_active"],
                regenerated_energies[ncore:],
                atol=tolerance["mo_energy_max_abs"],
            )
            occupied_count = int(nelec[0])
            subspace_diagnostics = {
                "selected_active_space": orbital_subspace_certificate(
                    frozen_active,
                    regenerated_active,
                    ao_overlap,
                    atol=tolerance["orbital_overlap"],
                ),
                "occupied_active_space": orbital_subspace_certificate(
                    frozen_active[:, :occupied_count],
                    regenerated_mo[
                        :, ncore : ncore + occupied_count
                    ],
                    ao_overlap,
                    atol=tolerance["orbital_overlap"],
                ),
                "active_virtuals_within_regenerated_virtual_space": (
                    orbital_subspace_certificate(
                        frozen_active[:, occupied_count:],
                        regenerated_mo[:, ncore + occupied_count :],
                        ao_overlap,
                        atol=tolerance["orbital_overlap"],
                    )
                ),
            }
            frozen_basis_mo = mo_coefficients_with_frozen_active_basis(
                regenerated_mo, frozen_active, ncore
            )

            occupations = occupation_order_certificate(
                arrays["mo_occ_active"],
                regenerated_occupations[active_slice],
                nelec,
                atol=tolerance["occupation_max_abs"],
            )
            mo_energy_difference = _max_abs_difference(
                arrays["mo_energy_active"], regenerated_energies[active_slice]
            )

            cas = mcscf.CASCI(mean_field, ncas=norb, nelecas=nelec)
            regenerated_h1e, regenerated_ecore = cas.get_h1eff(
                frozen_basis_mo
            )
            regenerated_eri = ao2mo.restore(
                1, cas.get_h2eff(frozen_basis_mo), norb
            )
            regenerated_h1e = np.asarray(regenerated_h1e, dtype=np.float64)
            regenerated_eri = np.asarray(regenerated_eri, dtype=np.float64)
            integral_differences = {
                "basis": (
                    "checksum-bound frozen mo_coeff_active with regenerated "
                    "core orbitals"
                ),
                "h1e_max_abs": _max_abs_difference(
                    arrays["h1e"], regenerated_h1e
                ),
                "eri_max_abs": _max_abs_difference(
                    arrays["eri"], regenerated_eri
                ),
                "ecore_abs": abs(float(bundle["ecore"]) - float(regenerated_ecore)),
            }
            pspace = full_pspace_certificate(
                regenerated_h1e,
                regenerated_eri,
                norb,
                nelec,
                float(regenerated_ecore),
                residual_atol=tolerance["eigen_residual_hartree"],
            )
            rhf_difference = abs(float(bundle["e_hf"]) - rhf_energy)
            casci_difference = abs(
                float(bundle["e_casci"]) - float(pspace["total_energy_hartree"])
            )
            basis_invariant_passed = bool(frozen_basis["passed"])
            checks = {
                "bundle_checksum": bool(bundle["bundle_checksum_match"]),
                "frozen_active_orbital_checksum": bool(
                    bundle["frozen_active_orbitals"]["checksum_match"]
                ),
                "bundle_electron_sector": bool(
                    sector["bundle_matches_manifest"]
                ),
                "rhf_converged": bool(mean_field.converged),
                "rhf_energy": rhf_difference
                <= tolerance["rhf_energy_hartree"],
                "frozen_occupation_order": bool(occupations["passed"]),
                "frozen_active_s_orthonormality": bool(
                    frozen_basis.get("s_orthonormality", {}).get("passed")
                ),
                "frozen_core_active_orthogonality": bool(
                    frozen_basis.get(
                        "core_active_orthogonality", {}
                    ).get("passed")
                ),
                "generalized_fock_eigen_residual": bool(
                    frozen_basis.get(
                        "generalized_fock_eigen_residual", {}
                    ).get("passed")
                ),
                # Backward-compatible validator key.  It now means the strict,
                # ordering-invariant frozen-basis certificate above; the old
                # sign-only order comparison is retained as a diagnostic only.
                "canonical_orbital_order": basis_invariant_passed,
                # The stored energies are certified through the generalized
                # eigen equation.  Nearest-spectrum matching is diagnostic
                # because a regenerated virtual cutoff/order is not unique.
                "mo_energies": bool(
                    frozen_basis.get(
                        "generalized_fock_eigen_residual", {}
                    ).get("passed")
                ),
                "h1e": integral_differences["h1e_max_abs"]
                <= tolerance["integral_max_abs"],
                "eri": integral_differences["eri_max_abs"]
                <= tolerance["integral_max_abs"],
                "ecore": integral_differences["ecore_abs"]
                <= tolerance["ecore_hartree"],
                "casci_declared_core_count": int(cas.ncore) == ncore,
                "declared_sector_dimension": bool(
                    sector["declared_dimension_matches_combinatorial"]
                ),
                "full_determinant_pspace": bool(pspace["converged"]),
                "pspace_sector_dimension": int(
                    pspace["full_determinant_dimension"]
                )
                == int(sector["combinatorial_sector_dimension"]),
                "casci_energy": casci_difference
                <= tolerance["casci_energy_hartree"],
            }
            records[name] = {
                "status": "PASS" if all(checks.values()) else "FAIL",
                "source": source,
                "norb": norb,
                "n_qubits": 2 * norb,
                "nelec_alpha_beta": list(nelec),
                "sector_dimension": int(record["sector_dimension"]),
                "sector_consistency": sector,
                "bundle": bundle,
                "checks": checks,
                "check_semantics": {
                    "canonical_orbital_order": (
                        "legacy validator key: strict ordering-invariant "
                        "frozen-basis S/core/Fock certificate"
                    ),
                    "mo_energies": (
                        "stored orbital energies satisfy the generalized "
                        "Fock eigen equation in the frozen basis"
                    ),
                },
                "rhf": {
                    "frozen_energy_hartree": float(bundle["e_hf"]),
                    "regenerated_energy_hartree": rhf_energy,
                    "absolute_difference_hartree": rhf_difference,
                    "absolute_tolerance_hartree": tolerance[
                        "rhf_energy_hartree"
                    ],
                    "converged": bool(mean_field.converged),
                },
                "occupation_order": occupations,
                "frozen_active_basis": {
                    **frozen_basis,
                    "coefficient_sha256": bundle[
                        "frozen_active_orbitals"
                    ]["sha256"],
                    "expected_coefficient_sha256": bundle[
                        "frozen_active_orbitals"
                    ]["expected_sha256"],
                    "coefficient_checksum_match": bundle[
                        "frozen_active_orbitals"
                    ]["checksum_match"],
                },
                "order_sensitive_canonical_orbital_diagnostic": (
                    orbital_order
                ),
                "canonical_subspace_diagnostics": {
                    "gating": False,
                    "reason": (
                        "regenerated near-degenerate virtual orbitals may "
                        "differ by sign, permutation, or subspace rotation"
                    ),
                    **subspace_diagnostics,
                },
                "orbital_spectrum": spectrum,
                "mo_energy_max_abs_difference": mo_energy_difference,
                "integral_differences": integral_differences,
                "active_space_regeneration": {
                    "basis": (
                        "checksum-bound frozen active orbitals inserted after "
                        "the regenerated core"
                    ),
                    "declared_ncore": ncore,
                    "pyscf_casci_ncore": int(cas.ncore),
                    "declared_core_count_matches": int(cas.ncore) == ncore,
                },
                "pspace": pspace,
                "casci": {
                    "frozen_energy_hartree": float(bundle["e_casci"]),
                    "regenerated_energy_hartree": float(
                        pspace["total_energy_hartree"]
                    ),
                    "absolute_difference_hartree": casci_difference,
                    "absolute_tolerance_hartree": tolerance[
                        "casci_energy_hartree"
                    ],
                },
            }

    status = "PASS" if all(item["status"] == "PASS" for item in records.values()) else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "scope": (
            "independent geometry-to-RHF regeneration; checksum-bound "
            "frozen-basis S/core/Fock certification; frozen-basis integral "
            "regeneration; and complete determinant p-space CASCI eigensolve; "
            "no frozen scalar is accepted as proof"
        ),
        "manifest_file": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "thread_controls": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "tolerances": tolerance,
        "mean_fields": mean_fields,
        "problem_count": len(records),
        "passed": sum(item["status"] == "PASS" for item in records.values()),
        "failed": sum(item["status"] == "FAIL" for item in records.values()),
        "problems": records,
    }
