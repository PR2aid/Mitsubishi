"""Versioned, checksum-verified molecular inputs for deterministic replay.

An exploratory implementation rebuilt canonical orbitals and active-space
integrals in every process.  Near-degenerate orbital subspaces then made the
reported table sensitive to the threaded linear-algebra implementation.  This
module turns the complete scientific input to the sector engine into a frozen
first-class artifact.  Loading a bundle performs no SCF or orbital choice.

The bundle intentionally stores more than the contractions strictly require:
the geometry specification, active MO coefficients, Fock diagonal, pair seed,
integrals, reference energies and electron sector are all hashed and checked.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .chemistry import CASProblem


SCHEMA_VERSION = 1
REQUIRED_ARRAYS = (
    "h1e",
    "eri",
    "nelec",
    "fock_diag",
    "t_pair",
    "mo_coeff_active",
    "mo_energy_active",
    "mo_occ_active",
)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_fingerprint(arrays: dict[str, np.ndarray], scalars: dict[str, Any]) -> str:
    """Hash all numerical inputs in a platform-independent field order."""

    digest = hashlib.sha256()
    for key in sorted(arrays):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_array_sha256(np.asarray(arrays[key])).encode("ascii"))
        digest.update(b"\n")
    digest.update(
        json.dumps(scalars, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )
    return digest.hexdigest()


def problem_record(prob: CASProblem, source: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return JSON-safe provenance for a frozen problem."""

    meta = dict(prob.meta)
    for key in (
        "fock_diag",
        "t_pair",
        "mo_coeff_active",
        "mo_energy_active",
        "mo_occ_active",
    ):
        meta.pop(key, None)
    record = {
        "schema_version": SCHEMA_VERSION,
        "name": prob.name,
        "norb": int(prob.norb),
        "n_qubits": int(prob.n_qubits),
        "nelec_alpha_beta": [int(x) for x in prob.nelec],
        "sector_dimension": int(np.prod(prob.dims)),
        "ecore_hartree": float(prob.ecore),
        "rhf_energy_hartree": float(prob.e_hf),
        "casci_energy_hartree": (
            None if prob.e_casci is None else float(prob.e_casci)
        ),
        "meta": meta,
        "source": dict(source or {}),
    }
    return record


def save_frozen_problem(
    prob: CASProblem,
    path: str | Path,
    *,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize one deterministic problem and return its manifest record."""

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fock = np.asarray(prob.meta.get("fock_diag", np.diag(prob.h1e)), dtype=np.float64)
    t_pair = np.asarray(
        prob.meta.get("t_pair", np.zeros((prob.norb, prob.norb))), dtype=np.float64
    )
    mo_active = np.asarray(
        prob.meta.get("mo_coeff_active", np.empty((0, prob.norb))), dtype=np.float64
    )
    mo_energy = np.asarray(
        prob.meta.get("mo_energy_active", np.full(prob.norb, np.nan)),
        dtype=np.float64,
    )
    mo_occ = np.asarray(
        prob.meta.get("mo_occ_active", np.full(prob.norb, np.nan)),
        dtype=np.float64,
    )
    arrays = {
        "h1e": np.asarray(prob.h1e, dtype=np.float64),
        "eri": np.asarray(prob.eri, dtype=np.float64),
        "nelec": np.asarray(prob.nelec, dtype=np.int64),
        "fock_diag": fock,
        "t_pair": t_pair,
        "mo_coeff_active": mo_active,
        "mo_energy_active": mo_energy,
        "mo_occ_active": mo_occ,
    }
    record = problem_record(prob, source)
    scalars = {
        "name": record["name"],
        "norb": record["norb"],
        "ecore_hartree": record["ecore_hartree"],
        "rhf_energy_hartree": record["rhf_energy_hartree"],
        "casci_energy_hartree": record["casci_energy_hartree"],
    }
    record["array_sha256"] = {
        key: _array_sha256(value) for key, value in sorted(arrays.items())
    }
    record["scientific_fingerprint_sha256"] = scientific_fingerprint(arrays, scalars)
    metadata_json = json.dumps(record, sort_keys=True, separators=(",", ":"))
    np.savez_compressed(
        path,
        **arrays,
        ecore=np.asarray(prob.ecore, dtype=np.float64),
        e_hf=np.asarray(prob.e_hf, dtype=np.float64),
        e_casci=np.asarray(
            np.nan if prob.e_casci is None else prob.e_casci, dtype=np.float64
        ),
        metadata_json=np.asarray(metadata_json),
    )
    record["bundle_sha256"] = sha256_file(path)
    record["bundle_file"] = path.name
    return record


def load_frozen_problem(
    path: str | Path,
    *,
    expected_bundle_sha256: str | None = None,
    verify: bool = True,
) -> CASProblem:
    """Load and validate a problem without rebuilding molecular orbitals."""

    path = Path(path).expanduser().resolve()
    if expected_bundle_sha256 is not None:
        actual = sha256_file(path)
        if actual != expected_bundle_sha256:
            raise ValueError(
                f"frozen bundle checksum mismatch for {path.name}: {actual}"
            )
    with np.load(path, allow_pickle=False) as data:
        record = json.loads(str(data["metadata_json"].item()))
        if int(record.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(f"unsupported frozen schema in {path}")
        arrays = {key: np.asarray(data[key]).copy() for key in REQUIRED_ARRAYS}
        if verify:
            for key, value in arrays.items():
                expected = record["array_sha256"].get(key)
                actual = _array_sha256(value)
                if expected != actual:
                    raise ValueError(
                        f"scientific array checksum mismatch: {path.name}:{key}"
                    )
        e_casci_raw = float(np.asarray(data["e_casci"]).item())
        meta = dict(record.get("meta", {}))
        meta.update(
            {
                "fock_diag": arrays["fock_diag"],
                "t_pair": arrays["t_pair"],
                "mo_coeff_active": arrays["mo_coeff_active"],
                "mo_energy_active": arrays["mo_energy_active"],
                "mo_occ_active": arrays["mo_occ_active"],
                "frozen_bundle": path.name,
                "frozen_bundle_sha256": sha256_file(path),
                "scientific_fingerprint_sha256": record[
                    "scientific_fingerprint_sha256"
                ],
                "source": record.get("source", {}),
            }
        )
        problem = CASProblem(
            name=str(record["name"]),
            norb=int(record["norb"]),
            nelec=tuple(int(x) for x in arrays["nelec"]),
            h1e=np.asarray(arrays["h1e"], dtype=np.float64),
            eri=np.asarray(arrays["eri"], dtype=np.float64),
            ecore=float(np.asarray(data["ecore"]).item()),
            e_hf=float(np.asarray(data["e_hf"]).item()),
            e_casci=None if np.isnan(e_casci_raw) else e_casci_raw,
            meta=meta,
        )
    if verify:
        if problem.h1e.shape != (problem.norb, problem.norb):
            raise ValueError("invalid frozen one-electron integral shape")
        if problem.eri.shape != (problem.norb,) * 4:
            raise ValueError("invalid frozen two-electron integral shape")
        if int(np.prod(problem.dims)) != int(record["sector_dimension"]):
            raise ValueError("frozen determinant-sector dimension mismatch")
    return problem


def load_manifest(directory: str | Path) -> dict[str, Any]:
    directory = Path(directory).expanduser().resolve()
    return json.loads((directory / "MANIFEST.json").read_text(encoding="utf-8"))


def load_named_problem(
    directory: str | Path,
    molecule: str,
    norb: int,
    *,
    verify: bool = True,
) -> CASProblem:
    """Resolve a named frozen rung through its checksum manifest."""

    directory = Path(directory).expanduser().resolve()
    manifest = load_manifest(directory)
    key = f"{molecule}_no{int(norb)}"
    if key not in manifest["problems"]:
        # Some validation-only widths (notably BeH2 no=5) sit between release
        # rungs.  Derive them deterministically from the smallest declared
        # larger member of the same nested orbital chain.  This performs no
        # SCF/orbital choice and keeps the release manifest itself immutable.
        prefix = f"{molecule}_no"
        larger = sorted(
            (
                int(name[len(prefix):]),
                item,
            )
            for name, item in manifest["problems"].items()
            if name.startswith(prefix) and int(name[len(prefix):]) > int(norb)
        )
        if not larger:
            raise FileNotFoundError(f"frozen problem not declared: {key}")
        parent_norb, parent_item = larger[0]
        parent = load_frozen_problem(
            directory / parent_item["bundle_file"],
            expected_bundle_sha256=parent_item["bundle_sha256"],
            verify=verify,
        )
        if max(parent.nelec) > int(norb):
            raise FileNotFoundError(
                f"cannot derive {key}: electron sector exceeds requested orbitals"
            )
        meta = dict(parent.meta)
        meta.update(
            {
                "fock_diag": np.asarray(parent.meta["fock_diag"][:norb]).copy(),
                "t_pair": np.asarray(parent.meta["t_pair"][:norb, :norb]).copy(),
                "mo_coeff_active": np.asarray(
                    parent.meta["mo_coeff_active"][:, :norb]
                ).copy(),
                "mo_energy_active": np.asarray(
                    parent.meta["mo_energy_active"][:norb]
                ).copy(),
                "mo_occ_active": np.asarray(
                    parent.meta["mo_occ_active"][:norb]
                ).copy(),
                "frozen_bundle": None,
                "frozen_bundle_sha256": None,
                "derived_from_frozen_parent": parent.meta["frozen_bundle"],
                "derived_parent_norb": int(parent_norb),
                "derived_without_scf": True,
            }
        )
        h1e = np.ascontiguousarray(parent.h1e[:norb, :norb])
        eri = np.ascontiguousarray(parent.eri[:norb, :norb, :norb, :norb])
        arrays = {
            "h1e": h1e,
            "eri": eri,
            "nelec": np.asarray(parent.nelec, dtype=np.int64),
            "fock_diag": meta["fock_diag"],
            "t_pair": meta["t_pair"],
            "mo_coeff_active": meta["mo_coeff_active"],
            "mo_energy_active": meta["mo_energy_active"],
            "mo_occ_active": meta["mo_occ_active"],
        }
        scalars = {
            "name": f"{molecule}_no{int(norb)}_q{2 * int(norb)}",
            "norb": int(norb),
            "ecore_hartree": float(parent.ecore),
            "rhf_energy_hartree": float(parent.e_hf),
            "casci_energy_hartree": None,
        }
        meta["scientific_fingerprint_sha256"] = scientific_fingerprint(
            arrays, scalars
        )
        return CASProblem(
            name=scalars["name"],
            norb=int(norb),
            nelec=parent.nelec,
            h1e=h1e,
            eri=eri,
            ecore=float(parent.ecore),
            e_hf=float(parent.e_hf),
            e_casci=None,
            meta=meta,
        )
    item = manifest["problems"][key]
    return load_frozen_problem(
        directory / item["bundle_file"],
        expected_bundle_sha256=item["bundle_sha256"],
        verify=verify,
    )
