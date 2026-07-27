"""Molecule definitions and CAS(ne, no) Hamiltonian construction via PySCF.

All escalation rungs reuse the *same* molecules and geometries as the
manuscript benchmarks; only the active space (hence qubit count = 2*no)
grows. Active orbitals are the lowest-energy canonical RHF MOs above a
frozen core, i.e. a plain CASCI(ne, no) truncation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import numpy as np
from pyscf import gto, scf, mcscf, ao2mo, lib as pyscf_lib
from pyscf.fci import direct_spin1


# PySCF's optional memory diagnostic reads ``/proc/<pid>/statm``.  Some
# managed notebook sandboxes intentionally hide that entry from nested
# processes.  The molecular instances below are explicitly forced in-core, so
# a zero-memory diagnostic is a safe deterministic fallback when /proc is not
# exposed; on ordinary Linux the original diagnostic is preserved.
_pyscf_current_memory = pyscf_lib.current_memory


def _sandbox_safe_current_memory():
    try:
        return _pyscf_current_memory()
    except (FileNotFoundError, OSError):
        return 0.0, 0.0


pyscf_lib.current_memory = _sandbox_safe_current_memory

# Geometries follow the benchmark conventions used in the manuscript / npz files.
MOLECULES: dict[str, dict] = {
    # LiH at 2.2 A (the primary LiH-6 JW row geometry).
    "LiH": dict(atom="Li 0 0 0; H 0 0 2.2", basis="aug-cc-pvdz", ncore=1),
    # BeH2 public-specification geometry: H at +/- 1.33 A.
    "BeH2": dict(atom="Be 0 0 0; H 0 0 1.33; H 0 0 -1.33", basis="cc-pvdz", ncore=1),
    # H2O geometry exactly as encoded in the benchmark npz filename.
    "H2O": dict(
        atom="H -0.021 -0.002 0; O 0.835 0.452 0; H 1.477 -0.273 0",
        basis="cc-pvdz",
        ncore=1,
    ),
    # Stretched LiH stress geometry (3.4 A), same basis as LiH.
    "LiH_stretch": dict(atom="Li 0 0 0; H 0 0 3.4", basis="aug-cc-pvdz", ncore=1),
}


@dataclass
class CASProblem:
    """A CAS(ne, no) problem: integrals in the active MO basis + metadata."""

    name: str
    norb: int                      # active spatial orbitals
    nelec: tuple[int, int]         # (n_alpha, n_beta) active electrons
    h1e: np.ndarray                # (norb, norb)
    eri: np.ndarray                # (norb,)*4, chemist notation (pq|rs)
    ecore: float                   # core + nuclear scalar shift
    e_hf: float                    # RHF total energy (full molecule)
    e_casci: float | None = None   # exact CASCI total energy (reference)
    meta: dict = field(default_factory=dict)

    @property
    def n_qubits(self) -> int:
        return 2 * self.norb

    @property
    def dims(self) -> tuple[int, int]:
        from pyscf.fci import cistring

        return (
            cistring.num_strings(self.norb, self.nelec[0]),
            cistring.num_strings(self.norb, self.nelec[1]),
        )

    def hdiag(self) -> np.ndarray:
        """Diagonal <det|H|det> (electronic, no ecore), shape (dimA*dimB,)."""
        return direct_spin1.make_hdiag(self.h1e, self.eri, self.norb, self.nelec)


def build_cas_from_spec(spec: dict, no: int, name: str = "custom",
                        run_casci: bool = True,
                        casci_max_dim: int = 40_000_000,
                        orbital_basis: str = "canonical") -> CASProblem:
    """Molecule-agnostic CAS(ne, no) builder from an explicit specification.

    spec keys: atom (PySCF geometry string), basis, and optionally charge,
    spin (2S), unit, ncore. orbital_basis: "canonical" (RHF/ROHF MOs) or
    "mp2nat" (MP2 natural orbitals; closed-shell only) — the latter compacts
    correlation into fewer orbital pairs, which improves pruning, budget
    cost, and accuracy simultaneously. EVERYTHING downstream (partition,
    budget, initial state, MP2 seeds, CASCI reference) is derived from this
    spec at run time — nothing is borrowed from previous molecules or
    stored parameters.
    """
    spin = spec.get("spin", 0)
    mol = gto.M(atom=spec["atom"], basis=spec["basis"], charge=spec.get("charge", 0),
                spin=spin, unit=spec.get("unit", "Angstrom"), verbose=0)
    # Every benchmark in this submission is small enough for in-core AO
    # integrals.  Forcing that deterministic path also avoids PySCF's optional
    # Linux /proc memory probe, which is unavailable in some managed notebook
    # and container sandboxes (including restricted qBraid-style runners).
    mol.incore_anyway = True
    mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
    mf.conv_tol = 1e-12
    e_hf = mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF did not converge for {name}")

    ncore = spec.get("ncore", 0)
    nelec_act = mol.nelectron - 2 * ncore
    if nelec_act <= 0:
        raise ValueError("No active electrons after freezing the core")
    if ncore + no > mol.nao:
        raise ValueError(
            f"{name}: requested no={no} active orbitals but basis only has "
            f"{mol.nao} MOs with ncore={ncore} (max no={mol.nao - ncore})")
    na = (nelec_act + spin) // 2
    nb = nelec_act - na
    if nb < 0 or (nelec_act - spin) % 2:
        raise ValueError(f"{name}: inconsistent electron count/spin")
    if max(na, nb) > no:
        raise ValueError(
            f"{name}: no={no} active orbitals cannot host ({na},{nb}) electrons")
    from scipy.special import comb as _comb
    if _comb(no, na, exact=True) * _comb(no, nb, exact=True) <= 1:
        raise ValueError(
            f"{name}: CAS({nelec_act},{no}) sector is trivial (dim 1); "
            f"increase no")

    mo = np.asarray(mf.mo_coeff)
    if orbital_basis == "mp2nat":
        if spin != 0:
            raise ValueError("mp2nat orbitals are closed-shell only in v1")
        from pyscf import mp as _mp

        pt = _mp.MP2(mf, frozen=(ncore if ncore else None))
        pt.kernel()
        dm = np.asarray(pt.make_rdm1())   # full MO basis (frozen block ~ 2)
        blk = dm[ncore:, ncore:]          # keep the frozen core untouched
        occ_n, u = np.linalg.eigh(blk)
        order = np.argsort(occ_n)[::-1]
        mo = np.hstack([mo[:, :ncore], mo[:, ncore:] @ u[:, order]])
    elif orbital_basis != "canonical":
        raise ValueError(orbital_basis)

    mc = mcscf.CASCI(mf, ncas=no, nelecas=(na, nb))
    h1e, ecore = mc.get_h1eff(mo)
    eri = ao2mo.restore(1, mc.get_h2eff(mo), no)  # full (pq|rs), chemist notation

    # Runtime data for MP2-style pair-double seeding (competition-clean:
    # derived from this molecule's own integrals/Fock at build time).
    fao = mf.get_fock()
    c_act = mo[:, ncore:ncore + no]
    f_act = c_act.T @ fao @ c_act
    f_diag = np.diag(f_act).copy()
    k_exch = np.einsum("pqpq->pq", eri).copy()   # (pq|pq)
    t_pair = np.zeros((no, no))
    nocc_act = min(na, nb)
    for p in range(nocc_act):
        for q in range(nocc_act, no):
            denom = 2.0 * (f_diag[p] - f_diag[q])
            if abs(denom) > 1e-8:
                t_pair[p, q] = np.clip(k_exch[p, q] / denom, -0.3, 0.3)

    from pyscf.fci import cistring
    dim = cistring.num_strings(no, na) * cistring.num_strings(no, nb)
    e_casci = None
    if run_casci and dim <= casci_max_dim:
        mc.fcisolver.conv_tol = 1e-12
        e_casci = float(mc.kernel(mo)[0])

    return CASProblem(
        name=f"{name}_no{no}_q{2*no}",
        norb=no,
        nelec=(na, nb),
        h1e=np.asarray(h1e, dtype=np.float64),
        eri=np.asarray(eri, dtype=np.float64),
        ecore=float(ecore),
        e_hf=float(e_hf),
        e_casci=e_casci,
        meta=dict(
            molecule=name,
            atom=spec["atom"],
            basis=spec["basis"],
            charge=int(spec.get("charge", 0)),
            spin=int(spin),
            unit=spec.get("unit", "Angstrom"),
            ncore=ncore,
            nelec_active=nelec_act,
            sector_dim=int(dim),
            orbital_basis=orbital_basis,
            fock_diag=f_diag,
            t_pair=t_pair,
            # These coefficients are provenance, not a runtime dependency.
            # Once serialized they fix the orbital phase/order that produced
            # h1e and eri, including any near-degenerate subspace choice.
            mo_coeff_active=np.asarray(c_act, dtype=np.float64),
            mo_energy_active=np.asarray(
                mf.mo_energy[ncore:ncore + no], dtype=np.float64
            ),
            mo_occ_active=np.asarray(
                mf.mo_occ[ncore:ncore + no], dtype=np.float64
            ),
        ),
    )


def build_cas(molname: str, no: int, run_casci: bool = True,
              casci_max_dim: int = 40_000_000) -> CASProblem:
    """Preset wrapper with an opt-in deterministic frozen-input route.

    Set ``GQE_FROZEN_INPUT_DIR`` to a directory containing ``MANIFEST.json``
    to bypass SCF/orbital reconstruction.  ``GQE_FROZEN_INPUT_MODE=required``
    makes a missing declared rung a hard error; the default mode is ``auto``.
    """

    frozen_dir = os.environ.get("GQE_FROZEN_INPUT_DIR")
    mode = os.environ.get("GQE_FROZEN_INPUT_MODE", "auto").lower()
    if frozen_dir:
        from .frozen_problem import load_named_problem

        try:
            return load_named_problem(Path(frozen_dir), molname, no)
        except FileNotFoundError:
            if mode == "required":
                raise
    elif mode == "required":
        raise RuntimeError(
            "GQE_FROZEN_INPUT_MODE=required but GQE_FROZEN_INPUT_DIR is unset"
        )
    return build_cas_from_spec(MOLECULES[molname], no, name=molname,
                               run_casci=run_casci, casci_max_dim=casci_max_dim)


def build_from_fcidump(path: str, name: str = "fcidump", run_casci: bool = True,
                       casci_max_dim: int = 40_000_000) -> CASProblem:
    """Load a problem from an FCIDUMP file (standard interchange format).

    Supports competition settings where organizers provide integrals rather
    than a geometry. The active space is whatever the FCIDUMP defines; the
    HF energy is estimated as <HF-det|H|HF-det> within that space.
    """
    from pyscf import fci as _fci
    from pyscf.tools import fcidump as _fcid
    from pyscf.fci import cistring

    data = _fcid.read(path, verbose=False)
    norb = int(data["NORB"])
    nelec = int(data["NELEC"])
    ms2 = int(data.get("MS2", 0))
    na = (nelec + ms2) // 2
    nb = nelec - na
    if na != nb:
        raise ValueError("Open-shell FCIDUMP not supported in v1")
    h1e = np.asarray(data["H1"], dtype=np.float64)
    eri = ao2mo.restore(1, data["H2"], norb).astype(np.float64)
    ecore = float(data["ECORE"])

    hf_e = None
    dim = cistring.num_strings(norb, na) * cistring.num_strings(norb, nb)
    # HF determinant energy within the space (diagonal element of H)
    hd = direct_spin1.make_hdiag(h1e, eri, norb, (na, nb))
    hf_addr = 0  # lowest orbitals occupied = address 0 in cistring order
    hf_e = float(hd.reshape(-1)[hf_addr * cistring.num_strings(norb, nb) + hf_addr]) + ecore

    e_casci = None
    if run_casci and dim <= casci_max_dim:
        e_elec, _ = _fci.direct_spin1.kernel(h1e, eri, norb, (na, nb),
                                             conv_tol=1e-12, max_cycle=400)
        e_casci = float(e_elec) + ecore

    return CASProblem(
        name=f"{name}_no{norb}_q{2*norb}", norb=norb, nelec=(na, nb),
        h1e=h1e, eri=eri, ecore=ecore, e_hf=hf_e, e_casci=e_casci,
        meta=dict(molecule=name, basis="fcidump", ncore=0,
                  nelec_active=nelec, sector_dim=int(dim)),
    )
