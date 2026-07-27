"""givens40: scalable Givens-exchange VQE package (sector-restricted, cutting-aware).

Tracks
------
1. Dense statevector track (<= ~14 qubits): the paper-faithful ansatz
   (RY layers + all-pair qubit Givens) on dense npz Hamiltonians.
2. Sector track (the 40-qubit engine): particle-conserving variant
   (qubit-convention Givens singles + pair-double exchanges) simulated
   exactly in the fixed (Na, Nb) determinant sector, with energies via
   PySCF's FCI contraction kernels. Memory/time scale with the sector
   dimension C(no,na)*C(no,nb), not 2^n.
3. Cutting-overhead accounting (Nakamura & Sanji, arXiv:2509.08351):
   per-gate log-overhead u, circuit overhead phi, budgeted cross-partition
   topologies from a Hamiltonian-informed pair score.
"""

__version__ = "0.1.0"
