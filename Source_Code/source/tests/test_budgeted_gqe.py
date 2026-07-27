"""Fast, dependency-light checks for the budgeted GQE vocabulary.

Run from ``source/``::

    python -m tests.test_budgeted_gqe
"""
from __future__ import annotations

import numpy as np

from givens40 import overhead as oh
from givens40.budgeted_gqe import (
    ResidualToken,
    base_u_from_phi,
    blocked_to_interleaved_fock_state,
    build_residual_pool,
    sequence_u,
    token_cost,
    validate_sequence_budget,
)


def check(name: str, condition: bool, detail: str = "") -> None:
    """Record one budgeted-vocabulary invariant."""

    if not condition:
        raise AssertionError(f"[FAIL] {name}: {detail}")
    suffix = f"  ({detail})" if detail else ""
    print(f"[PASS] {name}{suffix}")


def main() -> int:
    """Run deterministic masking, ordering, and budget checks."""

    for u in (0.0, 0.1, 0.5, oh.u_max_from_phi(15.0)):
        recovered = base_u_from_phi(oh.phi_from_u(u))
        check(
            f"phi/u inverse at u={u:.6f}",
            abs(recovered - u) < 1e-12,
            f"recovered={recovered:.12f}",
        )

    class FakeProblem:
        norb = 4
        h1e = np.array(
            [
                [0.0, 0.1, 0.9, 0.2],
                [0.1, 0.0, 0.3, 0.8],
                [0.9, 0.3, 0.0, 0.1],
                [0.2, 0.8, 0.1, 0.0],
            ],
            dtype=float,
        )
        eri = np.zeros((4, 4, 4, 4), dtype=float)

    pool, pairs = build_residual_pool(
        FakeProblem(),
        left_block=[0, 1],
        max_pairs=2,
        angle_bins=(0.025,),
        include_pair_doubles=True,
    )
    check("identity token is pool index zero", pool[0].kind == "identity")
    check("top two cross pairs selected", pairs == [(0, 2), (1, 3)], str(pairs))
    check("expected signed vocabulary size", len(pool) == 13, f"size={len(pool)}")
    check(
        "all non-identity tokens cross the cut",
        all((token.p < 2) != (token.q < 2) for token in pool[1:]),
    )
    check(
        "token costs are exact",
        all(
            abs(token.u_cost - token_cost(token.kind, token.angle)) < 1e-15
            for token in pool
        ),
    )

    feasible = [
        ResidualToken("single", 0, 2, 0.05, "a", token_cost("single", 0.05))
        for _ in range(3)
    ]
    accounting = validate_sequence_budget(feasible, base_phi=1.0, phi_max=15.0)
    check("feasible sequence accepted", accounting["within_budget"])
    check(
        "sequence_u is additive",
        abs(sequence_u(feasible) - 3.0 * token_cost("single", 0.05)) < 1e-15,
    )

    expensive = [
        ResidualToken("double", 0, 2, 0.5, None, token_cost("double", 0.5))
        for _ in range(4)
    ]
    rejected = False
    try:
        validate_sequence_budget(expensive, base_phi=1.0, phi_max=15.0)
    except ValueError:
        rejected = True
    check("over-budget sequence rejected", rejected)

    # Fermionic mode reordering is not a plain bit permutation.  For two
    # spatial orbitals, occupied alpha_1 crosses occupied beta_0, producing a
    # minus sign; occupied alpha_0 and beta_1 do not cross.
    blocked = np.zeros(16, dtype=np.complex128)
    blocked[0b0110] = 3.0 + 4.0j  # alpha_1, beta_0
    blocked[0b1001] = 2.0         # alpha_0, beta_1
    interleaved = blocked_to_interleaved_fock_state(blocked, 2)
    check(
        "blocked/interleaved fermionic parity phase",
        interleaved[0b0110] == -(3.0 + 4.0j),
    )
    check(
        "noncrossing blocked/interleaved phase",
        interleaved[0b1001] == 2.0,
    )
    check(
        "blocked/interleaved norm preserved",
        abs(np.vdot(interleaved, interleaved) - np.vdot(blocked, blocked))
        < 1e-14,
    )

    print("\nAll budgeted-GQE vocabulary checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
