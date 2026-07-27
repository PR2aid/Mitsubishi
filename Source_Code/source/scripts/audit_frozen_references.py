#!/usr/bin/env python3
"""Independently regenerate and certify all frozen RHF/CASCI references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


SOURCE = Path(__file__).resolve().parents[1]
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from givens40.reference_audit import audit_frozen_references


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    certificate = audit_frozen_references(args.frozen_inputs)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, record in certificate["problems"].items():
        residual = record["pspace"]["eigen_residual_norm_hartree"]
        delta = record["casci"]["absolute_difference_hartree"]
        print(
            f"[{record['status']}] {name}: "
            f"|CASCI delta|={delta:.3e} Ha, residual={residual:.3e} Ha",
            flush=True,
        )
    print(
        f"REFERENCE AUDIT {certificate['status']}: "
        f"{certificate['passed']} passed, {certificate['failed']} failed",
        flush=True,
    )
    print(output, flush=True)
    return 0 if certificate["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
