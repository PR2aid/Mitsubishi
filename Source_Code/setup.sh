#!/usr/bin/env bash
# Create the isolated pinned environment used by QBRAID_RUNME.ipynb.
# This script is path-independent and may be run from any working directory.
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESOLVE_ARGS=(resolve --root "${PACKAGE_ROOT}")
if [[ -n "${QBRAID_GQE_ENV:-}" ]]; then
  RESOLVE_ARGS+=(--override "${QBRAID_GQE_ENV}")
fi
RESOLVED_ENV="$("${PYTHON_BIN}" -I -B \
  "${PACKAGE_ROOT}/environment_contract.py" "${RESOLVE_ARGS[@]}")"
export QBRAID_GQE_ENV="${RESOLVED_ENV}"
exec bash "${PACKAGE_ROOT}/source/scripts/setup_qbraid_gqe.sh" --setup-only
