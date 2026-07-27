#!/usr/bin/env bash
# Run the hard-budget CUDA-QX GQE experiment in the verified qBraid
# virtual environment.
#
# Examples (run from source/):
#   bash scripts/run_budgeted_gqe.sh --system beh2-6 --smoke
#   bash scripts/run_budgeted_gqe.sh --system beh2-12 --smoke
#   bash scripts/run_budgeted_gqe.sh --system lih-40 --smoke
#   bash scripts/run_budgeted_gqe.sh --system lih-40 --full
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_ROOT="$(cd "${SOURCE_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESOLVE_ARGS=(resolve --root "${PACKAGE_ROOT}")
if [[ -n "${QBRAID_GQE_ENV:-}" ]]; then
  RESOLVE_ARGS+=(--override "${QBRAID_GQE_ENV}")
fi
ENV_DIR="$("${PYTHON_BIN}" -I -B \
  "${PACKAGE_ROOT}/environment_contract.py" "${RESOLVE_ARGS[@]}")"
PYTHON="${ENV_DIR}/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "CUDA-QX GQE environment not found: ${ENV_DIR}" >&2
  echo "Create it with: bash \"${PACKAGE_ROOT}/setup.sh\"" >&2
  exit 1
fi

SITE_PACKAGES="$("${PYTHON}" -I -B - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
MPI_LIBRARY_DIR="${SITE_PACKAGES}/lib"

exec env \
  -u PYTHONBREAKPOINT \
  -u PYTHONHOME \
  -u PYTHONINSPECT \
  -u PYTHONPATH \
  -u PYTHONPLATLIBDIR \
  -u PYTHONSTARTUP \
  -u PYTHONUSERBASE \
  -u PYTHONWARNINGS \
  PIP_CONFIG_FILE=/dev/null \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_INPUT=1 \
  PYTHONNOUSERSITE=1 \
  PYTHONSAFEPATH=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  LD_LIBRARY_PATH="${MPI_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  MPIR_CVAR_CH4_NETMOD=ofi \
  FI_PROVIDER=tcp \
  MPIR_CVAR_CH4_SHM_ENABLE=0 \
  UCX_TLS=self \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  NUMEXPR_NUM_THREADS=1 \
  MPLBACKEND=Agg \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  QBRAID_GQE_ENV="${ENV_DIR}" \
  MPLCONFIGDIR="${ENV_DIR}/.cache/matplotlib" \
  PYTHONUNBUFFERED=1 \
  "${PYTHON}" -I -B -c \
  'import runpy,sys; source=sys.argv.pop(1); script=sys.argv.pop(1); sys.path.insert(0,source); sys.argv[0]=script; runpy.run_path(script,run_name="__main__")' \
  "${SOURCE_DIR}" "${SOURCE_DIR}/scripts/qbraid_budgeted_gqe.py" "$@"
