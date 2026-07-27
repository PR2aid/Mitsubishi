#!/usr/bin/env bash
# Internal implementation used by Source_Code/setup.sh.
# Usage:
#   bash source/scripts/setup_qbraid_gqe.sh --setup-only
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_ROOT="$(cd "${SOURCE_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:---smoke}"

case "${MODE}" in
  --smoke|--full|--setup-only) ;;
  *)
    echo "Usage: bash source/scripts/setup_qbraid_gqe.sh [--smoke|--full|--setup-only]" >&2
    exit 2
    ;;
esac

RESOLVE_ARGS=(resolve --root "${PACKAGE_ROOT}")
if [[ -n "${QBRAID_GQE_ENV:-}" ]]; then
  RESOLVE_ARGS+=(--override "${QBRAID_GQE_ENV}")
fi
ENV_DIR="$("${PYTHON_BIN}" -I -B \
  "${PACKAGE_ROOT}/environment_contract.py" "${RESOLVE_ARGS[@]}")"
export QBRAID_GQE_ENV="${ENV_DIR}"

# Remove every ambient pip control, including future/unknown PIP_* names, before
# venv/ensurepip or any explicit pip child can start. Re-add only our contract.
while IFS= read -r PIP_VARIABLE; do
  if [[ "${PIP_VARIABLE}" == PIP_* ]]; then
    unset "${PIP_VARIABLE}"
  fi
done < <(compgen -e)
export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

CLEAN_ENV=(
  env
  -u PYTHONBREAKPOINT
  -u PYTHONHOME
  -u PYTHONINSPECT
  -u PYTHONPATH
  -u PYTHONPLATLIBDIR
  -u PYTHONSTARTUP
  -u PYTHONUSERBASE
  -u PYTHONWARNINGS
  PIP_CONFIG_FILE=/dev/null
  PIP_DISABLE_PIP_VERSION_CHECK=1
  PIP_NO_INPUT=1
  PYTHONNOUSERSITE=1
  PYTHONSAFEPATH=1
  PYTHONDONTWRITEBYTECODE=1
  PYTHONUNBUFFERED=1
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  MPLBACKEND=Agg
  UCX_TLS=self
  QBRAID_GQE_ENV="${ENV_DIR}"
)

"${CLEAN_ENV[@]}" "${PYTHON_BIN}" -I -B - <<'PY'
import platform
import os
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"This release lock is validated for Python 3.12; found {platform.python_version()}. "
        "Select a qBraid Python 3.12 kernel or rerun with "
        "`PYTHON_BIN=python3.12 bash setup.sh`."
    )
if platform.system() != "Linux" or platform.machine().lower() not in {"x86_64", "amd64"}:
    raise SystemExit(
        "This release lock is validated for Linux x86-64; found "
        f"{platform.system()} {platform.machine()}. Launch a Linux x86-64 "
        "qBraid instance and rerun setup."
    )
cpu_count = os.cpu_count() or 0
if cpu_count < 4:
    raise SystemExit(
        "This release requires at least 4 visible vCPU to preserve the "
        f"validated thread contract; found {cpu_count}"
    )
print(f"Using Python {sys.version.split()[0]} on {platform.system()} {platform.machine()}")
print(f"Visible vCPU: {cpu_count}")
PY

if [[ -e "${ENV_DIR}" ]]; then
  if [[ ! -d "${ENV_DIR}" || ! -x "${ENV_DIR}/bin/python" ]]; then
    echo "Existing environment is incomplete and will not be repaired in place: ${ENV_DIR}" >&2
    echo "Choose a new QBRAID_GQE_ENV path and rerun setup." >&2
    exit 1
  fi
  "${CLEAN_ENV[@]}" "${PYTHON_BIN}" -I -B \
    "${PACKAGE_ROOT}/environment_contract.py" check-disk \
    --root "${PACKAGE_ROOT}" --override "${ENV_DIR}" --mode reuse
  echo "Verifying existing environment without modifying it: ${ENV_DIR}"
  if ! "${CLEAN_ENV[@]}" "${PYTHON_BIN}" -I -B \
      "${PACKAGE_ROOT}/install_locked_requirements.py" check \
      --python "${ENV_DIR}/bin/python" \
    || ! "${CLEAN_ENV[@]}" "${ENV_DIR}/bin/python" -I -B \
      "${PACKAGE_ROOT}/verify_environment.py" --smoke; then
    echo "Existing environment failed verification and was left unchanged: ${ENV_DIR}" >&2
    echo "Choose a new QBRAID_GQE_ENV path; in-place repair is intentionally disabled." >&2
    exit 1
  fi
else
  "${CLEAN_ENV[@]}" "${PYTHON_BIN}" -I -B \
    "${PACKAGE_ROOT}/environment_contract.py" check-disk \
    --root "${PACKAGE_ROOT}" --override "${ENV_DIR}" --mode fresh
  # Copies avoid host-specific interpreter symlinks when a qBraid project is
  # moved, snapshotted, or restored under a different workspace path.
  "${CLEAN_ENV[@]}" "${PYTHON_BIN}" -I -B -m venv --copies "${ENV_DIR}"
  "${CLEAN_ENV[@]}" "${PYTHON_BIN}" -I -B \
    "${PACKAGE_ROOT}/install_locked_requirements.py" install \
    --python "${ENV_DIR}/bin/python" \
    --lock "${PACKAGE_ROOT}/requirements.lock" \
    --batch-size 4

  "${CLEAN_ENV[@]}" "${PYTHON_BIN}" -I -B \
    "${PACKAGE_ROOT}/install_locked_requirements.py" check \
    --python "${ENV_DIR}/bin/python"
  "${CLEAN_ENV[@]}" "${ENV_DIR}/bin/python" -I -B \
    "${PACKAGE_ROOT}/verify_environment.py" --smoke
fi

echo "Environment ready: ${ENV_DIR}"

if [[ "${MODE}" == "--setup-only" ]]; then
  echo "Run the judge workflow with:"
  echo "  source \"${ENV_DIR}/bin/activate\""
  echo "  python -I -B \"${PACKAGE_ROOT}/certify_release.py\" --full"
  exit 0
fi

exec bash "${SOURCE_DIR}/scripts/run_qbraid_gqe.sh" "${MODE}"
