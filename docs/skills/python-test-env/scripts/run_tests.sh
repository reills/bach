#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

pick_python() {
  if CONDA_NO_PLUGINS=true conda run -n bach python -V >/dev/null 2>&1; then
    echo "CONDA_NO_PLUGINS=true conda run -n bach python"
    return 0
  fi

  if [[ -x ".venv/bin/python" ]]; then
    echo ".venv/bin/python"
    return 0
  fi

  return 1
}

if ! PYTHON_CMD="$(pick_python)"; then
  echo "No usable Python runtime found. Expected conda env 'bach' or executable .venv/bin/python." >&2
  exit 1
fi

VERIFY_CMD="${PYTHON_CMD} -m pytest -q"
ACTION="${1:-run}"

if [[ "$ACTION" == "--print-cmd" ]]; then
  echo "$VERIFY_CMD"
  exit 0
fi

if [[ "$ACTION" == "--check" ]]; then
  # Validate core imports before running full test suite.
  eval "$PYTHON_CMD - <<'PY'
import importlib.util
import sys

required = ['pytest', 'torch', 'music21', 'pandas', 'pyarrow']
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print('Missing packages:', ', '.join(missing))
    sys.exit(1)
PY"
  echo "Environment check passed."
  echo "Using verify command: $VERIFY_CMD"
  exit 0
fi

shift || true
eval "$VERIFY_CMD $*"
