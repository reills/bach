#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

pick_python() {
  if CONDA_NO_PLUGINS=true conda run -n bach python -V >/dev/null 2>&1; then
    echo "CONDA_NO_PLUGINS=true conda run -n bach python"
    return 0
  fi

  # Fallback: venv with a real (non-broken) python binary
  for candidate in ".venv/bin/python" ".venv/Scripts/python.exe"; do
    if [[ -x "$candidate" ]] && "$candidate" -V >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

if ! PYTHON_CMD="$(pick_python)"; then
  echo "No usable Python runtime found. Expected conda env 'bach-gen' or a working .venv." >&2
  exit 1
fi

VERIFY_CMD="${PYTHON_CMD} -m pytest -q"
ACTION="${1:-}"

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

if [[ -n "$ACTION" && "$ACTION" == --* ]]; then
  echo "Unknown option: $ACTION" >&2
  exit 1
fi

eval "$VERIFY_CMD $*"
