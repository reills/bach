#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-gpt-5-codex}"
MAX_ITERS="${MAX_ITERS:-20}"
NO_PROGRESS_LIMIT="${NO_PROGRESS_LIMIT:-2}"
USE_LIVE_SEARCH="${USE_LIVE_SEARCH:-0}"

choose_verify_cmd() {
  if CONDA_NO_PLUGINS=true conda run -n bach python -V >/dev/null 2>&1; then
    echo "CONDA_NO_PLUGINS=true conda run -n bach python -m pytest -q"
    return 0
  fi

  if [[ -x ".venv/bin/python" ]]; then
    echo ".venv/bin/python -m pytest -q"
    return 0
  fi

  echo "pytest -q"
}

VERIFY_CMD="${VERIFY_CMD:-$(choose_verify_cmd)}"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "Using verify command: $VERIFY_CMD"

CODEX_HELP="$(codex exec --help 2>&1 || true)"
EXEC_MODE_ARGS=(-s workspace-write)
APPROVAL_ARGS=()
if printf '%s' "$CODEX_HELP" | rg -q -- '--ask-for-approval'; then
  APPROVAL_ARGS=(--ask-for-approval never)
elif printf '%s' "$CODEX_HELP" | rg -q -- '(^|[[:space:]])-a,'; then
  APPROVAL_ARGS=(-a never)
elif printf '%s' "$CODEX_HELP" | rg -q -- '--full-auto'; then
  # Newer CLI versions removed explicit approval flag. --full-auto provides non-interactive behavior.
  EXEC_MODE_ARGS=(--full-auto)
else
  APPROVAL_ARGS=()
fi

# Safety: new branch (or reuse if it already exists)
BRANCH="${BRANCH:-codex-loop-$(date +%Y%m%d-%H%M%S)}"
if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
  git switch "$BRANCH" >/dev/null
else
  git switch -c "$BRANCH" >/dev/null
fi

OUTDIR=".codex/loop_runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

SCHEMA_PATH="loop_output.schema.json"
no_progress_count=0

for ((i=1; i<=MAX_ITERS; i++)); do
  echo "=== iter $i / $MAX_ITERS ==="

  SNAP="$OUTDIR/context_$i.md"
  {
    echo "# Context (iter $i) — $(date -Is)"
    echo
    echo "## Git status"
    git status -sb || true
    echo
    echo "## Diffstat"
    git --no-pager diff --stat || true
    echo
    echo "## Verify (may fail)"
    ( eval "$VERIFY_CMD" ) || true
  } > "$SNAP" 2>&1

  PROMPT="$OUTDIR/prompt_$i.md"
  cat > "$PROMPT" <<PROMPT_EOF
You are working in this repo.

Read:
- AGENTS.md
- SPEC.md
- TODO.md
- PROGRESS.md
- docs/exec/EXECPLAN_stage1.md
- docs/skills/python-test-env/SKILL.md
- $SNAP (current context snapshot)

Before coding, run:
- bash docs/skills/python-test-env/scripts/run_tests.sh --check

Do the next unchecked item in docs/exec/EXECPLAN_stage1.md.
Keep TODO.md in sync.
Run the verify command mentioned in AGENTS.md (or reuse the failing one from the context).
Update TODO.md (check completed items) and append notes to PROGRESS.md.

Your final message MUST be valid JSON matching loop_output.schema.json.
If you made no progress, set progress_made=false and explain why in summary.
PROMPT_EOF

  OUT_JSON="$OUTDIR/out_$i.json"
  OUT_LOG="$OUTDIR/iter_${i}.out"
  ERR_LOG="$OUTDIR/iter_${i}.err"

  SEARCH_FLAG=""
  if [[ "$USE_LIVE_SEARCH" == "1" ]]; then
    SEARCH_FLAG="--search"
  fi

  codex exec -m "$MODEL" "${EXEC_MODE_ARGS[@]}" "${APPROVAL_ARGS[@]}" \
    --output-schema "$SCHEMA_PATH" \
    --output-last-message "$OUT_JSON" \
    $SEARCH_FLAG - < "$PROMPT" \
    >"$OUT_LOG" 2>"$ERR_LOG" || true

  # Fail fast on CLI/config errors instead of silently looping.
  if [[ ! -s "$OUT_JSON" && -s "$ERR_LOG" ]] && rg -q "^error:|^Usage: codex exec" "$ERR_LOG"; then
    echo "codex exec failed before producing output. See: $ERR_LOG"
    sed -n '1,40p' "$ERR_LOG"
    break
  fi

  # Commit any changes for rollback points
  if ! git diff --quiet; then
    git add -A
    git commit -m "codex loop iter $i" >/dev/null || true
  fi

  # Parse JSON output (fallback-safe)
  if [[ ! -s "$OUT_JSON" ]]; then
    cat > "$OUT_JSON" <<'EOF_JSON'
{"done": false, "progress_made": false, "summary": "No JSON output produced.", "tests_ran": false, "tests_status": "not_run", "next_step": "Check logs and rerun.", "stop_reason": "no_output"}
EOF_JSON
  fi

  done_flag="$(python - <<PY
import json
with open("$OUT_JSON", "r", encoding="utf-8") as f:
    data = json.load(f)
print(str(data.get("done", False)).lower())
PY
)"

  progress_flag="$(python - <<PY
import json
with open("$OUT_JSON", "r", encoding="utf-8") as f:
    data = json.load(f)
print(str(data.get("progress_made", False)).lower())
PY
)"

  tests_status="$(python - <<PY
import json
with open("$OUT_JSON", "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get("tests_status", "not_run"))
PY
)"

  echo "done=$done_flag progress_made=$progress_flag tests_status=$tests_status"

  if [[ "$progress_flag" != "true" ]]; then
    no_progress_count=$((no_progress_count+1))
  else
    no_progress_count=0
  fi

  if [[ "$done_flag" == "true" ]]; then
    echo "done=true reported. stopping."
    break
  fi

  if [[ "$no_progress_count" -ge "$NO_PROGRESS_LIMIT" ]]; then
    echo "no progress for $NO_PROGRESS_LIMIT iterations. stopping."
    break
  fi

done

echo "Branch: $BRANCH"
echo "Logs in: $OUTDIR"
