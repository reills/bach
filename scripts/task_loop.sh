#!/usr/bin/env bash
# scripts/task_loop.sh — Automated coding-agent task loop for bach-gen
#
# USAGE:
#   bash scripts/task_loop.sh [OPTIONS]
#
# OPTIONS:
#   --dry-run           Print what would happen; do not execute agent commands
#   --init              Initialize (or reset) state file from prompts.md
#   --prompt-file FILE  Path to prompt markdown file (default: prompts.md)
#   --state-file FILE   Path to persistent state TSV (default: .task_runner_state.tsv)
#   --task TASKID       Force-run exactly this task ID (bypasses dependency check)
#   --help              Show this help and exit
#
# ENVIRONMENT VARIABLES:
#   AGENT_MODEL     Default model for both agents: "claude" or "codex" (default: codex)
#   IMPLEMENT_MODEL Override model for implement agent
#   REVIEW_MODEL    Override model for review agent
#   IMPLEMENT_CMD   Shell command for the implement agent
#   REVIEW_CMD      Shell command for the review agent
#   MAX_RETRIES     Max FAIL retries before a task is marked blocked (default: 3)
#   AUTO_COMMIT     Set to 1 to git-commit after each PASS (default: 0)
#   STOP_ON_BLOCKED Set to 1 to exit when a task is blocked (default: 1)
#   DRY_RUN         Set to 1 to enable dry-run mode (default: 0)
#
# FILES CREATED / MANAGED:
#   .task_runner_state.tsv            Persistent state: task_id TAB status TAB retries
#   TODO.md                           Active task prompt (written before each agent run)
#   finished.md                       Written by implement agent (required)
#   review.md                         Written by review agent with VERDICT: PASS|FAIL
#   finished_prompt_summary/
#     PXX-impl-NN.md                  Archived implementation summary per attempt
#     PXX-review-NN.md                Archived review summary per attempt
#   .runner/
#     implement_instructions.txt      Template with {{TASK_ID}} {{TASK_TITLE}} placeholders
#     review_instructions.txt         Template with {{TASK_ID}} {{TASK_TITLE}} {{ATTEMPT}}
#     implement_active_prompt.txt     Generated prompt sent to implement agent (temp)
#     review_active_prompt.txt        Generated prompt sent to review agent (temp)
#
# HOW RESUME WORKS:
#   Re-run the script after an interruption. It reads .task_runner_state.tsv to find
#   where it left off. Tasks with status 'in_progress' will be restarted from the top
#   of that task (implement + review). Tasks marked 'completed' are skipped.
#
# PROMPT FILE FORMAT (prompts.md):
#   ### PXX - Task title
#   - Dependency: `Independent`
#   - Dependency: `Dependent on PYY and PZZ`
#   - Goal: ...
#   - Files: ...
#   - Prompt:
#   ```text
#   ... prompt body ...
#   ```
#
# EXAMPLE RUN:
#   # Initialize state for the first time:
#   bash scripts/task_loop.sh --init
#
#   # Run the loop:
#   IMPLEMENT_CMD='claude --dangerously-skip-permissions' \
#   REVIEW_CMD='claude --dangerously-skip-permissions' \
#   AUTO_COMMIT=1 bash scripts/task_loop.sh
#
#   # Dry-run to see what would happen:
#   bash scripts/task_loop.sh --dry-run
#
#   # Force-run a single task (re-run even if completed):
#   bash scripts/task_loop.sh --task P04

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Defaults (override with environment variables)
# ─────────────────────────────────────────────────────────────────────────────
#   claude → claude --dangerously-skip-permissions -p "<prompt>"
#   codex  → codex exec --full-auto -C . - < prompt_file
AGENT_MODEL="${AGENT_MODEL:-codex}"
IMPLEMENT_MODEL="${IMPLEMENT_MODEL:-$AGENT_MODEL}"
REVIEW_MODEL="${REVIEW_MODEL:-$AGENT_MODEL}"

default_cmd_for_model() {
  local model="$1"
  case "$model" in
    claude) printf '%s\n' 'claude --dangerously-skip-permissions -p' ;;
    codex) printf '%s\n' 'codex exec --full-auto -C . -' ;;
    *)
      echo "[ERROR] Unknown model: $model (expected 'claude' or 'codex')" >&2
      exit 1
      ;;
  esac
}

IMPLEMENT_CMD="${IMPLEMENT_CMD:-$(default_cmd_for_model "$IMPLEMENT_MODEL")}"
REVIEW_CMD="${REVIEW_CMD:-$(default_cmd_for_model "$REVIEW_MODEL")}"

MAX_RETRIES="${MAX_RETRIES:-3}"
AUTO_COMMIT="${AUTO_COMMIT:-1}"
STOP_ON_BLOCKED="${STOP_ON_BLOCKED:-1}"
DRY_RUN="${DRY_RUN:-0}"
LAST_TASK="${LAST_TASK:-}"          # stop after this task completes (e.g. LAST_TASK=P24)

PROMPT_FILE="prompts.md"
STATE_FILE=".task_runner_state.tsv"
RUNNER_DIR=".runner"
ARCHIVE_DIR="finished_prompt_summary"
TODO_FILE="TODO.md"
FINISHED_FILE="finished.md"
REVIEW_FILE="review.md"

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
INIT_MODE=0
FORCE_TASK=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)        DRY_RUN=1 ;;
    --init)           INIT_MODE=1 ;;
    --prompt-file)    PROMPT_FILE="$2"; shift ;;
    --state-file)     STATE_FILE="$2"; shift ;;
    --task)           FORCE_TASK="$2"; shift ;;
    --help|-h)
      grep '^#' "$0" | head -70 | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done

# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────
log()  { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
info() { log "INFO  $*"; }
warn() { log "WARN  $*"; }
err()  { log "ERROR $*" >&2; }
die()  { err "$*"; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight checks
# ─────────────────────────────────────────────────────────────────────────────
[[ -f "$PROMPT_FILE" ]] || die "Prompt file not found: $PROMPT_FILE"
[[ -f "$RUNNER_DIR/implement_instructions.txt" ]] || \
  die "Missing $RUNNER_DIR/implement_instructions.txt — run from repo root after setup"
[[ -f "$RUNNER_DIR/review_instructions.txt" ]] || \
  die "Missing $RUNNER_DIR/review_instructions.txt — run from repo root after setup"

mkdir -p "$ARCHIVE_DIR" "$RUNNER_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Prompt file parsing
#
# The prompts.md format (abbreviated):
#   ### P02 - Create backend service skeleton
#   - Dependency: `Independent`
#   - Prompt:
#   ```text
#   ... body ...
#   ```
#   ### P03 - Next task
#   ...
#
# We use awk to extract per-task sections (start at "### PXX", end at next "### P").
# ─────────────────────────────────────────────────────────────────────────────

# get_task_ids: print all task IDs in document order (e.g., P02, P03, ...)
get_task_ids() {
  grep -E '^### P[0-9]+ ' "$PROMPT_FILE" \
    | sed 's/^### \(P[0-9]\+\) .*/\1/'
}

# get_task_title TASKID: print the one-line title (after "### PXX - ")
get_task_title() {
  local id="$1"
  grep -E "^### ${id} " "$PROMPT_FILE" \
    | sed "s/^### ${id} - //" \
    | head -1
}

# _extract_section TASKID: print the raw lines of just this task's section
# (from "### PXX" up to but not including the next "### P" heading)
_extract_section() {
  local id="$1"
  awk "
    /^### ${id} / { in_section=1 }
    in_section && /^### P[0-9]+ / && !/^### ${id} / { in_section=0 }
    in_section { print }
  " "$PROMPT_FILE"
}

# get_task_deps TASKID: print one dep ID per line; empty output = Independent
get_task_deps() {
  local id="$1"
  local dep_line dep_val

  dep_line=$(_extract_section "$id" | grep -m1 -e '- Dependency:' || true)

  if [[ -z "$dep_line" ]]; then
    warn "No Dependency line found for $id; treating as Independent"
    return
  fi

  # Extract the content inside backticks after "Dependency:"
  # e.g.  "- Dependency: `Dependent on P06 and P10`"  ->  "Dependent on P06 and P10"
  dep_val=$(printf '%s' "$dep_line" \
    | sed "s/.*Dependency: \`//; s/\`.*//")

  if [[ "$dep_val" == "Independent" ]]; then
    return  # no output = no deps
  fi

  # Extract every PXX token from the dep value, one per line
  # Handles: "P02", "P06 and P10", "P04, P07, P11, P15, and P16"
  printf '%s\n' "$dep_val" | grep -oE 'P[0-9]+'
}

# get_task_tests TASKID: print the test file paths from "- Tests:" line
# Returns space-separated test paths, or empty if none specified
get_task_tests() {
  local id="$1"
  local tests_line
  tests_line=$(_extract_section "$id" | grep -m1 -e '- Tests:' || true)
  [[ -z "$tests_line" ]] && return
  # Extract backtick-delimited paths: `tests/foo.py`, `tests/bar.py`
  printf '%s' "$tests_line" | grep -oE '`[^`]+`' | tr -d '`' | tr '\n' ' '
}

# get_task_prompt TASKID: print the body between ```text ... ``` in this section
get_task_prompt() {
  local id="$1"
  _extract_section "$id" | awk '
    /^```text/ { inblock=1; next }
    inblock && /^```/ { inblock=0; next }
    inblock { print }
  '
}

# ─────────────────────────────────────────────────────────────────────────────
# State file management
#
# Format: tab-separated, no header
#   task_id <TAB> status <TAB> retries
# Status values: pending | in_progress | completed | blocked
# ─────────────────────────────────────────────────────────────────────────────

# init_state: create state file from prompt IDs (skips if file exists and --init not set)
init_state() {
  if [[ -f "$STATE_FILE" && "$INIT_MODE" -eq 0 ]]; then
    return  # keep existing state
  fi

  if [[ -f "$STATE_FILE" && "$INIT_MODE" -eq 1 ]]; then
    warn "Reinitializing state file. Existing state backed up."
    cp "$STATE_FILE" "${STATE_FILE}.bak.$(date +%s)"
  fi

  info "Writing state file: $STATE_FILE"
  : > "$STATE_FILE"
  while IFS= read -r id; do
    printf '%s\tpending\t0\n' "$id" >> "$STATE_FILE"
  done < <(get_task_ids)
}

# _state_field TASKID FIELD: FIELD is 2 (status) or 3 (retries)
_state_field() {
  local id="$1" field="$2"
  awk -F'\t' -v id="$id" -v f="$field" '$1==id { print $f }' "$STATE_FILE"
}

get_task_status()  { _state_field "$1" 2; }
get_task_retries() { _state_field "$1" 3; }

# _update_state TASKID STATUS RETRIES: atomically rewrite one row
_update_state() {
  local id="$1" status="$2" retries="$3"
  local tmp
  tmp=$(mktemp)
  awk -F'\t' -v id="$id" -v st="$status" -v ret="$retries" \
    'BEGIN{OFS="\t"} $1==id { $2=st; $3=ret } { print }' \
    "$STATE_FILE" > "$tmp"
  mv "$tmp" "$STATE_FILE"
}

set_task_status() {
  local id="$1" status="$2"
  local retries
  retries=$(get_task_retries "$id")
  _update_state "$id" "$status" "$retries"
}

increment_task_retries() {
  local id="$1"
  local status retries
  status=$(get_task_status "$id")
  retries=$(get_task_retries "$id")
  retries=$(( retries + 1 ))
  _update_state "$id" "$status" "$retries"
}

# find_eligible_task: print the first task that is pending/in_progress with all deps done
find_eligible_task() {
  local ids
  ids=$(get_task_ids)

  while IFS= read -r id; do
    local status
    status=$(get_task_status "$id")
    [[ "$status" == "pending" || "$status" == "in_progress" ]] || continue

    # Verify every declared dependency is completed
    local all_met=1
    while IFS= read -r dep; do
      [[ -z "$dep" ]] && continue
      local dep_status
      dep_status=$(get_task_status "$dep")
      if [[ "$dep_status" != "completed" ]]; then
        all_met=0
        break
      fi
    done < <(get_task_deps "$id")

    if [[ "$all_met" -eq 1 ]]; then
      printf '%s\n' "$id"
      return
    fi
  done <<< "$ids"
}

# any_pending: exit 0 if at least one task is pending or in_progress
any_pending() {
  grep -qE $'\t''(pending|in_progress)'$'\t' "$STATE_FILE"
}

# ─────────────────────────────────────────────────────────────────────────────
# Instruction prompt builder
# ─────────────────────────────────────────────────────────────────────────────

_render_template() {
  local template_file="$1" task_id="$2" task_title="$3" attempt="${4:-1}"
  sed \
    -e "s|{{TASK_ID}}|${task_id}|g" \
    -e "s|{{TASK_TITLE}}|${task_title}|g" \
    -e "s|{{ATTEMPT}}|${attempt}|g" \
    "$template_file"
}

# ─────────────────────────────────────────────────────────────────────────────
# Agent runner
#
# The generated prompt is written to a .runner/*_active_prompt.txt file and
# piped to the agent command on stdin. This allows IMPLEMENT_CMD / REVIEW_CMD
# to be any shell command that reads a prompt from stdin.
#
# Example: IMPLEMENT_CMD='claude --dangerously-skip-permissions'
# ─────────────────────────────────────────────────────────────────────────────

run_agent() {
  local mode="$1"    # "implement" or "review"
  local prompt="$2"
  local active_prompt_file="${RUNNER_DIR}/${mode}_active_prompt.txt"
  local cmd
  local model

  if [[ "$mode" == "implement" ]]; then
    cmd="$IMPLEMENT_CMD"
    model="$IMPLEMENT_MODEL"
  else
    cmd="$REVIEW_CMD"
    model="$REVIEW_MODEL"
  fi

  printf '%s\n' "$prompt" > "$active_prompt_file"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    info "[DRY-RUN] Would run: $cmd"
    info "[DRY-RUN] Prompt preview (first 5 lines):"
    head -5 "$active_prompt_file" | sed 's/^/  > /'
    return 0
  fi

  info "Running $mode agent ($model)..."
  if [[ "$model" == "codex" ]]; then
    $cmd < "$active_prompt_file"
  else
    $cmd "$(cat "$active_prompt_file")"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Review verdict parsing
# ─────────────────────────────────────────────────────────────────────────────

# parse_verdict: extract "PASS" or "FAIL" from review.md
# The first non-blank line must be "VERDICT: PASS" or "VERDICT: FAIL"
parse_verdict() {
  local verdict
  verdict=$(grep -m1 '^VERDICT:' "$REVIEW_FILE" 2>/dev/null \
    | awk '{print $2}' \
    || true)
  printf '%s' "${verdict:-UNKNOWN}"
}

# parse_remaining_work: extract lines after "REMAINING_WORK:" in review.md
parse_remaining_work() {
  awk '/^REMAINING_WORK:/{ found=1; next } found{ print }' "$REVIEW_FILE" 2>/dev/null || true
}

# ─────────────────────────────────────────────────────────────────────────────
# Archive helpers
# ─────────────────────────────────────────────────────────────────────────────

archive_file() {
  local src="$1" dest="$2"
  cp "$src" "$dest"
  info "Archived: $src -> $dest"
}

# ─────────────────────────────────────────────────────────────────────────────
# Git commit helper
# ─────────────────────────────────────────────────────────────────────────────

do_commit() {
  local task_id="$1" task_title="$2"
  [[ "$AUTO_COMMIT" -eq 1 ]] || return 0
  [[ "$DRY_RUN"    -eq 1 ]] && { info "[DRY-RUN] Would commit: ${task_id}: ${task_title}"; return 0; }

  info "Auto-committing for $task_id"
  git add -A
  git commit -m "$(cat <<MSG
${task_id}: ${task_title}

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
MSG
)"
}

# ─────────────────────────────────────────────────────────────────────────────
# Core task runner
#
# Returns:
#   0 = completed (PASS)
#   2 = retry needed (FAIL, retries remaining)
#   1 = hard error (agent error, blocked, unexpected)
# ─────────────────────────────────────────────────────────────────────────────

run_task() {
  local task_id="$1"
  local task_title retries attempt impl_prompt review_prompt

  task_title=$(get_task_title "$task_id")
  retries=$(get_task_retries "$task_id")
  attempt=$(( retries + 1 ))
  local original_status
  original_status=$(get_task_status "$task_id")

  info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  info "Task ${task_id} — attempt ${attempt} / max ${MAX_RETRIES}"
  info "Title: ${task_title}"
  info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  # Guard: already at max retries?
  if [[ "$attempt" -gt "$MAX_RETRIES" ]]; then
    warn "Task $task_id exceeded max retries ($MAX_RETRIES). Marking blocked."
    set_task_status "$task_id" "blocked"
    return 1
  fi

  set_task_status "$task_id" "in_progress"

  # ── Write TODO.md ──────────────────────────────────────────────────────────
  local task_prompt task_tests
  task_prompt=$(get_task_prompt "$task_id")
  task_tests=$(get_task_tests "$task_id")
  if [[ -z "$task_prompt" ]]; then
    die "Could not extract prompt body for $task_id from $PROMPT_FILE"
  fi

  {
    printf '# TODO — Active Task: %s\n\n' "$task_id"
    printf '## %s — %s\n\n' "$task_id" "$task_title"
    printf '%s\n' "$task_prompt"
    if [[ -n "$task_tests" ]]; then
      printf '\n## Test Command\n\n'
      printf 'Run ONLY these targeted tests (do NOT run the full suite):\n\n'
      printf '```bash\nbash docs/skills/python-test-env/scripts/run_tests.sh %s\n```\n' "$task_tests"
    fi
  } > "$TODO_FILE"
  info "Wrote $TODO_FILE"

  # ── Pre-flight: if targeted tests already pass, skip the agent entirely ────
  if [[ "$DRY_RUN" -eq 0 && -n "$task_tests" && "$attempt" -gt 1 ]]; then
    info "Pre-flight: running targeted tests to check if work is already done..."
    if bash docs/skills/python-test-env/scripts/run_tests.sh $task_tests >/dev/null 2>&1; then
      info "Pre-flight PASS — tests already green. Marking $task_id completed."
      set_task_status "$task_id" "completed"
      do_commit "$task_id" "$task_title"
      printf '' > "$TODO_FILE"
      return 0
    else
      info "Pre-flight: tests not passing yet. Proceeding with agent."
    fi
  fi

  # ── Clean previous run artifacts ───────────────────────────────────────────
  rm -f "$FINISHED_FILE" "$REVIEW_FILE"

  # ── IMPLEMENT PASS ─────────────────────────────────────────────────────────
  info "--- IMPLEMENT PASS ---"
  impl_prompt=$(_render_template \
    "${RUNNER_DIR}/implement_instructions.txt" \
    "$task_id" "$task_title" "$attempt")

  if ! run_agent "implement" "$impl_prompt"; then
    err "Implement agent exited non-zero for $task_id"
    return 1
  fi

  if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ ! -f "$FINISHED_FILE" ]]; then
      err "Implement agent did not write $FINISHED_FILE for $task_id"
      err "Re-run after the agent writes this file, or fix the agent command."
      return 1
    fi
    archive_file "$FINISHED_FILE" \
      "${ARCHIVE_DIR}/${task_id}-impl-$(printf '%02d' "$attempt").md"
  fi

  # ── REVIEW PASS ───────────────────────────────────────────────────────────
  info "--- REVIEW PASS ---"
  review_prompt=$(_render_template \
    "${RUNNER_DIR}/review_instructions.txt" \
    "$task_id" "$task_title" "$attempt")

  if ! run_agent "review" "$review_prompt"; then
    err "Review agent exited non-zero for $task_id"
    return 1
  fi

  if [[ "$DRY_RUN" -eq 0 ]]; then
    if [[ ! -f "$REVIEW_FILE" ]]; then
      err "Review agent did not write $REVIEW_FILE for $task_id"
      return 1
    fi
    archive_file "$REVIEW_FILE" \
      "${ARCHIVE_DIR}/${task_id}-review-$(printf '%02d' "$attempt").md"
  fi

  # ── Dry-run short-circuit ──────────────────────────────────────────────────
  if [[ "$DRY_RUN" -eq 1 ]]; then
    info "[DRY-RUN] Would parse verdict and advance state"
    # Restore original status so the state file is not polluted
    set_task_status "$task_id" "$original_status"
    return 0
  fi

  # ── Parse verdict ──────────────────────────────────────────────────────────
  local verdict
  verdict=$(parse_verdict)
  info "Verdict for $task_id: $verdict"

  case "$verdict" in
    PASS)
      info "PASS — marking $task_id completed"
      set_task_status "$task_id" "completed"
      do_commit "$task_id" "$task_title"
      printf '' > "$TODO_FILE"   # clear without deleting
      return 0
      ;;

    FAIL)
      warn "FAIL — preparing remediation for $task_id"
      increment_task_retries "$task_id"
      local new_retries
      new_retries=$(get_task_retries "$task_id")

      if [[ "$new_retries" -ge "$MAX_RETRIES" ]]; then
        err "Task $task_id has hit max retries ($MAX_RETRIES). Marking blocked."
        set_task_status "$task_id" "blocked"
        return 1
      fi

      # Write a focused remediation TODO for the next iteration
      local remaining
      remaining=$(parse_remaining_work)

      {
        printf '# TODO — Remediation: %s (attempt %s)\n\n' "$task_id" "$(( new_retries + 1 ))"
        printf '## Original Task: %s — %s\n\n' "$task_id" "$task_title"
        printf 'This is a remediation run. The previous attempt received a FAIL verdict.\n'
        printf 'Address ONLY the remaining work items listed below. Do not redo work that passed.\n\n'
        printf '## Remaining Work\n\n%s\n\n' "$remaining"
        printf '## Original Task Prompt (reference only)\n\n%s\n' "$task_prompt"
      } > "$TODO_FILE"
      info "Wrote remediation $TODO_FILE (retry $(( new_retries + 1 )) of $MAX_RETRIES)"

      set_task_status "$task_id" "pending"
      return 2   # caller: retry this task
      ;;

    *)
      err "Unexpected verdict '${verdict}' in $REVIEW_FILE"
      err "Expected first line: 'VERDICT: PASS' or 'VERDICT: FAIL'"
      return 1
      ;;
  esac
}

# ─────────────────────────────────────────────────────────────────────────────
# Print current state table (for --init confirmation and debugging)
# ─────────────────────────────────────────────────────────────────────────────

print_state() {
  printf '\n%-6s  %-12s  %s\n' "Task" "Status" "Retries"
  printf '%-6s  %-12s  %s\n' "------" "------------" "-------"
  while IFS=$'\t' read -r id status retries; do
    printf '%-6s  %-12s  %s\n' "$id" "$status" "$retries"
  done < "$STATE_FILE"
  printf '\n'
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

main() {
  info "task_loop.sh starting"
  [[ "$DRY_RUN" -eq 1 ]] && info "(DRY-RUN mode active)"

  init_state

  if [[ "$INIT_MODE" -eq 1 ]]; then
    info "State initialized:"
    print_state
    info "Run without --init to start the task loop."
    exit 0
  fi

  [[ -f "$STATE_FILE" ]] || die "State file missing: $STATE_FILE  (run --init first)"

  local current_task=""
  local exit_code

  while true; do
    # Determine which task to run
    if [[ -n "$FORCE_TASK" ]]; then
      current_task="$FORCE_TASK"
      local forced_status
      forced_status=$(get_task_status "$current_task")
      if [[ "$forced_status" == "completed" ]]; then
        info "Task $current_task is already completed. Marking pending for forced re-run."
        set_task_status "$current_task" "pending"
      fi
    else
      current_task=$(find_eligible_task || true)
    fi

    if [[ -z "$current_task" ]]; then
      if any_pending; then
        warn "No eligible tasks found, but pending tasks remain."
        warn "All remaining pending tasks have unmet (or blocked) dependencies."
        info "Current state:"
        print_state
        exit 1
      else
        info "All tasks completed. Workflow done!"
        print_state
        exit 0
      fi
    fi

    exit_code=0
    run_task "$current_task" || exit_code=$?

    case "$exit_code" in
      0)
        info "Task $current_task PASSED."
        if [[ -n "$LAST_TASK" && "$current_task" == "$LAST_TASK" ]]; then
          info "Reached LAST_TASK=$LAST_TASK. Stopping."
          print_state
          exit 0
        fi
        ;;
      2)
        info "Task $current_task will be retried on next iteration."
        ;;
      *)
        local blocked_status
        blocked_status=$(get_task_status "$current_task")
        if [[ "$blocked_status" == "blocked" ]]; then
          if [[ "$STOP_ON_BLOCKED" -eq 1 ]]; then
            err "Task $current_task is blocked. STOP_ON_BLOCKED=1. Halting."
            print_state
            exit 1
          else
            warn "Task $current_task is blocked. Continuing to next eligible task."
          fi
        else
          err "Task $current_task failed with exit code $exit_code. Runner stopping."
          print_state
          exit 1
        fi
        ;;
    esac

    # If --task was given or dry-run, run exactly once and exit
    [[ -n "$FORCE_TASK" || "$DRY_RUN" -eq 1 ]] && {
      [[ "$DRY_RUN" -eq 1 ]] && { info "[DRY-RUN] Showing next eligible task only. Re-run without --dry-run to execute."; print_state; }
      break
    }

  done
}

main
