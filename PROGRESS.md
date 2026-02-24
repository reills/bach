# PROGRESS

- 2026-02-24: Initialized AGENTS.md, SPEC.md, TODO.md, and PROGRESS.md for Stage-1 loop setup.
- 2026-02-24: Added ExecPlan, Codex loop script, config, and schema for structured loop output.
- 2026-02-24: Added python-test-env skill and test helper script; loop now defaults to conda env `bach` and includes environment precheck in prompt.
- 2026-02-24: Fixed loop compatibility with installed Codex CLI (removed hard dependency on `-a`, added auto-detect for approval flags, and fail-fast on CLI usage errors).
- 2026-02-24: Updated loop to use `--full-auto` on newer Codex CLI builds where explicit approval flags are unavailable.
- 2026-02-24: Completed Stage-1 ExecPlan item 1 by defining versioned `EventSpec`/`DescriptorSpec` in `src/tokens/schema.py` (`remi_tab_v1`, `tpq=24`) and added `tests/test_schema.py`.
- 2026-02-24: Completed Stage-1 ExecPlan item 2 by adding `src/tokens/intervals.py` (melodic/harmonic helpers with QA vs production behavior), wiring repairs into `eventize_musicxml`, updating harmonic validation to shared math helpers, and adding `tests/test_intervals.py`; verify run: `53 passed, 1 skipped`.
- 2026-02-24: Completed Stage-1 ExecPlan item 3 by adding `src/tokens/tokenizer.py` (voice-event parsing/serialization, stream canonicalization helpers), wiring `eventize_musicxml` pitched-onset emission through tokenizer serialization for canonical ordering, and adding `tests/test_tokenizer.py`; verify run: `59 passed, 1 skipped`.
- 2026-02-24: Completed Stage-1 ExecPlan item 4 by adding `tests/test_roundtrip_pitch.py` with random-bar pitch reconstruction round-trip checks (anchors + `MEL_INT12`) and sustained-reference `HARM_*` consistency assertions; verify run: `63 passed, 1 skipped`.
