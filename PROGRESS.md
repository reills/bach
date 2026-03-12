## 2026-03-12 - P25

- Added `frontend/src/components/FingeringPicker.tsx`, a lightweight picker component that lists alternate string/fret positions, marks the currently-selected one, and delegates selection to a callback.
- Extended `frontend/src/App.tsx`: made `handleNoteClick` async so it calls `/alt_positions` whenever an `eventHitMap` lookup succeeds; added `handleApplyFingering` which calls `/apply_fingering` and updates `scoreXml` plus `revision` from the response; renders `<FingeringPicker>` in the control panel when alt-position data is available.
- Added `frontend/src/state/types.test.ts` (12 vitest tests) covering `toHitKey`, `getMeasureId`, and `getEventId` pure state helpers, including null/undefined guard paths.
- Added `test` section to `frontend/vite.config.ts` so `npx vitest run` targets only `src/**/*.test.ts` files.
- Ran `npx vitest run` inside `frontend/`: 12 passed. Ran backend `tests/test_api_fingering.py` via `conda run -n bach-gen python -m pytest -q`: 4 passed.

## 2026-03-12 - P24

- Added `/apply_fingering` to `src/api/routes/scores.py`, validating the caller's `revision`, applying one or more fingering selections by `eventId` through the existing canonical fingering helper, and committing the updated score back through the repository so the response returns re-exported MusicXML plus the new revision.
- Extended `tests/test_api_fingering.py` with an end-to-end apply-fingering case that proves the returned MusicXML updates only fingering-related technical tags while preserving rendered pitch content, plus a stale-revision conflict case that returns HTTP `409`.
- Ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_api_fingering.py` and confirmed it passes with `4 passed in 1.20s`.

## 2026-03-12 - P14

- Added a minimal NoteLM inference loader in `src/models/notelm/inference.py` that reads the saved checkpoint config and model weights, resolves and loads the vocab file, and returns an eval-mode model on CPU by default.
- Exported the loader from `src/models/notelm/__init__.py` and `src/models/__init__.py` without changing the existing training checkpoint format in `scripts/train_v1.py`.
- Added `tests/test_load_checkpoint.py`, which saves a tiny synthetic checkpoint plus relative `vocab.json`, reloads it through the new helper, and confirms the restored model produces the same logits.
- Ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_load_checkpoint.py` and confirmed it passes with `1 passed in 0.75s`.

## 2026-03-11 - P02

- Verified the existing FastAPI skeleton under `src/api` already provides `create_app()` and a `/healthz` route with a simple JSON health payload.
- Verified the existing API test uses `fastapi.testclient.TestClient` to assert `/healthz` returns HTTP 200 and `{"status": "ok"}`.
- Did not run `bash docs/skills/python-test-env/scripts/run_tests.sh` in this environment because the active task instructions explicitly said to skip tests here.

## 2026-03-12 - P02

- Kept the existing FastAPI service skeleton under `src/api` and left `create_app()` plus `/healthz` as the backend entry point for later routes.
- Updated the health route to `async def` and replaced the health check test with an `httpx.ASGITransport` request path after reproducing a deadlock in the installed `fastapi.testclient.TestClient` stack.
- Ran `bash docs/skills/python-test-env/scripts/run_tests.sh -- tests/test_api_health.py` and got `1 passed in 0.20s`.

## 2026-03-12 - P02

- Confirmed the FastAPI service skeleton remains in place under `src/api`, with `create_app()` registering the `/healthz` router for later backend routes.
- Kept the in-process health check coverage in `tests/test_api_health.py` and documented why it uses `httpx.ASGITransport`: `fastapi.testclient.TestClient` hangs in the installed Starlette/AnyIO stack, even against a minimal one-route app.
- Verified the targeted health test directly with `CONDA_NO_PLUGINS=true conda run -n bach python -m pytest -q tests/test_api_health.py`, which passed with `1 passed in 0.15s`.
- Checked the current `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_api_health.py` wrapper and found it is presently parsing the positional test path as an action, so it does not execute the targeted test as written.

## 2026-03-12 - P02

- Kept the existing FastAPI backend skeleton in `src/api` unchanged because it already provides the required `create_app()` factory and `/healthz` route for later score routes.
- Updated `tests/test_api_health.py` to spin up the app with a small `TestClient` compatibility shim, because the installed Starlette/httpx stack hangs on `TestClient.get()` even for a one-route FastAPI app.
- Fixed `docs/skills/python-test-env/scripts/run_tests.sh` so positional test paths are passed through to `pytest`, which makes the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_api_health.py` run only the targeted test file.

## 2026-03-12 - P02

- Verified the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_api_health.py` now completes and reports `1 passed in 0.19s`.
- Kept the health check assertion focused on real app behavior: the app factory is instantiated, `/healthz` is requested in-process, and the response body remains `{"status": "ok"}`.

## 2026-03-12 - P02

- Re-read `TODO.md` and confirmed the required backend service skeleton is already present under `src/api` with `create_app()` and a `/healthz` endpoint.
- Re-ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_api_health.py`; it passed with `1 passed in 0.18s`.
- Added this task completion entry and wrote the required handoff files for the runner.

## 2026-03-12 - P03

- Tightened `src/api/canonical/types.py` so canonical score timing fields are explicitly integer ticks and per-part `voice_id` values must be contiguous from `0..N-1`, matching the frontend canonical-score invariants.
- Kept cross-bar `dur_tick` support and optional guitar fingering metadata intact, while clarifying in `frontend-readme.md` that the Python backend model uses snake_case names for the same canonical fields.
- Ran `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_canonical_types.py` and confirmed the targeted canonical types tests pass with `6 passed in 0.29s`.

## 2026-03-12 - P03

- Re-read `TODO.md` and verified the current canonical score model under `src/api/canonical/types.py` already satisfies the task shape and invariants for headers, parts, measures, events, optional guitar fingering, cross-bar sustains, and per-part `voice_id` numbering.
- Re-ran the exact targeted task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_canonical_types.py`; it passed with `6 passed in 0.27s`.
- Wrote the required task handoff files for this run without changing the canonical-score implementation further.

## 2026-03-12 - P02

- Verified the existing FastAPI backend skeleton still matches `TODO.md`: `src/api/app.py` exposes `create_app()` and registers the `/healthz` router from `src/api/routes/health.py`.
- Re-ran the exact targeted task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_api_health.py`; it passed with `1 passed in 0.19s`.
- Wrote the required task handoff files for this run without changing unrelated code.

## 2026-03-12 - P03

- Tightened `src/api/canonical/types.py` so `Part` and `CanonicalScore` reject non-canonical nested objects with explicit `ValueError`s instead of failing later through attribute access.
- Added focused construction tests in `tests/test_canonical_types.py` for invalid nested `Part`, `ScoreHeader`, `Measure`, and `Part` container entries, keeping the existing timing, ID, and voice invariants intact.
- Clarified `frontend-readme.md` so the documented canonical schema now notes that backend `Part` objects store `info: PartInfo` plus `events`, while describing the same logical fields for the frontend.
- Ran `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_canonical_types.py` and confirmed the targeted canonical types tests pass with `8 passed in 0.28s`.

## 2026-03-12 - P04

- Updated `src/api/canonical/from_tokens.py` so raw token voices are remapped onto contiguous canonical `voice_id` values before events are emitted, while still deriving measures from `BAR` and `TIME_SIG` tokens and rebuilding pitches from the existing anchor plus `MEL_INT12` semantics.
- Tightened `tests/test_tokens_to_canonical.py` so the monophonic bridge case uses sparse token voice `3` and proves the canonical score now normalizes it to `voice_id == 0`; the cross-bar sustain coverage remains intact.
- Ran `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_tokens_to_canonical.py tests/test_canonical_types.py` and confirmed the targeted tests pass with `10 passed in 0.29s`.

## 2026-03-12 - P05

- Added pure canonical score lookup/query helpers in `src/api/canonical/ops.py` for measure ID lookup, event ID lookup, events that start inside a measure, carry-in events active at a measure boundary, and replacing just the events that start within one measure.
- Exported the new helpers from `src/api/canonical/__init__.py` without changing the existing token-to-canonical conversion path.
- Added focused coverage in `tests/test_canonical_ops.py` for measure/event lookup, carry-in detection versus in-measure starts, and one-measure event replacement that preserves carry-in and later-measure events.
- Ran `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_canonical_ops.py` and confirmed the targeted tests pass with `3 passed in 0.27s`.

## 2026-03-12 - P06

- Added a minimal canonical-score MusicXML exporter in `src/api/render/musicxml.py` for a single part, emitting measure and note `xml:id` values, `divisions` from `header.tpq`, and barline-aware note splitting with MusicXML tie start/stop markers.
- Added focused coverage in `tests/test_musicxml_export.py` for exported `xml:id` values, `divisions`, and cross-bar note splitting that keeps the same logical canonical event ID on both tied note fragments.
- Ran `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_musicxml_export.py` and confirmed the targeted tests pass with `2 passed in 0.27s`.

## 2026-03-12 - P07

- Added an isolated canonical-score MIDI adapter in `src/api/render/midi.py` that converts canonical header metadata and per-part note events into a `music21` score, then emits MIDI bytes with the canonical `tpq` preserved in the MIDI header.
- Exported the new MIDI renderer from `src/api/render/__init__.py` and added focused coverage in `tests/test_midi_export.py` that builds a simple canonical score, exports MIDI bytes, and verifies the payload is parseable with a valid MIDI header and track chunk.
- Ran `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_midi_export.py` and confirmed the targeted test passes with `1 passed in 0.27s`.

## 2026-03-12 - P09

- Extended `src/api/render/musicxml.py` so pitched canonical events with `fingering` metadata emit MusicXML `<notations><technical><string>/<fret></technical></notations>` using the frontend contract’s high-E=`1` to low-E=`6` numbering derived from the part tuning.
- Added a small hand-authored golden fixture at `tests/fixtures/musicxml/canonical_bridge.xml` plus `tests/test_musicxml_golden.py`, covering measure `xml:id`, note `xml:id`, technical string/fret tags, and cross-bar tie splitting in one backend contract example.
- Ran `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_musicxml_golden.py tests/test_musicxml_export.py` and confirmed the targeted tests pass with `3 passed in 0.27s`.

## 2026-03-12 - P10

- Added `src/tabber/heuristic.py` with a first-pass guitar tabbing heuristic for standard six-string tuning, covering canonical `Event` input plus a narrow `TabNote` DTO, candidate fret generation inside a playable range, and same-onset unique-string assignment via a small brute-force search.
- Preferred low-fret positions by cost, which naturally favors open strings when available, and raised explicit `ValueError`s when a same-onset voicing can only be realized by reusing one string.
- Added focused coverage in `tests/test_tabber_heuristic.py` for open-string preference, a simple simultaneous chord, and an impossible voicing case, then ran `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_tabber_heuristic.py` and confirmed it passes with `3 passed in 0.28s`.

## 2026-03-12 - P08

- Added `src/tabber/ascii.py` with a deterministic ASCII tab renderer for fingered canonical `Event` sequences, emitting labeled string rows from high string to low string and aligning each onset by the widest fret label at that onset.
- Exported the renderer from `src/tabber/__init__.py` and added focused coverage in `tests/test_ascii_tab.py` for a short multi-string phrase plus the required fingering-validation failure path.
- Ran `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_ascii_tab.py` and confirmed the targeted tests pass with `2 passed in 0.28s`.

## 2026-03-12 - P11

- Verified `src/api/render/musicxml.py` already emits MusicXML `<notations><technical><string>/<fret></technical></notations>` from canonical `fingering` metadata using the documented high-E=`1` to low-E=`6` numbering convention.
- Added the missing targeted coverage in `tests/test_musicxml_tab_encoding.py` to assert backend string index `5` exports as MusicXML string `1` and backend string index `0` exports as MusicXML string `6`, both with the expected fret tags.
- Ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_musicxml_tab_encoding.py tests/test_musicxml_export.py` and confirmed it passes with `4 passed in 0.28s`.

## 2026-03-12 - P12

- Added `alternate_fingerings_for_event()` in `src/tabber/heuristic.py`, exposing the existing bounded per-string search as a public helper for a single pitched canonical `Event` while preserving the current deterministic low-fret ordering.
- Exported the helper from `src/tabber/__init__.py` and added focused coverage in `tests/test_tabber_alternates.py` for a pitch with multiple valid positions plus a high pitch that is limited to one in-range position at the current fret cap.
- Ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_tabber_alternates.py` and confirmed it passes with `2 passed in 0.27s`.

## 2026-03-12 - P13

- Added `src/api/canonical/fingering.py` with `FingeringSelection` plus `apply_fingering_selections()`, a pure canonical-score helper that matches updates by `event_id`, rebuilds only `fingering`, and rejects missing events or selections that change pitch or timing.
- Exported the helper from `src/api/canonical/__init__.py` and added focused coverage in `tests/test_canonical_fingering.py` for a successful fingering change plus the required unknown-`event_id` rejection.
- Ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_canonical_fingering.py` and confirmed it passes with `2 passed in 0.27s`.

## 2026-03-12 - P14

- Verified the NoteLM checkpoint inference helper in `src/models/notelm/inference.py` loads checkpoint config, model weights, and vocab metadata into an eval-mode model on CPU by default, while keeping the interface minimal.
- Confirmed the focused regression coverage in `tests/test_load_checkpoint.py` saves a tiny synthetic checkpoint plus `vocab.json`, reloads it through `load_notelm_checkpoint()`, and checks both restored metadata and logits.
- Re-ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_load_checkpoint.py` and confirmed it passes with `1 passed in 0.76s`.

## 2026-03-12 - P14

- Re-read `TODO.md` and confirmed the existing checkpoint loader in `src/models/notelm/inference.py` still matches the task: it restores `NoteLM` weights plus vocab metadata and returns an eval-mode CPU model by default.
- Re-verified `tests/test_load_checkpoint.py` covers the required synthetic save-and-reload path with a tiny checkpoint and `vocab.json`, checking restored config, vocab path, step metadata, and logits.
- Ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_load_checkpoint.py` and confirmed it passes with `1 passed in 0.73s`.

## 2026-03-12 - P14

- Re-read `TODO.md` carefully and verified the existing `load_notelm_checkpoint()` helper in `src/models/notelm/inference.py` already satisfies the task requirements without further code changes.
- Re-ran the exact targeted task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_load_checkpoint.py` and confirmed it passes with `1 passed in 0.74s`.
- Wrote the required handoff file for this run and left the implementation unchanged because the task was already complete in the workspace.

## 2026-03-12 - P14

- Re-read `TODO.md` and verified the existing inference checkpoint loader in `src/models/notelm/inference.py` still matches the task requirements without additional code changes.
- Ran the exact targeted task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_load_checkpoint.py` and confirmed it passes with `1 passed in 0.73s`.
- Wrote the required `finished.md` handoff for this run.

## 2026-03-12 - P15

- Added a minimal autoregressive generation entry point in `src/inference/generate_v1.py` that loads a NoteLM checkpoint, resolves string or integer seed tokens against the checkpoint vocab, steps the model one token at a time, and stops on `max_length` or an inferred/configured EOS token.
- Reused the existing decoding helpers by building either `Sampler` or `SCGSampler` from the requested decoding settings instead of introducing a separate generation-specific sampling path.
- Added focused coverage in `tests/test_generate_v1.py` with a tiny mocked autoregressive model to verify EOS termination, SCG-guided sampling, and context cropping to the model `max_seq_len` without requiring training.
- Ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_generate_v1.py tests/test_load_checkpoint.py` and confirmed it passes with `3 passed in 0.76s`.

## 2026-03-12 - P16

- Added `src/inference/controls.py`, a thin compose-facing wrapper that normalizes user-entered keys like `f# minor` into the training token format and reuses the existing prefix-token builder so compose controls stay in the same `KEY`, `STYLE`, `DIFFICULTY`, `MEAS` order as training.
- Added focused coverage in `tests/test_controls.py` for key normalization, normalized style/difficulty token labels, token ordering, and rejection of non-positive measure counts.
- Updated `README.md` with a small example showing how compose controls map to prefix tokens.
- Ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_controls.py` and confirmed it passes with `3 passed in 0.95s`.

## 2026-03-12 - P17

- Added `src/api/compose_service.py`, an internal compose pipeline that calls the generation loop, converts generated tokens into a canonical score, tabs the events, renders MusicXML and MIDI, and returns frontend-ready `measure_map` and `event_hit_map` metadata alongside the generated score.
- Added `tests/test_compose_service.py`, which stubs the generation result and verifies the full transformation path including generator arguments, canonical measure IDs, tabbed MusicXML note IDs and technical tags, cross-bar event hit mapping, and parseable MIDI bytes.
- Ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_compose_service.py` and confirmed it passes with `1 passed in 1.09s`.

## 2026-03-12 - P17

- Kept the existing task-scoped compose pipeline in `src/api/compose_service.py`, which already wires generation output through canonical conversion, tab assignment, MusicXML rendering, MIDI rendering, and frontend measure/event maps.
- Exported `compose_baseline` and `ComposeServiceResult` from `src/api/__init__.py` so the internal compose service has a clean package-level entry point without adding any HTTP route surface.
- Re-ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_compose_service.py` and confirmed it passes with `1 passed in 1.14s`.

## 2026-03-12 - P18

- Added `src/api/store.py` with a small replaceable repository surface plus an in-memory implementation that creates score and draft IDs, tracks integer revisions, commits or discards drafts, and raises `StaleRevisionError` on stale base revisions or stale draft commits.
- Exported the store types from `src/api/__init__.py` and added focused coverage in `tests/test_store.py` for the full draft lifecycle and stale revision rejection.
- Ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_store.py` and confirmed it passes with `2 passed in 1.17s`.

## 2026-03-12 - P19

- Added `src/api/services/inpaint.py`, a minimal window-mode inpaint preview service that creates a draft from the in-memory score repository, identifies carry-in events at the target measure boundary, preserves those carry-ins, replaces only events whose `start_tick` falls inside the selected measure, and returns draft metadata including `changed_measure_ids` and `locked_event_ids`.
- Added `tests/test_inpaint_service.py` to cover carry-in preservation plus locked-event reporting, and one-measure replacement that leaves earlier and later measures untouched while exercising the replacement callback contract.
- Ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_inpaint_service.py tests/test_canonical_ops.py` and confirmed it passes with `5 passed in 1.18s`.

## 2026-03-12 - P19

- Updated `src/api/services/inpaint.py` so `changed_measure_ids` is derived from the actual replacement-event spans instead of always reporting only the selected measure; cross-bar replacements now mark every downstream measure they overlap.
- Extended `tests/test_inpaint_service.py` with a focused regression case for a replacement event that starts in the selected measure and sustains into the next measure, while keeping the existing carry-in preservation and single-measure replacement coverage intact.
- Ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_inpaint_service.py tests/test_canonical_ops.py` and confirmed it passes with `6 passed in 1.17s`.

## 2026-03-12 - P20

- Added `src/api/routes/scores.py` and updated `src/api/app.py` so FastAPI now exposes `/compose`, `/inpaint_preview`, `/commit_draft`, and `/discard_draft`, with payloads matching the existing frontend camelCase API types and `409` responses for stale revision conflicts.
- Reused the existing compose and inpaint services by storing composed scores in the in-memory draft repository, returning frontend `measureMap` and `eventHitMap` metadata from canonical scores, and validating score/draft ownership on commit or discard.
- Added `tests/test_api_scores.py` to cover one compose request, one full preview-to-commit flow, and a stale draft commit conflict; ran `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_api_scores.py tests/test_api_health.py` and confirmed it passes with `4 passed in 1.18s`.

## 2026-03-12 - P20

- Re-read `TODO.md` and verified the current workspace already contains the requested FastAPI compose and draft routes in `src/api/routes/scores.py`, wired through `src/api/app.py` with frontend-aligned camelCase payload fields and `409` handling for stale revisions.
- Verified the existing route coverage in `tests/test_api_scores.py` still matches the task requirements: one compose request, one preview-to-commit flow, and a stale draft commit conflict, alongside the existing `tests/test_api_health.py` health check.
- Re-ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_api_scores.py tests/test_api_health.py` and confirmed it passes with `4 passed in 1.21s`.

## 2026-03-12 - P20

- Re-validated the existing P20 FastAPI implementation against `TODO.md`: `/compose`, `/inpaint_preview`, `/commit_draft`, and `/discard_draft` are already wired through the app factory with frontend-aligned camelCase payloads and stale-revision conflicts mapped to HTTP `409`.
- Confirmed the focused route coverage remains in place in `tests/test_api_scores.py` for one compose request, one preview-to-commit draft flow, and a stale draft conflict, with `tests/test_api_health.py` still covering `/healthz`.
- Ran the exact task test command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_api_scores.py tests/test_api_health.py` and confirmed it passes with `4 passed in 1.17s`.

## 2026-03-12 - P21

- Updated `src/api/compose_service.py` so `measureMap` and `eventHitMap` are generated by traversing the exported MusicXML structure directly, which keeps the bar/voice/beat/note hit keys deterministic for the same rendered score layout instead of inferring them from canonical event timing alone.
- Added `tests/test_hit_map.py` with a real polyphonic canonical-score export case covering per-voice beat indexing, leading/trailing rest gaps, and cross-measure carry notes, plus a focused MusicXML contract case for non-zero `noteIndex` on chord notes.
- Ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_hit_map.py` and confirmed it passes with `2 passed in 1.23s`.

## 2026-03-12 - P22

- Added a shared `export_score()` helper in `src/api/compose_service.py` so the MusicXML export, `measureMap`, and `eventHitMap` are generated together from one centralized export path.
- Updated the compose and inpaint preview flows to return those precomputed maps from their service results, and switched the draft commit route to reuse the same helper instead of rebuilding hit maps in the route layer.
- Extended `tests/test_api_scores.py` to assert `measureMap` and `eventHitMap` are present on compose, preview, and commit responses with valid hit-key structure, and updated `tests/test_hit_map.py` to cover the shared export helper.
- Ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_api_scores.py tests/test_hit_map.py` and confirmed it passes with `5 passed in 1.16s`.

## 2026-03-12 - P23

- Added `/alt_positions` to `src/api/routes/scores.py`, resolving the frontend `eventHitKey` through the current exported hit map for the stored score, then mapping that canonical `eventId` back to the score event and returning compact picker options with `stringIndex`, `fret`, and `selected`.
- Added `tests/test_api_fingering.py` to cover a valid carry-note lookup that only resolves correctly via the exported hit map and a missing-hit-key `404` case, keeping the route contract focused on the frontend click flow.
- Updated `frontend/src/api/types.ts` and `frontend/src/api/client.ts` with a concrete `AltPositionsResponse` type for the picker API.
- Ran the exact task command `bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_api_fingering.py` and confirmed it passes with `2 passed in 1.15s`.
