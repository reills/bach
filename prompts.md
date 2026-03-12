# Coding-Agent Backlog

This file breaks the remaining project work into coding-agent-sized tasks.
Each task is intentionally narrow enough for one run. Prompts are written so
an agent can pick one item, make the change, add tests, and stop.

## Current Baseline

These parts already exist and should usually be reused, not rewritten:

- Token schema and tokenizer in `src/tokens/schema.py`, `src/tokens/tokenizer.py`
- MusicXML eventization in `src/tokens/eventizer.py`
- Round-trip helpers in `src/tokens/roundtrip.py`
- Dataset builder and vocab builder in `scripts/make_dataset.py`, `scripts/build_vocab.py`
- Bar descriptor logic in `src/dataio/descriptors.py`
- Dataset loading and packing in `src/dataio/dataset.py`, `src/dataio/collate_miditok.py`
- NoteLM model skeleton in `src/models/notelm/model.py`
- Basic decoding helpers in `src/utils/decoding/sampler.py`, `src/utils/decoding/rules.py`, `src/utils/decoding/scg.py`
- Frontend shell and AlphaTab viewer in `frontend/src/App.tsx`, `frontend/src/components/ScoreViewer.tsx`
- Local frontend mock mode in `frontend/src/mock/localData.ts`

## Dependency Labels

- `Independent`: can be started now without waiting for another unfinished task.
- `Dependent on Pxx`: should wait for the referenced unfinished task.

## Working Rules For Agents

- Keep diffs small and focused to the single prompt.
- Reuse existing modules before adding new abstractions.
- Add or update tests in the same run for behavior you change.
- Run the relevant tests with `bash docs/skills/python-test-env/scripts/run_tests.sh`.

## Prompts
 

### P02 - Create backend service skeleton
- Dependency: `Independent`
- Goal: add a minimal FastAPI app entrypoint and health check so the backend has a stable place to grow.
- Files: `src/api/__init__.py`, `src/api/app.py`, `src/api/routes/health.py`, optional `scripts/run_api.py`, `tests/test_api_health.py`
- Tests: `tests/test_api_health.py`
- Prompt:

```text
Add a minimal FastAPI service skeleton under src/api with an app factory and a /healthz endpoint. Keep the structure small and ready for later score routes. Add a tiny test that spins up the app with TestClient and asserts /healthz returns 200 with a simple JSON payload. Reuse existing project style and keep imports straightforward. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P03 - Define canonical score types
- Dependency: `Dependent on P02`
- Goal: implement the backend canonical score model described in frontend-readme.md.
- Files: `src/api/canonical/types.py`, `src/api/canonical/__init__.py`, `tests/test_canonical_types.py`, `frontend-readme.md`
- Tests: `tests/test_canonical_types.py`
- Prompt:

```text
Implement canonical score data structures under src/api/canonical/types.py using dataclasses or pydantic models. Include header fields, part info, measures, events, and optional fingering data needed for guitar output. Match the invariants in frontend-readme.md: quantized timing, durTick across barlines, stable measure IDs, stable event IDs, and voiceId per part. Add focused tests for construction and basic invariants. If frontend-readme.md needs a small clarification to match the final model shape, update it. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P04 - Token stream to canonical score bridge
- Dependency: `Dependent on P03`
- Goal: convert the existing token/event stream into canonical score objects with stable measure and event IDs.
- Files: `src/api/canonical/from_tokens.py`, `src/tokens/roundtrip.py`, `tests/test_tokens_to_canonical.py`
- Tests: `tests/test_tokens_to_canonical.py` `tests/test_canonical_types.py`
- Prompt:

```text
Add a converter from the existing token stream to the canonical score model. Reuse token semantics already implemented in src/tokens and do not duplicate interval logic. The converter should derive measures from BAR and TIME_SIG tokens, rebuild pitches from ABS_VOICE and MEL_INT12, assign stable event IDs, and map VOICE_v to canonical voiceId. Add tests that cover a simple monophonic example and a cross-bar sustained note. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P05 - Canonical score utilities for measure and event lookup
- Dependency: `Dependent on P03`
- Goal: centralize bar lookup, event lookup, and revision-safe mutation helpers.
- Files: `src/api/canonical/ops.py`, `tests/test_canonical_ops.py`
- Tests: `tests/test_canonical_ops.py`
- Prompt:

```text
Implement small canonical score utility functions for looking up measures by measureId, events by eventId, events starting inside a measure, and carry-in events active at a measure start. Keep the functions pure and easy to test because later API routes will depend on them. Add tests for measure lookup, carry-in detection, and event replacement inside one measure. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P06 - MusicXML exporter from canonical score
- Dependency: `Dependent on P04`
- Goal: produce MusicXML from canonical score, including stable IDs and barline tie splitting.
- Files: `src/api/render/musicxml.py`, `src/api/render/__init__.py`, `tests/test_musicxml_export.py`
- Tests: `tests/test_musicxml_export.py`
- Prompt:

```text
Implement canonical score to MusicXML export in src/api/render/musicxml.py. Support measure xml:id, note xml:id, divisions derived from tpq, and tie splitting when durTick crosses a barline. Keep the exporter MVP-focused for one guitar part. Add tests that assert a cross-bar note becomes two MusicXML notes with tie start/stop and preserves the same logical event identity. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P07 - MIDI exporter from canonical score
- Dependency: `Dependent on P04`
- Goal: generate MIDI directly from canonical score so API routes can return playable output without depending on the frontend.
- Files: `src/api/render/midi.py`, `tests/test_midi_export.py`
- Tests: `tests/test_midi_export.py`
- Prompt:

```text
Add a canonical score to MIDI exporter. Reuse music21 if that is the shortest path, but keep the adapter isolated in src/api/render/midi.py. Cover one test that exports a simple score and verifies the output file or byte payload is non-empty and structurally valid enough for the current test harness. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P08 - ASCII tab renderer
- Dependency: `Dependent on P10`
- Goal: satisfy the root README deliverable for ASCII tab output.
- Files: `src/api/render/ascii_tab.py`, `tests/test_ascii_tab.py`, `README.md`
- Tests: `tests/test_ascii_tab.py`
- Prompt:

```text
Implement a simple ASCII tab renderer that consumes canonical events with assigned string/fret positions and emits a readable six-line tab block. Keep the first version deterministic and easy to reason about; it does not need advanced engraving. Add tests for a short phrase across multiple strings and update README.md only if needed to describe the concrete output format. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P09 - Golden tests for canonical -> MusicXML behavior
- Dependency: `Dependent on P06`
- Goal: lock down the specific frontend-readme.md invariants before API work expands.
- Files: `tests/test_musicxml_golden.py`, optional fixtures under `tests/fixtures/`
- Tests: `tests/test_musicxml_golden.py` `tests/test_musicxml_export.py`
- Prompt:

```text
Add golden-style tests for the canonical-to-MusicXML bridge. Cover measure IDs, event IDs, string/fret technical tags for fretted notes, and cross-bar tie splitting. Keep fixtures small and hand-authored. The goal is to protect the backend contract expected by the frontend, not to build a large snapshot suite. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P10 - Heuristic guitar tabber core
- Dependency: `Independent`
- Goal: assign playable string/fret positions to symbolic pitches using a deterministic heuristic.
- Files: `src/tabber/__init__.py`, `src/tabber/heuristic.py`, `tests/test_tabber_heuristic.py`, `README.md`
- Tests: `tests/test_tabber_heuristic.py`
- Prompt:

```text
Implement a first-pass heuristic tabber in src/tabber/heuristic.py. Input should be canonical score events or a narrow intermediate structure with pitch, onset, duration, and voice information. Output should assign string/fret positions for standard six-string guitar tuning. Use a pragmatic heuristic first: prefer lower fret movement, keep notes within playable fret range, and reject impossible duplicate string use at the same onset. Add focused tests for open-string preference, basic chord assignment, and an impossible voicing case. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P11 - Attach tab positions to MusicXML export
- Dependency: `Dependent on P06 and P10`
- Goal: ensure rendered MusicXML includes `<technical><string>/<fret>` for fretted notes.
- Files: `src/api/render/musicxml.py`, `tests/test_musicxml_tab_encoding.py`, `frontend-readme.md`
- Tests: `tests/test_musicxml_tab_encoding.py` `tests/test_musicxml_export.py`
- Prompt:

```text
Extend the MusicXML exporter so canonical events with fingering data emit <technical><string> and <fret> tags using the MusicXML and AlphaTab numbering convention documented in frontend-readme.md. Add tests that assert string numbering is correct for high-E and low-E cases. Update frontend-readme.md only if clarification is needed, not to change scope. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P12 - Alternate fingering search
- Dependency: `Dependent on P10`
- Goal: return alternate same-pitch string/fret positions for one event.
- Files: `src/tabber/alternates.py`, `tests/test_tabber_alternates.py`
- Tests: `tests/test_tabber_alternates.py`
- Prompt:

```text
Add a helper that returns alternate guitar positions for a single pitch event while preserving pitch. Keep the search bounded and deterministic. Use standard tuning and current fret limits, and return results in a stable order that is useful for a UI picker. Add tests for a pitch with multiple valid positions and one with very limited options. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P13 - Canonical fingering mutation helper
- Dependency: `Dependent on P03 and P12`
- Goal: safely apply selected fingering changes without altering pitch or timing.
- Files: `src/api/canonical/fingering.py`, `tests/test_canonical_fingering.py`
- Tests: `tests/test_canonical_fingering.py`
- Prompt:

```text
Implement a canonical score helper that applies fingering selections to events by eventId. The helper must only update fingering metadata and must reject changes that alter pitch, timing, or target a missing event. Add tests covering a successful update and a rejected invalid eventId. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P14 - Inference checkpoint loading
- Dependency: `Independent`
- Goal: load a trained NoteLM checkpoint and associated vocab cleanly for generation scripts and API routes.
- Files: `src/inference/__init__.py`, `src/inference/load_checkpoint.py`, `tests/test_load_checkpoint.py`
- Tests: `tests/test_load_checkpoint.py`
- Prompt:

```text
Add a small inference helper that loads a NoteLM checkpoint plus vocab and returns a ready-to-run model on CPU by default. Keep the interface minimal and avoid introducing a heavy configuration system. Add a test that saves a tiny synthetic checkpoint and reloads it successfully. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P15 - Autoregressive generation loop
- Dependency: `Dependent on P14`
- Goal: turn the existing NoteLM model and sampler utilities into a working token generator.
- Files: `src/inference/generate_v1.py`, `src/utils/decoding/sampler.py`, `src/utils/decoding/scg.py`, `tests/test_generate_v1.py`
- Tests: `tests/test_generate_v1.py` `tests/test_load_checkpoint.py`
- Prompt:

```text
Implement a minimal autoregressive generation loop for NoteLM in src/inference/generate_v1.py. It should load a checkpoint, accept seed tokens and decoding settings, step the model forward, sample next tokens, and stop on max length or EOS if present. Reuse existing sampler and SCG helper modules instead of rewriting them. Add tests using a tiny synthetic model or mocked logits so generation logic is exercised without expensive training. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P16 - Prompt/control token mapping
- Dependency: `Dependent on P15`
- Goal: map user-facing compose options into control tokens the generator can consume.
- Files: `src/inference/controls.py`, `tests/test_controls.py`, `README.md`
- Tests: `tests/test_controls.py`
- Prompt:

```text
Implement a thin control-token layer that maps compose inputs like key, style, difficulty, and requested measure count into token prefixes expected by the training pipeline. Reuse the token conventions already present in scripts/train_v1.py and collate_miditok.py. Add small tests for normalization and token ordering. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P17 - Baseline compose pipeline
- Dependency: `Dependent on P04, P07, P11, P15, and P16`
- Goal: build the backend compose service path from model output to canonical score to XML and MIDI.
- Files: `src/api/services/compose.py`, `tests/test_compose_service.py`
- Tests: `tests/test_compose_service.py`
- Prompt:

```text
Implement a compose service that calls the generation loop, converts generated tokens to canonical score, tabs the result, and returns MusicXML, MIDI, and measure/event maps needed by the frontend. Keep the API surface internal for now; just build a service function with a clean return object. Add tests around the service using a stubbed generation result so the transformation pipeline is covered even without a real trained model. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P18 - Draft/revision repository
- Dependency: `Dependent on P03`
- Goal: manage score revisions and inpaint drafts server-side.
- Files: `src/api/store.py`, `tests/test_store.py`
- Tests: `tests/test_store.py`
- Prompt:

```text
Implement a small in-memory repository for scores and drafts. It should create score IDs, track integer revisions, create draft IDs tied to a base revision, commit drafts, discard drafts, and reject stale writes. Keep it in-memory for MVP and structure it so a later persistence layer can replace it. Add tests for happy path and stale revision rejection. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P19 - Inpaint window-mode service
- Dependency: `Dependent on P05, P11, and P18`
- Goal: implement the core inpaint_preview behavior for one measure while preserving carry-in notes.
- Files: `src/api/services/inpaint.py`, `tests/test_inpaint_service.py`, `frontend-readme.md`
- Tests: `tests/test_inpaint_service.py` `tests/test_canonical_ops.py`
- Prompt:

```text
Implement an MVP inpaint service for window mode. It should accept a canonical score plus measureId, compute carry-in events active at the measure start, preserve those carry-in events, regenerate or replace only events whose startTick falls inside the selected measure, and return a draft result with changed measure IDs and lockedEventIds. Keep the first version deterministic if needed; correctness of measure boundaries and carry-in preservation matters more than model sophistication. Add tests for carry-in preservation and single-measure replacement. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P20 - FastAPI compose and draft routes
- Dependency: `Dependent on P02, P17, P18, and P19`
- Goal: expose the core backend API expected by the frontend.
- Files: `src/api/routes/scores.py`, `src/api/app.py`, `tests/test_api_scores.py`, `frontend/src/api/types.ts`
- Tests: `tests/test_api_scores.py` `tests/test_api_health.py`
- Prompt:

```text
Add FastAPI routes for /compose, /inpaint_preview, /commit_draft, and /discard_draft. Wire them to the compose and draft services, keep request and response payloads aligned with frontend/src/api/types.ts, and return 409 on stale revision conflicts. Add route tests with TestClient that cover one compose call and one full preview -> commit flow. If the TypeScript payload types drift from the backend contract, update frontend/src/api/types.ts in the same change. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P21 - Event hit map generation
- Dependency: `Dependent on P06`
- Goal: generate deterministic `measureMap` and `eventHitMap` payloads for AlphaTab click resolution.
- Files: `src/api/render/hit_map.py`, `tests/test_hit_map.py`, `frontend-readme.md`
- Tests: `tests/test_hit_map.py`
- Prompt:

```text
Implement deterministic measureMap and eventHitMap generation for the exported score. The mapping should support the frontend hit key shape of barIndex, voiceIndex, beatIndex, and noteIndex, and it should remain stable for the same exported MusicXML structure. Keep the implementation explicit and well-tested because the UI depends on it for note-level actions. Add tests for a small polyphonic example. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P22 - Wire hit maps into compose and inpaint responses
- Dependency: `Dependent on P20 and P21`
- Goal: complete the backend contract the frontend already expects.
- Files: `src/api/services/compose.py`, `src/api/services/inpaint.py`, `src/api/routes/scores.py`, `tests/test_api_scores.py`
- Tests: `tests/test_api_scores.py` `tests/test_hit_map.py`
- Prompt:

```text
Update the compose and inpaint service/route responses so they always include measureMap and eventHitMap when export succeeds. Keep the mapping generation centralized and avoid duplicating logic in the route layer. Extend the route tests to assert these fields are present and structurally valid. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P23 - Alternate positions API
- Dependency: `Dependent on P12, P18, and P22`
- Goal: implement the backend endpoint for note-level fingering alternatives.
- Files: `src/api/routes/fingering.py`, `src/api/services/fingering.py`, `tests/test_api_fingering.py`, `frontend/src/api/types.ts`
- Tests: `tests/test_api_fingering.py`
- Prompt:

```text
Implement the /alt_positions endpoint. It should resolve the frontend eventHitKey to a canonical eventId using the stored hit map, compute alternate fingerings for that event, and return a compact response suitable for a picker UI. Add tests covering a valid request and a missing-event case. Update frontend/src/api/types.ts if a concrete response type is needed. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P24 - Apply fingering API
- Dependency: `Dependent on P13, P18, and P23`
- Goal: implement server-side fingering edits that only update technical positions.
- Files: `src/api/routes/fingering.py`, `src/api/services/fingering.py`, `tests/test_api_fingering.py`
- Tests: `tests/test_api_fingering.py`
- Prompt:

```text
Implement the /apply_fingering endpoint. It should validate revision, apply one or more fingering selections by eventId, re-export MusicXML, and return the new revision. Add tests asserting the response updates fingering-related MusicXML while leaving pitch content unchanged. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P25 - Frontend fingering picker UI
- Dependency: `Dependent on P23 and P24`
- Goal: finish the note-level editing flow that is only stubbed today.
- Files: `frontend/src/App.tsx`, `frontend/src/api/client.ts`, `frontend/src/api/types.ts`, `frontend/src/state/types.ts`, optional new component under `frontend/src/components/`
- Prompt:

```text
Add a small fingering picker flow to the frontend. When a note is clicked and an eventHitMap lookup succeeds, call /alt_positions, show the returned options in a lightweight picker, and on selection call /apply_fingering then refresh the displayed MusicXML and revision. Keep the UI intentionally small and reuse existing state in App.tsx where possible. Add targeted frontend tests if the repo already has a lightweight setup; otherwise add at least pure-state tests around the mapping and selection logic. Append a PROGRESS.md entry and run the relevant frontend and backend tests.
```

### P26 - Frontend draft state polish
- Dependency: `Dependent on P20 and P22`
- Goal: surface locked events, changed measures, and clearer preview state in the UI.
- Files: `frontend/src/App.tsx`, `frontend/src/components/ScoreViewer.tsx`, `frontend/src/App.css`, `frontend-readme.md`
- Prompt:

```text
Improve the existing frontend draft workflow so it displays lockedEventIds, changedMeasureIds, and clearer preview state returned by the backend. Keep the styling minimal and fit it into the current UI rather than redesigning the app. Update frontend-readme.md only if the implemented UI behavior needs small clarification. Append a PROGRESS.md entry and run any relevant frontend tests if present.
```

### P27 - Frontend integration tests for compose/inpaint flow
- Dependency: `Dependent on P20 and P22`
- Goal: lock down the main UI workflow against regressions.
- Files: `frontend/src/`, `frontend/package.json`, test files under `frontend/src/`
- Prompt:

```text
Add lightweight frontend tests for the main workflow: load a score response, select a measure, request an inpaint preview, and commit or discard the draft. Mock the API layer rather than requiring a live backend. Keep the test surface focused on state transitions and visible UI text. Update frontend/package.json only if an additional small test utility is required. Append a PROGRESS.md entry and run the frontend test command.
```

### P28 - Training script hardening
- Dependency: `Independent`
- Goal: make `scripts/train_v1.py` practical for producing a real checkpoint.
- Files: `scripts/train_v1.py`, optional `tests/test_train_v1_smoke.py`, `README.md`
- Tests: `tests/test_train_v1_smoke.py`
- Prompt:

```text
Review scripts/train_v1.py and harden it for actual use. Add missing basics only if needed for a practical first training run: clearer logging, checkpoint save metadata, resume support, validation split support, or a small dry-run mode. Do not redesign the trainer. Add a smoke test if feasible with a tiny synthetic dataset. Update README.md only for concrete run instructions that match the script. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P29 - Evaluation script for generated outputs
- Dependency: `Dependent on P15`
- Goal: fill the missing evaluation/reporting path referenced in the root README.
- Files: `scripts/eval_basic.py`, `tests/test_eval_basic.py`, `README.md`
- Tests: `tests/test_eval_basic.py`
- Prompt:

```text
Implement a small evaluation script that can score generated token streams or exported pieces with pragmatic metrics: bar count, interval range sanity, token validity, and simple tab/playability summaries if tab data exists. Keep it lightweight and file-based so it can be run after generation without extra infrastructure. Add tests for the CLI on a tiny fixture. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P30 - End-to-end example generation script
- Dependency: `Dependent on P17, P20, and P29`
- Goal: provide one reproducible command that creates example outputs for demos and manual QA.
- Files: `scripts/generate_example.py`, `README.md`, optional fixture paths under `out/examples/`
- Tests: `tests/test_compose_service.py` `tests/test_eval_basic.py`
- Prompt:

```text
Add a small end-to-end example generation script that exercises the compose pipeline and writes MusicXML, MIDI, and any ASCII tab output to an output directory. The purpose is manual QA and demo preparation, not a production batch tool. Update README.md with one concrete example command using the script. Append a PROGRESS.md entry and run the relevant tests.
```

### P31 - API conflict and error contract cleanup
- Dependency: `Dependent on P20, P23, and P24`
- Goal: make backend failures predictable for the frontend and tests.
- Files: `src/api/routes/`, `tests/test_api_errors.py`, `frontend/src/api/client.ts`
- Tests: `tests/test_api_errors.py` `tests/test_api_scores.py` `tests/test_api_fingering.py`
- Prompt:

```text
Standardize API error handling across compose, draft, and fingering routes. Return clear JSON errors for missing score IDs, stale revisions, bad measure IDs, and bad event IDs. Keep status codes conventional and update frontend/src/api/client.ts only if the client needs a small improvement to surface backend error text consistently. Add focused API tests for the error cases. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.
```

### P32 - Final docs pass for shippable MVP
- Dependency: `Dependent on P20, P24, P27, P29, and P30`
- Goal: leave the repo with accurate setup, architecture, and usage docs.
- Files: `README.md`, `frontend/README.md`, `frontend-readme.md`, `TODO.md`, `PROGRESS.md`
- Prompt:

```text
Do a final documentation pass for the MVP that now exists. Update README.md and frontend/README.md so a new developer can train, run the backend, run the frontend, and exercise compose/inpaint/fingering flows without reading source code first. Keep frontend-readme.md as the architecture contract and adjust it only where implementation details have concretely settled. Update TODO.md to reflect genuinely remaining post-MVP work. Append a PROGRESS.md entry and note any tests or manual checks you ran.
```

## Suggested Critical Path

If the goal is to finish the project with the fewest blocked tasks, do the work in roughly this order:

1. `P02` -> `P03` -> `P04` -> `P06` -> `P21`
2. `P10` -> `P11` -> `P12` -> `P13`
3. `P14` -> `P15` -> `P16` -> `P17`
4. `P18` -> `P19` -> `P20` -> `P22`
5. `P23` -> `P24` -> `P25` -> `P26` -> `P27`
6. `P28` -> `P29` -> `P30` -> `P31` -> `P32`