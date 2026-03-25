# Fix Plan

Issues in `fix.md` are directionally correct, but a few proposed remedies need adjustment to fit the
current codebase:

- The biggest gap is observability. Right now the repo cannot reliably tell you whether a change
  improved tonality, harmonic consistency, or repetition.
- `HARM_*` is not absolute pitch metadata in this repo. It is derived relative to a per-position
  reference pitch via `compute_reference_pitch()` and `harmonic_tokens_for_pitch()`. Any plan that
  rewrites those tokens as `pitch % 12` / `pitch // 12` would be wrong.
- The current generator has no explicit token-grammar state, so "inject deterministic tokens when
  the next slot is known" is harder than it sounds. Grammar-aware decoding should come first.

Ordered below by ROI and dependency, not by how easy each change sounds in isolation.

---

## 1. Build a Reproducible Audit Loop First

**Why this is first:** the repo currently has anecdotal evidence (`fix.md`) but no repeatable
before/after harness for musical quality. Every later change needs a stable scorecard.

**Files:** `scripts/eval_basic.py`, `tests/test_eval_basic.py`

**Implementation spec:**
- Keep the existing CLI shape and existing output keys in `scripts/eval_basic.py` working as-is.
- Add only optional inputs:
  - `evaluate(tokens, vocab=None, key_override=None)`
  - CLI flag `--key` for manual key override
- Add small helpers inside `scripts/eval_basic.py` rather than rewriting the evaluator:
  - `_split_bars(tokens) -> List[List[str]]`
  - `_count_duplicate_bars(tokens) -> int`
  - `_count_grammar_violations(tokens) -> int`
  - `_infer_onset_pitches(tokens) -> List[Tuple[int, int, int]]`
    - return `(voice_idx, abs_tick, pitch)` only for pitched onsets
  - `_count_off_key_onsets(tokens, key_override=None) -> Tuple[Optional[int], Optional[int]]`
    - return `(off_key_count, total_pitched_onsets)` or `(None, None)` when no usable key exists

**Metric semantics:**
- `token_grammar_violations`
  - Parse voiced events with `src.tokens.tokenizer.parse_voice_event()` instead of a custom ad hoc
    parser.
  - Walk the stream left to right.
  - Every time a `VOICE_*` token is encountered, try `parse_voice_event(tokens, idx)`.
  - On success, jump to the returned `next_idx`.
  - On `ValueError`, increment the violation counter by 1 and advance by one token so one malformed
    region does not abort the entire audit.
  - Do not count unknown structural tokens as grammar violations in this metric. This metric is only
    for malformed voice-event structure.
- `harm_mismatch_count`
  - Do not reimplement the harmonic logic heuristically.
  - Call `src.tokens.validator.validate_harm_tokens(tokens)` and report `len(errors)`.
  - This keeps the metric aligned with the repo's actual `active_until` + `compute_reference_pitch()`
    behavior.
- `duplicate_bar_rate`
  - Split the stream into exact token slices per bar, excluding the `BAR` delimiter itself.
  - Count how many bars are exact duplicates of an earlier bar.
  - Report `duplicate_bar_rate = duplicate_bar_count / bar_count`.
  - Empty bar bodies should still count as bars if the stream contains `BAR`.
- `off_key_rate`
  - This is an onset-based heuristic, not a full tonal analysis.
  - Reconstruct pitches using the same anchor and `MEL_INT12_*` state used by roundtrip:
    - update per-voice base pitch from `ABS_VOICE_*`, `ABS_BASS_*`, `ABS_SOP_*`
    - when a pitched `VOICE_*` event is parsed, compute `pitch = prev_pitch[voice] + mel_int`
    - update `prev_pitch[voice] = pitch`
  - Determine key from `--key` when supplied; otherwise use the latest in-stream `KEY_*` token.
  - If neither exists, return `off_key_rate = null`.
  - Start with simple pitch-class membership:
    - major: `{0, 2, 4, 5, 7, 9, 11}`
    - natural minor: `{0, 2, 3, 5, 7, 8, 10}`
  - Evaluate only pitched onsets, not sustained notes.
  - Report both `off_key_rate` and supporting counts:
    - `off_key_count`
    - `pitched_onset_count`
- `cadence_proxy_rate`
  - Keep this intentionally crude in step 1.
  - Only inspect the final pitched onset in each bar that has at least one pitched onset.
  - Mark a bar as a cadence proxy hit when that final onset lands on scale degree 1 or 5 relative to
    the active key.
  - If no usable key exists for a bar, skip that bar from the cadence denominator.
  - Report both:
    - `cadence_proxy_rate`
    - `cadence_proxy_hits`
    - `cadence_proxy_eligible_bars`

**Result contract:**
- Preserve existing keys already returned by `evaluate()`.
- Add new keys only when they are computable:
  - `harm_mismatch_count`
  - `token_grammar_violations`
  - `duplicate_bar_count`
  - `duplicate_bar_rate`
  - `off_key_count`
  - `pitched_onset_count`
  - `off_key_rate`
  - `cadence_proxy_hits`
  - `cadence_proxy_eligible_bars`
  - `cadence_proxy_rate`
- When a metric is not computable because key context is absent, return `None` for that metric rather
  than guessing.

**Tests to add in `tests/test_eval_basic.py`:**
- Keep all existing tests passing unchanged.
- Add one keyed single-voice fixture with anchors so `off_key_rate == 0.0`.
- Add one keyed fixture with a clear chromatic onset so `off_key_count == 1`.
- Add one fixture with repeated bar token slices so `duplicate_bar_count` and `duplicate_bar_rate`
  are non-zero.
- Add one fixture with a deliberately broken voiced event, for example `VOICE_0 DUR_12 HARM_OCT_0`
  missing `MEL_INT12_*`, and assert `token_grammar_violations > 0`.
- Add one fixture with a forced wrong `HARM_*` pair and assert `harm_mismatch_count > 0`.
- Add one CLI test for `--key` to prove manual override changes the emitted JSON.

**Definition of done for step 1:**
- `scripts/eval_basic.py` still works on old inputs with no new required flags.
- `tests/test_eval_basic.py` covers each new metric path.
- The evaluator never crashes on malformed streams; it should return counts and `None` values where
  appropriate.

**Important note:** the exact compose-failure JSON cited in `fix.md` is not present in this working
tree. Treat that file as anecdotal evidence, not as a reproducible benchmark.

---

## 2. Tighten Decoding, but Start with Grammar Before Key Penalties

**Problem:** current generation uses permissive sampling with weak rule-based rescoring. Lowering
temperature helps, but it does not solve malformed event structure or inconsistent state tracking.

**Files:** `src/inference/generate_v1.py`, `src/utils/decoding/sampler.py`,
`src/utils/decoding/rules.py`, `src/utils/decoding/scg.py`

**Implementation goal:** prevent the sampler from choosing token categories that cannot possibly
follow the current partial stream. This step is about syntax/state validity first, not musical
preference shaping.

**What this step should not do yet:**
- do not hard-ban notes for being "off key"
- do not force guitar/tab feasibility yet
- do not try to deterministically inject `HARM_*` yet
- do not redesign the tokenizer

**Phase 2A: lower the entropy baseline**
- In `src/inference/generate_v1.py`, reduce default generation settings to:
  - `temperature=0.75`
  - `top_p=0.85`
- Do not change user-provided overrides; only adjust defaults.
- This should be a one-line baseline improvement, not the main fix.

**Phase 2B: add explicit next-token category masking**
- Add a helper in `src/utils/decoding/rules.py` or `src/utils/decoding/scg.py` that inspects the
  current decoded prefix and returns the allowed next token categories.
- The helper should operate on token categories/prefixes, not on full token IDs at first. For
  example:
  - `BAR`
  - `POS_*`
  - `VOICE_*`
  - `REST_*`
  - `DUR_*`
  - `DUP_*`
  - `MEL_INT12_*`
  - `HARM_OCT_*`
  - `HARM_CLASS_*`
  - `STR_*`
  - `FRET_*`
  - anchors such as `ABS_VOICE_*`, `ABS_BASS_*`, `ABS_SOP_*`
  - structural metadata such as `TIME_SIG_*`, `KEY_*`, `TEMPO_*`
- Then map those allowed categories to a token-ID mask using the vocab already loaded by inference.

**Required grammar state machine:**
- Treat generation as alternating between structural context and voice-event context.
- At minimum, enforce these local transitions:
  - after `VOICE_*`, allow `REST_*` or `DUR_*`
  - after `REST_*`, allow structural tokens or another `VOICE_*`
  - after `DUR_*`, allow optional `DUP_*` or `MEL_INT12_*`
  - after `DUP_*`, allow only `MEL_INT12_*`
  - after `MEL_INT12_*`, allow only `HARM_OCT_*`
  - after `HARM_OCT_*`, allow only `HARM_CLASS_*`
  - after `HARM_CLASS_*`, allow `STR_*`, structural tokens, or another `VOICE_*`
  - after `STR_*`, allow only `FRET_*`
  - after `FRET_*`, allow structural tokens or another `VOICE_*`

**Required structural rules:**
- `BAR` can begin a new bar.
- Inside a bar, `POS_*` should appear before voice events at a position.
- `VOICE_*` events should only be allowed after a bar has started.
- Optional structural tokens such as `TIME_SIG_*`, `KEY_*`, and anchors should be allowed only in
  places that match the existing token streams:
  - near bar starts
  - before pitched events that depend on anchors
- Do not over-constrain bar layout beyond what the current dataset clearly uses. The first version
  should enforce obvious impossibilities, not a perfect formal grammar.

**Recommended implementation shape:**
- In `src/utils/decoding/rules.py`:
  - add token-category helpers such as `token_category(token: str) -> str`
  - add `allowed_next_categories(prefix_tokens: List[str]) -> Set[str]`
- In `src/utils/decoding/scg.py`:
  - convert allowed categories into a mask over candidate token IDs
  - intersect that mask with any existing score adjustments, rather than replacing the sampler
- In `src/inference/generate_v1.py`:
  - apply the mask during each step before sampling
  - if the mask would eliminate everything, fall back to the old behavior plus a warning/log counter
    rather than crashing

**Minimum state the decoder must track:**
- whether a bar has started
- whether the current position is inside a partially completed voice event
- whether a `STR_*` token is awaiting `FRET_*`
- whether the current prefix has seen an anchor for a voice before allowing a pitched event for that
  voice

**Important pragmatic shortcut:**
- You do not need a full parser in the generation loop.
- A small finite-state machine over recent token categories is enough for the first pass.
- Keep it local and cheap so it can run every decoding step.

**Tests to add before calling step 2 done:**
- add unit tests for allowed-next-category logic in `src/utils/decoding/*` tests or a new decoding
  test file
- cover at least:
  - `VOICE_* -> DUR_* | REST_*`
  - `DUR_* -> DUP_* | MEL_INT12_*`
  - `DUP_* -> MEL_INT12_*`
  - `MEL_INT12_* -> HARM_OCT_*`
  - `HARM_OCT_* -> HARM_CLASS_*`
  - `STR_* -> FRET_*`
  - malformed transitions are masked out
- add one generation-level smoke test proving decoding no longer emits impossible local sequences

**How to roll this out safely:**
- first add the rule helper and tests
- then enable the mask behind a config flag such as `use_grammar_mask=True`
- compare baseline vs masked decoding with the step-1 audit harness
- only after that consider stronger constraints like soft key penalties

**Why grammar comes before key penalties:**
- malformed local syntax is a hard failure mode
- off-key notes are a soft musical error
- if the stream shape is broken, later key-aware penalties and harmonic repairs become unreliable

---

## 3. Repair `HARM_*` Deterministically in Post-Process First

**Problem:** `HARM_*` mismatches are real, but in-loop deterministic insertion is not the safest
first patch because the current generator does not maintain reliable per-position harmonic state.

**Files:** `src/tokens/validator.py`, `src/tokens/intervals.py`, likely a new utility module such as
`src/tokens/repair.py`, plus tests

**Implementation goal:** given an already generated token stream, deterministically rewrite only the
`HARM_OCT_*` / `HARM_CLASS_*` pair for pitched voice events so they match the repo's actual
reference-pitch logic. This is a cleanup pass after generation, not a new decoding strategy.

**What this step should not do:**
- do not change `MEL_INT12_*`
- do not insert missing anchors
- do not try to fix malformed event grammar
- do not change bar/position layout
- do not derive `HARM_*` from raw pitch alone

**Implementation shape:**
- Add a new helper module, preferably `src/tokens/repair.py`, rather than overloading
  `src/tokens/validator.py` with mutation logic.
- Keep `validate_harm_tokens()` as the read-only checker.
- Put the actual rewrite pass in:
  - `repair_harm_tokens(tokens: List[str], tpq: int = 24) -> HarmRepairResult`
- Add a small result dataclass:
  - `HarmRepairResult`
  - fields:
    - `tokens: List[str]`
    - `repaired_event_count: int`
    - `mismatch_count_before: int`
    - `mismatch_count_after: int`
    - `skipped_event_count: int`
    - optionally `errors_before: List[str]`
    - optionally `errors_after: List[str]`

**Core repair algorithm:**
- Reuse the same state model as `validate_harm_tokens()`:
  - `prev_pitch` per voice
  - `active_until` per voice
  - current `bar_start`
  - current bar length from `TIME_SIG_*`
- Walk the stream left to right.
- Ignore tokens that do not affect harmonic repair state:
  - `KEY_*`
  - `TEMPO_*`
  - `ABS_LOW_*`
  - `ABS_HIGH_*`
  - `REF_VOICE_*`
- Update anchor state from:
  - `ABS_VOICE_*`
  - `ABS_BASS_*`
  - `ABS_SOP_*`
- At each `POS_*`:
  - do a lookahead across that position, exactly like validation
  - collect all pitched onsets that begin at this position
  - reconstruct each onset pitch from anchor state plus `MEL_INT12_*`
  - combine:
    - still-active sustained pitches from `active_until`
    - newly reconstructed onset pitches at this same position
  - compute `ref_pitch = compute_reference_pitch(active_pitches)`
- Then do a second pass through the same position payload:
  - for each pitched `VOICE_*` event:
    - identify the exact `HARM_OCT_*` token index
    - identify the exact `HARM_CLASS_*` token index
    - recompute expected tokens with `harmonic_tokens_for_pitch(pitch, ref_pitch, qa_mode=True)`
    - if the existing pair differs, overwrite those two token strings in a copied token list
    - increment `repaired_event_count`
  - for each rest event:
    - leave tokens unchanged
  - update `prev_pitch` and `active_until` exactly as validation does

**Important semantic detail:**
- The repair pass must use the same per-position reference definition as validation:
  - reference pitch is the minimum of:
    - pitches still sounding at that absolute tick
    - pitches that onset at that `POS_*`
- This means the repair helper should mirror the two-pass logic already present in
  `validate_harm_tokens()`, not invent a simpler one-pass approximation.

**Error handling and skip policy:**
- If a voice event is malformed or truncated, do not guess.
- If a pitched event has no anchor and therefore pitch reconstruction is impossible:
  - leave that event's `HARM_*` tokens unchanged
  - increment `skipped_event_count`
- If `harmonic_tokens_for_pitch(..., qa_mode=True)` raises because the event is outside supported
  harmonic range:
  - leave the original pair unchanged
  - increment `skipped_event_count`
- The helper should never crash on a partially bad stream; it should repair what it can and report
  what it skipped.

**CLI/inference integration after the helper exists:**
- Do not bake this into the core generator first.
- Add it as an optional post-process hook after `generate_v1()` returns tokens, for example:
  - `repair_harm_tokens(result.tokens)` in the compose path or a separate script
- Keep the original unmodified tokens available for debugging and audit comparison.

**Metrics to expose:**
- Before returning from `repair_harm_tokens()`, run `validate_harm_tokens()` on:
  - original tokens
  - repaired tokens
- Report:
  - `mismatch_count_before = len(errors_before)`
  - `mismatch_count_after = len(errors_after)`
- Step 3 is only worth keeping if `mismatch_count_after` reliably drops, ideally to zero on streams
  that are otherwise grammatically valid.

**Tests required before step 3 is done:**
- Add a unit test with one pitched event whose `HARM_*` pair is wrong and assert:
  - the pair is rewritten
  - `mismatch_count_before > 0`
  - `mismatch_count_after == 0`
- Add a multi-voice same-position fixture where reference pitch comes from the minimum active pitch,
  not from the current voice, and assert repair uses the shared reference correctly.
- Add a sustained-note fixture where a previous voice is still active at the next `POS_*`, and assert
  repair includes sustained pitches in the reference set.
- Add a malformed/truncated event fixture and assert:
  - the helper does not crash
  - `skipped_event_count > 0`
  - unreconstructable tokens remain unchanged
- Add a rest-event fixture and assert no `HARM_*` rewrite is attempted.

**Definition of done for step 3:**
- There is a standalone `repair_harm_tokens()` helper with tests.
- It rewrites only `HARM_OCT_*` / `HARM_CLASS_*` tokens.
- It mirrors validation semantics closely enough that:
  - repaired valid streams produce `mismatch_count_after == 0`
  - malformed streams are reported, not guessed through
- It is callable as an optional post-process and is not yet entangled with the decoding loop.

**Why this is the right scope:**
- It directly addresses one known failure mode.
- It is deterministic and cheap.
- It avoids mixing harmonic cleanup with sampling-state bugs.
- It gives you a measurable intermediate win before deciding whether `HARM_*` should later be
  generated, repaired, down-weighted, or removed from the training target.

---

## 4. Wire Bar-Level Descriptor Embeddings End-to-End

**Problem:** descriptor conditioning exists in the model but is not used in training, and the
current packed-sequence path only preserves the first bar's `plan`.

**Files:** `src/dataio/collate_miditok.py`, `src/dataio/dataset.py`, `scripts/train_v1.py`,
`src/models/notelm/model.py`, tests around training/collation

**Implementation goal:** keep the current `BarPlan`-based conditioning idea, but make it survive
packing, batching, and both train/val forward passes so `NoteLM._inject_desc()` is actually used.

**What already exists and should be reused:**
- `BarPlan` already has the right source fields at `src/tokens/schema.py`.
- descriptor-relevant bar metadata is already computed in `src/dataio/descriptors.py`.
- the model already supports `desc_embed` at [src/models/notelm/model.py](/mnt/c/Users/Admin/dev/bach_gen/src/models/notelm/model.py#L178).
- do not invent a second planning format or a parallel token-prefix control system for this step.

**Phase 4A: preserve all bar plans through packed sequences**
- In `src/dataio/collate_miditok.py`, change `SequenceSample` so it carries:
  - `plans: List[Optional[BarPlan]]`
  - not just a single `plan` copied from the first bar
- In `PackedBarDataset.__getitem__()`:
  - collect one `plan` per packed bar in the same order as `ids` are concatenated
  - preserve `piece_id`, first `bar_index`, and `bar_count` as they already work
  - when truncation is enabled, truncate the `plans` list to the number of bars whose token slices
    are still present in the packed sequence
- Do not infer plans from tokens at collate time; reuse the `plan` objects already loaded by
  `BarDataset`.

**Phase 4B: define one deterministic descriptor vector format**
- Add a small helper in `src/dataio/collate_miditok.py` or a nearby dataio utility module:
  - `bar_plan_to_desc_vector(plan: Optional[BarPlan]) -> List[float]`
- Keep the representation simple, fixed-width, and deterministic.
- Recommended first version:
  - one-hot key over the keys actually represented by `BarPlan.key`
    - use a fixed canonical ordering such as
      `["C","G","D","A","E","B","F#","C#","F","Bb","Eb","Ab","Db","Gb","Cb","Am","Em","Bm","F#m","C#m","G#m","D#m","A#m","Dm","Gm","Cm","Fm","Bbm","Ebm","Abm"]`
    - if the corpus uses fewer names, keep only the names that appear in training data plus a final
      `UNK_KEY` slot
  - one-hot time signature over the values already seen in `BarPlan.time_sig`
    - for example `4/4`, `3/4`, `6/8`, plus `UNK_TIME_SIG`
  - one-hot density bucket from existing labels:
    - `DENSITY_LOW`
    - `DENSITY_MED`
    - `DENSITY_HIGH`
    - plus an `UNK_DENSITY` slot only if needed
  - pitch range bucket derived from `BarPlan.pitch_range`
    - recommended bins:
      - `NONE` when `pitch_range is None`
      - `LE_7`
      - `LE_12`
      - `LE_19`
      - `GT_19`
  - polyphony bucket derived from `BarPlan.polyphony_max`
    - recommended bins:
      - `0`
      - `1`
      - `2`
      - `3PLUS`
- Keep this vector purely numeric `float32`.
- Do not normalize or learn an encoder in this step; let `desc_proj` do the projection.

**Phase 4C: batch descriptor tensors alongside token IDs**
- Extend `MidiTokBatch` with:
  - `desc_embed: Optional[torch.Tensor]`
- Expected batch shape:
  - `desc_embed.shape == (batch, num_bars_in_sample, desc_dim)`
- In the collator:
  - build one descriptor vector per bar plan in each `SequenceSample`
  - pad descriptor rows across the batch to the maximum `num_bars_in_sample`
  - use zeros for padded descriptor rows
  - keep `bar_count` so the model can still sanity-check descriptor/bar alignment if needed
- `desc_embed` must line up with actual `BAR` tokens in sequence order, because
  `NoteLM._inject_desc()` uses cumulative `BAR` positions to gather the per-bar embedding.

**Phase 4D: pass descriptor embeddings through train and validation**
- In `scripts/train_v1.py`:
  - move `batch.desc_embed` to device when present
  - pass it in the training forward call:
    - `logits = model(inputs, attn_mask=attn_mask, desc_embed=desc_embed)`
  - if `inputs = ids[:, :-1]`, keep `desc_embed` unshifted
    - it is indexed by `BAR` positions, not by target labels
- In validation loss code, do the same so train and val use the same conditioning path.
- Only pass `desc_embed` when the batch actually has it.

**Phase 4E: config and compatibility**
- Ensure `NoteLMConfig.desc_embed_dim` matches the actual descriptor vector width.
- Do not make descriptor conditioning mandatory:
  - if `desc_embed_dim == 0`, existing runs should still work
  - if plans are missing for some samples, emit a zero vector for that bar rather than crashing
- If `strict_bar_count` is enabled, descriptor padding must still provide at least as many bar rows
  as there are `BAR` tokens in the sample.

**Tests required before calling step 4 done:**
- In `tests/test_collate_miditok.py` or a new nearby test file:
  - packed multi-bar sample returns `plans` in the same bar order as concatenated token IDs
  - collator emits `desc_embed` with shape `(batch, max_bars, desc_dim)`
  - descriptor rows stay aligned when two samples have different `bar_count`
  - missing `plan` yields a zero descriptor row, not a crash
- In model-level tests:
  - forward pass with matching `desc_embed` shape succeeds
  - forward pass with fewer descriptor rows than `BAR` tokens raises when `strict_bar_count=True`
- In training-path tests:
  - one small batch goes through both train and validation loss computation with `desc_embed`
  - verify the batch path uses `desc_embed` when `desc_embed_dim > 0`

**Definition of done for step 4:**
- packed samples preserve every bar's `BarPlan`
- collated batches expose `desc_embed`
- train and val both pass `desc_embed` into `NoteLM`
- descriptor/bar alignment is covered by tests
- older runs with `desc_embed_dim == 0` still work unchanged

**Why this matters:** this is the cleanest way to strengthen conditioning without changing the
tokenizer or the decoding contract.

---

## 5. Run a Controlled Training Matrix Before Major Data Surgery

**Problem:** the current checkpoint is likely undertrained, but retraining without a measurement
loop will just produce longer runs with ambiguous conclusions.

**Changes:**
- Keep one baseline experiment that changes only training duration:
  - `max_steps=20000` as the first checkpointing target
  - if loss and samples still improve, extend toward `50000`
- Save and audit fixed prompts every few thousand steps, not just validation loss.
- After the longer-run baseline, compare:
  - baseline longer training
  - longer training + descriptor embeddings
  - longer training + reduced `HARM_*` dependence (for example loss masking/down-weighting or input
    dropout)

**Why this order:** it isolates whether the main problem is simply undertraining or whether the
representation/conditioning is the real bottleneck.

---

## 6. Defer Canonical-Key Transposition Until Product Goals Are Clear

**Problem:** transposing everything to `C` / `Am` may help learning, but it changes the task. It
also interacts with guitar range, transposition-back logic, and whether you actually want natural
chromaticism in the final output.

**Do this only after answering the questions below.**

If you do proceed, treat it as an experiment branch:
- canonical-key corpus
- original-key corpus
- compare with the same audit harness

**Reason for deferral:** this is a data-definition decision, not just a training tweak.

---

## 7. Treat Guitar/Tab Constraints as a Separate Layer

**Problem:** tonal quality and tab feasibility are related but not the same problem. Mixing them too
early will make debugging harder.

**Do later:**
- add a soft pitch-range penalty during decoding
- add string/fret feasibility checks after tonal metrics are stable

**Goal:** first get musically coherent token streams, then make them more guitar-friendly.

---

## Research Worth Doing Before a Bigger Redesign

You do not need a long literature review before steps 1-5. Those are pragmatic repo-local fixes.
Research becomes more important if you want to change the representation, the control scheme, or
the decoding formalism.

Recommended reading areas:
- event/token representations for symbolic music transformers
- control-token and conditioning strategies for symbolic generation
- constrained decoding with explicit automata/state machines

Concrete papers worth reading:
- `Music Transformer: Generating Music with Long-Term Structure` (Huang et al., 2019)
- `Pop Music Transformer` / `REMI` representation work (Huang and Yang, 2020)
- `MMM: Exploring Conditional Multi-Track Music Generation with the Transformer` (Ens and
  Pasquier, 2020)
- `Automata-based constraints for language model decoding` (Koo, Liu, He, 2024)

Why these matter here:
- they give you better priors for representation choices
- they support grammar-aware or control-aware decoding
- they help decide whether corpus canonicalization is a good tradeoff for your goal

---

## Questions To Answer Before Step 6

These are the product/experiment questions that matter more than additional general reading:

1. Is the goal "pleasant samples in a small set of keys" or "native generation in arbitrary keys"?
2. Should the model allow chromatic accidentals and tonicizations, or do you want strict
   mostly-diatonic output?
3. Is guitar/tab feasibility a hard requirement during generation, or acceptable as a downstream
   repair/filtering step?
4. Are you willing to retrain from a regenerated corpus, or do you want improvements that mostly
   preserve the current dataset/checkpoint path?

If those answers are not settled yet, do steps 1-5 first and postpone 6-7.
