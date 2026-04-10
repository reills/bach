# Piano-First Plan

## Project Decision

The near-term goal is no longer "Bach-style guitar generation."

The priority is:

1. Generate piano-ready symbolic music.
2. Preserve and generate real polyphony, not mostly single-line melody.
3. Improve counterpoint and overall musicality before worrying about guitar/tab.

That changes the order of work. Guitar-specific constraints stay deferred until the model can already produce convincing multi-voice piano texture.

## Combined Diagnosis

Both prior plans are useful, but they solve different layers of the problem:

- Plan 2 identified the highest-impact data bug: the current eventizer destroys simultaneity.
- Plan 1 identified the highest-impact generation gap after data is fixed: the decoder/evaluator does not reason about forbidden parallels or voice-leading.

Both are true.

Current code evidence:

- `src/tokens/eventizer.py:_extract_events()` was keeping only the top pitch of a chord. **Fixed**: now emits one `NoteEvent` per chord tone.
- `src/tokens/eventizer.py:_collapse_events()` was collapsing all simultaneous onsets to one event (highest pitch). **Fixed**: now only dedupes exact duplicates (same onset + same pitch).
- `src/tokens/eventizer.py:_collapse_unison_octaves()` is **correctly** scoped: it only trims redundant octave doublings of the same pitch class (controlled by `max_octaves_per_pitch_class`, default 3). Different pitch classes at the same onset are preserved. This function was not a bug.
- `README.md:29-40` confirms that pitch is reconstructed from anchors plus `MEL_INT12`, while `HARM_*` is only derived metadata.
- `src/utils/decoding/rules.py:250-294` only penalizes large leaps and dissonant `HARM_CLASS`; it does not check parallels, direct fifths/octaves, voice crossing, or spacing.

Conclusion:

- The data pipeline was destroying chord tones and same-onset polyphony before training. Now fixed.
- Fixing the data alone is still not enough, because parallel fifths and octaves are not local token properties and will still slip through generation without explicit checking.

## Strategy

Take the best of both worlds in this order:

1. Preserve polyphony in the dataset.
2. Add objective musical/counterpoint metrics.
3. Retrain on piano-first, polyphony-preserving data.
4. Add counterpoint-aware constrained decoding.
5. Compare checkpoints using the new metrics and listening tests.

## Phase 1: Fix Data Fidelity First

This is the first step because the current pipeline is likely teaching the model the wrong task.

### 1A. Expand chords instead of collapsing them to the top note — DONE

`_extract_events()` now emits one `NoteEvent` per chord tone instead of taking `max(p.midi ...)`.

### 1B. Stop collapsing simultaneous onsets to a single event — DONE

`_collapse_events()` now only dedupes exact duplicates (same onset + same pitch, keeping longest duration). Distinct pitches at the same onset are preserved.

Note: `_collapse_unison_octaves()` (used in the default `auto`/`events` path) was already correctly scoped — it only trims redundant octave doublings of the same pitch class, not different pitch classes. No change needed there.

### 1C. Add data sanity checks before retraining

Files:

- likely `scripts/make_dataset.py` and/or a small audit helper
- tests for any new helper

Required metrics:

- average notes per onset
- average distinct voices per bar
- percent of bars with 2+ voices
- percent of bars with 3+ voices

Definition of done:

- Running dataset build or audit shows whether the corpus is actually polyphonic after the fix.

## Phase 2: Curate A Piano-First Training Corpus

The model should not learn "general Bach soup" if the product target is polyphonic piano music.

### 2A. Create a keyboard-first subset

Priority repertoire:

- Inventions
- Sinfonias
- WTC
- Suites
- Partitas
- Goldberg

Why:

- These are keyboard-native and align with the piano-first goal.

### 2B. Keep chorales as a counterpoint-control subset

Why:

- Chorales are still useful because they provide the cleanest voice-leading signal.
- They should inform counterpoint evaluation and possibly fine-tuning, even if the final product is piano rather than SATB chorale output.

Recommended practical setup:

- Primary training run: keyboard-only or keyboard-heavy.
- Secondary comparison run: keyboard-heavy plus chorale-weighted fine-tune or mixture.

Definition of done:

- We can run at least two comparable experiments:
  - keyboard-focused
  - keyboard-focused plus stronger chorale counterpoint signal

## Phase 3: Add A Real Musical Audit Loop

This is where Plan 1 becomes essential.

The repo needs to score the specific failures we care about, not just rely on listening impressions.

### 3A. Add counterpoint metrics

Target checks:

- parallel fifths
- parallel octaves/unisons
- direct fifths/octaves
- voice crossing
- excessive spacing between adjacent voices
- voice-dropout density per bar/position

Implementation direction:

- Reconstruct candidate pitches per voice from the existing token stream.
- Compare intervals between the same voice pairs across adjacent positions.

Important constraint:

- Do not treat `HARM_*` as pitch truth. Use reconstructed pitches from anchors plus melodic intervals.

### 3B. Use the audit in both evaluation and checkpoint selection

Where to use it:

- dataset sanity checks
- offline sample evaluation
- checkpoint comparison/ranking

Core output metrics:

- average voices per position
- parallel fifth count
- parallel octave count
- crossing count
- spacing violation count
- percent of positions that are monophonic vs polyphonic

Definition of done:

- Two checkpoints can be compared with a repeatable counterpoint/polyphony scorecard.

## Phase 4: Retrain On Fixed Data Before Fancy Decoder Tricks

Once the data is corrected and measurable, retrain before spending a lot of time on inference tuning.

### 4A. Rebuild dataset and verify the fix

Before training:

- rebuild the dataset
- verify voice-count metrics improved materially
- inspect a few rendered examples manually

### 4B. Train longer on the corrected corpus

Keep:

- existing token schema
- existing NoteLM skeleton
- current sequence packing approach unless measurements force a redesign

Change:

- train on the corrected corpus, not the flattened one
- run longer than the current modest baseline
- increase batch size if hardware allows

Recommended first comparison:

- old dataset + old checkpoint baseline
- fixed dataset + same training recipe
- fixed dataset + longer training

### 4C. Log simple polyphony metrics during training

Useful low-cost signals:

- greedy decode voices-per-position on validation prompts
- percent of generated bars with 2+ voices
- percent of generated bars with 3+ voices

Definition of done:

- Training curves are paired with basic musicality/polyphony telemetry, not just loss.

## Phase 5: Add Counterpoint-Aware Constrained Decoding

This is the second high-leverage step after the data fix.

The current decoding rules only shape local token preferences. They do not prevent forbidden motion.

### 5A. Reuse the existing decoding stack, extend it

Files:

- `src/utils/decoding/rules.py`
- `src/utils/decoding/scg.py`
- generation entrypoints

Do not redesign the tokenizer first. Extend the current sampler/state tracking.

### 5B. Add a hard mask for forbidden parallels

Required behavior:

- Reconstruct active pitches for the current candidate continuation.
- When sampling `MEL_INT12_*` for a voice, mask options that would create:
  - parallel fifths
  - parallel octaves/unisons
  - obvious direct fifths/octaves where both voices leap in similar motion

### 5C. Add basic polyphony/voice-leading constraints

Also block or penalize:

- voice crossing
- obviously illegal spacing between adjacent voices
- dropping to one voice for long stretches when the target texture is multi-voice

Implementation note:

- Hard-mask true rule violations.
- Keep softer musical preferences, like leap penalties, as logit penalties.

Definition of done:

- Generation can no longer freely emit the most obvious contrapuntal violations even when the raw model distribution would.

## Phase 6: Tune Generation For Piano Texture

Only do this after Phases 1-5 are in place.

Low-risk tuning to try:

- lower temperature
- moderately high `top_p`
- seed with genuinely polyphonic context
- optionally bias missing voices back in when a position should likely stay multi-voice

This is a finishing layer, not the foundation.

## Recommended Execution Order

1. Fix chord extraction in `src/tokens/eventizer.py`.
2. Fix same-onset collapse logic in `src/tokens/eventizer.py`.
3. Add tests proving simultaneous notes survive preprocessing.
4. Rebuild the dataset and audit voice density.
5. Create keyboard-first and chorale-control subsets.
6. Add counterpoint/polyphony evaluation metrics.
7. Retrain on the corrected dataset.
8. Compare samples and metrics against the current baseline.
9. Add hard decoding masks for parallel fifths/octaves, crossing, and spacing.
10. Tune sampling only after the above is working.

## What Is Explicitly Deferred

Not a priority right now:

- guitar/tab generation quality
- string/fret-specific constraints
- broad refactors of the token schema
- solving every music-theory issue in one pass

Those can come later. Right now the project wins by producing convincing multi-voice piano music with stronger counterpoint.

## Bottom Line

Plan 2 is right about the first move: fix the eventizer so the model can actually see polyphony.

Plan 1 is right about the next major move: add counterpoint-aware evaluation and constrained decoding so the system stops producing obvious forbidden parallels.

The merged plan is:

- fix the data first
- train on piano-first polyphonic repertoire
- measure counterpoint explicitly
- enforce the worst counterpoint violations during decoding
- ignore guitar until piano output is musically credible
