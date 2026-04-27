# Improving Bach Generation

## Executive Summary

The UI is not the problem. The current model is producing weak counterpoint because the music representation and generation loop do not actually model counterpoint strongly enough.

The existing system is a flat autoregressive token model over bars:

```text
BAR TIME_SIG KEY POS ABS_VOICE VOICE DUR MEL_INT12 HARM_OCT HARM_CLASS ...
```

That is enough to produce syntactically parseable symbolic music, but it is not enough to reliably produce Bach-like counterpoint. The most important issue is that the model is not being trained or decoded around persistent voice-leading relationships. It can emit `HARM_CLASS_*` tokens, but those tokens are metadata, not pitch truth, and the renderer reconstructs pitch from `ABS_VOICE_*` plus `MEL_INT12_*`. So the model can generate harmonic labels that look plausible while the actual pitches violate the labels or create bad voice-leading.

The fix is not "make the transformer bigger." The fix is:

1. Rebuild the dataset cleanly.
2. Make voice identity stable.
3. Add real counterpoint metrics.
4. Use counterpoint-aware constrained decoding/reranking.
5. Retrain on a curated Bach subset, not indiscriminate "all Bach."
6. Consider a better factorized model only after the above is measurable.

## What I Found In The Repo

### 1. The Processed Dataset Looks Stale Or Invalid

Current processed data:

- `data/processed/events.parquet`: 247,834 bars, 3,154 pieces.
- Average voices per bar is about 4.12.
- Average notes per onset is about 1.93.
- 95.59% of bars have 2+ voices.
- 81.61% of bars have 3+ voices.

That sounds polyphonic, but there is a serious mismatch:

- Schema says `MEL_INT12` should be `-24..+24`.
- Current `data/processed/vocab.json` contains `MEL_INT12` values from `-40` to `+60`.
- Current parquet contains 10,498 out-of-range `MEL_INT12` tokens.

This means at least one trained checkpoint was trained on stale or bad eventization output. The current `src/tokens/eventizer.py` appears to repair large intervals correctly now; for example, re-eventizing `BWV_1081.xml` produced no out-of-range melodic interval tokens. But `data/processed/events.parquet` was not rebuilt after that fix.

Do not train another model until the dataset and vocab are regenerated.

### 2. `HARM_*` Tokens Do Not Enforce Counterpoint

The parser/export path reconstructs actual notes from:

- `ABS_VOICE_{v}_{pitch}`
- `MEL_INT12_{delta}`
- `DUR_*`
- `POS_*`

The `HARM_OCT_*` and `HARM_CLASS_*` tokens are only derived features. They are validated in some places, but they are not the source of truth.

Generated examples already show this problem:

- `out/examples/notelm_v2_t08_p085/metrics.json` reported `harm_mismatch_count = 20`.
- `out/examples/notelm_v2_step50000_t09_p09/metrics.json` reported `harm_mismatch_count = 6`.

So the model can generate harmonic metadata that disagrees with the actual reconstructed pitches. This is fatal if the goal is interval-focused counterpoint.

### 3. The Current Decoding Rules Are Too Local

`src/utils/decoding/rules.py` only applies simple token-level penalties:

- penalize melodic leaps larger than a fifth/seventh-ish threshold
- bias `HARM_CLASS_*` toward consonant classes

It does not check:

- parallel fifths
- parallel octaves/unisons
- direct/hidden fifths or octaves
- voice crossing
- voice spacing
- prepared/resolved dissonance
- suspensions
- cadential motion
- whether voices are staying alive over time

Those are not single-token properties. They require a decoded score state across voice pairs and adjacent onsets.

Also, the API path in `src/api/compose_launcher.py` exposes `use_scg`, but not `use_grammar_mask`, even though `GenerationConfig` supports grammar masking. So even basic grammar masking is not part of the main compose runtime yet.

### 4. Voice Identity Is Not Strong Enough

The current `voice_mode="auto"` path merges events and assigns voices by continuity with a greedy pitch-distance heuristic. That helps recover polyphony, but it does not create musically stable voices in the counterpoint sense.

For Bach-style counterpoint, "voice 0" should mean a persistent line with range, register, history, and voice-leading responsibility. In the current output, generated voices can be sparse, arbitrary, and non-contiguous. Some failure reports show missing anchors and sparse voice IDs. Some samples use many voices but behave like repeated block chords. Other samples collapse into one repeated note stream.

That is why the generated music can look polyphonic by count while not sounding contrapuntal.

### 5. The Training Objective Is Too Weak

The trainer optimizes next-token cross entropy. It does not optimize any musical objectives:

- no parallel fifth/octave metric in validation
- no generated-sample scorecard during training
- no checkpoint ranking by counterpoint quality
- no penalty for static repeated notes
- no penalty for inconsistent harmonic metadata
- no reward for imitative motion, contrary motion, suspensions, or cadential syntax

Loss can improve while musical quality remains poor.

## Immediate Actions

### Step 1: Rebuild The Dataset And Vocab

Run a clean rebuild from the current eventizer:

```bash
CONDA_NO_PLUGINS=true conda run -n bach python scripts/make_dataset.py \
  --input data/tobis_xml \
  --output data/processed_rebuilt \
  --voice-mode auto \
  --max-voices 8 \
  --validate-roundtrip 20

CONDA_NO_PLUGINS=true conda run -n bach python scripts/build_vocab.py \
  --events data/processed_rebuilt/events.parquet \
  --output data/processed_rebuilt/vocab.json
```

Then add a hard audit before training:

- zero `MEL_INT12` tokens outside `-24..+24`
- zero malformed `VOICE` events
- zero harmonic mismatches on the dataset
- polyphony stats written to `stats.json`
- piece-level train/val split

If any of these fail, do not train.

### Step 2: Add A Counterpoint Evaluator

Create a real counterpoint module, probably:

```text
src/music/counterpoint.py
tests/test_counterpoint.py
```

It should reconstruct actual pitches per voice and score adjacent sonorities. Minimum metrics:

- `parallel_fifths`
- `parallel_octaves`
- `parallel_unisons`
- `direct_fifths`
- `direct_octaves`
- `voice_crossings`
- `spacing_violations`
- `dissonance_on_strong_beat`
- `unresolved_dissonances`
- `avg_active_voices`
- `monophonic_position_rate`
- `static_voice_rate`
- `harmonic_metadata_mismatches`

Use this evaluator on:

- the training dataset
- validation pieces
- every generated sample
- checkpoint comparison

You need this before deciding whether a new model is better.

### Step 3: Turn On Grammar Masking In The Backend

Wire `use_grammar_mask` through `src/api/compose_launcher.py` and the frontend constraints. Then default it on for model-backed generation.

This will not make Bach counterpoint by itself, but it reduces malformed token streams and missing-anchor failures.

### Step 4: Lower Sampling Chaos

For current checkpoints, use conservative settings:

```text
temperature: 0.65-0.8
top_p: 0.8-0.9
use_grammar_mask: true
use_scg: true only after it is stateful
```

Avoid pure top-p sampling for final output. Use it to produce candidates, then rerank.

## Better Generation Strategy

### Use The Model As A Proposal Engine, Not The Final Authority

For this domain, a pure language model sampler is the wrong final generator. Bach-style counterpoint is rule-heavy. A better architecture is:

```text
prompt/seed
  -> model proposes N candidate continuations
  -> parser reconstructs pitches
  -> counterpoint evaluator scores candidates
  -> reject impossible candidates
  -> rerank remaining candidates
  -> continue bar by bar
```

Start with bar-level or half-bar-level generation. Generate 16-64 candidates per step, parse each one, reject failures, and choose the best by:

- low parallel fifth/octave count
- low crossing count
- low harmonic mismatch count
- desired active voice count
- reasonable melodic motion
- key conformity
- non-repetition

This will improve output faster than building a bigger transformer.

### Add Stateful Counterpoint Masks

After reranking works, add hard masks during token sampling.

When the next token category is `MEL_INT12`, the decoder knows:

- current voice
- previous pitch for that voice
- current onset
- other active pitches
- previous sonority

So it can mask melodic intervals that would create:

- parallel fifths
- parallel octaves
- direct fifths/octaves
- voice crossing
- impossible pitch range

Keep style preferences as soft penalties, but make true contrapuntal violations hard constraints.

## Better Data Strategy

### Do Not Train On All Bach As One Undifferentiated Corpus

"All Bach" includes:

- chorales
- organ works
- keyboard works
- vocal works
- orchestral/chamber scores
- doubtful/fragmentary works
- arrangements
- pieces with many parts that do not map cleanly to keyboard counterpoint

That mixture teaches the model inconsistent tasks. If the product goal is keyboard-like Bach counterpoint, start with a curated corpus:

Primary:

- Inventions
- Sinfonias
- Well-Tempered Clavier fugues/preludes
- keyboard suites/partitas
- Goldberg variations
- Art of Fugue / Musical Offering where representation is clean

Secondary:

- chorales as a counterpoint-control or fine-tune set

Exclude initially:

- doubtful works
- fragments
- large orchestral/vocal scores
- broken OMR
- pieces with extreme voice counts unless specifically handled

### Add Transposition Augmentation

Bach-only data is small for training a transformer from scratch. Add musically valid transpositions:

- transpose each clean piece to several nearby keys
- preserve intervals and voice-leading
- avoid impossible ranges for the target instrument
- keep train/val splits by original work, not augmented copy

This gives the model more examples of the same contrapuntal patterns without leaving Bach.

## Better Representation

### Make Voice Identity Explicit

The current automatic continuity assignment is useful for preprocessing, but it is not enough as the core training target.

For chorales:

- preserve SATB identities where part names exist
- enforce stable voice order: bass, tenor, alto, soprano
- reject or repair crossing in preprocessing

For keyboard:

- separate into persistent layers, not arbitrary chord tones
- consider 2-4 voices as explicit contrapuntal lines
- keep voice range priors

### Replace Flat Tokens With Factorized Events

The current flat token stream makes the model learn grammar and music at the same time. A better v3 model should predict structured event attributes:

```text
time_step
voice_id
duration
pitch_delta_or_absolute_pitch
articulation/tie/rest
```

With separate heads:

- next time/onset
- active voice mask
- pitch delta per voice
- duration per voice
- optional harmonic descriptor

This makes it much easier to enforce legal combinations and score candidates.

Do not jump here first. Build the evaluator and constrained reranker first. That will tell you exactly which representation failures matter.

## Training Plan

### Baseline Retrain

After rebuilding data:

```bash
CONDA_NO_PLUGINS=true conda run -n bach python scripts/train_v1.py \
  --events data/processed_rebuilt/events.parquet \
  --vocab data/processed_rebuilt/vocab.json \
  --output-dir out/notelm_clean_v1 \
  --batch-size 4 \
  --max-seq-len 1024 \
  --bars-per-seq 8 \
  --val-split 0.1 \
  --val-every 500 \
  --max-steps 50000 \
  --prepend-bos \
  --append-eos \
  --bos-token '<bos>' \
  --eos-token '<eos>' \
  --mask-prefix-loss
```

But do not select checkpoints by loss alone. Every saved checkpoint should generate a fixed sample set and produce a counterpoint scorecard.

### Experiments To Compare

Run these as separate controlled experiments:

1. Clean full corpus.
2. Clean keyboard-only corpus.
3. Keyboard-only plus chorale fine-tune.
4. Keyboard-only plus transposition augmentation.
5. Same best corpus with counterpoint reranking enabled.

Only after that consider a larger or different model.

## Definition Of "Better"

A checkpoint is better only if it improves measurable musical behavior:

- no invalid token events
- no out-of-range melodic interval tokens
- harmonic metadata mismatch near zero
- lower parallel fifth/octave rate than current generated samples
- active voices match requested texture
- lower static repeated-note rate
- lower duplicate-bar rate
- off-key rate is low unless chromaticism is explained by context
- rendered examples pass listening review

Use a fixed prompt suite:

- 2-voice invention-like prompt
- 3-voice sinfonia-like prompt
- 4-voice chorale prompt
- free 8-bar keyboard generation in C, G, Dm, and Am
- inpaint a middle bar from a real Bach excerpt

## Practical Priority Order

1. Rebuild `data/processed` from the current eventizer.
2. Add dataset audit checks that fail on bad intervals and harmonic mismatches.
3. Add `src/music/counterpoint.py` and tests.
4. Extend `scripts/eval_basic.py` to report real counterpoint metrics.
5. Wire `use_grammar_mask` through the backend.
6. Build candidate reranking around the counterpoint evaluator.
7. Retrain clean baselines and compare by generated scorecards.
8. Curate keyboard/chorale subsets and add transposition augmentation.
9. Only then consider a factorized v3 model.

## Bottom Line

The model is not hopeless. The current system is asking a next-token transformer to infer counterpoint from a representation that does not preserve enough voice-leading semantics and a decoder that does not enforce the rules you care about.

Make the training data clean, make voices stable, measure real contrapuntal failures, and put a rule-aware rejection/reranking layer around generation. That is the fastest path to output that actually sounds like Bach instead of token-shaped music.
