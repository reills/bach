• What’s Wrong

  - The model looks undertrained for the amount of data you have. out/notelm_v1/train_args.json:19 shows
    max_steps=5000, batch_size=4, bars_per_seq=4, while data/processed/stats.json says the dataset has 247,834
    bars across 3,154 pieces. Roughly, you only exposed it to about a third of one full pass at best.
  - Conditioning is weak. The collator only prepends a tiny prefix like KEY_* and MEAS_* for the whole packed
    sequence in src/dataio/collate_miditok.py:145 and src/dataio/collate_miditok.py:215. The model supports
    descriptor embeddings in src/models/notelm/model.py:178, but training never passes desc_embed in scripts/
    train_v1.py:518.
  - You’re making the LM predict derived harmonic tokens as if they were creative choices. HARM_* is generated
    token-by-token, but one failure report already shows 49 harmonic mismatches in out/
    compose_failures/20260323T070106Z_5814c16a_tab.json. That wastes capacity and adds noise.
  - Decoding is very loose. Defaults are temperature=1.0, top_p=0.9, no EOS, no real no-repeat logic in src/
    inference/generate_v1.py:17 and src/utils/decoding/sampler.py:32. The optional SCG rules are too weak and
    not key-aware in src/utils/decoding/rules.py:16 and src/utils/decoding/scg.py:19.
  - Your eval is mostly syntax-level. scripts/eval_basic.py:94 doesn’t measure key fit, cadence quality,
    harmonic consistency, or repetition. So you can get “valid” output that still sounds bad.

  One live KEY_C sample from the current checkpoint had 39 note onsets over 5 bars, and about 51% of them were
  outside the C-major pitch-class set. That matches your complaint pretty directly.

  What I’d Do First

  1. Train longer before changing architecture. I’d push this checkpoint toward at least 20k-50k steps and
     watch validation loss plus sample quality every few thousand steps.
  2. Tighten inference immediately: try temperature 0.7-0.85 and top_p 0.8-0.9. Right now sampling is too
     permissive for a still-weak model.
  3. Normalize the corpus more aggressively. If the goal is “sounds in-key and Bach-ish,” transpose training
     pieces to a small canonical set first, ideally C major / A minor, instead of asking a lightly conditioned
     model to learn all keys equally well.
  4. Stop predicting HARM_OCT and HARM_CLASS directly. Derive them from the pitches during training/inference,
     or at minimum remove their loss weight. They should be consistency features, not freeform sampled content.
  5. Add real musical eval: off-key rate by declared key, harmonic-token mismatch count, cadence-ending
     quality, repeated-pattern rate, and tab/playability failure rate.
  6. If guitar output matters, add guitar awareness before tabbing. Right now the LM generates symbolic notes
     and the tabber often fails afterward, which is a separate problem from tonality.

  Best Next Patch
  If you want me to implement changes, I’d do them in this order:

  1. Add better eval metrics and a generation-audit script.
  2. Make HARM_* deterministic instead of sampled.
  3. Wire real bar descriptors into training, or at least strengthen conditioning.
  4. Add constrained decoding for key consistency and token grammar.

  If you want, I can take the first pass and patch eval_basic.py plus generation so you can measure exactly why
  samples are sounding bad.
