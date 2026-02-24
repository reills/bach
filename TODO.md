# TODO

## Milestone: Stage-1 Tokenizer & Dataset
- [x] Implement `src/tokens/schema.py` (EventSpec + DescriptorSpec, versioned)
- [x] Implement tokenizer interval math + round-trip tests
- [x] Implement event parsing/serialization in `src/tokens/tokenizer.py` with canonical ordering
- [x] Add round-trip tests for pitch reconstruction and `HARM_*` consistency
- [ ] Implement `scripts/make_dataset.py` basic pipeline
- [ ] Produce `data/processed/{events.parquet,barplans.parquet,stats.json}`
- [x] CI: `pytest -q` passes
