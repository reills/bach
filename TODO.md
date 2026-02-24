# TODO

## Milestone: Stage-1 Tokenizer & Dataset
- [x] Implement `src/tokens/schema.py` (EventSpec + DescriptorSpec, versioned)
- [x] Implement tokenizer interval math + round-trip tests
- [ ] Implement `scripts/make_dataset.py` basic pipeline
- [ ] Produce `data/processed/{events.parquet,barplans.parquet,stats.json}`
- [ ] CI: `pytest -q` passes
