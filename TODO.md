# TODO — Active Task: P04

## P04 — Token stream to canonical score bridge

Add a converter from the existing token stream to the canonical score model. Reuse token semantics already implemented in src/tokens and do not duplicate interval logic. The converter should derive measures from BAR and TIME_SIG tokens, rebuild pitches from ABS_VOICE and MEL_INT12, assign stable event IDs, and map VOICE_v to canonical voiceId. Add tests that cover a simple monophonic example and a cross-bar sustained note. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_tokens_to_canonical.py tests/test_canonical_types.py 
```
