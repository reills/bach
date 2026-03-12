# TODO — Active Task: P10

## P10 — Heuristic guitar tabber core

Implement a first-pass heuristic tabber in src/tabber/heuristic.py. Input should be canonical score events or a narrow intermediate structure with pitch, onset, duration, and voice information. Output should assign string/fret positions for standard six-string guitar tuning. Use a pragmatic heuristic first: prefer lower fret movement, keep notes within playable fret range, and reject impossible duplicate string use at the same onset. Add focused tests for open-string preference, basic chord assignment, and an impossible voicing case. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_tabber_heuristic.py 
```
