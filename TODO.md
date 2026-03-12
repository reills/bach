# TODO — Active Task: P07

## P07 — MIDI exporter from canonical score

Add a canonical score to MIDI exporter. Reuse music21 if that is the shortest path, but keep the adapter isolated in src/api/render/midi.py. Cover one test that exports a simple score and verifies the output file or byte payload is non-empty and structurally valid enough for the current test harness. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_midi_export.py 
```
