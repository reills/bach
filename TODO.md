# TODO — Active Task: P09

## P09 — Golden tests for canonical -> MusicXML behavior

Add golden-style tests for the canonical-to-MusicXML bridge. Cover measure IDs, event IDs, string/fret technical tags for fretted notes, and cross-bar tie splitting. Keep fixtures small and hand-authored. The goal is to protect the backend contract expected by the frontend, not to build a large snapshot suite. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_musicxml_golden.py tests/test_musicxml_export.py 
```
