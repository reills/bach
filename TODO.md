# TODO — Active Task: P21

## P21 — Event hit map generation

Implement deterministic measureMap and eventHitMap generation for the exported score. The mapping should support the frontend hit key shape of barIndex, voiceIndex, beatIndex, and noteIndex, and it should remain stable for the same exported MusicXML structure. Keep the implementation explicit and well-tested because the UI depends on it for note-level actions. Add tests for a small polyphonic example. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_hit_map.py 
```
