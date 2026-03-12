# TODO — Active Task: P06

## P06 — MusicXML exporter from canonical score

Implement canonical score to MusicXML export in src/api/render/musicxml.py. Support measure xml:id, note xml:id, divisions derived from tpq, and tie splitting when durTick crosses a barline. Keep the exporter MVP-focused for one guitar part. Add tests that assert a cross-bar note becomes two MusicXML notes with tie start/stop and preserves the same logical event identity. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_musicxml_export.py 
```
