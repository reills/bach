# TODO — Active Task: P11

## P11 — Attach tab positions to MusicXML export

Extend the MusicXML exporter so canonical events with fingering data emit <technical><string> and <fret> tags using the MusicXML and AlphaTab numbering convention documented in frontend-readme.md. Add tests that assert string numbering is correct for high-E and low-E cases. Update frontend-readme.md only if clarification is needed, not to change scope. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_musicxml_tab_encoding.py tests/test_musicxml_export.py 
```
