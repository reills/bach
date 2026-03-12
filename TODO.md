# TODO — Active Task: P24

## P24 — Apply fingering API

Implement the /apply_fingering endpoint. It should validate revision, apply one or more fingering selections by eventId, re-export MusicXML, and return the new revision. Add tests asserting the response updates fingering-related MusicXML while leaving pitch content unchanged. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_api_fingering.py 
```
