# TODO — Active Task: P17

## P17 — Baseline compose pipeline

Implement a compose service that calls the generation loop, converts generated tokens to canonical score, tabs the result, and returns MusicXML, MIDI, and measure/event maps needed by the frontend. Keep the API surface internal for now; just build a service function with a clean return object. Add tests around the service using a stubbed generation result so the transformation pipeline is covered even without a real trained model. Append a PROGRESS.md entry and run bash docs/skills/python-test-env/scripts/run_tests.sh.

## Test Command

Run ONLY these targeted tests (do NOT run the full suite):

```bash
bash docs/skills/python-test-env/scripts/run_tests.sh tests/test_compose_service.py 
```
