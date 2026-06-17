---
VERDICT: FAIL
REMAINING_WORK:
- Make `event_hit_map` account for explicit rest events as well as implicit gaps using the same per-voice note/rest ordering that the MusicXML renderer emits, so frontend `bar|voice|beat|note` keys stay aligned for generated scores that contain rests.
- Add a targeted test in `tests/test_compose_service.py` with a stubbed generation result that produces an explicit rest between pitched notes and assert the later note maps to the rendered beat index in `event_hit_map`.
---
