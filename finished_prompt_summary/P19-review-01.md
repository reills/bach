---
VERDICT: FAIL
REMAINING_WORK:
- Compute `changed_measure_ids` from the actual regenerated event spans instead of always returning only the selected measure; when a replacement note starting in the target measure extends into a later measure, include that later measure ID in the draft result.
- Add a targeted test that inpaints one measure with a replacement event crossing the barline and asserts that the downstream measure is reported in `changed_measure_ids`.
---
