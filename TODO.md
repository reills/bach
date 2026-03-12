# TODO — Active Task: P25

## P25 — Frontend fingering picker UI

Add a small fingering picker flow to the frontend. When a note is clicked and an eventHitMap lookup succeeds, call /alt_positions, show the returned options in a lightweight picker, and on selection call /apply_fingering then refresh the displayed MusicXML and revision. Keep the UI intentionally small and reuse existing state in App.tsx where possible. Add targeted frontend tests if the repo already has a lightweight setup; otherwise add at least pure-state tests around the mapping and selection logic. Append a PROGRESS.md entry and run the relevant frontend and backend tests.
