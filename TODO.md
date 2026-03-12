# TODO — Active Task: P27

## P27 — Frontend integration tests for compose/inpaint flow

Add lightweight frontend tests for the main workflow: load a score response, select a measure, request an inpaint preview, and commit or discard the draft. Mock the API layer rather than requiring a live backend. Keep the test surface focused on state transitions and visible UI text. Update frontend/package.json only if an additional small test utility is required. Append a PROGRESS.md entry and run the frontend test command.
