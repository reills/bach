VERDICT: FAIL
REMAINING_WORK:
- Configure `create_app()`/`app` with a real compose-service adapter so the default FastAPI application can serve `POST /compose` successfully instead of returning HTTP 503.
- Wire the `/compose` request payload into the existing compose pipeline (for example via a wrapper around `compose_baseline`) and add route coverage that exercises that default wiring rather than only an injected fake compose handler.
