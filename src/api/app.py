from fastapi import FastAPI

from src.api.canonical import CanonicalScore
from src.api.routes.health import router as health_router
from src.api.routes.scores import ComposeHandler, create_router as create_scores_router
from src.api.store import ScoreDraftRepository


def create_app(
    *,
    compose_service: ComposeHandler | None = None,
    repository: ScoreDraftRepository[CanonicalScore] | None = None,
) -> FastAPI:
    app = FastAPI(title="bach-gen API")
    app.include_router(health_router)
    app.include_router(
        create_scores_router(
            compose_service=compose_service,
            repository=repository,
        )
    )
    return app


app = create_app()
