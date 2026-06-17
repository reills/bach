import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health_router)
    app.include_router(
        create_scores_router(
            compose_service=compose_service,
            repository=repository,
        )
    )
    return app


def _default_repository() -> ScoreDraftRepository[CanonicalScore]:
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        from src.api.store_postgres import PostgresScoreRepository
        return PostgresScoreRepository(database_url)
    from src.api.store import InMemoryScoreRepository
    return InMemoryScoreRepository[CanonicalScore]()


app = create_app(repository=_default_repository())
