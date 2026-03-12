import asyncio

import httpx

from src.api import create_app


def test_healthz_returns_ok():
    async def run_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=create_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/healthz")

    response = asyncio.run(run_request())

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
