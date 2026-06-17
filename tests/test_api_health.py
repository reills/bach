import asyncio

import httpx
from fastapi.testclient import TestClient

from src.api import create_app


class CompatTestClient(TestClient):
    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        async def run_request() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url=str(self.base_url)) as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(run_request())


def test_healthz_returns_ok():
    client = CompatTestClient(create_app())
    try:
        response = client.get("/healthz")
    finally:
        client.close()

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
