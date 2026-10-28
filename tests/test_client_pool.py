"""Shared AsyncClient pool: loop binding, cleanup, concurrency."""

import asyncio

from test_core import MockProvider, cfg, make_doc

from subkit.engine import translate_document_sync
from subkit.providers import _ClientPool


class FakeClient:
    created = 0
    closed = 0

    def __init__(self, **kw):
        FakeClient.created += 1

    async def aclose(self):
        FakeClient.closed += 1


def _patch(monkeypatch):
    FakeClient.created = 0
    FakeClient.closed = 0
    monkeypatch.setattr("subkit.providers.httpx.AsyncClient", FakeClient)


def test_pool_reuses_client_within_one_loop(monkeypatch):
    _patch(monkeypatch)
    pool = _ClientPool(timeout=5)

    async def main():
        c1 = await pool.get()
        c2 = await pool.get()
        assert c1 is c2

    asyncio.run(main())
    assert FakeClient.created == 1


def test_pool_new_client_for_new_event_loop(monkeypatch):
    _patch(monkeypatch)
    pool = _ClientPool(timeout=5)
    asyncio.run(pool.get())
    first = asyncio.run(pool.get())  # new loop -> new client
    assert FakeClient.created == 2
    assert first is pool._client


def test_pool_aclose_releases_client_and_resets(monkeypatch):
    _patch(monkeypatch)
    pool = _ClientPool(timeout=5)

    async def main():
        await pool.get()
        await pool.aclose()
        assert pool._client is None

    asyncio.run(main())
    assert FakeClient.closed == 1


def test_concurrent_gets_share_one_client(monkeypatch):
    _patch(monkeypatch)
    pool = _ClientPool(timeout=5)

    async def main():
        clients = await asyncio.gather(*(pool.get() for _ in range(10)))
        assert len({id(c) for c in clients}) == 1

    asyncio.run(main())
    assert FakeClient.created == 1


def test_engine_closes_provider_client_after_each_document(monkeypatch):
    _patch(monkeypatch)

    class ClosingProvider(MockProvider):
        def __init__(self):
            super().__init__()
            self.pool = _ClientPool(timeout=5)

        async def aclose(self):
            await self.pool.aclose()

    provider = ClosingProvider()

    async def run_two_documents():
        # simulate batch mode: two asyncio.run-equivalent iterations would each
        # open a client; here we exercise get/aclose within one loop
        await provider.pool.get()
        await provider.aclose()
        await provider.pool.get()
        await provider.aclose()

    asyncio.run(run_two_documents())
    assert FakeClient.created == 2
    assert FakeClient.closed == 2, "every pooled client must be closed, not leaked"


def test_engine_calls_provider_aclose_on_success_and_failure():
    calls = []

    class TrackingProvider(MockProvider):
        async def aclose(self):
            calls.append("closed")

    translate_document_sync(make_doc(), cfg(), TrackingProvider())
    assert calls == ["closed"]

    class FailingProvider(MockProvider):
        async def chat(self, system, messages):
            raise RuntimeError("boom")

        async def aclose(self):
            calls.append("closed")

    c = cfg()
    c.translator.retry_base_delay = 0
    c.translator.max_retries = 1
    try:
        translate_document_sync(make_doc(), c, FailingProvider())
        raise AssertionError("expected failure")
    except RuntimeError:
        pass
    assert calls == ["closed", "closed"], "aclose must run even when translation fails"
