"""Provider error surfaces must not leak credentials.

Provider/proxy error bodies are untrusted text: OpenAI's 401 is known to echo
the API key back in the message (sometimes truncated/obfuscated). That body
reaches logs and tracebacks through ProviderHTTPError, so it is redacted at the
source rather than at each print site.
"""

import asyncio

import pytest

from subkit.config import ProviderConfig
from subkit.providers import OpenAIProvider, ProviderHTTPError

FULL_KEY = "sk-live-SECRET-KEY-1234567890"
# Shape OpenAI actually returns: first 8 + asterisks + last 4.
TRUNCATED = "sk-EhmTs**************************************RcQh"


class _Resp:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


class _Client:
    def __init__(self, resp):
        self._resp = resp

    async def post(self, url, json=None, headers=None):
        return self._resp


def _provider(monkeypatch, body, api_key):
    monkeypatch.setattr(
        "subkit.providers.httpx.AsyncClient", lambda **kw: _Client(_Resp(401, body))
    )
    return OpenAIProvider(
        ProviderConfig(kind="openai", model="m", api_key=api_key, base_url="http://127.0.0.1:1/v1")
    )


@pytest.mark.parametrize(
    "echo,fragments",
    [
        (FULL_KEY, [FULL_KEY, "SECRET-KEY"]),
        (TRUNCATED, ["EhmTs", "RcQh"]),
        ("Bearer " + FULL_KEY, [FULL_KEY]),
    ],
)
def test_provider_error_redacts_api_key(monkeypatch, echo, fragments):
    body = f'{{"error": {{"message": "Incorrect API key provided: {echo}"}}}}'
    provider = _provider(monkeypatch, body, FULL_KEY)
    with pytest.raises(ProviderHTTPError) as ei:
        asyncio.run(provider.chat("s", [{"role": "user", "content": "hi"}]))
    msg = str(ei.value)
    for fragment in fragments:
        assert fragment not in msg, f"credential fragment {fragment!r} leaked: {msg}"
    assert "Incorrect API key provided" in msg  # useful detail survives redaction


def test_redact_credentials_covers_custom_secret_and_keeps_plain_text():
    from subkit.providers import _redact_credentials

    assert _redact_credentials("token=hunter2", secrets=("hunter2",)) == "token=[REDACTED]"
    assert _redact_credentials("bad request: max_tokens invalid") == (
        "bad request: max_tokens invalid"
    )
