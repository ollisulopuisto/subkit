"""Translation backends. One async chat-style interface over two providers:

- GeminiProvider: Google Gemini generateContent REST API (no SDK dependency,
  so the Synology NAS install stays small).
- OpenAIProvider: any OpenAI-compatible /v1/chat/completions endpoint
  (OpenAI itself, llama.cpp / llama-swap, Ollama, ...).

Both take a system prompt plus a list of user/assistant messages and return
the reply text. HTTP errors raise; the engine owns retry policy.
"""

from __future__ import annotations

import logging
from typing import Protocol

import httpx

from subkit.config import ProviderConfig

logger = logging.getLogger(__name__)

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class ChatProvider(Protocol):
    name: str

    async def chat(self, system: str, messages: list[dict[str, str]]) -> str:
        """Return the model's reply text for system + user/assistant messages."""
        ...


class GeminiProvider:
    """Gemini generateContent. System prompt goes to systemInstruction; the
    user/assistant message list maps to contents with roles user/model."""

    def __init__(self, cfg: ProviderConfig) -> None:
        if not cfg.api_key:
            raise ValueError("Gemini provider needs an API key (GEMINI_API_KEY)")
        # thinkingLevel is a Gemini 3.x-only parameter. If the user explicitly
        # set it for a non-3.x model, fail with a clean config error; if it is
        # just the default, auto-omit it instead of sending a rejected field.
        import os

        level = (cfg.thinking_level or "").lower()
        wants_thinking = level not in ("", "default", "off", "none")
        is_gemini3 = "gemini-3" in cfg.model.lower()
        if wants_thinking and not is_gemini3:
            if os.getenv("GEMINI_THINKING_LEVEL") is not None:
                raise ValueError(
                    f"GEMINI_THINKING_LEVEL={cfg.thinking_level} is only valid on Gemini 3.x "
                    f"models; SUBKIT_MODEL={cfg.model} would reject it. "
                    "Set GEMINI_THINKING_LEVEL=off or use a Gemini 3.x model."
                )
            cfg.thinking_level = "off"  # default not applicable to this model: omit
        self._cfg = cfg
        self.name = f"gemini/{cfg.model}"

    async def chat(self, system: str, messages: list[dict[str, str]]) -> str:
        cfg = self._cfg
        contents = [
            {
                "role": "user" if m["role"] == "user" else "model",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
        ]
        body: dict = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {
                "temperature": cfg.temperature,
                "maxOutputTokens": cfg.max_output_tokens,
            },
        }
        # Thinking level is supported by Gemini 3.x Flash; older models may
        # reject it, so it is only sent when explicitly non-default.
        if cfg.thinking_level and cfg.thinking_level.lower() not in ("default", "off", "none"):
            body["generationConfig"]["thinkingConfig"] = {"thinkingLevel": cfg.thinking_level}
        url = f"{GEMINI_API_BASE}/models/{cfg.model}:generateContent"
        async with httpx.AsyncClient(timeout=cfg.request_timeout) as client:
            resp = await client.post(url, json=body, headers={"x-goog-api-key": cfg.api_key})
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini API {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        candidate = (data.get("candidates") or [{}])[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        finish = candidate.get("finishReason")
        if finish == "MAX_TOKENS":
            logger.warning("Gemini hit MAX_TOKENS - output truncated")
        if not text.strip():
            raise RuntimeError(f"Gemini returned no text (finishReason={finish})")
        return text


class OpenAIProvider:
    """OpenAI-compatible chat.completions (OpenAI, llama-swap, Ollama, ...)."""

    def __init__(self, cfg: ProviderConfig) -> None:
        self._cfg = cfg
        base = (cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        self._url = f"{base}/chat/completions"
        self.name = f"openai/{cfg.model}@{base}"

    async def chat(self, system: str, messages: list[dict[str, str]]) -> str:
        cfg = self._cfg
        headers = {}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        body = {
            "model": cfg.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_output_tokens,
        }
        async with httpx.AsyncClient(timeout=cfg.request_timeout) as client:
            resp = await client.post(self._url, json=body, headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI-compatible API {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        if not text.strip():
            raise RuntimeError("OpenAI-compatible API returned no text")
        return text


def build_provider(cfg: ProviderConfig) -> ChatProvider:
    if cfg.kind == "gemini":
        return GeminiProvider(cfg)
    if cfg.kind == "openai":
        return OpenAIProvider(cfg)
    raise ValueError(f"Unknown provider kind: {cfg.kind!r} (expected gemini|openai)")
