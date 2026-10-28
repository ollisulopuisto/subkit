"""Configuration for subkit. Values resolve: CLI flag > env var > default."""

from __future__ import annotations

from dataclasses import dataclass, field
import os


def _env(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key, default)
    return val if val else default


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    return int(val) if val else default


def _env_float(key: str, default: float) -> float:
    val = os.getenv(key)
    return float(val) if val else default


def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class ChunkConfig:
    max_lines: int = _env_int("CHUNK_MAX_LINES", 18)
    max_chars: int = _env_int("CHUNK_MAX_CHARS", 1500)
    time_gap_threshold_ms: int = _env_int("CHUNK_TIME_GAP_MS", 2500)
    context_lines: int = _env_int("CHUNK_CONTEXT_LINES", 3)


@dataclass
class ProviderConfig:
    """One translation backend.

    kind: "gemini" | "openai"
      - gemini: Google Gemini generateContent API. Needs api_key (GEMINI_API_KEY).
      - openai: any OpenAI-compatible /v1/chat/completions endpoint
        (OpenAI, llama.cpp / llama-swap, Ollama, ...). base_url required for
        non-OpenAI hosts; api_key optional for local endpoints.
    """

    kind: str = _env("SUBKIT_PROVIDER", "gemini")  # type: ignore[assignment]
    model: str = _env("SUBKIT_MODEL", "gemini-3.8-flash")  # type: ignore[assignment]
    api_key: str | None = _env("GEMINI_API_KEY") or _env("OPENAI_API_KEY")
    base_url: str | None = _env("OPENAI_BASE_URL")
    # Gemini generation knobs (ignored by the openai provider).
    # thinkingLevel is a Gemini 3.x parameter; other models reject it.
    # Set GEMINI_THINKING_LEVEL=off to omit thinkingConfig entirely.
    thinking_level: str = _env("GEMINI_THINKING_LEVEL", "low")  # type: ignore[assignment]
    temperature: float = _env_float("SUBKIT_TEMPERATURE", 0.3)
    max_output_tokens: int = _env_int("SUBKIT_MAX_OUTPUT_TOKENS", 16384)
    request_timeout: float = _env_float("SUBKIT_REQUEST_TIMEOUT", 120.0)


@dataclass
class TranslatorConfig:
    target_language: str = "Finnish"
    source_language: str | None = None
    log_level: str = _env("LOG_LEVEL", "INFO")  # type: ignore[assignment]
    max_concurrency: int = _env_int("MAX_CONCURRENCY", 3)
    max_retries: int = _env_int("MAX_RETRIES", 3)
    retry_base_delay: float = _env_float("RETRY_BASE_DELAY", 1.5)
    # Reading-speed budget: characters per second of on-screen time.
    cps_target: float = _env_float("CPS_TARGET", 15.0)
    cps_min_chars: int = _env_int("CPS_MIN_CHARS", 20)
    enforce_cps: bool = _env_bool("ENFORCE_CPS", True)
    cps_tolerance: float = _env_float("CPS_TOLERANCE", 1.25)
    # Glossary lifecycle
    glossary_path: str | None = _env("GLOSSARY_PATH")
    rolling_glossary: bool = _env_bool("ROLLING_GLOSSARY", True)
    rolling_glossary_save: str | None = _env("ROLLING_GLOSSARY_SAVE")
    enforce_glossary: bool = _env_bool("ENFORCE_GLOSSARY", True)
    # Line-kind handling
    translate_cues: bool = _env_bool("TRANSLATE_CUES", True)
    translate_lyrics: bool = _env_bool("TRANSLATE_LYRICS", True)
    # Resume: "auto" resumes a matching status file, "fresh" ignores it.
    resume_mode: str = _env("RESUME_MODE", "auto")  # type: ignore[assignment]
    status_dir: str | None = _env("STATUS_DIR")
    # Rate/cost control.
    request_delay: float = _env_float("REQUEST_DELAY", 0.0)  # pause between provider calls
    max_requests: int = _env_int("MAX_REQUESTS", 0)  # 0 = unlimited; hard stop when hit
    # Send the whole original transcript as read-only context in every call.
    full_context: bool = _env_bool("FULL_CONTEXT", True)
    # Hard fail before sending when the transcript context exceeds this size.
    # 800k chars is comfortably inside gemini-3.8-flash's 1,048,576-token
    # window; it is a deliberate ceiling, not a claim of unlimited support.
    max_context_chars: int = _env_int("MAX_CONTEXT_CHARS", 800_000)


@dataclass
class AppConfig:
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
