"""ASR backend protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Segment:
    start: float  # seconds
    end: float    # seconds
    text: str


@dataclass
class TranscriptResult:
    segments: list[Segment]
    language: str | None


def transcribe(audio_path: str, engine: str = "sherpa", **kwargs) -> TranscriptResult:
    if engine == "sherpa":
        from subkit.asr.sherpa_backend import transcribe_sherpa
        return transcribe_sherpa(audio_path, **kwargs)
    if engine == "mlx":
        from subkit.asr.mlx_backend import transcribe_mlx
        return transcribe_mlx(audio_path, **kwargs)
    raise ValueError(f"Unknown ASR engine: {engine!r} (expected sherpa|mlx)")
