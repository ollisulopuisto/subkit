"""parakeet-mlx backend (Apple Silicon only).

Covers nvidia/parakeet-tdt-0.6b-v3 and MLX-converted derivatives. Orukeet has
no published MLX conversion at build time, so use the sherpa backend for it.

NOTE: untested in this build; parakeet-mlx exposes from_pretrained().transcribe()
returning sentence-level timestamps in its documented API.
"""

from __future__ import annotations

from subkit.asr.base import Segment, TranscriptResult


def transcribe_mlx(
    audio_path: str,
    model: str = "mlx-community/parakeet-tdt-0.6b-v3",
    language: str | None = None,
    **_: object,
) -> TranscriptResult:
    from parakeet_mlx import from_pretrained

    mdl = from_pretrained(model)
    result = mdl.transcribe(audio_path)
    segments = [
        Segment(start=s.start, end=s.end, text=s.text.strip())
        for s in result.sentences
        if s.text and s.text.strip()
    ]
    return TranscriptResult(
        segments=segments, language=language or getattr(result, "language", None)
    )
