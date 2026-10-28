"""Audio extraction via ffmpeg: any media file -> 16 kHz mono float32."""

from __future__ import annotations

import subprocess

import numpy as np


def extract_audio(media_path: str, sample_rate: int = 16000) -> np.ndarray:
    """Decode the first audio stream to mono f32 samples with ffmpeg."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        media_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on {media_path}: {proc.stderr.decode(errors='replace')[:400]}"
        )
    samples = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    del proc
    return samples
