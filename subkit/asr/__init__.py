"""ASR backends: media file -> timed text segments.

Backends:
- "sherpa": sherpa-onnx offline transducer. Runs the Parakeet TDT v3 family,
  including Orukeet (oruk/orukeet) INT8, on CPU - works on x86_64 Synology NAS
  and Apple Silicon Macs alike. Needs a model directory and a Silero VAD model,
  both user-provided (no automatic downloads).
- "mlx": parakeet-mlx, Apple Silicon only, faster on Mac.
"""

from subkit.asr.base import Segment, TranscriptResult, transcribe
