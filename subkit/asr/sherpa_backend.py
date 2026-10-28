"""sherpa-onnx backend for the Parakeet TDT family (incl. Orukeet INT8).

Segmentation comes from Silero VAD; each speech segment is decoded by the
offline transducer. Both model directories are user-supplied paths - nothing
is downloaded at runtime.

Model layout (sherpa-onnx separate-graph transducer export):
  encoder*.onnx, decoder*.onnx, joiner*.onnx, tokens.txt
For Orukeet use the "sherpa-v0.1.0-int8" export
(https://huggingface.co/oruk/orukeet/tree/main/onnx/sherpa-v0.1.0-int8, also
as sherpa-onnx-orukeet-v0.1.0-int8.tar.bz2). The "combined" Orukeet export
merges decoder+joiner into one graph for Rust engines and CANNOT be loaded
through sherpa-onnx's OfflineRecognizer.from_transducer - this backend
detects that layout and refuses with a pointer to the right download.

Parakeet TDT uses 128 mel features (nemo128), so feature_dim defaults to 128.

API verified against sherpa-onnx 2.0.1 (installed wheel introspection):
VoiceActivityDetector(VadModelConfig, buffer_size_in_seconds), SpeechSegment
(start, samples), OfflineRecognizer.from_transducer(encoder, decoder,
joiner, tokens, num_threads, sample_rate, feature_dim, decoding_method).

NOTE: untested end-to-end on real hardware in this build - validate on one
short clip before a library run.
"""

from __future__ import annotations

import logging
from pathlib import Path

from subkit.asr.base import Segment, TranscriptResult

logger = logging.getLogger(__name__)

_MAX_SEGMENT_SEC = 30.0
_MERGE_GAP_SEC = 0.6
_FEED_SEC = 5.0


def _find_model_files(model_dir: str) -> tuple[str, str, str, str]:
    d = Path(model_dir)
    if (d / "decoder_joint-model.int8.onnx").exists() or (d / "decoder_joint.onnx").exists():
        raise FileNotFoundError(
            f"{model_dir} looks like the COMBINED Orukeet export "
            "(decoder_joint single graph), which sherpa-onnx cannot load. "
            "Download the sherpa layout instead: "
            "https://huggingface.co/oruk/orukeet/tree/main/onnx/sherpa-v0.1.0-int8"
        )
    triples = [
        ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx"),
        ("encoder-model.int8.onnx", "decoder-model.int8.onnx", "joiner-model.int8.onnx"),
        ("encoder.onnx", "decoder.onnx", "joiner.onnx"),
    ]
    for enc, dec, joi in triples:
        if (d / enc).exists() and (d / dec).exists() and (d / joi).exists():
            tokens = d / "tokens.txt"
            if not tokens.exists():
                raise FileNotFoundError(f"missing tokens.txt in {model_dir}")
            return str(d / enc), str(d / dec), str(d / joi), str(tokens)
    raise FileNotFoundError(
        f"no encoder/decoder/joiner ONNX triple found in {model_dir}; "
        "expected the sherpa-onnx transducer layout"
    )


def transcribe_sherpa(
    audio_path: str,
    model_dir: str,
    vad_model: str,
    num_threads: int = 4,
    sample_rate: int = 16000,
    feature_dim: int = 128,
    language: str | None = None,
) -> TranscriptResult:
    import sherpa_onnx

    from subkit.asr.audio import extract_audio

    samples = extract_audio(audio_path, sample_rate)
    if samples.size == 0:
        return TranscriptResult(segments=[], language=language)

    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = vad_model
    vad_config.silero_vad.min_speech_duration = 0.25
    vad_config.silero_vad.min_silence_duration = 0.5
    vad_config.sample_rate = sample_rate
    vad_config.num_threads = 1
    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=120)

    encoder, decoder, joiner, tokens = _find_model_files(model_dir)
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        tokens=tokens,
        num_threads=num_threads,
        sample_rate=sample_rate,
        feature_dim=feature_dim,
        decoding_method="greedy_search",
    )

    # Feed audio in small slices, draining finished speech segments as we go.
    windows: list[tuple[float, float]] = []

    def drain() -> None:
        while not vad.empty():
            seg = vad.front()
            windows.append((seg.start / sample_rate,
                            (seg.start + len(seg.samples)) / sample_rate))
            vad.pop()

    feed = int(_FEED_SEC * sample_rate)
    for i in range(0, samples.size, feed):
        vad.accept_waveform(samples[i:i + feed])
        drain()
    vad.flush()
    drain()

    merged: list[list[float]] = []
    for start, end in windows:
        if merged and start - merged[-1][1] <= _MERGE_GAP_SEC and end - merged[-1][0] <= _MAX_SEGMENT_SEC:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    segments: list[Segment] = []
    for start, end in merged:
        clip = samples[int(start * sample_rate):int(end * sample_rate)]
        if clip.size == 0:
            continue
        stream = recognizer.create_stream()
        stream.accept_waveform(sample_rate, clip)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        if text:
            segments.append(Segment(start=start, end=end, text=text))
        logger.info("ASR %.1f-%.1fs: %s", start, end, text[:60])

    return TranscriptResult(segments=segments, language=language)
