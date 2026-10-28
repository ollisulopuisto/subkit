# subkit

Subtitle pipeline: transcribe with Parakeet/Orukeet ASR, translate with an LLM
(Gemini API or any OpenAI-compatible endpoint such as a local llama-swap),
write Jellyfin-compatible sidecar subtitles.

## Provenance and licenses

- Code derived from ai-subtitle-translator
  (https://github.com/surbiks/ai-subtitle-translator, commit 2e33056).
  The repo carries NO LICENSE file; its README states "License: MIT".
  Parsing, chunking, ASS style handling and file writing come from it.
- Sync-guarantee and glossary mechanisms inspired by Subly
  (https://github.com/zakariaf/Subly, commit dce9348). Subly carries NO
  license file, so none of its code was copied; the mechanisms were
  reimplemented from its README description.

What those mechanisms are (Subly's "Why it stays in sync", reimplemented):

- stable integer cue ids go to the model, timestamps never do
- the model returns the same ids back; output is matched to input by id
- any cue the model drops falls back to its original text and is reported as
  INCOMPLETE - the SRT is never shorter and the gap is visible
- terms are pinned, translated, self-reviewed within one call; a small
  glossary of term decisions is carried from each chunk into the next
- each cue also carries a max_chars budget from its duration and a CPS
  reading-speed cap; over-budget translations get one compression retry

## Status: what is tested

| part | state |
|---|---|
| parser, chunker, engine (id alignment, dropped-id fallback flagging, rolling glossary carry, full-transcript context, CPS budget, correction retry, request budget across ALL call kinds, resume + stale-state invalidation, context budget), Jellyfin categorization/force/no-overwrite rules, Gemini request shape and thinking-model gating | 24 pytest tests with a mock provider (no network, no models) |
| full CLI (`translate`, `library --translate`) | exercised end-to-end against a localhost stub server in the build sandbox; resume verified (second run made zero provider calls) |
| Gemini/OpenAI providers against live APIs | untested (no API key was used) |
| sherpa-onnx ASR backend | API usage verified by introspecting the installed sherpa-onnx 2.0.1 wheel (VoiceActivityDetector, OfflineRecognizer.from_transducer signatures); no model was downloaded, so no audio has been transcribed |
| parakeet-mlx backend (Apple Silicon) | API usage verified against the parakeet-mlx source (from_pretrained, model.transcribe, AlignedResult.sentences); untested - requires a Mac |
| Docker image | Dockerfile statically reviewed only - no container runtime was available in the build environment, image NOT built |

## Install

```bash
uv sync                      # core only
uv sync --extra asr-sherpa   # + sherpa-onnx transcription (x86_64/arm64 Linux, Mac)
uv sync --extra asr-mlx      # + parakeet-mlx (Apple Silicon only)
uv sync --extra dev          # + pytest
```

## Translate an existing subtitle

```bash
export GEMINI_API_KEY=...
uv run subkit translate movie.srt --target Finnish --target-code fi \
    --glossary series-glossary.json --save-glossary series-glossary.json
# writes movie.fi.srt; INCOMPLETE cue ids are printed if any fell back
```

To use a local OpenAI-compatible server (llama-swap, Ollama, ...) instead:

```bash
export SUBKIT_PROVIDER=openai
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_MODEL=gemma4
uv run subkit translate movie.srt --target Finnish
```

## Transcribe with Parakeet/Orukeet

Orukeet (https://huggingface.co/oruk/orukeet) is a Parakeet TDT 0.6B v3
fine-tune covering 25 languages including Finnish, CC BY-SA 4.0. For
sherpa-onnx use its sherpa-layout export
(https://huggingface.co/oruk/orukeet/tree/main/onnx/sherpa-v0.1.0-int8, or the
sherpa-onnx-orukeet-v0.1.0-int8.tar.bz2 archive). The "combined" export merges
decoder+joiner into one graph for Rust engines; sherpa-onnx cannot load it and
subkit will refuse it with a pointer to the right download. Models are not
bundled - download them yourself.

```bash
uv run subkit transcribe movie.mkv \
    --engine sherpa --model-dir ./orukeet-sherpa-int8 --vad-model ./silero_vad.onnx
# writes movie.srt
```

Note: Parakeet/Orukeet are offline (non-streaming) models; this tool is for
batch use (https://github.com/k2-fsa/sherpa-onnx/issues/2918).

## Batch a Jellyfin library

Per video, subkit decides:

- Finnish subs already exist (sidecar or embedded) -> skip
- English (or --source-code) subs exist but no Finnish -> TRANSLATE from them
- no usable subs at all -> TRANSCRIBE (needs --transcribe-engine)

Existing English subtitles are a translation source, never a reason to skip.
`--force` only regenerates files subkit itself wrote (`.generated.srt`);
user-owned subtitles are never overwritten. Writes are atomic.

```bash
uv run subkit library /volume1/media \
    --translate --target Finnish --target-code fi --source-code en \
    --transcribe-engine sherpa --model-dir ./orukeet-sherpa-int8 --vad-model ./silero_vad.onnx \
    --dry-run        # list planned actions first
```

Output is `Name.fi.generated.srt`; Jellyfin picks it up on the next scan or
item refresh. The `.generated` token becomes the stream title in Jellyfin's
subtitle picker (https://jellyfin.org/docs/general/server/media/external-files).

## Reliability: retries, rate limits, resume, cost control

- Transport/API errors retry with exponential backoff (MAX_RETRIES, default 3).
- REQUEST_DELAY=<seconds> paces provider calls (rate-limit friendly).
- MAX_REQUESTS=<n> sets a hard cap on provider calls per run; when hit, the
  run stops with an explicit error - already completed chunks are kept.
- MAX_REQUESTS counts every provider call: translation, correction AND
  compression retries. Hitting the budget raises a distinct error that is
  never retried or swallowed.
- Every run writes `<input>.subkit-status.json` after each chunk. The file is
  fingerprinted with the source text, languages, provider/model/endpoint,
  glossary and prompt; ANY change to those invalidates it (stale translations
  are never silently reused) and the run starts fresh with a warning.
  RESUME_MODE=fresh ignores state entirely. At library level, finished items
  are skipped because their `.generated.srt` already exists.
- Context: every provider call carries the FULL original transcript as
  read-only context plus the id-pinned chunk to translate. This does NOT scale
  without limit: if the transcript exceeds MAX_CONTEXT_CHARS (default
  800,000) the run refuses BEFORE sending, with instructions (FULL_CONTEXT=off
  falls back to a previous-chunk window). gemini-3.8-flash allows 1,048,576
  input / 65,536 output tokens per call
  (https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash).
- thinkingLevel is a Gemini 3.x-only parameter. With a non-3.x SUBKIT_MODEL it
  is auto-omitted when left at the default; setting GEMINI_THINKING_LEVEL
  explicitly with a non-3.x model is a clean config error at startup, never a
  rejected API call.

## Docker / NAS

```bash
docker build -t subkit .
docker run --rm -v /volume1/media:/media -v $PWD/models:/models \
    -e GEMINI_API_KEY=... \
    subkit library /media --translate --target Finnish \
    --transcribe-engine sherpa --model-dir /models/orukeet-sherpa-int8 \
    --vad-model /models/silero_vad.onnx
```

The image targets x86_64 Linux. Check your NAS CPU with `uname -m` first;
arm64 needs the base image and sherpa-onnx wheel to match.

## First run

No API key, no models, no writes - see what would happen:

```bash
uv sync   # minimal install: httpx is the only runtime dependency
uv run subkit library /path/to/media --dry-run
```

`uv.lock` is included for reproducible installs; the CLI entry point is
`uv run subkit` (installed as the `subkit` script from subkit.cli:main).

## Configuration

See .env.sample for every variable: provider, model, endpoints, glossary
paths, CPS limits, chunk sizes, retry policy, request pacing and budget.
Nothing is hardcoded to a specific machine.
