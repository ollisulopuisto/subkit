"""subkit command line. Run with: uv run subkit <command> ...

Commands:
  translate   Translate an existing .srt/.ass file.
  transcribe  Transcribe a media file to a subtitle file (no translation).
  library     Walk a media library: transcribe missing subs and/or translate.

Every command prints a run report; untranslated cue fallbacks are listed as
INCOMPLETE ids, never silently counted as done.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
import sys

from subkit.config import AppConfig
from subkit.engine import TranslationEngine
from subkit.jellyfin import (
    extract_embedded_srt,
    plan_library,
    sidecar_covers_lang,
    write_sidecar_atomic,
)
from subkit.merger import format_srt, write_subtitle_file
from subkit.parser import Subtitle, SubtitleDocument, parse_subtitle_file
from subkit.providers import build_provider


def _segments_to_doc(segments) -> SubtitleDocument:

    def fmt(sec: float) -> str:
        ms = max(0, int(round(sec * 1000)))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    subs = [
        Subtitle(id=i + 1, start=fmt(seg.start), end=fmt(seg.end), text=seg.text)
        for i, seg in enumerate(segments)
    ]
    return SubtitleDocument(format="srt", subtitles=subs)


def _report(report, out_path: Path | None) -> None:
    print(f"chunks: {report.chunks_done}/{report.chunks_total}")
    if report.dropped_ids:
        print(
            f"INCOMPLETE: {len(report.dropped_ids)} cues kept the original "
            f"text (model returned no translation): ids {report.dropped_ids[:50]}"
            + (" ..." if len(report.dropped_ids) > 50 else "")
        )
    if report.cps_violations:
        print(
            f"reading-speed: {len(report.cps_violations)} cues still over budget "
            f"after compression retry: ids {report.cps_violations[:50]}"
        )
    if report.glossary:
        print(f"glossary: {len(report.glossary)} terms")
    if out_path:
        print(f"wrote: {out_path}")


def cmd_translate(args: argparse.Namespace) -> int:
    cfg = AppConfig()
    cfg.translator.target_language = args.target
    cfg.translator.source_language = args.source
    if args.glossary:
        cfg.translator.glossary_path = args.glossary
    if args.save_glossary:
        cfg.translator.rolling_glossary_save = args.save_glossary
    provider = build_provider(cfg.provider)
    doc = parse_subtitle_file(args.input)
    report = asyncio.run(
        TranslationEngine(cfg, provider).translate_document(doc, resume_id=args.input)
    )
    out = (
        Path(args.output)
        if args.output
        else Path(args.input).with_suffix(f".{args.target_code or args.target.lower()}.srt")
    )
    write_subtitle_file(doc, report.subtitles, out)
    _report(report, out)
    return 0 if report.ok else 2


def cmd_transcribe(args: argparse.Namespace) -> int:
    from subkit.asr.base import transcribe

    kwargs = {}
    if args.engine == "sherpa":
        if not args.model_dir or not args.vad_model:
            print("sherpa engine needs --model-dir and --vad-model", file=sys.stderr)
            return 1
        kwargs = {
            "model_dir": args.model_dir,
            "vad_model": args.vad_model,
            "num_threads": args.threads,
        }
    else:
        kwargs = {"model": args.model}
    result = transcribe(args.input, engine=args.engine, language=args.language, **kwargs)
    doc = _segments_to_doc(result.segments)
    out = Path(args.output) if args.output else Path(args.input).with_suffix(".srt")
    out.write_text(format_srt(doc.subtitles), encoding="utf-8")
    print(f"{len(result.segments)} segments -> {out}")
    return 0


def cmd_library(args: argparse.Namespace) -> int:
    root = Path(args.root)
    plan = plan_library(root, args.target_code, args.source_code, force=args.force)
    print(plan.summary())
    if args.dry_run:
        for p in plan.translate:
            print(f"  translate (source subs found): {p}")
        for p in plan.transcribe:
            print(f"  transcribe (no source subs):   {p}")
        return 0

    cfg = AppConfig()
    cfg.translator.target_language = args.target
    provider = build_provider(cfg.provider) if args.translate else None
    engine = TranslationEngine(cfg, provider) if provider else None
    failures = 0

    def finish(video: Path, doc, resume_key: str) -> None:
        if engine is not None:
            report = asyncio.run(engine.translate_document(doc, resume_id=resume_key))
            subs = report.subtitles
            if report.dropped_ids:
                print(f"INCOMPLETE {video.name}: {len(report.dropped_ids)} cues kept original text")
        else:
            subs = doc.subtitles
        out = write_sidecar_atomic(
            video,
            args.target_code,
            format_srt(subs),
            generated_marker=not args.no_generated_marker,
            force=args.force,
        )
        print(f"wrote: {out}")

    if plan.transcribe and not args.transcribe_engine:
        print(
            f"NOTE: {len(plan.transcribe)} items have no source subtitles and need ASR; "
            "pass --transcribe-engine sherpa (or mlx) to process them"
        )

    for video in plan.translate:
        try:
            src = sidecar_covers_lang(video, args.source_code)
            tmp = None
            if src is None:
                tmp = extract_embedded_srt(video, args.source_code)
                src = tmp
            if src is None:
                print(f"FAILED {video}: no extractable {args.source_code} subtitles")
                failures += 1
                continue
            try:
                doc = parse_subtitle_file(src)
            finally:
                if tmp:
                    tmp.unlink(missing_ok=True)
            if engine is None:
                print(f"skip (needs --translate): {video}")
                continue
            finish(video, doc, str(video))
        except Exception as exc:
            failures += 1
            print(f"FAILED {video}: {exc}")

    if args.transcribe_engine:
        for video in plan.transcribe:
            try:
                from subkit.asr.base import transcribe

                kwargs = (
                    {
                        "model_dir": args.model_dir,
                        "vad_model": args.vad_model,
                        "num_threads": args.threads,
                    }
                    if args.transcribe_engine == "sherpa"
                    else {}
                )
                result = transcribe(str(video), engine=args.transcribe_engine, **kwargs)
                doc = _segments_to_doc(result.segments)
                finish(video, doc, str(video))
            except Exception as exc:
                failures += 1
                print(f"FAILED {video}: {exc}")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="subkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("translate", help="translate an existing .srt/.ass file")
    t.add_argument("input")
    t.add_argument("-o", "--output")
    t.add_argument("--target", default="Finnish")
    t.add_argument("--target-code", default="fi")
    t.add_argument("--source", default=None)
    t.add_argument("--glossary", default=None, help="static per-series glossary JSON")
    t.add_argument(
        "--save-glossary", default=None, help="write the rolling glossary here after the run"
    )
    t.set_defaults(func=cmd_translate)

    tr = sub.add_parser("transcribe", help="transcribe media to a subtitle file")
    tr.add_argument("input")
    tr.add_argument("-o", "--output")
    tr.add_argument("--engine", choices=["sherpa", "mlx"], default="sherpa")
    tr.add_argument("--model-dir", help="sherpa-onnx model directory (Orukeet/Parakeet INT8)")
    tr.add_argument("--vad-model", help="silero VAD .onnx path")
    tr.add_argument(
        "--model", default="mlx-community/parakeet-tdt-0.6b-v3", help="mlx engine model id"
    )
    tr.add_argument("--threads", type=int, default=4)
    tr.add_argument("--language", default=None)
    tr.set_defaults(func=cmd_transcribe)

    li = sub.add_parser("library", help="batch-process a media library for Jellyfin")
    li.add_argument("root")
    li.add_argument("--target", default="Finnish")
    li.add_argument("--target-code", default="fi")
    li.add_argument(
        "--source-code",
        default="en",
        help="source subtitle language to translate from when present",
    )
    li.add_argument(
        "--translate",
        action="store_true",
        help="translate source subtitles (and transcripts) to the target language",
    )
    li.add_argument("--transcribe-engine", choices=["sherpa", "mlx"], default=None)
    li.add_argument("--model-dir")
    li.add_argument("--vad-model")
    li.add_argument("--threads", type=int, default=4)
    li.add_argument(
        "--force",
        action="store_true",
        help="regenerate subkit-written (.generated) subs; never overwrites user-owned subtitles",
    )
    li.add_argument(
        "--no-generated-marker",
        action="store_true",
        help="name output Name.fi.srt instead of Name.fi.generated.srt",
    )
    li.add_argument("--dry-run", action="store_true")
    li.set_defaults(func=cmd_library)

    args = ap.parse_args()
    logging.basicConfig(
        level=getattr(logging, AppConfig().translator.log_level, logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
