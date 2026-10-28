"""The translation engine: chunks -> provider calls -> validated subtitles.

Merged design:
- stable integer cue ids; the model must return the same ids (Subly pattern).
- any id the model drops falls back to the ORIGINAL text, and the id is recorded
  in the run's `dropped_ids` - untranslated cues are visible incomplete work,
  never silently counted as finished.
- single-pass terms -> translate -> review prompt (Subly process) with per-line
  kind and max_chars budgets (ai-subtitle-translator payload).
- rolling glossary threaded chunk -> chunk and optionally saved for series reuse.
- CPS (reading-speed) check with one targeted compression retry.
- one targeted glossary-compliance retry when a pinned term is missing.

Chunks are processed SEQUENTIALLY when the rolling glossary is on (default):
each chunk's term decisions feed the next. Set rolling_glossary=False to allow
concurrency.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import logging
from pathlib import Path

from subkit.ass_tags import has_tags, restore_tags, strip_tags
from subkit.chunker import build_context_window, chunk_subtitles
from subkit.classify import LineKind, classify
from subkit.config import AppConfig
from subkit.glossary import Glossary
from subkit.parser import Subtitle, SubtitleDocument
from subkit.prompts import COMPRESSION_PROMPT, CORRECTION_PROMPT, SYSTEM_PROMPT
from subkit.providers import ChatProvider

logger = logging.getLogger(__name__)


class BudgetExhausted(RuntimeError):
    """Raised when the MAX_REQUESTS budget is hit. Never swallowed or retried."""


@dataclass
class RunReport:
    subtitles: list[Subtitle] = field(default_factory=list)
    dropped_ids: list[int] = field(default_factory=list)
    cps_violations: list[int] = field(default_factory=list)
    glossary: dict[str, str] = field(default_factory=dict)
    chunks_total: int = 0
    chunks_done: int = 0

    @property
    def ok(self) -> bool:
        return not self.dropped_ids


def _extract_block(content: str, tag: str) -> str:
    low = content.lower()
    open_t, close_t = f"<{tag}>", f"</{tag}>"
    i = low.find(open_t)
    if i == -1:
        return ""
    start = i + len(open_t)
    j = low.find(close_t, start)
    return content[start : (j if j != -1 else len(content))].strip()


def _loads_object(text: str):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].startswith("```"):
            text = "\n".join(lines[1:-1]).strip()
    text = text.strip("`").strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None


def _parse_segments(block: str) -> dict[int, str]:
    out: dict[int, str] = {}
    data = _loads_object(block)
    if isinstance(data, dict):
        for item in data.get("segments", []):
            try:
                text = str(item["text"]).strip()
                if text:
                    out[int(item["id"])] = text
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _parse_glossary(block: str) -> dict[str, str]:
    data = _loads_object(block)
    if not isinstance(data, dict):
        return {}
    return {
        str(k).strip(): str(v).strip() for k, v in data.items() if str(k).strip() and str(v).strip()
    }


def _char_budget(sub: Subtitle, cps_target: float, cps_min: int) -> int:
    duration_sec = max(0.5, (sub.end_ms - sub.start_ms) / 1000.0)
    return max(cps_min, int(duration_sec * cps_target))


class TranslationEngine:
    def __init__(self, cfg: AppConfig, provider: ChatProvider) -> None:
        self.cfg = cfg
        self.provider = provider
        self._glossary = (
            Glossary.from_file(cfg.translator.glossary_path)
            if cfg.translator.glossary_path
            else Glossary()
        )
        self._requests = 0

    @property
    def glossary(self) -> Glossary:
        return self._glossary

    async def _call(self, system: str, messages: list[dict[str, str]]) -> str:
        """One provider call with pacing and a hard request budget."""
        cfg = self.cfg.translator
        self._requests += 1
        if cfg.max_requests and self._requests > cfg.max_requests:
            raise BudgetExhausted(
                f"Request budget exhausted ({cfg.max_requests} calls, counting every "
                "translation, correction and compression request). Raise MAX_REQUESTS "
                "or rerun - completed chunks resume from the status file."
            )
        if cfg.request_delay > 0 and self._requests > 1:
            await asyncio.sleep(cfg.request_delay)
        return await self.provider.chat(system, messages)

    async def _chat_with_retries(
        self, system: str, messages: list[dict[str, str]], expect_ids: list[int]
    ) -> tuple[dict[int, str], dict[str, str]]:
        """Call the provider; on unparsable or incomplete output, send one
        correction request. Retries transport errors with backoff."""
        cfg = self.cfg.translator
        last_err: Exception | None = None
        for attempt in range(cfg.max_retries):
            call_messages = list(messages)
            try:
                content = await self._call(system, call_messages)
                translations = _parse_segments(_extract_block(content, "translation") or content)
                new_terms = _parse_glossary(_extract_block(content, "glossary"))
                missing = [i for i in expect_ids if i not in translations]
                if missing:
                    # One correction round for malformed/incomplete replies.
                    corr_messages = [
                        *call_messages,
                        {"role": "assistant", "content": content},
                        {"role": "user", "content": CORRECTION_PROMPT},
                    ]
                    content = await self._call(system, corr_messages)
                    fixed = _parse_segments(_extract_block(content, "translation") or content)
                    translations.update(fixed)
                    new_terms.update(_parse_glossary(_extract_block(content, "glossary")))
                return translations, new_terms
            except BudgetExhausted:
                raise
            except Exception as exc:  # transport/API errors
                last_err = exc
                wait = cfg.retry_base_delay * (2**attempt)
                logger.warning(
                    "Provider call failed (attempt %d): %s; retrying in %.1fs",
                    attempt + 1,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
        raise RuntimeError(f"Provider call failed after {cfg.max_retries} attempts: {last_err}")

    def _build_user_message(
        self,
        payload: list[dict],
        context: list[Subtitle] | None,
        full_transcript_json: str | None = None,
    ) -> str:
        cfg = self.cfg.translator
        parts = [f"TARGET LANGUAGE: {cfg.target_language}"]
        parts.append(f"SOURCE LANGUAGE: {cfg.source_language or 'auto-detect'}")
        parts.append(f"GLOSSARY: {json.dumps(self._glossary.entries, ensure_ascii=False)}")
        if full_transcript_json:
            parts.append(
                "FULL TRANSCRIPT (read-only, whole video for context; do NOT "
                "translate these, translate only SEGMENTS):\n" + full_transcript_json
            )
        elif context:
            ctx = [{"id": s.id, "text": s.text.replace(chr(10), " ")} for s in context]
            parts.append(
                "CONTEXT (read-only, do NOT translate):\n" + json.dumps(ctx, ensure_ascii=False)
            )
        parts.append("SEGMENTS:\n" + json.dumps(payload, ensure_ascii=False))
        return "\n".join(parts)

    async def _compress_lines(
        self, by_id: dict[int, Subtitle], translated: dict[int, str], over: list[int]
    ) -> dict[int, str]:
        """One targeted retry asking the model to shorten over-budget lines."""
        cfg = self.cfg.translator
        lines = json.dumps(
            [
                {
                    "id": i,
                    "text": translated[i],
                    "max_chars": _char_budget(by_id[i], cfg.cps_target, cfg.cps_min_chars),
                }
                for i in over
            ],
            ensure_ascii=False,
        )
        try:
            content = await self._call(
                "You shorten subtitle lines. Return only the required block.",
                [{"role": "user", "content": COMPRESSION_PROMPT.format(lines=lines)}],
            )
            fixed = _parse_segments(_extract_block(content, "translation") or content)
            for i, text in fixed.items():
                if i in over:
                    translated[i] = text
        except BudgetExhausted:
            raise
        except Exception as exc:
            logger.warning("Compression retry failed: %s", exc)
        return translated

    async def translate_document(
        self, doc: SubtitleDocument, resume_id: str | None = None
    ) -> RunReport:
        cfg = self.cfg.translator
        chunks = chunk_subtitles(doc.subtitles, self.cfg.chunk)
        contexts = build_context_window(chunks, self.cfg.chunk.context_lines)
        report = RunReport(chunks_total=len(chunks))
        results: dict[int, str] = {}

        state_path = _status_path(cfg, resume_id)
        done_chunks: set[int] = set()
        fingerprint = _fingerprint(self, doc)
        if state_path and cfg.resume_mode == "auto":
            done_chunks, results = _load_state(state_path, fingerprint)
            if done_chunks:
                logger.info("Resuming: %d chunks already done", len(done_chunks))
            saved_glossary = _load_state_glossary(state_path, fingerprint)
            if saved_glossary:
                self._glossary.extend(saved_glossary)
        full = doc.subtitles if cfg.full_context else None
        full_json: str | None = None
        if full is not None:
            context_chars = sum(len(s.text) for s in full)
            if context_chars > cfg.max_context_chars:
                raise RuntimeError(
                    f"Full transcript is {context_chars:,} chars, over the context budget "
                    f"MAX_CONTEXT_CHARS={cfg.max_context_chars:,}. Refusing to send. "
                    "Set FULL_CONTEXT=off to use the smaller previous-chunk context window, "
                    "or raise MAX_CONTEXT_CHARS deliberately."
                )
            ft = [{"id": s.id, "text": s.text.replace(chr(10), " ")} for s in full]
            full_json = json.dumps(ft, ensure_ascii=False)

        for idx, (chunk, context) in enumerate(zip(chunks, contexts, strict=False)):
            if idx in done_chunks:
                report.chunks_done += 1
                continue
            kinds = [classify(s.text) for s in chunk]
            payload: list[dict] = []
            ass_tags_info: dict[int, tuple[list[tuple[int, str]], int]] = {}
            for s, kind in zip(chunk, kinds, strict=False):
                if (
                    kind in (LineKind.SOUND_CUE, LineKind.STAGE_DIR) and not cfg.translate_cues
                ) or (kind == LineKind.LYRICS and not cfg.translate_lyrics):
                    continue
                clean_text = s.text
                if doc.format == "ass" and has_tags(s.text):
                    clean_text, tags = strip_tags(s.text)
                    ass_tags_info[s.id] = (tags, len(clean_text))

                item: dict = {"id": s.id, "text": clean_text.replace("\n", " ")}
                if s.speaker:
                    item["speaker"] = s.speaker
                if kind != LineKind.DIALOG:
                    item["kind"] = kind.value
                if cfg.enforce_cps:
                    item["max_chars"] = _char_budget(s, cfg.cps_target, cfg.cps_min_chars)
                payload.append(item)

            if payload:
                expect_ids = [p["id"] for p in payload]
                translated, new_terms = await self._chat_with_retries(
                    SYSTEM_PROMPT,
                    [{"role": "user", "content": self._build_user_message(payload, context, full_json)}],
                    expect_ids,
                )
                if cfg.rolling_glossary and new_terms:
                    added = self._glossary.extend(new_terms)
                    if added:
                        logger.info("Chunk %d: rolling glossary +%d terms", idx, added)

                # Flagged fallback: missing ids keep the original text and are
                # recorded as incomplete work.
                by_id = {s.id: s for s in chunk}
                for pid in expect_ids:
                    if pid not in translated:
                        report.dropped_ids.append(pid)
                        translated[pid] = by_id[pid].text

                if cfg.enforce_cps:
                    over = [
                        pid
                        for pid in expect_ids
                        if len(translated[pid])
                        > _char_budget(by_id[pid], cfg.cps_target, cfg.cps_min_chars)
                        * cfg.cps_tolerance
                    ]
                    if over:
                        translated = await self._compress_lines(by_id, translated, over)
                        still = [
                            pid
                            for pid in over
                            if len(translated[pid])
                            > _char_budget(by_id[pid], cfg.cps_target, cfg.cps_min_chars)
                            * cfg.cps_tolerance
                        ]
                        report.cps_violations.extend(still)

                # Restore ASS override tags if present
                if doc.format == "ass" and ass_tags_info:
                    for pid in expect_ids:
                        if pid in ass_tags_info and pid in translated:
                            tags, orig_len = ass_tags_info[pid]
                            translated[pid] = restore_tags(translated[pid], tags, orig_len)

                # Glossary compliance: one targeted note for the next chunk is
                # cheaper than a re-call; violations surface in the log.
                if cfg.enforce_glossary and not self._glossary.is_empty:
                    for pid in expect_ids:
                        violations = self._glossary.check_compliance(
                            by_id[pid].text, translated[pid]
                        )
                        if violations:
                            logger.info(
                                "Chunk %d id %d: glossary terms not applied: %s",
                                idx,
                                pid,
                                violations,
                            )

                results.update(translated)

            report.chunks_done += 1
            done_chunks.add(idx)
            if state_path:
                _save_state(state_path, fingerprint, done_chunks, results, self._glossary.entries)
            logger.info("Chunk %d/%d done", idx + 1, report.chunks_total)

        report.subtitles = [
            Subtitle(
                id=s.id,
                start=s.start,
                end=s.end,
                text=results.get(s.id, s.text),
                speaker=s.speaker,
                metadata=s.metadata,
            )
            for s in doc.subtitles
        ]
        report.glossary = dict(self._glossary.entries)
        if cfg.rolling_glossary and cfg.rolling_glossary_save:
            self._glossary.save(cfg.rolling_glossary_save)
        return report


def _fingerprint(engine: TranslationEngine, doc: SubtitleDocument) -> str:
    """Resume state is only valid for exactly this source text + config. Any
    change to the subtitles, languages, provider/model/endpoint, static
    glossary or system prompt invalidates it."""
    cfg = engine.cfg.translator
    p = engine.cfg.provider
    h = hashlib.sha256()
    h.update("\n".join(f"{s.id}:{s.text}" for s in doc.subtitles).encode("utf-8"))
    h.update(
        f"|{cfg.target_language}|{cfg.source_language}"
        f"|{p.kind}|{p.model}|{p.base_url}|{p.thinking_level}".encode()
    )
    h.update(
        f"|{json.dumps(engine._glossary.entries, sort_keys=True, ensure_ascii=False)}"
        f"|{SYSTEM_PROMPT}".encode()
    )
    return h.hexdigest()


def _status_path(cfg, resume_id: str | None) -> Path | None:
    if not resume_id:
        return None
    base = Path(cfg.status_dir) if cfg.status_dir else Path(resume_id).parent
    name = Path(resume_id).name + ".subkit-status.json"
    return base / name


def _load_state(path: Path, fingerprint: str) -> tuple[set[int], dict[int, str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return set(), {}
    if data.get("fingerprint") != fingerprint:
        if data:
            logger.warning(
                "Ignoring stale status file %s: source text or translation "
                "config changed since it was written",
                path,
            )
        return set(), {}
    return (
        set(data.get("done_chunks", [])),
        {int(k): str(v) for k, v in data.get("translations", {}).items()},
    )


def _load_state_glossary(path: Path, fingerprint: str) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    if data.get("fingerprint") != fingerprint:
        return {}
    return {str(k): str(v) for k, v in (data.get("glossary") or {}).items()}


def _save_state(
    path: Path,
    fingerprint: str,
    done_chunks: set[int],
    results: dict[int, str],
    glossary: dict[str, str],
) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "done_chunks": sorted(done_chunks),
                "translations": {str(k): v for k, v in results.items()},
                "glossary": glossary,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)


def translate_document_sync(
    doc: SubtitleDocument, cfg: AppConfig, provider: ChatProvider
) -> RunReport:
    return asyncio.run(TranslationEngine(cfg, provider).translate_document(doc))
