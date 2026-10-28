"""Jellyfin-aware library handling.

Categorization (per video, per target language):
- skip: the target language is already covered by a sidecar (Jellyfin
  external-subtitle naming) or an embedded subtitle stream. Under --force an
  item is still skipped when a USER-OWNED target subtitle exists - force only
  regenerates subtitles this tool previously wrote (the `.generated` marker).
- translate: a source-language sidecar or embedded stream exists to translate
  from (e.g. English subs -> Finnish subs). Existing English subtitles never
  cause a skip; they are the translation source.
- transcribe: no usable source subtitles; speech-to-text is needed.

Naming: <VideoName>.<lang>[.generated].srt next to the video, per
https://jellyfin.org/docs/general/server/media/external-files
The "generated" title token shows up in Jellyfin's subtitle picker and marks
machine-made subs. Writes are atomic (temp file + rename) and never overwrite
a file this tool did not create.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mkv", ".mp4", ".m4v", ".avi", ".mov", ".ts", ".webm", ".mpg", ".mpeg"}
SUBTITLE_EXTENSIONS = {".srt", ".ass", ".ssa", ".vtt", ".sub"}

# ISO 639-1 -> common 639-2 codes used in file names and streams.
LANG_ALIASES: dict[str, set[str]] = {
    "fi": {"fi", "fin", "finnish"},
    "en": {"en", "eng", "english"},
    "sv": {"sv", "swe", "swedish"},
    "de": {"de", "ger", "deu", "german"},
    "fr": {"fr", "fre", "fra", "french"},
}


def lang_codes(lang: str) -> set[str]:
    return LANG_ALIASES.get(lang.lower(), {lang.lower()})


def sidecar_path(video: Path, lang: str, generated_marker: bool = True) -> Path:
    suffix = f".{lang}.generated.srt" if generated_marker else f".{lang}.srt"
    return video.with_suffix(suffix)


def _sidecar_lang_tokens(video: Path, sub: Path) -> list[str]:
    """Tokens after the video base name: 'Movie.fi.generated.srt' -> ['fi','generated']."""
    tail = sub.name[len(video.stem) :]  # '.fi.generated.srt'
    return [
        t.lower() for t in tail.split(".") if t and t.lower() not in SUBTITLE_EXTENSIONS and t != ""
    ]


def existing_sidecars(video: Path) -> list[Path]:
    base = video.stem
    if not video.parent.exists():
        return []
    return [
        p
        for p in video.parent.iterdir()
        if p.is_file() and p.suffix.lower() in SUBTITLE_EXTENSIONS and p.name.startswith(base + ".")
    ]


def sidecar_covers_lang(video: Path, lang: str) -> Path | None:
    """First sidecar carrying the language code in Jellyfin flag position."""
    codes = lang_codes(lang)
    for path in existing_sidecars(video):
        if any(tok in codes for tok in _sidecar_lang_tokens(video, path)):
            return path
    return None


def embedded_sub_langs(video: Path) -> set[str]:
    """Languages of embedded subtitle streams, via ffprobe."""
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "s",
                "-show_entries",
                "stream_tags=language",
                "-of",
                "json",
                str(video),
            ],
            capture_output=True,
            timeout=60,
        )
        if proc.returncode != 0:
            logger.warning("ffprobe failed on %s", video)
            return set()
        data = json.loads(proc.stdout or b"{}")
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        logger.warning("ffprobe unusable on %s: %s", video, exc)
        return set()
    langs = set()
    for stream in data.get("streams", []):
        tag = ((stream.get("tags") or {}).get("language") or "").lower()
        if tag and tag != "und":
            langs.add(tag)
    return langs


def categorize_item(
    video: Path, target_lang: str, source_lang: str = "en", force: bool = False
) -> str:
    """-> "skip" | "translate" | "transcribe"."""
    target_codes = lang_codes(target_lang)
    gen = sidecar_path(video, target_lang, generated_marker=True)
    covering = sidecar_covers_lang(video, target_lang)
    embedded = embedded_sub_langs(video)

    if force:
        # Force only redoes items whose target sub is OUR generated file (or
        # absent). A user-owned target subtitle is never regenerated over.
        if (covering is not None and covering != gen) or (target_codes & embedded):
            return "skip"
    else:
        if covering is not None or (target_codes & embedded):
            return "skip"

    if sidecar_covers_lang(video, source_lang) or (lang_codes(source_lang) & embedded):
        return "translate"
    return "transcribe"


def item_needs_work(video: Path, lang: str, force: bool = False) -> bool:
    return categorize_item(video, lang, force=force) != "skip"


@dataclass
class LibraryPlan:
    translate: list[Path] = field(default_factory=list)  # source subs exist
    transcribe: list[Path] = field(default_factory=list)  # need ASR
    skipped: list[Path] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{len(self.translate)} to translate, {len(self.transcribe)} to transcribe, "
            f"{len(self.skipped)} already covered"
        )


def plan_library(
    root: Path, target_lang: str, source_lang: str = "en", force: bool = False
) -> LibraryPlan:
    plan = LibraryPlan()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        action = categorize_item(path, target_lang, source_lang, force)
        if action == "skip":
            plan.skipped.append(path)
        else:
            getattr(plan, action).append(path)
    return plan


def scan_library(root: Path, lang: str, force: bool = False) -> tuple[list[Path], list[Path]]:
    plan = plan_library(root, lang, force=force)
    return plan.translate + plan.transcribe, plan.skipped


def extract_embedded_srt(video: Path, lang: str) -> Path | None:
    """Extract the first embedded subtitle stream tagged with lang to a temp SRT."""
    for code in sorted(lang_codes(lang)):
        fd, tmp = tempfile.mkstemp(prefix=".subkit-src-", suffix=".srt")
        os.close(fd)
        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(video),
                "-map",
                f"0:s:m:language:{code}",
                tmp,
            ],
            capture_output=True,
            timeout=300,
        )
        if proc.returncode == 0 and Path(tmp).stat().st_size > 0:
            return Path(tmp)
        Path(tmp).unlink(missing_ok=True)
    logger.warning("no extractable embedded %s subtitle in %s", lang, video)
    return None


def write_sidecar_atomic(
    video: Path, lang: str, content: str, generated_marker: bool = True, force: bool = False
) -> Path:
    """Write the sidecar atomically. An existing target is only replaced when
    force=True AND the target is one of our `.generated` files - user-owned
    subtitles are never overwritten."""
    target = sidecar_path(video, lang, generated_marker)
    if target.exists() and not (force and generated_marker):
        raise FileExistsError(f"Refusing to overwrite existing subtitle: {target}")
    fd, tmp = tempfile.mkstemp(dir=video.parent, prefix=".subkit-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return target


def refresh_item(api_url: str, api_key: str, item_id: str, timeout: float = 30.0) -> bool:
    """Ask Jellyfin to refresh one item's metadata so a new sidecar appears.
    item_id comes from the Jellyfin API; this helper does not map paths to ids."""
    import httpx

    try:
        resp = httpx.post(
            f"{api_url.rstrip('/')}/Items/{item_id}/Refresh",
            headers={"X-Emby-Token": api_key},
            params={
                "metadataRefreshMode": "Default",
                "imageRefreshMode": "None",
                "replaceAllMetadata": "false",
                "replaceAllImages": "false",
            },
            timeout=timeout,
        )
        return resp.status_code in (200, 204)
    except Exception as exc:
        logger.warning("Jellyfin refresh failed for item %s: %s", item_id, exc)
        return False
