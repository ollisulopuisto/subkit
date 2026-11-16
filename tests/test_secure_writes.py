"""Output and resume-state writes must not follow symlinks planted at the target.

A media library can contain files this tool did not create (shared/NAS folders,
downloaded releases). Opening the destination path directly makes a planted
symlink redirect the write: the link target is truncated and overwritten with
generated subtitle text or resume JSON. Writes must create a fresh file next to
the destination and rename it over the path, so the link itself is replaced
instead of followed.
"""

import asyncio
import json
from pathlib import Path

from subkit.config import AppConfig
from subkit.engine import TranslationEngine
from subkit.glossary import Glossary
from subkit.merger import format_srt, write_subtitle_file
from subkit.parser import Subtitle, SubtitleDocument


class EchoProvider:
    name = "mock"

    async def chat(self, system, messages):
        return '{"segments": [{"id": 1, "text": "Hei."}]}'

    async def aclose(self):
        pass


def make_doc() -> SubtitleDocument:
    return SubtitleDocument(
        format="srt",
        subtitles=[Subtitle(id=1, start="00:00:01,000", end="00:00:02,000", text="Hello.")],
    )


def _planted_link(target: Path, victim: Path) -> None:
    target.symlink_to(victim)


def test_output_write_does_not_follow_symlink(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("USER DATA", encoding="utf-8")
    out = media / "movie.fi.srt"
    _planted_link(out, victim)

    doc = make_doc()
    write_subtitle_file(doc, doc.subtitles, out)

    assert victim.read_text(encoding="utf-8") == "USER DATA"
    assert not out.is_symlink()
    assert out.read_text(encoding="utf-8") == format_srt(doc.subtitles)
    assert list(media.glob(".subkit-*")) == []


def test_resume_state_write_does_not_follow_symlink(tmp_path):
    media = tmp_path / "media"
    media.mkdir()
    video = media / "movie.mkv"
    video.write_text("", encoding="utf-8")
    victim = tmp_path / "victim.txt"
    victim.write_text("USER DATA", encoding="utf-8")
    _planted_link(media / "movie.mkv.subkit-status.tmp", victim)

    asyncio.run(
        TranslationEngine(AppConfig(), EchoProvider()).translate_document(
            make_doc(), resume_id=str(video)
        )
    )

    assert victim.read_text(encoding="utf-8") == "USER DATA"
    status = media / "movie.mkv.subkit-status.json"
    assert json.loads(status.read_text(encoding="utf-8"))["version"] == 2
    assert list(media.glob(".subkit-*")) == []


def test_glossary_save_does_not_follow_symlink(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_text("USER DATA", encoding="utf-8")
    path = tmp_path / "glossary.json"
    _planted_link(path, victim)

    Glossary({"Shire": "Kontu"}).save(path)

    assert victim.read_text(encoding="utf-8") == "USER DATA"
    assert not path.is_symlink()
    assert json.loads(path.read_text(encoding="utf-8")) == {"Shire": "Kontu"}
    assert list(tmp_path.glob(".subkit-*")) == []
