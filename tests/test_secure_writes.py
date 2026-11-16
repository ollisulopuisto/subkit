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


def test_atomic_write_preserves_existing_file_mode(tmp_path):
    """An existing destination keeps its own permissions after a rewrite."""
    import stat as _stat

    from subkit.fsutil import write_text_atomic

    path = tmp_path / "out.srt"
    path.write_text("old", encoding="utf-8")
    path.chmod(0o660)
    write_text_atomic(path, "new")
    assert _stat.S_IMODE(path.stat().st_mode) == 0o660
    assert path.read_text(encoding="utf-8") == "new"


def test_atomic_write_new_file_is_world_readable(tmp_path):
    """New outputs get 0644 so media servers (other local users) can read them."""
    import stat as _stat

    from subkit.fsutil import write_text_atomic

    path = tmp_path / "new.srt"
    write_text_atomic(path, "data")
    assert _stat.S_IMODE(path.stat().st_mode) == 0o644


def test_atomic_write_planted_symlink_gets_default_mode(tmp_path):
    """A replaced symlink must not leak the victim's mode onto the new file."""
    import stat as _stat

    from subkit.fsutil import write_text_atomic

    victim = tmp_path / "victim.txt"
    victim.write_text("USER DATA", encoding="utf-8")
    victim.chmod(0o600)
    out = tmp_path / "out.srt"
    out.symlink_to(victim)
    write_text_atomic(out, "new")
    assert not out.is_symlink()
    assert _stat.S_IMODE(out.stat().st_mode) == 0o644
    assert victim.read_text(encoding="utf-8") == "USER DATA"


def test_atomic_write_refuses_read_only_destination(tmp_path):
    """A read-only existing file must keep its protection (Path.write_text parity)."""
    import pytest

    from subkit.fsutil import write_text_atomic

    path = tmp_path / "out.srt"
    path.write_text("protected", encoding="utf-8")
    path.chmod(0o444)
    with pytest.raises(PermissionError):
        write_text_atomic(path, "new")
    assert path.read_text(encoding="utf-8") == "protected"
