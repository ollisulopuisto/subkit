"""Write translated subtitles as SRT or ASS.

SRT/ASS merge and formatting derived from ai-subtitle-translator (MIT,
https://github.com/surbiks/ai-subtitle-translator).
"""

from __future__ import annotations

from pathlib import Path

from subkit.parser import Subtitle, SubtitleDocument


def format_srt(subtitles: list[Subtitle]) -> str:
    if not subtitles:
        return ""
    blocks = [f"{s.id}\n{s.start} --> {s.end}\n{s.text.replace(chr(10), ' ')}" for s in subtitles]
    return "\n\n".join(blocks) + "\n"


def format_ass(document: SubtitleDocument, subtitles: list[Subtitle]) -> str:
    if document.ass_lines is None:
        raise ValueError("ASS document is missing original file lines")
    lines = list(document.ass_lines)
    for subtitle in subtitles:
        metadata = subtitle.metadata or {}
        line_index = metadata.get("ass_line_index")
        prefix = metadata.get("ass_prefix")
        newline_token = metadata.get("ass_newline_token", "\\N")
        if not isinstance(line_index, int) or not isinstance(prefix, str):
            raise TypeError("ASS subtitle is missing reconstruction metadata")
        lines[line_index] = prefix + subtitle.text.replace("\n", newline_token)
    return "\n".join(lines) + "\n"


def write_subtitle_file(
    document: SubtitleDocument, subtitles: list[Subtitle], path: str | Path
) -> None:
    content = format_ass(document, subtitles) if document.format == "ass" else format_srt(subtitles)
    Path(path).write_text(content, encoding="utf-8")
