"""Filesystem helpers shared by subkit's writers."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


def write_text_atomic(path: str | Path, content: str, encoding: str = "utf-8") -> None:
    """Write `content` to `path` via a fresh temp file + os.replace.

    Security (CWE-59): opening the destination directly follows a symlink
    planted there and truncates/overwrites whatever it points at. Building the
    file under a new name and renaming it over the path replaces the link
    itself. The rename also makes the write atomic, so a crash never leaves a
    half-written subtitle or resume-state file.
    """
    target = Path(path)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".subkit-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(content)
        os.replace(tmp, target)
    finally:
        Path(tmp).unlink(missing_ok=True)
