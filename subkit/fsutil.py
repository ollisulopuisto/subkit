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
        # mkstemp forces mode 0600; the renamed file would be unreadable to
        # other users on a shared media library (e.g. a Jellyfin service
        # account). Give the output the same 0644 a normal create gets under a
        # typical umask instead of keeping the restrictive temp-file mode.
        os.chmod(fd, 0o644)
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(content)
        os.replace(tmp, target)
    finally:
        Path(tmp).unlink(missing_ok=True)
