"""Filesystem helpers shared by subkit's writers."""

from __future__ import annotations

import os
import stat
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
    # mkstemp forces mode 0600; a renamed file must end up with the
    # permissions the output is expected to have. An existing destination
    # keeps its own mode (e.g. 0664 group-shared, or 0444 read-only, which
    # also preserves the old Path.write_text failure mode of leaving a
    # protected file protected). A planted symlink is replaced, so its target
    # is irrelevant - new files get 0644, the same mode Path.write_text
    # produced under a typical umask.
    mode = None
    try:
        if not target.is_symlink():
            mode = stat.S_IMODE(target.stat().st_mode)
    except OSError:
        mode = None
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".subkit-", suffix=".tmp")
    try:
        os.chmod(fd, mode if mode is not None else 0o644)
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(content)
        os.replace(tmp, target)
    finally:
        Path(tmp).unlink(missing_ok=True)
