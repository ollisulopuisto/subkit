"""Filesystem helpers shared by subkit's writers."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile


def write_text_atomic(path: str | Path, content: str, encoding: str = "utf-8") -> None:
    """Write `content` to `path` via a fresh temp file + os.replace.

    Security (CWE-59): opening the destination directly follows a symlink
    planted there and truncates/overwrites whatever it points at. Building the
    file under a new name and renaming it over the path replaces the link
    itself. The rename also makes the write atomic, so a crash never leaves a
    half-written subtitle or resume-state file.

    The rename swaps the destination inode, so the new file must not silently
    change what the old one was: an existing destination keeps its permission
    bits and group (a group-writable file stays writable by that group, a
    read-only file keeps its mode), and a destination the caller could not
    have written directly (e.g. mode 0444) raises PermissionError, matching
    Path.write_text. A planted symlink is replaced outright and gets the
    default 0644, never the link target's mode.
    """
    target = Path(path)
    st = None
    try:
        if not target.is_symlink():
            st = target.stat()
    except OSError:
        st = None
    if st is not None and not os.access(target, os.W_OK):
        # Path.write_text refused this write; keep refusing it.
        raise PermissionError(f"destination is not writable: {target}")
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".subkit-", suffix=".tmp")
    try:
        if st is not None:
            os.chmod(fd, stat.S_IMODE(st.st_mode))
            try:
                # fchown on the fd, not the path: in a shared directory the
                # temp name could be swapped for a symlink between mkstemp and
                # chown, and path-based chown would follow it to a victim file.
                os.fchown(fd, -1, st.st_gid)
            except OSError:
                pass  # not a member of the destination's group
        else:
            # New file: 0644, the mode Path.write_text produced under a
            # typical umask, so media servers (other local users) can read it.
            os.chmod(fd, 0o644)
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            fh.write(content)
        os.replace(tmp, target)
    finally:
        Path(tmp).unlink(missing_ok=True)
