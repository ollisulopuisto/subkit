"""Persistent per-series glossary library.

One JSON file per series (default directory: $SERIES_DIR or
~/.local/share/subkit/series). Every term carries provenance:

- origin "user": set or confirmed by the user. User terms always win and are
  never modified by automatic extraction.
- origin "auto": extracted by the model during a translation run. These are
  CANDIDATES, not canonical decisions - they are never promoted to user terms
  silently and never overwrite a user term. `subkit glossary confirm` is the
  only promotion path.

Terms can be inspected, corrected, confirmed and deleted from the CLI:
  subkit glossary show <series>
  subkit glossary set <series> <source> <target>
  subkit glossary confirm <series> [source ...]
  subkit glossary delete <series> <source>
  subkit glossary list

Series are fully separate files: no terms leak between series.
"""

from __future__ import annotations

from datetime import date
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def default_series_dir() -> Path:
    env = os.getenv("SERIES_DIR")
    if env:
        return Path(env)
    return Path.home() / ".local" / "share" / "subkit" / "series"


def series_path(name: str, directory: str | Path | None = None) -> Path:
    if not name or not _NAME_RE.match(name):
        raise ValueError(
            f"Invalid series name {name!r}: use letters, digits, '.', '_' or '-' "
            "(no spaces or path separators)"
        )
    base = Path(directory) if directory else default_series_dir()
    return base / f"{name}.json"


def list_series(directory: str | Path | None = None) -> list[str]:
    base = Path(directory) if directory else default_series_dir()
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.json"))


class SeriesGlossary:
    """Term map with provenance. terms: src -> {target, origin, count, updated}."""

    def __init__(self, name: str, terms: dict | None = None) -> None:
        self.name = name
        self.terms: dict[str, dict] = terms or {}

    @classmethod
    def load(cls, path: str | Path) -> SeriesGlossary:
        p = Path(path)
        if not p.exists():
            return cls(name=p.stem)
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError(f"Series glossary must be a JSON object: {p}")
        if "terms" in data:  # current format
            terms = {}
            for src, rec in (data.get("terms") or {}).items():
                terms[str(src)] = {
                    "target": str(rec.get("target", "")),
                    "origin": rec.get("origin")
                    if rec.get("origin") in ("user", "auto")
                    else "auto",
                    "count": int(rec.get("count", 1)),
                    "updated": str(rec.get("updated", "")),
                }
            return cls(name=str(data.get("series") or p.stem), terms=terms)
        # legacy flat {source: target} files are USER-authored decisions.
        return cls(
            name=p.stem,
            terms={
                str(k): {"target": str(v), "origin": "user", "count": 1, "updated": ""}
                for k, v in data.items()
            },
        )

    def entries(self) -> dict[str, str]:
        """All terms as source -> target, for prompt injection."""
        return {src: rec["target"] for src, rec in self.terms.items() if rec["target"]}

    def user_entries(self) -> dict[str, str]:
        """User-set or user-confirmed terms - canonical decisions."""
        return {
            src: rec["target"]
            for src, rec in self.terms.items()
            if rec["target"] and rec["origin"] == "user"
        }

    def auto_entries(self) -> dict[str, str]:
        """Auto-extracted candidates - suggestions, never canonical."""
        return {
            src: rec["target"]
            for src, rec in self.terms.items()
            if rec["target"] and rec["origin"] == "auto"
        }

    def revision(self) -> str:
        """Content hash - any term edit changes it (used to invalidate resume state)."""
        blob = json.dumps(self.terms, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def merge_auto(self, terms: dict[str, str]) -> int:
        """Fold model-extracted terms in as candidates. Returns new-term count.
        Existing user terms are never touched; existing auto terms with the same
        rendering just bump their count; a CONFLICTING auto rendering is kept
        (first candidate wins) and logged for user review."""
        added = 0
        for src, tgt in terms.items():
            src, tgt = src.strip(), (tgt or "").strip()
            if not src or not tgt:
                continue
            rec = self.terms.get(src)
            if rec is None:
                self.terms[src] = {
                    "target": tgt,
                    "origin": "auto",
                    "count": 1,
                    "updated": date.today().isoformat(),
                }
                added += 1
            elif rec["origin"] == "user":
                continue  # user decision stands
            elif rec["target"] == tgt:
                rec["count"] = int(rec.get("count", 1)) + 1
                rec["updated"] = date.today().isoformat()
            else:
                logger.info(
                    "Series %s: model proposed %r -> %r but candidate %r exists; keeping "
                    "candidate (review with 'subkit glossary show %s')",
                    self.name,
                    src,
                    tgt,
                    rec["target"],
                    self.name,
                )
        return added

    def set_user(self, source: str, target: str) -> None:
        self.terms[source] = {
            "target": target,
            "origin": "user",
            "count": 1,
            "updated": date.today().isoformat(),
        }

    def confirm(self, source: str) -> bool:
        rec = self.terms.get(source)
        if rec is None:
            return False
        rec["origin"] = "user"
        rec["updated"] = date.today().isoformat()
        return True

    def delete(self, source: str) -> bool:
        return self.terms.pop(source, None) is not None

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": FORMAT_VERSION,
            "series": self.name,
            "terms": self.terms,
        }
        fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=".subkit-series-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
