"""Glossary system: a static user glossary plus a rolling decisions glossary.

The static glossary is user-authored (per-series term decisions) and always wins
on conflict. The rolling glossary is learned chunk-by-chunk during a run - the
model reports the term renderings it fixed, and later chunks reuse them. It can
be saved at the end of a run and reused across episodes of the same series.

Rolling-glossary carry pattern adapted from Subly (https://github.com/zakariaf/Subly),
reimplemented here; static-glossary mechanics derived from
ai-subtitle-translator (MIT, https://github.com/surbiks/ai-subtitle-translator).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class Glossary:
    """A term->translation mapping for prompt injection."""

    def __init__(self, entries: dict[str, str] | None = None) -> None:
        self._entries: dict[str, str] = dict(entries or {})

    @classmethod
    def from_file(cls, path: str | Path) -> Glossary:
        p = Path(path)
        if not p.exists():
            logger.warning("Glossary file not found: %s - using empty glossary", p)
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Glossary must be a JSON object, got {type(data).__name__}")
        logger.info("Loaded glossary with %d entries from %s", len(data), p)
        return cls(entries={str(k): str(v) for k, v in data.items()})

    @property
    def entries(self) -> dict[str, str]:
        return self._entries

    @property
    def is_empty(self) -> bool:
        return not self._entries

    def extend(self, other: dict[str, str], existing_wins: bool = True) -> int:
        """Merge another map; existing entries are preserved when existing_wins."""
        added = 0
        for term, translation in other.items():
            term, translation = term.strip(), (translation or "").strip()
            if not term or not translation:
                continue
            if existing_wins and term in self._entries:
                continue
            if self._entries.get(term) == translation:
                continue
            self._entries[term] = translation
            added += 1
        return added

    def check_compliance(self, source: str, translated: str) -> list[tuple[str, str]]:
        """(term, expected) pairs present in source but missing from translation."""
        violations: list[tuple[str, str]] = []
        for term, expected in self._entries.items():
            if term and expected and term in source and expected not in translated:
                violations.append((term, expected))
        return violations

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self._entries, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
