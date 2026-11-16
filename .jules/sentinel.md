## 2026-11-16 - Subtitle/state writes followed symlinks in media directories

**Vulnerability:** The subtitle-output, resume-state and glossary writers opened the destination path
directly. A symlink planted in a media directory redirected the write, so the link target was
truncated and replaced with generated subtitle text or resume JSON.

**Learning:** The codebase already treats media directories as untrusted input (parsing, categorization,
sidecar scanning), but only `jellyfin.write_sidecar_atomic` and `series.SeriesGlossary.save` used the
temp-file + rename pattern. The safe pattern existed but was not shared, so newer writers silently
reintroduced direct writes.

**Prevention:** All text writers now go through `subkit.fsutil.write_text_atomic` (mkstemp in the
destination directory, then `os.replace`), which replaces a planted link instead of following it and
keeps writes atomic. New writers must use that helper instead of `Path.write_text`.
