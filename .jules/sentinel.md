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

## 2026-11-16 - Provider error bodies echo credentials into logs
**Vulnerability:** Provider HTTP error responses are untrusted text passed verbatim into `ProviderHTTPError`, which surfaces in CLI stderr and retry logs. OpenAI's 401 echoes the API key back ("Incorrect API key provided: sk-..."), so a mistyped key leaks the live credential into the user's terminal and any log tailing it.
**Learning:** Error bodies from remote providers must be treated as untrusted and redacted at the raise site, not assumed safe because they look like machine JSON. This affects both `GeminiProvider` and `OpenAIProvider` and any future provider added to `subkit/providers.py`.
**Prevention:** `ProviderHTTPError` now accepts `secrets` and redacts credential-shaped runs (`sk-`, `AIza`, `gsk_`, `Bearer`, `x-goog-api-key`, `X-Emby-Token`) via `_redact_credentials` before the message is built. Every provider raise site passes its own key as a secret. Covered by `tests/test_provider_errors.py` with full-key, truncated-key, and Bearer-prefix cases; verified end-to-end against a local stub returning OpenAI's real 401 body over both `translate` and `library` paths.
