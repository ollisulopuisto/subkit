# Contributing to subkit

## Test-driven development (red-green)

subkit follows red-green TDD for all code changes:

1. **RED** — Write a failing test that captures the desired behavior. Run it and confirm
   it fails for the right reason (missing feature, not a typo or import error).
2. **GREEN** — Write the minimal code to make the test pass. No gold-plating.
3. **REFACTOR** — Clean up with the tests still green. Run lint + tests again.

Never commit or push work whose tests are not green.

## Commands

```bash
# Install dev dependencies (pytest, ruff)
uv sync --extra dev

# Run the test suite
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_core.py

# Lint (fix auto-fixable issues)
ruff check --fix subkit/ tests/

# Check formatting
ruff format --check subkit/ tests/

# Apply formatting
ruff format subkit/ tests/

# Full CI-equivalent check
ruff check subkit/ tests/ && ruff format --check subkit/ tests/ && uv run pytest tests/
```

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and PR to `main`:

1. `ruff check` — lint
2. `ruff format --check` — formatting
3. `pytest` — tests

All three must pass for the build to go green.

## Writing tests

Tests live in `tests/` and use plain `pytest` with a mock provider — no API keys, no
model downloads, no network. See `tests/test_core.py` for the mock provider pattern.

## Audit follow-through, 2026-10-29

Mock-provider resume regression: a dropped cue was correctly INCOMPLETE on the
first run, then reported complete on a zero-call resume because status saved
only text, not warning IDs. Version2 status preserves dropped IDs and CPS
warnings across resume. Legacy matching state without report evidence refuses before new provider
requests and explains keeping existing output or deliberately setting
RESUME_MODE=fresh to translate again. No live API calls were made to validate this change.

Baseline53tests, RED on missing resumed warning, final56tests pass. Ruff lint
and all22files format-check pass. Real provider/ASR/MLX/container acceptance
remains unverified. Human merge pending. This does not make incomplete chunks
automatically retry: it preserves their honest warning on zero-call resume.
