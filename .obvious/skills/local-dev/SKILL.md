---
name: local-dev
description: Durable record of LOCAL-DEV onboarding for wdm0006/categorical-encoding — environment setup, verification, and evidence (2026-09-05)
---

# Local dev onboarding — category_encoders

Recorded 2026-09-05 (UTC) by the onboarding worker on repo sandbox
`cmp_vOsrUUKm`. This is a pure Python library: no services to start, no ports,
no env vars, no secrets. "Local dev healthy" = deps installed, library
importable, test suite green, canonical lint clean.

## Environment

- Python 3.13.14 (pyproject requires `>=3.11`)
- Poetry 2.4.3 (installed via `pip3 install poetry` — not preinstalled on the box)
- Project installed as `category_encoders 2.10.0` via `poetry install`

## Steps to reproduce (verified)

1. `pip3 install poetry` — only needed if `poetry --version` fails
2. `poetry install` — installs runtime deps (numpy, pandas, scikit-learn,
   scipy, statsmodels, patsy) + dev deps (pytest, pytest-subtests, ruff,
   sphinx, numpydoc) from `poetry.lock`
3. `poetry run pytest tests` — full suite
4. `poetry run ruff check category_encoders` — canonical lint

## Evidence captured (2026-09-05)

- `poetry run pytest tests` → **225 passed, 2 skipped, 1086 subtests passed
  in 28.39 s** (exit 0)
- `poetry run ruff check category_encoders` → **All checks passed!** (exit 0)
- Primary user flow (README usage example), run via `poetry run python`:
  - `BinaryEncoder(cols=['gender', 'country']).fit(X).transform(X)` →
    numeric frame, shape (5, 5)
  - `TargetEncoder(cols=['gender', 'country']).fit(X_train, y_train)`
    `.transform(X_test)` → encoded test values
- `import category_encoders; category_encoders.__version__` → `2.10.0`

## Gotchas

- Encoders are sklearn **transformers**: use `transform()`, never `predict()`.
  Supervised encoders take `y` in `fit()`.
- `poetry run ruff check tests` reports 11 pre-existing findings (long lines /
  warnings in test files). The canonical lint command from CONTRIBUTING.md
  targets `category_encoders/` only, and CI has no lint step — do not "fix"
  these as part of unrelated work.
- Poetry prints a deprecation warning about the `[tool.poetry.dev-dependencies]`
  section (should become `[tool.poetry.group.dev.dependencies]`). Harmless.
- `tox.ini` at the root is a legacy matrix (py37–py310, old pandas/sklearn
  pins) that predates the current pyproject (`>=3.11`) — ignore it; CI and
  CONTRIBUTING use Poetry + pytest directly.

## Sandbox snapshot

- Snapshot captured 2026-09-05T21:15:58.044Z (after all verification above)
- Captured live session sandboxId: `ifssctpaln75z7zdwklui`
- New sandbox template: `2kpj8vxrybivw8lr1xq9:default`

Future workers booting from this snapshot skip straight to verification:
re-run `poetry run pytest tests` and `poetry run ruff check category_encoders`
to confirm the environment is intact.
