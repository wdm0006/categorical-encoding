# wdm0006/categorical-encoding — Agent Guide

`category_encoders` (v2.10.0) — a scikit-learn-contrib library of sklearn-style
transformers for encoding categorical variables into numeric features, using
unsupervised and supervised techniques.

Pure Python library. No web app, no servers, no external services (no
Postgres/Redis), no listening ports, no required env vars, no secrets.

## Stack

| Layer | Tool | Verified in sandbox (2026-09-05) |
|---|---|---|
| Language | Python `>=3.11` (pyproject.toml) | 3.13.14 |
| Package manager | Poetry (poetry.lock) | 2.4.3 |
| Runtime deps | numpy, pandas, scikit-learn, scipy, statsmodels, patsy | 2.3.4 / 2.3.3 / 1.7.2 / 1.16.3 / 0.14.5 / 1.0.2 |
| Tests | pytest + pytest-subtests | 8.4.2 / 0.15.0 |
| Lint | ruff (rules in `pyproject.toml` `[tool.ruff]`) | 0.14.3 |
| Docs | Sphinx + sphinx-rtd-theme (`docs/`) | 8.2.3 / 3.0.2 |

## Commands

Run from the repo root. Dependencies are already installed in the sandbox
snapshot (`poetry install` was run once at onboarding).

```shell
poetry install                                  # install runtime + dev deps (fresh env)
poetry run pytest tests                         # full test suite (~30 s; CI parity)
poetry run pytest tests/test_binary.py -q       # one encoder's tests
poetry run ruff check category_encoders         # lint — canonical command from CONTRIBUTING.md
poetry run python -c "import category_encoders; print(category_encoders.__version__)"
```

Notes:

- CI (`.github/workflows/test-suite.yml`) runs `poetry install` then
  `poetry run pytest tests` on Python 3.11 / 3.12 / 3.13. No lint step in CI.
- `poetry run ruff check tests` currently reports 11 pre-existing findings
  (long lines / warnings). Not a gate — the canonical lint target is
  `category_encoders/` only.
- Poetry emits a deprecation warning: `poetry.dev-dependencies` section should
  migrate to `poetry.group.dev.dependencies`. Harmless today.

## Codebase map

See `codebase-map.md`.

## Local Verification Summary

Verified 2026-09-05 in the repo sandbox (Poetry 2.4.3, Python 3.13.14):

- `poetry install` — OK; project installed as `category_encoders 2.10.0`.
- `poetry run pytest tests` — **225 passed, 2 skipped, 1086 subtests passed in 28.39 s**.
- `poetry run ruff check category_encoders` — **All checks passed!**
- Primary user flow (library end-to-end, from README usage): built a categorical
  DataFrame; `BinaryEncoder(cols=['gender', 'country']).fit(X).transform(X)`
  produced a (5, 5) numeric frame; `TargetEncoder(cols=['gender', 'country'])
  .fit(X_train, y_train).transform(X_test)` produced encoded test values.
  Ran via `poetry run python` — both flows succeeded.

No dev server, no ports, no env vars, no credentials required to develop.

## Snapshot

- Sandbox computer: `cmp_vOsrUUKm` (repo sandbox for wdm0006/categorical-encoding)
- Snapshot captured: **2026-09-05T21:15:58.044Z** (ISO-8601, UTC)
- Captured live session sandboxId: `ifssctpaln75z7zdwklui`
- New sandbox template: `2kpj8vxrybivw8lr1xq9:default`
- State baked in: Poetry 2.4.3, all runtime + dev deps installed, project
  importable, test suite green, canonical lint clean.

## Conventions

- ruff: single quotes, line length 100, target py311, numpy docstring
  convention, isort ordering (`pyproject.toml`).
- Every encoder implements the sklearn transformer API — `fit` / `transform`
  (there is **no `predict`**; supervised encoders take `y` in `fit`).
- Tests mirror the library: one `tests/test_<encoder>.py` per encoder module.
- Support numpy arrays and pandas DataFrames as inputs (project guideline).
