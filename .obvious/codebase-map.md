# Codebase Map — wdm0006/categorical-encoding

Folder-level overview, depth cap 2. `category_encoders` v2.10.0 — sklearn-style
categorical-encoding transformers. All encoder modules live at the top of
`category_encoders/`, one class each, all re-exported from
`category_encoders/__init__.py`.

| Path | Contents |
|---|---|
| `category_encoders/` | The library itself — one module per encoding technique |
| `category_encoders/__init__.py` | Public API: exports all encoder classes; `__version__` |
| `category_encoders/backward_difference.py` | `BackwardDifferenceEncoder` (unsupervised contrast) |
| `category_encoders/basen.py` | `BaseNEncoder` (generalizes OneHot/Binary) |
| `category_encoders/binary.py` | `BinaryEncoder` |
| `category_encoders/cat_boost.py` | `CatBoostEncoder` (supervised) |
| `category_encoders/count.py` | `CountEncoder` |
| `category_encoders/glmm.py` | `GLMMEncoder` (supervised, mixed models) |
| `category_encoders/gray.py` | `GrayEncoder` |
| `category_encoders/hashing.py` | `HashingEncoder` |
| `category_encoders/helmert.py` | `HelmertEncoder` (contrast) |
| `category_encoders/james_stein.py` | `JamesSteinEncoder` (supervised) |
| `category_encoders/leave_one_out.py` | `LeaveOneOutEncoder` (supervised) |
| `category_encoders/m_estimate.py` | `MEstimateEncoder` (supervised) |
| `category_encoders/one_hot.py` | `OneHotEncoder` |
| `category_encoders/ordinal.py` | `OrdinalEncoder` (base mapping used by many others) |
| `category_encoders/polynomial.py` | `PolynomialEncoder` (contrast) |
| `category_encoders/quantile_encoder.py` | `QuantileEncoder`, `SummaryEncoder` (supervised) |
| `category_encoders/rankhot.py` | `RankHotEncoder` |
| `category_encoders/sum_coding.py` | `SumEncoder` (contrast) |
| `category_encoders/target_encoder.py` | `TargetEncoder` (supervised) |
| `category_encoders/woe.py` | `WOEEncoder` (supervised, weight of evidence) |
| `category_encoders/base_contrast_encoder.py` | `BaseContrastEncoder` — shared base for all contrast-coding encoders |
| `category_encoders/utils.py` | Shared helpers (e.g. default `handle_unknown` behaviour) |
| `category_encoders/wrapper.py` | Wrapper enabling use inside sklearn `ColumnTransformer` |
| `category_encoders/datasets/` | Bundled dataset loader (`_base.py`, `get_dataset` helpers) |
| `category_encoders/datasets/data/` | CSV data files (`compass.csv`, `postcode_dataset_100.csv`) |
| `tests/` | pytest suite — `test_<encoder>.py` per encoder, plus `helpers.py`, `test_encoders.py`, `test_feature_names.py`, `test_utils.py`, `test_wrapper.py`, `test_helpers.py` |
| `examples/` | Runnable scripts (`encoding_examples.py`, `grid_search_example.py`, `column_transformer_example.py`) + `benchmarking*` dirs, `source_data/`, `img/` |
| `docs/` | Sphinx docs — `Makefile`, `requirements.txt`, `source/` with one `.rst` per encoder |
| `joss/` | JOSS paper sources (`paper.md`, `paper.bib`) |
| `.github/workflows/` | CI: `test-suite.yml` (pytest matrix 3.11–3.13), `docs.yml`, `test-docs-build.yml`, `pypi-publish.yml` |
| repo root | `pyproject.toml` + `poetry.lock` (Poetry, ruff config), `tox.ini` (legacy matrix), `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `CITATION.cff`, `LICENSE.md` |
