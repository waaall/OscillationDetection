# Repository Guidelines

## Project Structure & Module Organization
- Core modules: `src/core/fft_analyzer.py` (single-window FFT band analyzer) and `src/core/detection_pipeline.py` (input normalization, sampling-rate checks, windowing, status mapping). Helpers currently live in `src/core/SignalGenerator.py`.
- Offline layer: `src/offline/csv_replay.py` handles JSON-driven CSV replay and result export; `src/offline/result_visualizer.py` renders summary/spectrum plots from replay output; default offline config lives in `src/offline/csv_replay.default.json`.
- API layer: `src/api/__init__.py` is a placeholder package for the future FastAPI protocol layer; keep API-specific schemas/settings there instead of pushing them back into `core/` or `offline/`.
- Workflows: `tests/test_detection_pipeline.py` covers the core detection flow, `tests/test_csv_replay.py` covers offline CSV replay, and `tests/test_oscillation_detection.py` covers offline visualization.
- Data/logs: sample CSV fixtures live under `csv-data/`; plots and logs land in `plots/` and `log/` (created on demand). Keep `src/offline/csv_replay.default.json` aligned with any replay-config schema changes.
- Modbus prototype: `src/com/modbus-dcs.py` is a standalone RTU/DCS utility; keep changes isolated from FFT code paths.

## Build, Test, and Development Commands
- Environment + deps: `python -m venv .venv && source .venv/bin/activate && pip install numpy pandas matplotlib pymodbus pymysql`.
- Offline replay pipeline: `python -m src.offline.csv_replay` loads `src/offline/csv_replay.default.json`, runs the replay pipeline, writes result CSV, and optionally emits plots.
- Targeted tests: `python -m pytest tests/test_detection_pipeline.py tests/test_csv_replay.py tests/test_oscillation_detection.py`.
- Full suite: `python -m pytest tests`.

## Coding Style & Naming Conventions
- Python 3, PEP8-ish: 4-space indents, snake_case for functions/vars, CapWords for classes.
- Prefer type hints and short docstrings as used in analyzers; keep logs concise (current mix of zh/EN is acceptable).
- Module/file names should stay lower_snake and reflect layer ownership (`core/`, `offline/`, `api/`); avoid reintroducing old mixed-case filenames.
- Config keys stay lower_snake; preserve CSV headers; avoid breaking backward compatibility of `SignalGenerator` timestamp handling.

## Testing Guidelines
- Run `python -m pytest tests` after changes to the detection pipeline, offline replay, or output schema.
- For focused validation, use `tests/test_detection_pipeline.py` for status logic and resampling behavior, `tests/test_csv_replay.py` for replay/output coverage, and `tests/test_oscillation_detection.py` for plot generation.
- Add pytest-style assertions if expanding coverage; place new tests at repo root with `test_*.py` naming.
- Use small CSV fixtures under `csv-data/`; avoid committing large binaries; seed randomness for reproducibility if added.

## Commit & Pull Request Guidelines
- Match existing history: short, imperative, lowercase subjects (e.g., `add csv replay`, `fix sampling check`).
- PRs: describe scope, configs touched, sample commands to verify, and before/after behavior; include screenshots/plots if offline visualization changes.
- Link issues when available; call out backward-compatibility or sampling-rate impacts; mention new config fields, output-schema changes, or layer-boundary refactors.
