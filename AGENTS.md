# Repository Guidelines

## Project Structure & Module Organization
- Core analyzers: `src/core/FFT_analyzer.py` (single-window FFT), `src/Freq_dynamic_analyzer.py` (sliding-window FFT/ZC pipeline with optional refinement), `src/OscillationDetection.py` (visual + threshold detection); helpers live in `src/core/SignalGenerator.py`, `src/core/Zero_Cross_Freq.py`, `src/core/FrequencyRefinement.py`.
- Workflows: `tests/test_oscillation_detection.py` orchestrates the dev flow with CSV + config; `tests/test_dynamic_fft.py` scripts four demo scenarios for dynamic FFT; `tests/test_frequency_refinement.py` holds the unit tests.
- Config/data/logs: JSON configs live at `src/config_fft_dynamic.json` and `src/oscillate_dev_settings.json`; data in `csv-data/`; outputs land in `plots/` and `log/` (created on demand). Keep `config_fft_dynamic.json` aligned with any schema changes.
- Modbus prototype: `src/com/modbus-dcs.py` is a standalone RTU/DCS utility; keep changes isolated from FFT code paths.

## Build, Test, and Development Commands
- Environment + deps: `python -m venv .venv && source .venv/bin/activate && pip install numpy pandas matplotlib pymodbus pymysql`.
- Quick FFT sanity: `python -m src.core.FFT_analyzer` runs the built-in single-window FFT demo.
- Dynamic FFT pipeline (uses `src/config_fft_dynamic.json`): `python -m src.Freq_dynamic_analyzer` or call `FreqDynamicAnalyzer(...).run_pipeline()`.
- Oscillation dev flow: `python tests/test_oscillation_detection.py --create-config --config src/oscillate_dev_settings.json`; run `--mode animation` (default) or `--mode static` with the same config flag.
- Demo runs: `python tests/test_dynamic_fft.py` executes the scripted examples end-to-end; `python -m pytest tests/test_frequency_refinement.py` covers the refinement unit tests.

## Coding Style & Naming Conventions
- Python 3, PEP8-ish: 4-space indents, snake_case for functions/vars, CapWords for classes.
- Prefer type hints and short docstrings as used in analyzers; keep logs concise (current mix of zh/EN is acceptable).
- Config keys stay lower_snake; preserve CSV headers; avoid breaking backward compatibility of `SignalGenerator` timestamp handling.

## Testing Guidelines
- Current scripts double as smoke tests; run `python tests/test_dynamic_fft.py` and `python tests/test_oscillation_detection.py --mode static` after changes to analyzers/detectors.
- Add pytest-style assertions if expanding coverage; place new tests at repo root with `test_*.py` naming.
- Use small CSV fixtures under `csv-data/`; avoid committing large binaries; seed randomness for reproducibility if added.

## Commit & Pull Request Guidelines
- Match existing history: short, imperative, lowercase subjects (e.g., `add Freq_dynamic_analyzer`, `fix overlap calc`).
- PRs: describe scope, configs touched, sample commands to verify, and before/after behavior; include screenshots/plots if visuals change.
- Link issues when available; call out backward-compatibility or sampling-rate impacts; mention new config fields or data schema changes.
