# Repository Guidelines

## Project Structure & Module Organization
- Core analyzers: `FFT_analyzer.py` (single-window FFT), `FFT_dynamic_analyzer.py` (sliding-window pipeline), `OscillationDetection.py` (visual + threshold detection), helper `SignalGenerator.py`.
- Workflows: `test_oscillation_detection.py` orchestrates the dev flow with CSV + config; `test_dynamic_fft.py` scripts four demo scenarios for dynamic FFT.
- Config/data/logs: JSON configs (`config_fft_dynamic.json`, `oscillate_dev_settings.json`) live in repo root; data in `csv-data/`; outputs land in `plots/` and `log/` (created on demand). Keep `config_fft_dynamic.json` aligned with any schema changes.
- Modbus prototype: `modbus-dcs.py` is a standalone RTU/DCS utility; keep changes isolated from FFT code paths.

## Build, Test, and Development Commands
- Environment + deps: `python -m venv .venv && source .venv/bin/activate && pip install numpy pandas matplotlib pymodbus pymysql`.
- Quick FFT sanity: `python FFT_analyzer.py` plots a single-window spectrum.
- Dynamic FFT pipeline (uses `config_fft_dynamic.json`): `python FFT_dynamic_analyzer.py` or call `FFTDynamicAnalyzer(...).run_pipeline()`.
- Oscillation dev flow: `python test_oscillation_detection.py --create-config` to scaffold settings; `python test_oscillation_detection.py --mode animation` (default) or `--mode static` to process CSVs.
- Demo runs: `python test_dynamic_fft.py` executes the scripted examples end-to-end.

## Coding Style & Naming Conventions
- Python 3, PEP8-ish: 4-space indents, snake_case for functions/vars, CapWords for classes.
- Prefer type hints and short docstrings as used in analyzers; keep logs concise (current mix of zh/EN is acceptable).
- Config keys stay lower_snake; preserve CSV headers; avoid breaking backward compatibility of `SignalGenerator` timestamp handling.

## Testing Guidelines
- Current scripts double as smoke tests; run `python test_dynamic_fft.py` and `python test_oscillation_detection.py --mode static` after changes to analyzers/detectors.
- Add pytest-style assertions if expanding coverage; place new tests at repo root with `test_*.py` naming.
- Use small CSV fixtures under `csv-data/`; avoid committing large binaries; seed randomness for reproducibility if added.

## Commit & Pull Request Guidelines
- Match existing history: short, imperative, lowercase subjects (e.g., `add FFT_dynamic_analyzer`, `fix overlap calc`).
- PRs: describe scope, configs touched, sample commands to verify, and before/after behavior; include screenshots/plots if visuals change.
- Link issues when available; call out backward-compatibility or sampling-rate impacts; mention new config fields or data schema changes.
