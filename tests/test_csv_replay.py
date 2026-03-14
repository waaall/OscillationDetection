from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.offline.csv_replay import CsvReplay  # noqa: E402


def _write_timestamp_csv(path: Path, *, sample_count: int = 120) -> None:
    sampling_rate_hz = 10.0
    time_axis = np.arange(sample_count) / sampling_rate_hz
    signal = np.sin(2 * np.pi * 0.5 * time_axis)
    timestamps = pd.Timestamp("2026-03-14T10:00:00") + pd.to_timedelta(time_axis, unit="s")
    pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "signal": signal,
        }
    ).to_csv(path, index=False)


def test_csv_replay_outputs_all_windows(tmp_path: Path):
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "results.csv"
    _write_timestamp_csv(input_csv)

    analyzer = CsvReplay(
        sampling_rate_hz=10.0,
        target_freq_range_hz=(0.5, 1.0),
        window_duration_s=6.0,
        amplitude_threshold=0.7,
        step_duration_s=3.0,
        include_plot=False,
    )
    analyzer.input_csv_path = str(input_csv)
    analyzer.time_column = "timestamp"
    analyzer.value_column = "signal"
    analyzer.timestamp_format = "%Y-%m-%d %H:%M:%S.%f"
    analyzer.has_timestamp = True

    results_df = analyzer.run_pipeline(input_csv=str(input_csv), output_csv=str(output_csv))

    assert output_csv.exists()
    assert list(results_df["status"]) == ["alarm", "alarm", "alarm"]
    assert len(results_df) == 3


def test_csv_replay_generates_plots_from_debug_spectrum(tmp_path: Path):
    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "results.csv"
    plot_dir = tmp_path / "plots"
    _write_timestamp_csv(input_csv)

    analyzer = CsvReplay(
        sampling_rate_hz=10.0,
        target_freq_range_hz=(0.5, 1.0),
        window_duration_s=6.0,
        amplitude_threshold=0.7,
        step_duration_s=3.0,
        include_plot=True,
        plot_dir=str(plot_dir),
    )
    analyzer.input_csv_path = str(input_csv)
    analyzer.result_csv_path = str(output_csv)
    analyzer.time_column = "timestamp"
    analyzer.value_column = "signal"
    analyzer.timestamp_format = "%Y-%m-%d %H:%M:%S.%f"
    analyzer.has_timestamp = True

    analyzer.run_pipeline(input_csv=str(input_csv), output_csv=str(output_csv))

    assert "debug" in analyzer.last_results[0]
    assert (plot_dir / "dominant_frequency.png").exists()
    assert (plot_dir / "window_0_spectrum.png").exists()


def test_explicit_constructor_args_override_config_file(tmp_path: Path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "analysis": {
                    "sampling_rate_hz": 1.0,
                    "target_freq_range_hz": [0.0167, 0.2],
                    "window_duration_s": 180.0,
                    "amplitude_threshold": 0.5,
                    "allow_resample": False,
                },
                "replay": {
                    "step_duration_s": 1.0,
                },
                "output": {
                    "include_plot": True,
                    "plot_dir": "plots",
                },
                "logging": {
                    "log_level": "WARNING",
                },
            }
        ),
        encoding="utf-8",
    )

    analyzer = CsvReplay(
        config_path=str(config_path),
        sampling_rate_hz=10.0,
        target_freq_range_hz=(0.5, 1.0),
        window_duration_s=6.0,
        amplitude_threshold=0.7,
        allow_resample=True,
        step_duration_s=3.0,
        include_plot=False,
        plot_dir="custom-plots",
        log_level="INFO",
    )

    assert analyzer.sampling_rate_hz == 10.0
    assert analyzer.target_freq_range_hz == (0.5, 1.0)
    assert analyzer.window_duration_s == 6.0
    assert analyzer.amplitude_threshold == 0.7
    assert analyzer.allow_resample is True
    assert analyzer.step_duration_s == 3.0
    assert analyzer.include_plot is False
    assert analyzer.plot_dir == "custom-plots"
    assert analyzer.logger.level == logging.INFO
