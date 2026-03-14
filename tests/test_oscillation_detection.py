from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.offline.csv_replay import CsvReplay  # noqa: E402
from src.offline.result_visualizer import ResultVisualizer  # noqa: E402
from src.offline import SignalGenerator  # noqa: E402


def test_visualizer_generates_summary_and_spectrum_plots(tmp_path: Path):
    sampling_rate_hz = 10.0
    sample_count = 120
    time_axis = np.arange(sample_count) / sampling_rate_hz
    signal = np.sin(2 * np.pi * 0.5 * time_axis)
    timestamps = pd.Timestamp("2026-03-14T10:00:00") + pd.to_timedelta(time_axis, unit="s")
    input_csv = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "signal": signal,
        }
    ).to_csv(input_csv, index=False)

    analyzer = CsvReplay(
        sampling_rate_hz=10.0,
        target_freq_range_hz=(0.5, 1.0),
        window_duration_s=6.0,
        amplitude_threshold=0.7,
        step_duration_s=3.0,
        include_plot=True,
    )
    analyzer.input_csv_path = str(input_csv)
    analyzer.time_column = "timestamp"
    analyzer.value_column = "signal"
    analyzer.timestamp_format = "%Y-%m-%d %H:%M:%S.%f"
    analyzer.has_timestamp = True

    results_df = analyzer.analyze_dynamic(pd.read_csv(input_csv))
    visualizer = ResultVisualizer(
        results_df=results_df,
        raw_results=analyzer.last_results,
        amplitude_threshold=0.7,
    )

    summary_path = tmp_path / "summary.png"
    spectrum_path = tmp_path / "spectrum.png"
    visualizer.plot_summary(str(summary_path))
    visualizer.plot_window_spectrum(0, str(spectrum_path))

    assert summary_path.exists()
    assert spectrum_path.exists()


def test_signal_generator_is_available_from_offline_package():
    generator = SignalGenerator(
        sampling_rate=10,
        duration=1.0,
        noise_level=0.0,
        seed=1,
    )

    signal = generator.sine_wave(freqs=[1.0], amps=[1.0])

    assert len(signal) == 10
