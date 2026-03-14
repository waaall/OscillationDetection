from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.detection_pipeline import DetectionPipeline  # noqa: E402


def _make_signal(
    *,
    freq_hz: float,
    amplitude: float,
    sampling_rate_hz: float,
    sample_count: int,
) -> np.ndarray:
    time_axis = np.arange(sample_count) / sampling_rate_hz
    return amplitude * np.sin(2 * np.pi * freq_hz * time_axis)


def test_alarm_when_band_peak_exceeds_threshold():
    pipeline = DetectionPipeline(
        target_freq_range_hz=(0.5, 1.0),
        window_duration_s=6.0,
        amplitude_threshold=0.7,
        sampling_rate_hz=10.0,
    )
    signal = _make_signal(freq_hz=0.5, amplitude=1.0, sampling_rate_hz=10.0, sample_count=120)

    results = pipeline.analyze_samples(signal, sampling_rate_hz=10.0, step_duration_s=3.0)

    assert len(results) == 3
    assert all(result["status"] == "alarm" for result in results)


def test_ok_when_band_peak_is_below_threshold():
    pipeline = DetectionPipeline(
        target_freq_range_hz=(0.5, 1.0),
        window_duration_s=6.0,
        amplitude_threshold=0.7,
        sampling_rate_hz=10.0,
    )
    signal = _make_signal(freq_hz=0.5, amplitude=0.2, sampling_rate_hz=10.0, sample_count=120)

    results = pipeline.analyze_samples(signal, sampling_rate_hz=10.0, step_duration_s=3.0)

    assert len(results) == 3
    assert all(result["status"] == "ok" for result in results)


def test_out_of_band_when_strong_peak_is_outside_target_band():
    pipeline = DetectionPipeline(
        target_freq_range_hz=(0.5, 1.0),
        window_duration_s=6.0,
        amplitude_threshold=0.7,
        sampling_rate_hz=10.0,
    )
    signal = _make_signal(freq_hz=1.5, amplitude=1.0, sampling_rate_hz=10.0, sample_count=120)

    results = pipeline.analyze_samples(signal, sampling_rate_hz=10.0, step_duration_s=3.0)

    assert len(results) == 3
    assert all(result["status"] == "out_of_band" for result in results)


def test_window_duration_validation():
    with pytest.raises(ValueError, match="three periods"):
        DetectionPipeline(
            target_freq_range_hz=(0.5, 1.0),
            window_duration_s=5.0,
            amplitude_threshold=0.5,
            sampling_rate_hz=10.0,
        )


def test_analyze_samples_without_timestamps():
    pipeline = DetectionPipeline(
        target_freq_range_hz=(0.5, 1.0),
        window_duration_s=6.0,
        amplitude_threshold=0.7,
        sampling_rate_hz=10.0,
    )
    signal = _make_signal(freq_hz=0.5, amplitude=1.0, sampling_rate_hz=10.0, sample_count=120)

    results = pipeline.analyze_samples(signal, sampling_rate_hz=10.0, step_duration_s=3.0)

    assert results[0]["window"]["start_time"] is None
    assert results[0]["window"]["end_time"] is None


def test_analyze_samples_with_consistent_timestamps():
    pipeline = DetectionPipeline(
        target_freq_range_hz=(0.5, 1.0),
        window_duration_s=6.0,
        amplitude_threshold=0.7,
        sampling_rate_hz=10.0,
    )
    sample_count = 120
    time_axis = np.arange(sample_count) / 10.0
    timestamps = pd.Timestamp("2026-03-14T10:00:00") + pd.to_timedelta(time_axis, unit="s")
    signal = _make_signal(freq_hz=0.5, amplitude=1.0, sampling_rate_hz=10.0, sample_count=sample_count)
    results = pipeline.analyze_samples(
        signal,
        sampling_rate_hz=10.0,
        timestamps=timestamps,
        step_duration_s=3.0,
        source_mode="csv",
    )

    assert len(results) == 3
    assert results[0]["window"]["start_time"] is not None


def test_invalid_input_on_irregular_sampling_without_resample():
    pipeline = DetectionPipeline(
        target_freq_range_hz=(0.5, 1.0),
        window_duration_s=6.0,
        amplitude_threshold=0.7,
        sampling_rate_hz=10.0,
        allow_resample=False,
    )
    sample_count = 120
    time_axis = np.arange(sample_count) / 10.0
    time_axis[10:] += 0.15
    timestamps = pd.Timestamp("2026-03-14T10:00:00") + pd.to_timedelta(time_axis, unit="s")
    signal = _make_signal(freq_hz=0.5, amplitude=1.0, sampling_rate_hz=10.0, sample_count=sample_count)
    results = pipeline.analyze_samples(
        signal,
        sampling_rate_hz=10.0,
        timestamps=timestamps,
        step_duration_s=3.0,
        source_mode="csv",
    )

    assert len(results) == 1
    assert results[0]["status"] == "invalid_input"


def test_insufficient_data_returns_single_result():
    pipeline = DetectionPipeline(
        target_freq_range_hz=(0.5, 1.0),
        window_duration_s=6.0,
        amplitude_threshold=0.7,
        sampling_rate_hz=10.0,
    )
    signal = _make_signal(freq_hz=0.5, amplitude=1.0, sampling_rate_hz=10.0, sample_count=20)

    results = pipeline.analyze_samples(signal, sampling_rate_hz=10.0, step_duration_s=3.0)

    assert len(results) == 1
    assert results[0]["status"] == "insufficient_data"
