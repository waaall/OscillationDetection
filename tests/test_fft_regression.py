import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.Freq_dynamic_analyzer import FreqDynamicAnalyzer  # noqa: E402
from src.core.FFT_analyzer import FFTAnalyzer  # noqa: E402


def _build_timestamped_df(signal: np.ndarray,
                          sampling_rate: float,
                          start_time: datetime = datetime(2025, 1, 1)) -> pd.DataFrame:
    """为合成信号构造带时间戳的 DataFrame。"""
    timestamps = [
        start_time + timedelta(seconds=float(idx) / float(sampling_rate))
        for idx in range(len(signal))
    ]
    return pd.DataFrame({
        "timestamp": timestamps,
        "signal": signal,
    })


def test_fft_refinement_uses_raw_signal_under_windowing():
    """加窗只用于 FFT 检测，最小二乘精化应始终吃原始窗口数据。"""
    sampling_rate = 10000
    window_size = 2000
    target_freq = 50.0234
    analyzer = FFTAnalyzer(window_size=window_size, sampling_rate=sampling_rate, log_file=None)
    analyzer.logger.setLevel("WARNING")

    t = np.arange(window_size) / sampling_rate
    for phase in [0.0, 0.5, 1.0, 2.0]:
        signal = np.sin(2 * np.pi * target_freq * t + phase)
        success, freq, amp, phase_est = analyzer.fft_analyze(
            signal,
            use_window=True,
            IpDFT=True,
            refine_frequency=True,
            refine_config={
                "method": "grid_search",
                "search_range": 0.2,
                "step_size": 0.001,
            },
            peak_search_range=(49.9, 50.1),
        )

        assert success
        assert abs(freq - target_freq) < 1e-3
        assert abs(amp - 1.0) < 1e-3
        assert abs(((phase_est - phase + np.pi) % (2 * np.pi)) - np.pi) < 1e-3


def test_fft_ipdft_keeps_neighbor_bins_under_narrow_search_band():
    """窄搜索带只能限制选峰，不能裁掉 IpDFT 需要的相邻 bin。"""
    sampling_rate = 10000
    window_size = 2000
    target_freq = 50.0234
    analyzer = FFTAnalyzer(window_size=window_size, sampling_rate=sampling_rate, log_file=None)
    analyzer.logger.setLevel("WARNING")

    t = np.arange(window_size) / sampling_rate
    signal = np.sin(2 * np.pi * target_freq * t)

    success, freq, amp, phase = analyzer.fft_analyze(
        signal,
        use_window=True,
        IpDFT=True,
        refine_frequency=False,
        peak_search_range=(49.9, 50.1),
    )

    assert success
    assert abs(freq - target_freq) < 0.01
    assert abs(freq - 50.0) > 0.01


def test_fft_refinement_out_of_band_falls_back_to_ipdft(monkeypatch):
    """最小二乘精化越界时应回退到 IpDFT，而不是粗 FFT bin。"""
    from src.core.FrequencyRefinement import FrequencyRefinement

    sampling_rate = 10000
    window_size = 2000
    target_freq = 50.0234
    analyzer = FFTAnalyzer(window_size=window_size, sampling_rate=sampling_rate, log_file=None)
    analyzer.logger.setLevel("WARNING")

    t = np.arange(window_size) / sampling_rate
    signal = np.sin(2 * np.pi * target_freq * t)

    def fake_refine(self, signal_data, fs, freq_initial, return_all_params=False):
        return (50.2, 123.0, 0.0, 0.0, 0.0)

    monkeypatch.setattr(FrequencyRefinement, "refine", fake_refine)

    ipdft_result = analyzer.fft_analyze(
        signal,
        use_window=True,
        IpDFT=True,
        refine_frequency=False,
        peak_search_range=(49.9, 50.1),
    )
    refined_result = analyzer.fft_analyze(
        signal,
        use_window=True,
        IpDFT=True,
        refine_frequency=True,
        refine_config={
            "method": "grid_search",
            "search_range": 0.2,
            "step_size": 0.001,
        },
        peak_search_range=(49.9, 50.1),
    )

    assert refined_result[0]
    assert refined_result[1] == pytest.approx(ipdft_result[1])
    assert refined_result[2] == pytest.approx(ipdft_result[2])
    assert refined_result[3] == pytest.approx(ipdft_result[3])


def test_dynamic_fft_stays_on_fundamental_with_stronger_harmonic():
    """窄带搜索时，强谐波不应抢走 50Hz 基波的峰值。"""
    sampling_rate = 10000
    duration = 0.4
    t = np.arange(0, duration, 1 / sampling_rate)
    signal = np.sin(2 * np.pi * 50.0 * t) + 2.0 * np.sin(2 * np.pi * 150.0 * t)
    df = _build_timestamped_df(signal, sampling_rate)

    analyzer = FreqDynamicAnalyzer(
        window_duration_ms=200,
        step_duration_ms=100,
        sampling_rate=sampling_rate,
        freq_range=(49.5, 50.5),
        use_window=True,
        use_ipdft=True,
        refine_frequency=False,
        log_file=None,
    )
    analyzer.logger.setLevel("WARNING")

    results = analyzer.analyze_dynamic(df)

    assert not results.empty
    assert abs(results["frequency"].mean() - 50.0) < 1e-4


def test_dynamic_fft_uses_effective_sampling_rate_from_timestamps():
    """当时间戳可信时，应采用实测采样率重算窗口点数与频率。"""
    configured_sampling_rate = 10000
    actual_sampling_rate = 9600
    duration = 0.4
    t = np.arange(0, duration, 1 / actual_sampling_rate)
    signal = np.sin(2 * np.pi * 50.0 * t)
    df = _build_timestamped_df(signal, actual_sampling_rate)

    analyzer = FreqDynamicAnalyzer(
        window_duration_ms=200,
        step_duration_ms=100,
        sampling_rate=configured_sampling_rate,
        freq_range=(49.5, 50.5),
        use_window=True,
        use_ipdft=True,
        refine_frequency=False,
        log_file=None,
    )
    analyzer.logger.setLevel("WARNING")

    results = analyzer.analyze_dynamic(df)

    assert not results.empty
    assert abs(analyzer.effective_sampling_rate - actual_sampling_rate) < 0.1
    assert analyzer.window_samples == 1920
    assert analyzer.step_samples == 960
    assert abs(results["frequency"].mean() - 50.0) < 0.01
