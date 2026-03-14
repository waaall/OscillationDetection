import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.Zero_Cross_Freq import ZeroCrossFreq  # noqa: E402


def _phase_error(phase_a: float, phase_b: float) -> float:
    """返回 wrap 后的最小相位差。"""
    return abs(np.angle(np.exp(1j * (phase_a - phase_b))))


def test_zero_cross_double_edge_mode_uses_full_periods():
    """双边过零模式必须按整周期计算，不能把频率翻倍。"""
    sampling_rate = 2000
    window_size = 400
    t = np.arange(window_size) / sampling_rate
    signal = np.sin(2 * np.pi * 50.0 * t)

    analyzer = ZeroCrossFreq(
        window_size=window_size,
        sampling_rate=sampling_rate,
        config={
            "window_periods": 4,
            "min_freq_hz": 45.0,
            "max_freq_hz": 120.0,
            "rising_only": False,
        },
    )
    analyzer.logger.setLevel("WARNING")

    success, freq, amp, phase = analyzer.fft_analyze(signal)

    assert success
    assert abs(freq - 50.0) < 1e-9
    assert np.isfinite(phase)


def test_zero_cross_phase_estimate_is_consistent_across_edge_modes():
    """已知相位的纯正弦在单双边模式下都应返回一致的近似相位。"""
    sampling_rate = 2000
    window_size = 400
    target_phase = 1.0
    t = np.arange(window_size) / sampling_rate
    signal = np.sin(2 * np.pi * 50.0 * t + target_phase)

    for rising_only in [True, False]:
        analyzer = ZeroCrossFreq(
            window_size=window_size,
            sampling_rate=sampling_rate,
            config={
                "window_periods": 4,
                "min_freq_hz": 45.0,
                "max_freq_hz": 65.0,
                "rising_only": rising_only,
            },
        )
        analyzer.logger.setLevel("WARNING")

        success, freq, amp, phase = analyzer.fft_analyze(signal)

        assert success
        assert abs(freq - 50.0) < 1e-9
        assert np.isfinite(phase)
        assert _phase_error(phase, target_phase) < 1e-3
