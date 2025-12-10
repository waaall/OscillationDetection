"""
pytest test_frequency_refinement.py
"""
import logging
import time
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.FrequencyRefinement import FrequencyRefinement
from src.core.SignalGenerator import SignalGenerator

# 降低测试日志噪声
logging.getLogger("SignalGenerator").setLevel(logging.WARNING)
logging.getLogger("TestFFT").setLevel(logging.WARNING)
TEST_LOGGER = logging.getLogger("FrequencyRefinementTest")
TEST_LOGGER.setLevel(logging.WARNING)


def _make_tone(freq: float, sampling_rate: int, duration: float,
               noise_std: float = 0.0, seed: int = 0) -> np.ndarray:
    """生成单频测试信号，带可选高斯噪声。"""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / sampling_rate)
    signal = np.sin(2 * np.pi * freq * t)
    if noise_std > 0:
        signal = signal + rng.normal(scale=noise_std, size=signal.shape)
    return signal


def test_basic_accuracy_no_noise():
    """无噪声下应达 mHz 级误差。"""
    true_freq = 50.0234
    sampling_rate = 10000
    duration = 1.0

    gen = SignalGenerator(sampling_rate=sampling_rate,
                          duration=duration,
                          noise_level=0.0,
                          seed=1,
                          log_file=None)
    gen.logger.setLevel(logging.WARNING)
    signal = gen.sine_wave(freqs=[true_freq], amps=[1.0])

    refiner = FrequencyRefinement(logger=TEST_LOGGER)
    freq_est = refiner.refine(signal, sampling_rate, freq_initial=50.0)

    assert freq_est is not None
    assert abs(freq_est - true_freq) < 1e-3  # < 1 mHz


def test_noise_robustness():
    """不同 SNR 下仍保持低 mHz 误差。"""
    true_freq = 50.0
    sampling_rate = 10000
    duration = 0.5
    base_signal = _make_tone(true_freq, sampling_rate, duration, noise_std=0.0, seed=0)

    test_cases = [
        (60, 1.5),   # 60 dB, 期望 <1.5 mHz
        (40, 5.0),   # 40 dB, 期望 <5 mHz
        (30, 12.0),  # 30 dB, 期望 <12 mHz
    ]

    for snr_db, max_error_mhz in test_cases:
        noise_std = 10 ** (-snr_db / 20)  # 单位振幅噪声 std
        signal = base_signal + _make_tone(0, sampling_rate, duration,
                                          noise_std=noise_std, seed=snr_db)

        refiner = FrequencyRefinement(logger=TEST_LOGGER, search_range=0.1)
        freq_est = refiner.refine(signal, sampling_rate, freq_initial=50.0)

        assert freq_est is not None
        error_mhz = abs(freq_est - true_freq) * 1000
        assert error_mhz < max_error_mhz


def test_frequency_range():
    """49.8-50.2 Hz 多点测试，初值偏差仍能收敛。"""
    sampling_rate = 8000
    duration = 0.8
    freqs = [49.8, 49.9, 50.0, 50.1, 50.2]

    for target in freqs:
        signal = _make_tone(target, sampling_rate, duration, noise_std=0.0, seed=int(target * 10))
        refiner = FrequencyRefinement(logger=TEST_LOGGER, search_range=0.2)
        freq_est = refiner.refine(signal, sampling_rate, freq_initial=target + 0.02)

        assert freq_est is not None
        assert abs(freq_est - target) < 2e-3  # 2 mHz


def test_performance_benchmark():
    """性能基准：短数据下执行时间应在 20ms 以内。"""
    true_freq = 50.05
    sampling_rate = 8000
    duration = 0.25  # 约 2000 点
    signal = _make_tone(true_freq, sampling_rate, duration, noise_std=0.0, seed=7)

    refiner = FrequencyRefinement(
        logger=TEST_LOGGER,
        method="grid_search",
        search_range=0.05,
        step_size=0.001
    )

    start = time.perf_counter()
    freq_est = refiner.refine(signal, sampling_rate, freq_initial=50.0)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert freq_est is not None
    assert abs(freq_est - true_freq) < 2e-3
    assert elapsed_ms < 20.0


def test_convergence_with_poor_initial():
    """初值偏差 0.2 Hz 时仍应收敛。"""
    true_freq = 50.18
    sampling_rate = 12000
    duration = 0.6
    signal = _make_tone(true_freq, sampling_rate, duration, noise_std=0.0, seed=9)

    refiner = FrequencyRefinement(logger=TEST_LOGGER)
    freq_est = refiner.refine(signal, sampling_rate, freq_initial=true_freq - 0.2)

    assert freq_est is not None
    assert abs(freq_est - true_freq) < 2e-3


def test_data_length_boundary():
    """数据长度不足时返回 None，边界长度可正常工作。"""
    sampling_rate = 10000
    short_signal = np.zeros(80)
    refiner = FrequencyRefinement(logger=TEST_LOGGER)

    assert refiner.refine(short_signal, sampling_rate, freq_initial=50.0) is None

    boundary_signal = np.ones(160)
    assert refiner.refine(boundary_signal, sampling_rate, freq_initial=50.0) is not None
