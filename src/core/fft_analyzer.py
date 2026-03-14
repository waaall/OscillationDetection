from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FFTAnalysisResult:
    """单窗口 FFT 频带分析结果。"""

    dominant_freq_hz: Optional[float]       # 目标频带内主峰频率 (Hz)
    dominant_period_s: Optional[float]      # 目标频带内主峰周期 (s)
    peak_amplitude: Optional[float]         # 目标频带内主峰幅值
    overall_peak_freq_hz: Optional[float]   # 全频段（除 DC）最大峰频率
    overall_peak_amplitude: Optional[float]  # 全频段（除 DC）最大峰幅值
    threshold: float                        # 报警阈值（外部传入, 原样回传）
    spectrum_freqs: Optional[np.ndarray] = None  # 调试用：正频率轴
    spectrum_amps: Optional[np.ndarray] = None   # 调试用：对应幅值


class FFTAnalyzer:
    """
    单窗口FFT分析器, 主要用于检测低频振荡。
    该分析器的功能范围经过刻意限定：它仅接收均匀采样的数值, 并返回完整频谱以及目标频带的频谱峰值指标。
    """

    def __init__(
        self,
        sampling_rate_hz: float,
        target_freq_range_hz: Tuple[float, float],
        amplitude_threshold: float,
        include_spectrum: bool = False,
    ) -> None:
        if sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if amplitude_threshold < 0:
            raise ValueError("amplitude_threshold must be non-negative")

        low_hz, high_hz = target_freq_range_hz
        if low_hz <= 0 or high_hz <= 0 or low_hz >= high_hz:
            raise ValueError("target_freq_range_hz must satisfy 0 < low < high")
        # 目标频带上限不得超过 Nyquist 频率, 否则无法正确检测
        if high_hz >= sampling_rate_hz / 2.0:
            raise ValueError("target_freq_range_hz must be below Nyquist frequency")

        self.sampling_rate_hz = float(sampling_rate_hz)
        self.target_freq_range_hz = (float(low_hz), float(high_hz))
        self.amplitude_threshold = float(amplitude_threshold)
        self.include_spectrum = bool(include_spectrum)

    def analyze(self, samples: Sequence[float]) -> FFTAnalysisResult:
        signal = np.asarray(samples, dtype=float)
        if signal.ndim != 1:
            raise ValueError("samples must be a one-dimensional sequence")
        if signal.size < 2:
            raise ValueError("samples must contain at least two values")

        # 去均值：消除直流分量对频谱的影响
        signal = signal - np.mean(signal)

        # Hann 窗加窗, 减少频谱泄漏
        window = np.hanning(signal.size)
        # coherent_gain 补偿窗函数引入的幅值衰减（窗均值 / 矩形窗均值）
        coherent_gain = np.sum(window) / signal.size
        windowed_signal = signal * window

        # 实信号 rFFT, 只返回 0~Nyquist 的非负频率分量
        fft_vals = np.fft.rfft(windowed_signal)
        freqs = np.fft.rfftfreq(signal.size, d=1.0 / self.sampling_rate_hz)
        # 归一化幅值：除以 (N × coherent_gain) 得到真实信号幅度
        amps = np.abs(fft_vals) / (signal.size * coherent_gain)
        # 单边谱修正：rFFT 结果只含正频率, 需 ×2 还原双边能量
        # DC (index 0) 和 Nyquist (偶数长度末尾) 无对称分量, 不加倍
        if signal.size % 2 == 0:
            amps[1:-1] *= 2.0
        else:
            amps[1:] *= 2.0

        # 去掉 DC 分量 (freq=0), 只保留正频率
        positive_mask = freqs > 0.0
        positive_freqs = freqs[positive_mask]
        positive_amps = amps[positive_mask]
        if positive_freqs.size == 0:
            raise ValueError("samples do not produce any positive-frequency bins")

        # 全频段峰值（用于 out_of_band 状态判定）
        overall_idx = int(np.argmax(positive_amps))
        overall_peak_freq_hz = float(positive_freqs[overall_idx])
        overall_peak_amplitude = float(positive_amps[overall_idx])

        # 目标频带内峰值提取（用于 alarm/ok 状态判定）
        low_hz, high_hz = self.target_freq_range_hz
        band_mask = (positive_freqs >= low_hz) & (positive_freqs <= high_hz)
        dominant_freq_hz: Optional[float] = None
        dominant_period_s: Optional[float] = None
        peak_amplitude: Optional[float] = None
        if np.any(band_mask):
            band_freqs = positive_freqs[band_mask]
            band_amps = positive_amps[band_mask]
            band_idx = int(np.argmax(band_amps))
            dominant_freq_hz = float(band_freqs[band_idx])
            dominant_period_s = float(1.0 / dominant_freq_hz)
            peak_amplitude = float(band_amps[band_idx])

        spectrum_freqs: Optional[np.ndarray] = None
        spectrum_amps: Optional[np.ndarray] = None
        if self.include_spectrum:
            spectrum_freqs = positive_freqs.copy()
            spectrum_amps = positive_amps.copy()

        return FFTAnalysisResult(
            dominant_freq_hz=dominant_freq_hz,
            dominant_period_s=dominant_period_s,
            peak_amplitude=peak_amplitude,
            overall_peak_freq_hz=overall_peak_freq_hz,
            overall_peak_amplitude=overall_peak_amplitude,
            threshold=self.amplitude_threshold,
            spectrum_freqs=spectrum_freqs,
            spectrum_amps=spectrum_amps,
        )
