from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FFTAnalysisResult:
    """Result of a single-window FFT band analysis."""

    dominant_freq_hz: Optional[float]
    dominant_period_s: Optional[float]
    peak_amplitude: Optional[float]
    overall_peak_freq_hz: Optional[float]
    overall_peak_amplitude: Optional[float]
    threshold: float
    spectrum_freqs: Optional[np.ndarray] = None
    spectrum_amps: Optional[np.ndarray] = None


class FFTAnalyzer:
    """
    Single-window FFT analyzer for low-frequency oscillation detection.

    The analyzer is deliberately narrow in scope: it only receives evenly
    sampled values and returns spectral peak metrics for the full spectrum and
    for a target frequency band.
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

        signal = signal - np.mean(signal)
        window = np.hanning(signal.size)
        coherent_gain = np.sum(window) / signal.size
        windowed_signal = signal * window

        fft_vals = np.fft.rfft(windowed_signal)
        freqs = np.fft.rfftfreq(signal.size, d=1.0 / self.sampling_rate_hz)
        amps = np.abs(fft_vals) / (signal.size * coherent_gain)
        if signal.size % 2 == 0:
            amps[1:-1] *= 2.0
        else:
            amps[1:] *= 2.0

        positive_mask = freqs > 0.0
        positive_freqs = freqs[positive_mask]
        positive_amps = amps[positive_mask]
        if positive_freqs.size == 0:
            raise ValueError("samples do not produce any positive-frequency bins")

        overall_idx = int(np.argmax(positive_amps))
        overall_peak_freq_hz = float(positive_freqs[overall_idx])
        overall_peak_amplitude = float(positive_amps[overall_idx])

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
