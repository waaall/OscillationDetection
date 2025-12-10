"""
Zero crossing based frequency estimation.

The module mirrors the configuration knobs of the embedded `zero_crossing.c`
implementation: a configurable averaging window (period count), glitch
suppression via a fake/minimum period, and a reasonable frequency range check.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ZeroCrossConfig:
    """
    Zero-crossing configuration parsed from JSON or a plain dict.

    Attributes
    ----------
    window_periods: int
        Number of consecutive periods used to compute the averaged frequency.
    min_freq_hz / max_freq_hz: float
        Allowed frequency range; periods outside the derived bounds are ignored.
    fake_period_ms: float
        Minimum acceptable period (ms) to suppress glitches/spikes.
    min_cross_amplitude: float
        Reject windows whose peak-to-peak amplitude is below this value.
    remove_dc: bool
        Whether to remove the DC offset before detecting zero crossings.
    rising_only: bool
        If True, count only rising zero crossings; otherwise count both edges.
    """

    window_periods: int = 6
    min_freq_hz: float = 45.0
    max_freq_hz: float = 65.0
    fake_period_ms: float = 0.0
    min_cross_amplitude: float = 0.0
    remove_dc: bool = True
    rising_only: bool = True

    @classmethod
    def from_source(cls,
                    config: Optional[dict] = None,
                    config_path: Optional[str] = None) -> "ZeroCrossConfig":
        """Create configuration from a JSON file and/or mapping."""
        payload: dict = {}

        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                payload.update(json.load(f))

        if config:
            # Mapping provided directly has higher priority.
            payload.update(config)

        # Allow either a top-level dict or a nested `zero_cross` section.
        source = payload.get("zero_cross", payload)

        cfg = cls(
            window_periods=int(source.get("window_periods", cls.window_periods)),
            min_freq_hz=float(source.get("min_freq_hz", cls.min_freq_hz)),
            max_freq_hz=float(source.get("max_freq_hz", cls.max_freq_hz)),
            fake_period_ms=float(source.get("fake_period_ms", cls.fake_period_ms)),
            min_cross_amplitude=float(
                source.get("min_cross_amplitude", cls.min_cross_amplitude)
            ),
            remove_dc=bool(source.get("remove_dc", cls.remove_dc)),
            rising_only=bool(source.get("rising_only", cls.rising_only)),
        )

        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.window_periods <= 0:
            raise ValueError("window_periods must be positive")
        if self.min_freq_hz <= 0 or self.max_freq_hz <= 0:
            raise ValueError("min_freq_hz and max_freq_hz must be positive")
        if self.min_freq_hz >= self.max_freq_hz:
            raise ValueError("min_freq_hz must be smaller than max_freq_hz")
        if self.fake_period_ms < 0:
            raise ValueError("fake_period_ms must be non-negative")
        if self.min_cross_amplitude < 0:
            raise ValueError("min_cross_amplitude must be non-negative")


class ZeroCrossFreq:
    """
    Estimate frequency with a zero-crossing counter.

    API is kept drop-in compatible with the FFT-based analyzer: call
    `fft_analyze(window_data, ...)` and get `(success, freq, amplitude, phase)`.
    """

    def __init__(
        self,
        window_size: int,
        sampling_rate: float,
        *,
        config: Optional[dict] = None,
        config_path: Optional[str] = None,
        log_file: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")

        self.window_size = window_size
        self.sampling_rate = sampling_rate
        self.config = ZeroCrossConfig.from_source(config=config, config_path=config_path)
        self.logger = logger or self._build_logger(log_file)

        # Diagnostics counters similar to the embedded implementation.
        self.valid_count = 0
        self.error_count = 0

    def _build_logger(self, log_file: Optional[str]) -> logging.Logger:
        logger = logging.getLogger("ZeroCrossFreq")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.propagate = False

        if log_file:
            file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            formatter = logging.Formatter("[%(asctime)s] %(message)s",
                                          datefmt="%Y-%m-%d %H:%M:%S")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger

    def _detect_zero_crossings(self, signal: np.ndarray) -> np.ndarray:
        """Return fractional sample indices where the signal crosses zero."""
        crossings = []
        prev = signal[0]

        for idx in range(1, len(signal)):
            curr = signal[idx]

            if self.config.rising_only:
                condition = prev <= 0.0 < curr
            else:
                condition = (prev <= 0.0 < curr) or (prev >= 0.0 > curr)

            if not condition:
                prev = curr
                continue

            denom = curr - prev
            if denom == 0:
                frac = 0.0
            else:
                frac = (-prev) / denom

            crossings.append(idx - 1 + frac)
            prev = curr

        return np.array(crossings, dtype=float)

    def _filter_periods(self, period_samples: np.ndarray) -> np.ndarray:
        """Apply glitch and range filtering to raw period samples."""
        if period_samples.size == 0:
            return period_samples

        min_period_samples = self.sampling_rate / self.config.max_freq_hz
        max_period_samples = self.sampling_rate / self.config.min_freq_hz
        fake_period_samples = (
            self.config.fake_period_ms * self.sampling_rate / 1000.0
            if self.config.fake_period_ms > 0
            else 0.0
        )

        valid_periods = []
        for ps in period_samples:
            if ps <= 0:
                continue
            if fake_period_samples and ps < fake_period_samples:
                self.error_count += 1
                continue
            if ps < min_period_samples or ps > max_period_samples:
                self.error_count += 1
                continue
            valid_periods.append(ps)

        return np.array(valid_periods, dtype=float)

    def _estimate_phase(self, first_cross_idx: float) -> float:
        """
        Rough phase estimation (radians) relative to the window start.

        The calculation assumes a near-sinusoidal waveform; it is mainly kept for
        API compatibility with the FFT analyzer.
        """
        if self.window_size <= 0:
            return 0.0
        return -2.0 * np.pi * (first_cross_idx / self.window_size)

    def fft_analyze(  # noqa: N802 - keep name for compatibility
        self,
        window_data: Sequence[float],
        *,
        use_window: bool = False,
        IpDFT: bool = False,
        refine_frequency: bool = False,
        refine_config: Optional[dict] = None,
    ) -> Tuple[bool, float, float, float]:
        """
        Estimate frequency for a single window of samples.

        Parameters mirror the FFT analyzer so the caller side stays unchanged.
        """
        signal = np.asarray(window_data, dtype=float)
        if signal.size < 2:
            self.logger.debug("窗口数据太短，无法检测过零点")
            self.error_count += 1
            return False, 0.0, 0.0, 0.0

        if signal.size != self.window_size:
            self.logger.debug(
                "窗口长度与配置不一致: expected %d, got %d",
                self.window_size,
                signal.size,
            )

        # Basic amplitude and optional DC removal.
        amplitude = (signal.max() - signal.min()) / 2.0
        if amplitude < self.config.min_cross_amplitude:
            self.logger.debug(
                "信号幅值过小 (%.6f)，无法可靠检测过零点", amplitude
            )
            self.error_count += 1
            return False, 0.0, amplitude, 0.0

        if self.config.remove_dc:
            signal = signal - np.mean(signal)

        # Detect zero crossings with linear interpolation.
        crossings = self._detect_zero_crossings(signal)
        if crossings.size < 2:
            self.logger.debug("未检测到足够的过零点（%d）", crossings.size)
            self.error_count += 1
            return False, 0.0, amplitude, 0.0

        periods = np.diff(crossings)
        filtered_periods = self._filter_periods(periods)
        if filtered_periods.size == 0:
            self.logger.debug("全部周期被过滤，可能为毛刺或超出频率范围")
            return False, 0.0, amplitude, 0.0

        # Averaging over the latest N periods.
        window_len = min(self.config.window_periods, filtered_periods.size)
        period_avg_samples = float(np.mean(filtered_periods[-window_len:]))
        freq_hz = self.sampling_rate / period_avg_samples

        self.valid_count += 1
        phase = self._estimate_phase(crossings[0])
        return True, float(freq_hz), float(amplitude), float(phase)


# Alias for backward compatibility with the existing import name.
ZeroCross = ZeroCrossFreq

