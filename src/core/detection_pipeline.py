from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.core.fft_analyzer import FFTAnalysisResult, FFTAnalyzer


@dataclass(frozen=True)
class PreparedSignal:
    """经过清洗、排序、可选重采样后的信号，可直接送入FFTAnalyzer类"""
    values: np.ndarray
    timestamps: Optional[pd.Series]
    sampling_rate_hz: float


class DetectionPipeline:
    """信号输入规范化、窗口化处理、状态映射的振荡检测管道。"""

    def __init__(
        self,
        *,
        target_freq_range_hz: tuple[float, float],
        window_duration_s: float,
        amplitude_threshold: float,
        sampling_rate_hz: Optional[float] = None,
        allow_resample: bool = False,
        include_spectrum: bool = False,
    ) -> None:
        self.target_freq_range_hz = (
            float(target_freq_range_hz[0]),
            float(target_freq_range_hz[1]),
        )
        self.window_duration_s = float(window_duration_s)
        self.amplitude_threshold = float(amplitude_threshold)
        self.sampling_rate_hz = (
            None if sampling_rate_hz is None else float(sampling_rate_hz)
        )
        self.allow_resample = bool(allow_resample)
        self.include_spectrum = bool(include_spectrum)

        self._validate_config()

    def _validate_config(self) -> None:
        low_hz, high_hz = self.target_freq_range_hz
        if low_hz <= 0 or high_hz <= 0 or low_hz >= high_hz:
            raise ValueError("target_freq_range_hz must satisfy 0 < low < high")
        if self.window_duration_s <= 0:
            raise ValueError("window_duration_s must be positive")
        # 窗口时长必须 >= 目标最低频率的 3 个周期，否则频率分辨率不足
        min_window_duration_s = 3.0 / low_hz
        if self.window_duration_s < min_window_duration_s:
            raise ValueError(
                "window_duration_s 必须覆盖目标最低频率的至少三个周期"
            )
        if self.amplitude_threshold < 0:
            raise ValueError("amplitude_threshold must be non-negative")
        if self.sampling_rate_hz is not None and self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive when provided")
        if self.sampling_rate_hz is not None:
            self._validate_band_against_rate(self.sampling_rate_hz)

    def analyze_samples(
        self,
        values: Sequence[Any],
        *,
        sampling_rate_hz: Optional[float] = None,
        timestamps: Optional[Sequence[Any]] = None,
        step_duration_s: Optional[float] = None,
        source_mode: str = "samples",
        timestamp_format: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        # 信号预处理：清洗、时间戳解析、采样率推断/校验、可选重采样
        try:
            prepared = self._prepare_signal(
                values,
                sampling_rate_hz=sampling_rate_hz,
                timestamps=timestamps,
                timestamp_format=timestamp_format,
            )
        except ValueError as exc:
            # 预处理失败直接返回一条 invalid_input 结果，不进入窗口分析
            return [
                self._invalid_result(
                    reason=str(exc),
                    status="invalid_input",
                    source_mode=source_mode,
                )
            ]

        # 将窗口时长换算为样本数
        window_samples = max(
            int(round(self.window_duration_s * prepared.sampling_rate_hz)),
            1,
        )
        # 数据总长度不足一个完整窗口时，输出 insufficient_data 而非空结果
        if prepared.values.size < window_samples:
            return [
                self._result_from_metrics(
                    window_id=0,
                    status="insufficient_data",
                    reason="not_enough_samples",
                    start_index=0,
                    end_index=max(prepared.values.size - 1, 0),
                    sample_count=int(prepared.values.size),
                    sampling_rate_hz=prepared.sampling_rate_hz,
                    start_time=self._timestamp_to_iso(prepared.timestamps, 0),
                    end_time=self._timestamp_to_iso(
                        prepared.timestamps, prepared.values.size - 1
                    ),
                    metrics=None,
                    source_mode=source_mode,
                    debug=None,
                )
            ]

        # step_duration_s 为滑窗步进；None 时退化为不重叠的逐窗分析
        if step_duration_s is None:
            step_samples = window_samples
        else:
            if step_duration_s <= 0:
                raise ValueError("step_duration_s must be positive")
            step_samples = max(int(round(step_duration_s * prepared.sampling_rate_hz)), 1)

        analyzer = FFTAnalyzer(
            sampling_rate_hz=prepared.sampling_rate_hz,
            target_freq_range_hz=self.target_freq_range_hz,
            amplitude_threshold=self.amplitude_threshold,
            include_spectrum=self.include_spectrum,
        )

        # 滑窗遍历：逐窗口 FFT → 状态分类 → 组装结果
        results: list[dict[str, Any]] = []
        for window_id, start_index in enumerate(
            range(0, prepared.values.size - window_samples + 1, step_samples)
        ):
            end_index = start_index + window_samples - 1
            window_values = prepared.values[start_index: start_index + window_samples]
            metrics = analyzer.analyze(window_values)
            status, reason = self._classify_window(metrics)
            debug = None
            if self.include_spectrum:
                window_time_s = (
                    np.arange(window_values.size, dtype=float)
                    / prepared.sampling_rate_hz
                )
                debug = {
                    "window_time_s": window_time_s,
                    "window_values": window_values.copy(),
                    "spectrum_freqs": metrics.spectrum_freqs,
                    "spectrum_amps": metrics.spectrum_amps,
                }

            results.append(
                self._result_from_metrics(
                    window_id=window_id,
                    status=status,
                    reason=reason,
                    start_index=start_index,
                    end_index=end_index,
                    sample_count=window_samples,
                    sampling_rate_hz=prepared.sampling_rate_hz,
                    start_time=self._timestamp_to_iso(prepared.timestamps, start_index),
                    end_time=self._timestamp_to_iso(prepared.timestamps, end_index),
                    metrics=metrics,
                    source_mode=source_mode,
                    debug=debug,
                )
            )

        return results

    def _prepare_signal(
        self,
        values: Sequence[Any],
        *,
        sampling_rate_hz: Optional[float],
        timestamps: Optional[Sequence[Any]],
        timestamp_format: Optional[str],
    ) -> PreparedSignal:
        # 数值清洗：强制转数值 → 前向/后向填充缺失值
        raw_values = pd.Series(values, dtype="object")
        numeric_values = pd.to_numeric(raw_values, errors="coerce")
        if numeric_values.isna().all():
            raise ValueError("empty_signal_after_cleanup")
        clean_values = numeric_values.ffill().bfill()
        if clean_values.isna().any():
            raise ValueError("empty_signal_after_cleanup")

        resolved_rate = sampling_rate_hz or self.sampling_rate_hz
        parsed_timestamps: Optional[pd.Series] = None

        if timestamps is not None:
            parsed_timestamps = pd.to_datetime(
                pd.Series(timestamps),
                format=timestamp_format,
                errors="coerce",
            )
            if parsed_timestamps.isna().any():
                raise ValueError("timestamp_parse_failed")

            records = pd.DataFrame(
                {
                    "timestamp": parsed_timestamps,
                    "value": clean_values.astype(float),
                }
            )
            # 乱序时间戳先排序
            if not records["timestamp"].is_monotonic_increasing:
                records = records.sort_values("timestamp").reset_index(drop=True)

            # 重复时间戳：允许重采样时取均值合并，否则报错
            if records["timestamp"].duplicated().any():
                if not self.allow_resample:
                    raise ValueError("duplicate_timestamp_without_resample")
                records = (
                    records.groupby("timestamp", as_index=False)["value"]
                    .mean()
                    .sort_values("timestamp")
                    .reset_index(drop=True)
                )

            # 根据时间戳间隔推断采样率，并与外部传入值做一致性校验
            inferred_rate, relative_error, irregular = self._infer_sampling_context(
                records["timestamp"],
                expected_rate_hz=resolved_rate,
            )
            if resolved_rate is None:
                resolved_rate = inferred_rate

            if resolved_rate is None:
                raise ValueError("sampling_context_missing")

            self._validate_band_against_rate(resolved_rate)

            # 均值误差 >5% 或变异系数 >10% 视为采样不均匀，需要重采样
            has_rate_mismatch = (
                relative_error is not None and relative_error > 0.05
            )
            needs_resample = has_rate_mismatch or irregular
            if needs_resample:
                if not self.allow_resample:
                    if has_rate_mismatch and irregular:
                        raise ValueError(
                            "sampling_rate_mismatch_and_irregular_without_resample"
                        )
                    if has_rate_mismatch:
                        raise ValueError(
                            "sampling_rate_mismatch_without_resample"
                        )
                    raise ValueError("irregular_sampling_without_resample")
                records = self._resample_records(records, resolved_rate)

            parsed_timestamps = records["timestamp"]
            clean_values = records["value"]

        if resolved_rate is None:
            raise ValueError("sampling_context_missing")

        self._validate_band_against_rate(resolved_rate)
        return PreparedSignal(
            values=clean_values.astype(float).to_numpy(),
            timestamps=parsed_timestamps,
            sampling_rate_hz=float(resolved_rate),
        )

    def _infer_sampling_context(
        self,
        timestamps: pd.Series,
        *,
        expected_rate_hz: Optional[float],
    ) -> tuple[float, Optional[float], bool]:
        if timestamps.size < 2:
            raise ValueError("sampling_context_missing")

        intervals_s = timestamps.diff().dropna().dt.total_seconds()
        if intervals_s.empty or (intervals_s <= 0).any():
            raise ValueError("resample_failed")

        mean_interval_s = float(intervals_s.mean())
        std_interval_s = float(intervals_s.std(ddof=0))
        inferred_rate = 1.0 / mean_interval_s

        # 与外部指定的采样率做相对误差比较
        relative_error: Optional[float] = None
        if expected_rate_hz is not None:
            expected_interval_s = 1.0 / expected_rate_hz
            relative_error = abs(mean_interval_s - expected_interval_s) / expected_interval_s

        # 变异系数 >10% 判定为不规则采样
        irregular = bool(std_interval_s / mean_interval_s > 0.1)
        return inferred_rate, relative_error, irregular

    def _resample_records(
        self, records: pd.DataFrame, target_rate_hz: float
    ) -> pd.DataFrame:
        if records.shape[0] < 2:
            raise ValueError("resample_failed")

        start_time = records["timestamp"].iloc[0]
        seconds = (
            records["timestamp"] - start_time
        ).dt.total_seconds().to_numpy(dtype=float)
        if np.any(np.diff(seconds) <= 0):
            raise ValueError("resample_failed")

        # 按目标采样率生成等间隔时间轴，线性插值重采样
        step_s = 1.0 / target_rate_hz
        new_seconds = np.arange(0.0, seconds[-1] + (step_s * 0.5), step_s)
        new_values = np.interp(new_seconds, seconds, records["value"].to_numpy(dtype=float))
        new_timestamps = start_time + pd.to_timedelta(new_seconds, unit="s")

        return pd.DataFrame({"timestamp": new_timestamps, "value": new_values})

    def _classify_window(self, metrics: FFTAnalysisResult) -> tuple[str, str]:
        """窗口状态映射：alarm > out_of_band > ok，优先级从高到低。"""
        # 目标频带内峰值超阈 → alarm
        in_band_amplitude = metrics.peak_amplitude
        if in_band_amplitude is not None and in_band_amplitude >= self.amplitude_threshold:
            return "alarm", "peak_above_threshold"

        # 全频段峰值超阈但不在目标频带内 → out_of_band（提示带外异常）
        overall_freq_hz = metrics.overall_peak_freq_hz
        overall_amp = metrics.overall_peak_amplitude
        if (
            overall_freq_hz is not None
            and overall_amp is not None
            and overall_amp >= self.amplitude_threshold
            and not self._is_in_band(overall_freq_hz)
        ):
            return "out_of_band", "dominant_peak_outside_band"

        return "ok", "peak_below_threshold"

    def _result_from_metrics(
        self,
        *,
        window_id: int,
        status: str,
        reason: str,
        start_index: int,
        end_index: int,
        sample_count: int,
        sampling_rate_hz: float,
        start_time: Optional[str],
        end_time: Optional[str],
        metrics: Optional[FFTAnalysisResult],
        source_mode: str,
        debug: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        if metrics is None:
            metrics_payload = {
                "dominant_freq_hz": None,
                "dominant_period_s": None,
                "peak_amplitude": None,
                "overall_peak_freq_hz": None,
                "overall_peak_amplitude": None,
                "threshold": self.amplitude_threshold,
            }
        else:
            metrics_payload = {
                "dominant_freq_hz": metrics.dominant_freq_hz,
                "dominant_period_s": metrics.dominant_period_s,
                "peak_amplitude": metrics.peak_amplitude,
                "overall_peak_freq_hz": metrics.overall_peak_freq_hz,
                "overall_peak_amplitude": metrics.overall_peak_amplitude,
                "threshold": metrics.threshold,
            }

        result = {
            "status": status,
            "reason": reason,
            "window": {
                "window_id": window_id,
                "start_index": start_index,
                "end_index": end_index,
                "start_time": start_time,
                "end_time": end_time,
                "sample_count": sample_count,
                "sampling_rate_hz": float(sampling_rate_hz),
            },
            "metrics": metrics_payload,
            "meta": {
                "target_band_low_hz": self.target_freq_range_hz[0],
                "target_band_high_hz": self.target_freq_range_hz[1],
                "source_mode": source_mode,
            },
        }
        if debug is not None:
            result["debug"] = debug
        return result

    def _invalid_result(
        self,
        *,
        reason: str,
        status: str,
        source_mode: str,
    ) -> dict[str, Any]:
        return self._result_from_metrics(
            window_id=0,
            status=status,
            reason=reason,
            start_index=0,
            end_index=0,
            sample_count=0,
            sampling_rate_hz=float(self.sampling_rate_hz or 0.0),
            start_time=None,
            end_time=None,
            metrics=None,
            source_mode=source_mode,
            debug=None,
        )

    def _validate_band_against_rate(self, sampling_rate_hz: float) -> None:
        if self.target_freq_range_hz[1] >= sampling_rate_hz / 2.0:
            raise ValueError("invalid_frequency_range")

    def _timestamp_to_iso(
        self, timestamps: Optional[pd.Series], index: int
    ) -> Optional[str]:
        if timestamps is None or timestamps.empty or index < 0:
            return None
        index = min(index, len(timestamps) - 1)
        return timestamps.iloc[index].isoformat()

    def _is_in_band(self, freq_hz: float) -> bool:
        low_hz, high_hz = self.target_freq_range_hz
        return low_hz <= freq_hz <= high_hz
