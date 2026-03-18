from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd


class ResultVisualizer:

    def __init__(
        self,
        results_df: pd.DataFrame,
        raw_results: Optional[Sequence[dict[str, Any]]] = None,
        amplitude_threshold: Optional[float] = None,
        target_freq_range_hz: Optional[tuple[float, float]] = None,
    ) -> None:
        self.results_df = results_df.copy()
        # raw_results 用于获取 debug 中的频谱数据（仅 include_spectrum=True 时存在）
        self.raw_results = list(raw_results or [])
        self.target_freq_range_hz = target_freq_range_hz

        if amplitude_threshold is not None:
            self.amplitude_threshold = float(amplitude_threshold)
        elif "threshold" in self.results_df.columns and not self.results_df.empty:
            self.amplitude_threshold = float(self.results_df["threshold"].iloc[0])
        else:
            self.amplitude_threshold = 0.0

    @staticmethod
    def _plot_marker(
        ax: Any,
        *,
        freq_hz: Optional[float],
        amplitude: Optional[float],
        color: str,
        label: str,
        text_offset: tuple[float, float] = (6.0, 6.0),
    ) -> None:
        if freq_hz is None or amplitude is None:
            return
        ax.scatter(
            [freq_hz],
            [amplitude],
            color=color,
            s=28,
            zorder=3,
            label=label,
        )
        ax.annotate(
            f"({freq_hz:.4g}Hz, {amplitude:.4g})",
            xy=(freq_hz, amplitude),
            xytext=text_offset,
            textcoords="offset points",
            color=color,
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": color,
                "alpha": 0.85,
            },
        )

    def plot_summary(self, output_path: str) -> None:
        """三合一总览图：主频趋势 / 峰值幅度+阈值线 / 状态时序。"""
        import matplotlib.pyplot as plt

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        x_axis = self.results_df["window_id"]
        status_series = self.results_df["status"]

        axes[0].plot(
            x_axis,
            self.results_df["dominant_freq_hz"],
            color="#1b6ca8",
            linewidth=1.6,
        )
        axes[0].set_ylabel("Dominant Freq (Hz)")
        axes[0].grid(True, alpha=0.3)

        # 子图2：峰值幅度 + 报警阈值虚线
        axes[1].plot(
            x_axis,
            self.results_df["peak_amplitude"],
            color="#c84c09",
            linewidth=1.6,
        )
        axes[1].axhline(
            self.amplitude_threshold,
            color="red",
            linestyle="--",
            linewidth=1.0,
        )
        axes[1].set_ylabel("Peak Amp")
        axes[1].grid(True, alpha=0.3)

        # 子图3：状态编码为数值后绘制阶梯图
        status_code = status_series.map(
            {
                "ok": 0,
                "alarm": 1,
                "out_of_band": 2,
                "insufficient_data": 3,
                "invalid_input": 4,
            }
        ).fillna(5)
        axes[2].step(x_axis, status_code, where="mid", color="#5b6c5d", linewidth=1.6)
        axes[2].set_xlabel("Window ID")
        axes[2].set_ylabel("Status")
        axes[2].set_yticks([0, 1, 2, 3, 4])
        axes[2].set_yticklabels(
            ["ok", "alarm", "out_of_band", "insufficient", "invalid"]
        )
        axes[2].grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def plot_window_spectrum(self, window_id: int, output_path: str) -> None:
        """单窗口时域+频域图，数据来自 pipeline 返回的 debug 字段，不重复做 FFT。"""
        import matplotlib.pyplot as plt

        if window_id >= len(self.raw_results):
            raise ValueError("window_id is out of range")

        raw_result = self.raw_results[window_id]
        debug = raw_result.get("debug")
        if debug is None:
            raise ValueError("requested window does not include spectrum debug data")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(
            2,
            1,
            figsize=(10, 7),
            gridspec_kw={"height_ratios": [1.0, 1.2]},
        )
        signal_ax, spectrum_ax = axes

        signal_ax.plot(
            debug["window_time_s"],
            debug["window_values"],
            color="#1b6ca8",
            linewidth=1.2,
        )
        signal_ax.set_ylabel("Signal")
        signal_ax.set_xlabel("Time in Window (s)")
        signal_ax.grid(True, alpha=0.3)

        spectrum_ax.plot(
            debug["spectrum_freqs"],
            debug["spectrum_amps"],
            color="#5b6c5d",
            linewidth=1.6,
        )
        # 目标频带用半透明色带标注
        if self.target_freq_range_hz is not None:
            spectrum_ax.axvspan(
                self.target_freq_range_hz[0],
                self.target_freq_range_hz[1],
                color="#f3d9b1",
                alpha=0.35,
            )

        metrics = raw_result["metrics"]
        self._plot_marker(
            spectrum_ax,
            freq_hz=metrics.get("overall_peak_freq_hz"),
            amplitude=metrics.get("overall_peak_amplitude"),
            color="#c84c09",
            label="Overall Peak",
            text_offset=(8.0, 8.0),
        )
        self._plot_marker(
            spectrum_ax,
            freq_hz=metrics.get("dominant_freq_hz"),
            amplitude=metrics.get("peak_amplitude"),
            color="#8f2d56",
            label="Band Peak",
            text_offset=(8.0, -18.0),
        )
        if spectrum_ax.get_legend_handles_labels()[0]:
            spectrum_ax.legend(loc="upper right")

        spectrum_ax.set_xlabel("Frequency (Hz)")
        spectrum_ax.set_ylabel("Amplitude")
        spectrum_ax.grid(True, alpha=0.3)

        start_time = raw_result["window"].get("start_time")
        end_time = raw_result["window"].get("end_time")
        layout_rect = None
        if start_time is not None and end_time is not None:
            fig.suptitle(f"Window {window_id}: {start_time} to {end_time}")
            layout_rect = [0.0, 0.0, 1.0, 0.97]

        if layout_rect is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=layout_rect)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        plt.close(fig)
