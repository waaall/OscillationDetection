from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from src.core.detection_pipeline import DetectionPipeline


# 离线 CSV 输出 schema，以及从 DetectionPipeline 结果中提取值的路径
CSV_RESULT_FIELD_PATHS: dict[str, tuple[str, ...]] = {
    "window_id": ("window", "window_id"),
    "status": ("status",),
    "reason": ("reason",),
    "start_index": ("window", "start_index"),
    "end_index": ("window", "end_index"),
    "start_time": ("window", "start_time"),
    "end_time": ("window", "end_time"),
    "sample_count": ("window", "sample_count"),
    "sampling_rate_hz": ("window", "sampling_rate_hz"),
    "dominant_freq_hz": ("metrics", "dominant_freq_hz"),
    "dominant_period_s": ("metrics", "dominant_period_s"),
    "peak_amplitude": ("metrics", "peak_amplitude"),
    "overall_peak_freq_hz": ("metrics", "overall_peak_freq_hz"),
    "overall_peak_amplitude": ("metrics", "overall_peak_amplitude"),
    "threshold": ("metrics", "threshold"),
    "target_band_low_hz": ("meta", "target_band_low_hz"),
    "target_band_high_hz": ("meta", "target_band_high_hz"),
}
CSV_RESULT_COLUMNS = list(CSV_RESULT_FIELD_PATHS)

# 离线回放默认配置，可被 JSON 文件或构造函数参数覆盖
DEFAULT_CONFIG: dict[str, Any] = {
    "input": {
        "csv_path": "csv-data/input.csv",
        "time_column": "timestamp",
        "value_column": "signal",
        "timestamp_format": "%Y-%m-%d %H:%M:%S.%f",
        "has_timestamp": True,
    },
    "analysis": {
        "sampling_rate_hz": 1.0,
        "target_freq_range_hz": [0.0167, 0.2],
        "window_duration_s": 180.0,
        "amplitude_threshold": 0.5,
        "allow_resample": False,
    },
    "replay": {
        "step_duration_s": 1.0,
    },
    "output": {
        "result_csv_path": "csv-data/output.csv",
        "include_plot": True,
        "plot_dir": "plots",
    },
    "logging": {
        "log_file": "./log/csv_replay.log",
        "log_level": "INFO",
    },
}


def _resolve_option(
    explicit_value: Optional[Any],
    configured_value: Optional[Any],
    default_value: Any,
) -> Any:
    """三级优先级：构造函数显式传参 > JSON 配置文件 > 硬编码默认值。"""
    if explicit_value is not None:
        return explicit_value
    if configured_value is not None:
        return configured_value
    return default_value


def _extract_nested_value(data: dict[str, Any], path: Sequence[str]) -> Any:
    value: Any = data
    for key in path:
        value = value[key]
    return value


class ConfigLoader:

    @staticmethod
    def load(config_path: str) -> dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def create_default(output_path: str) -> None:
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(DEFAULT_CONFIG, file, ensure_ascii=False, indent=2)


class CsvReplay:
    """
    本类负责文件读写和配置管理，检测算法调用 DetectionPipeline
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        sampling_rate_hz: Optional[float] = None,
        target_freq_range_hz: Optional[tuple[float, float]] = None,
        window_duration_s: Optional[float] = None,
        amplitude_threshold: Optional[float] = None,
        allow_resample: Optional[bool] = None,
        step_duration_s: Optional[float] = None,
        include_plot: Optional[bool] = None,
        plot_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        log_level: Optional[str] = None,
    ) -> None:
        self.config: dict[str, Any] = {}
        if config_path:
            self.config = ConfigLoader.load(config_path)

        input_cfg = self.config.get("input", {})
        analysis_cfg = self.config.get("analysis", {})
        replay_cfg = self.config.get("replay", {})
        output_cfg = self.config.get("output", {})
        logging_cfg = self.config.get("logging", {})

        self.input_csv_path = input_cfg.get("csv_path")
        self.time_column = input_cfg.get("time_column", "timestamp")
        self.value_column = input_cfg.get("value_column", "signal")
        self.timestamp_format = input_cfg.get("timestamp_format")
        self.has_timestamp = bool(input_cfg.get("has_timestamp", True))

        self.sampling_rate_hz = _resolve_option(
            sampling_rate_hz,
            analysis_cfg.get("sampling_rate_hz"),
            None,
        )
        self.target_freq_range_hz = tuple(
            _resolve_option(
                target_freq_range_hz,
                analysis_cfg.get("target_freq_range_hz"),
                (0.0167, 0.2),
            )
        )
        self.window_duration_s = float(
            _resolve_option(
                window_duration_s,
                analysis_cfg.get("window_duration_s"),
                180.0,
            )
        )
        self.amplitude_threshold = float(
            _resolve_option(
                amplitude_threshold,
                analysis_cfg.get("amplitude_threshold"),
                0.5,
            )
        )
        self.allow_resample = bool(
            _resolve_option(
                allow_resample,
                analysis_cfg.get("allow_resample"),
                False,
            )
        )
        self.step_duration_s = float(
            _resolve_option(
                step_duration_s,
                replay_cfg.get("step_duration_s"),
                1.0,
            )
        )
        self.result_csv_path = output_cfg.get("result_csv_path")
        self.include_plot = bool(
            _resolve_option(
                include_plot,
                output_cfg.get("include_plot"),
                False,
            )
        )
        self.plot_dir = _resolve_option(
            plot_dir,
            output_cfg.get("plot_dir"),
            "plots",
        )
        resolved_log_file = _resolve_option(
            log_file,
            logging_cfg.get("log_file"),
            None,
        )
        resolved_log_level = str(
            _resolve_option(
                log_level,
                logging_cfg.get("log_level"),
                "INFO",
            )
        ).upper()
        self._setup_logger(resolved_log_file, resolved_log_level)

        # 将离线层参数注入检测流程层；include_plot 映射为 include_spectrum
        self.pipeline = DetectionPipeline(
            target_freq_range_hz=self.target_freq_range_hz,
            window_duration_s=self.window_duration_s,
            amplitude_threshold=self.amplitude_threshold,
            sampling_rate_hz=self.sampling_rate_hz,
            allow_resample=self.allow_resample,
            include_spectrum=self.include_plot,
        )
        self.last_results: list[dict[str, Any]] = []

    def _setup_logger(self, log_file: Optional[str], log_level: str) -> None:
        level = getattr(logging, log_level.upper(), None)
        if not isinstance(level, int):
            raise ValueError(f"invalid log level: {log_level}")

        self.logger = logging.getLogger("CsvReplay")
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self.logger.addHandler(console_handler)

        if log_file:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(
                logging.Formatter(
                    "[%(asctime)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self.logger.addHandler(file_handler)

        self.logger.propagate = False

    def load_csv(self, csv_path: Optional[str] = None) -> pd.DataFrame:
        resolved_csv_path = csv_path or self.input_csv_path
        if not resolved_csv_path:
            raise ValueError("input csv path is required")
        if not os.path.exists(resolved_csv_path):
            raise FileNotFoundError(f"CSV file not found: {resolved_csv_path}")

        self.logger.info("loading csv: %s", resolved_csv_path)
        return pd.read_csv(resolved_csv_path)

    def analyze_dynamic(self, df: pd.DataFrame) -> pd.DataFrame:
        """从 DataFrame 提取信号列和可选时间戳列，委托 DetectionPipeline 执行滑窗分析。"""
        if self.value_column not in df.columns:
            self.last_results = [self._invalid_result("empty_signal_after_cleanup")]
            return self.results_to_frame(self.last_results)

        timestamps: Optional[Sequence[Any]] = None
        if self.has_timestamp:
            if self.time_column not in df.columns:
                self.last_results = [self._invalid_result("timestamp_parse_failed")]
                return self.results_to_frame(self.last_results)
            timestamps = df[self.time_column].tolist()

        self.last_results = self.pipeline.analyze_samples(
            df[self.value_column].tolist(),
            sampling_rate_hz=self.sampling_rate_hz,
            timestamps=timestamps,
            step_duration_s=self.step_duration_s,
            source_mode="csv",
            timestamp_format=self.timestamp_format,
        )
        return self.results_to_frame(self.last_results)

    def results_to_frame(self, results: Sequence[dict[str, Any]]) -> pd.DataFrame:
        """将 DetectionPipeline 返回的嵌套 dict 列表展平为标准输出列的 DataFrame。"""
        records: list[dict[str, Any]] = []
        for result in results:
            records.append(
                {
                    column: _extract_nested_value(result, path)
                    for column, path in CSV_RESULT_FIELD_PATHS.items()
                }
            )

        return pd.DataFrame(records)

    def save_results(
        self,
        results_df: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> None:
        resolved_output_path = output_path or self.result_csv_path
        if not resolved_output_path:
            raise ValueError("result csv path is required")

        output_dir = os.path.dirname(resolved_output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 按标准列顺序写出，确保输出 schema 一致
        results_df = results_df.reindex(columns=CSV_RESULT_COLUMNS)
        results_df.to_csv(resolved_output_path, index=False)
        self.logger.info("saved results: %s", resolved_output_path)

    def generate_plots(
        self,
        results_df: pd.DataFrame,
        plot_dir: Optional[str] = None,
    ) -> None:
        if not self.include_plot or not self.last_results:
            return

        # 延迟导入：仅在需要绘图时加载 matplotlib 依赖
        from src.offline.result_visualizer import ResultVisualizer

        resolved_plot_dir = Path(plot_dir or self.plot_dir)
        resolved_plot_dir.mkdir(parents=True, exist_ok=True)

        visualizer = ResultVisualizer(
            results_df=results_df,
            raw_results=self.last_results,
            amplitude_threshold=self.amplitude_threshold,
            target_freq_range_hz=self.target_freq_range_hz,
        )
        visualizer.plot_summary(str(resolved_plot_dir / "dominant_frequency.png"))

        # debug 字段仅在 include_spectrum=True 时由 pipeline 填充
        if any("debug" in result for result in self.last_results):
            visualizer.plot_window_spectrum(
                0,
                str(resolved_plot_dir / "window_0_spectrum.png"),
            )

    def run_pipeline(
        self,
        input_csv: Optional[str] = None,
        output_csv: Optional[str] = None,
    ) -> pd.DataFrame:
        """一键执行：读 CSV → 滑窗检测 → 保存结果 → 可选绘图。"""
        df = self.load_csv(input_csv)
        results_df = self.analyze_dynamic(df)
        self.save_results(results_df, output_csv)
        self.generate_plots(results_df)
        return results_df

    def _invalid_result(self, reason: str) -> dict[str, Any]:
        return {
            "status": "invalid_input",
            "reason": reason,
            "window": {
                "window_id": 0,
                "start_index": 0,
                "end_index": 0,
                "start_time": None,
                "end_time": None,
                "sample_count": 0,
                "sampling_rate_hz": float(self.sampling_rate_hz or 0.0),
            },
            "metrics": {
                "dominant_freq_hz": None,
                "dominant_period_s": None,
                "peak_amplitude": None,
                "overall_peak_freq_hz": None,
                "overall_peak_amplitude": None,
                "threshold": self.amplitude_threshold,
            },
            "meta": {
                "target_band_low_hz": self.target_freq_range_hz[0],
                "target_band_high_hz": self.target_freq_range_hz[1],
                "source_mode": "csv",
            },
        }


def main() -> None:
    config_path = Path(__file__).with_name("csv_replay.default.json")
    analyzer = CsvReplay(config_path=str(config_path))
    results_df = analyzer.run_pipeline()
    print(results_df.head())


if __name__ == "__main__":
    main()
