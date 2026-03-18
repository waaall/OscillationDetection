"""
按指定采样间隔筛选 CSV 时间序列数据。

用途:
- 从原始 CSV 中筛出满足目标采样间隔的连续数据段。
- 默认保留最长的连续规则片段, 方便后续直接用于离线检测。
- 可切换为保留全部规则片段, 并额外写入 segment_id 列。

用法:
python -m src.offline.filter_by_sampling_interval \
  csv-data/test.csv \
  --time-column timestamp \
  --sampling-rate-hz 1 \
  --timestamp-format "%Y-%m-%d %H:%M:%S"

python -m src.offline.filter_by_sampling_interval \
  csv-data/test.csv \
  csv-data/test.filtered.csv \
  --time-column timestamp \
  --sampling-rate-hz 1 \
  --mode all_segments
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import pandas as pd


FilterMode = Literal["longest_segment", "all_segments"]


@dataclass(frozen=True)
class SamplingFilterSummary:
    input_rows: int
    kept_rows: int
    dropped_rows: int
    invalid_gap_count: int
    segment_count: int
    longest_segment_length: int
    expected_interval_s: float
    tolerance_s: float
    mode: FilterMode


def _resolve_expected_interval_s(
    *,
    sampling_rate_hz: Optional[float],
    interval_seconds: Optional[float],
) -> float:
    if sampling_rate_hz is None and interval_seconds is None:
        raise ValueError("sampling_rate_hz or interval_seconds is required")
    if sampling_rate_hz is not None and interval_seconds is not None:
        raise ValueError("provide only one of sampling_rate_hz or interval_seconds")
    if sampling_rate_hz is not None:
        if sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        return 1.0 / sampling_rate_hz
    assert interval_seconds is not None
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    return float(interval_seconds)


def _collect_regular_segments(
    valid_step_mask: pd.Series,
    *,
    min_segment_length: int,
) -> list[tuple[int, int]]:
    if min_segment_length < 2:
        raise ValueError("min_segment_length must be >= 2")
    if valid_step_mask.empty:
        return []

    segments: list[tuple[int, int]] = []
    current_start = 0
    row_count = valid_step_mask.shape[0]

    for index in range(1, row_count):
        if not bool(valid_step_mask.iloc[index]):
            if index - current_start >= min_segment_length:
                segments.append((current_start, index - 1))
            current_start = index

    if row_count - current_start >= min_segment_length:
        segments.append((current_start, row_count - 1))

    return segments


def filter_dataframe_by_sampling_interval(
    df: pd.DataFrame,
    *,
    time_column: str,
    sampling_rate_hz: Optional[float] = None,
    interval_seconds: Optional[float] = None,
    timestamp_format: Optional[str] = None,
    tolerance_seconds: float = 1e-6,
    mode: FilterMode = "longest_segment",
    min_segment_length: int = 2,
    sort_by_time: bool = False,
) -> tuple[pd.DataFrame, SamplingFilterSummary]:
    if time_column not in df.columns:
        raise ValueError(f"time column not found: {time_column}")
    if tolerance_seconds < 0:
        raise ValueError("tolerance_seconds must be >= 0")
    if mode not in {"longest_segment", "all_segments"}:
        raise ValueError(f"unsupported mode: {mode}")

    expected_interval_s = _resolve_expected_interval_s(
        sampling_rate_hz=sampling_rate_hz,
        interval_seconds=interval_seconds,
    )

    working_df = df.copy()
    parsed_timestamps = pd.to_datetime(
        working_df[time_column],
        format=timestamp_format,
        errors="coerce",
    )
    if parsed_timestamps.isna().any():
        raise ValueError("timestamp_parse_failed")

    working_df["_parsed_timestamp"] = parsed_timestamps
    if sort_by_time:
        working_df = working_df.sort_values("_parsed_timestamp").reset_index(drop=True)
    elif not working_df["_parsed_timestamp"].is_monotonic_increasing:
        raise ValueError("timestamps must be monotonic increasing or enable sort_by_time")

    step_seconds = working_df["_parsed_timestamp"].diff().dt.total_seconds()
    valid_step_mask = (step_seconds - expected_interval_s).abs() <= tolerance_seconds
    valid_step_mask.iloc[0] = False

    regular_segments = _collect_regular_segments(
        valid_step_mask,
        min_segment_length=min_segment_length,
    )
    if not regular_segments:
        raise ValueError("no_regular_segment_found")

    longest_segment_length = max(end - start + 1 for start, end in regular_segments)
    invalid_gap_count = int((~valid_step_mask.iloc[1:]).sum())

    if mode == "longest_segment":
        selected_start, selected_end = max(
            regular_segments,
            key=lambda segment: (segment[1] - segment[0] + 1, -segment[0]),
        )
        filtered_df = working_df.iloc[selected_start: selected_end + 1].copy()
    else:
        segment_frames: list[pd.DataFrame] = []
        for segment_id, (start, end) in enumerate(regular_segments):
            segment_df = working_df.iloc[start: end + 1].copy()
            segment_df.insert(0, "segment_id", segment_id)
            segment_frames.append(segment_df)
        filtered_df = pd.concat(segment_frames, ignore_index=True)

    filtered_df = filtered_df.drop(columns="_parsed_timestamp").reset_index(drop=True)
    summary = SamplingFilterSummary(
        input_rows=int(df.shape[0]),
        kept_rows=int(filtered_df.shape[0]),
        dropped_rows=int(df.shape[0] - filtered_df.shape[0]),
        invalid_gap_count=invalid_gap_count,
        segment_count=len(regular_segments),
        longest_segment_length=longest_segment_length,
        expected_interval_s=expected_interval_s,
        tolerance_s=float(tolerance_seconds),
        mode=mode,
    )
    return filtered_df, summary


def filter_csv_by_sampling_interval(
    input_csv: str,
    output_csv: str,
    *,
    time_column: str,
    sampling_rate_hz: Optional[float] = None,
    interval_seconds: Optional[float] = None,
    timestamp_format: Optional[str] = None,
    tolerance_seconds: float = 1e-6,
    mode: FilterMode = "longest_segment",
    min_segment_length: int = 2,
    sort_by_time: bool = False,
) -> SamplingFilterSummary:
    df = pd.read_csv(input_csv)
    filtered_df, summary = filter_dataframe_by_sampling_interval(
        df,
        time_column=time_column,
        sampling_rate_hz=sampling_rate_hz,
        interval_seconds=interval_seconds,
        timestamp_format=timestamp_format,
        tolerance_seconds=tolerance_seconds,
        mode=mode,
        min_segment_length=min_segment_length,
        sort_by_time=sort_by_time,
    )

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    filtered_df.to_csv(output_csv, index=False)
    return summary


def _default_output_path(input_csv: str) -> str:
    input_path = Path(input_csv)
    return str(input_path.with_name(f"{input_path.stem}.regular.csv"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter CSV rows into regular timestamp segments. "
            "Default mode keeps only the longest continuous segment so the "
            "output itself remains regular-sampled."
        )
    )
    parser.add_argument("input_csv", help="input CSV path")
    parser.add_argument(
        "output_csv",
        nargs="?",
        help="output CSV path, default: <input>.regular.csv",
    )
    parser.add_argument(
        "--time-column",
        default="timestamp",
        help="timestamp column name",
    )
    parser.add_argument(
        "--sampling-rate-hz",
        type=float,
        help="expected sampling rate in Hz",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        help="expected interval in seconds",
    )
    parser.add_argument(
        "--timestamp-format",
        default=None,
        help="optional datetime format passed to pandas.to_datetime",
    )
    parser.add_argument(
        "--tolerance-seconds",
        type=float,
        default=1e-6,
        help="allowed absolute error when comparing interval seconds",
    )
    parser.add_argument(
        "--mode",
        choices=["longest_segment", "all_segments"],
        default="longest_segment",
        help="longest_segment is safer for downstream replay/detection",
    )
    parser.add_argument(
        "--min-segment-length",
        type=int,
        default=2,
        help="minimum row count required for a valid segment",
    )
    parser.add_argument(
        "--sort-by-time",
        action="store_true",
        help="sort by timestamp before filtering",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    output_csv = args.output_csv or _default_output_path(args.input_csv)

    try:
        summary = filter_csv_by_sampling_interval(
            input_csv=args.input_csv,
            output_csv=output_csv,
            time_column=args.time_column,
            sampling_rate_hz=args.sampling_rate_hz,
            interval_seconds=args.interval_seconds,
            timestamp_format=args.timestamp_format,
            tolerance_seconds=args.tolerance_seconds,
            mode=args.mode,
            min_segment_length=args.min_segment_length,
            sort_by_time=args.sort_by_time,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[INFO] input rows: {summary.input_rows}")
    print(f"[INFO] kept rows: {summary.kept_rows}")
    print(f"[INFO] dropped rows: {summary.dropped_rows}")
    print(f"[INFO] invalid gaps: {summary.invalid_gap_count}")
    print(f"[INFO] regular segments: {summary.segment_count}")
    print(f"[INFO] longest segment rows: {summary.longest_segment_length}")
    print(f"[INFO] expected interval s: {summary.expected_interval_s}")
    print(f"[INFO] output csv: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
