"""Offline CSV replay, visualization, and signal generation helpers."""

from __future__ import annotations

from typing import Any

__all__ = ["CsvReplay", "ResultVisualizer", "SignalGenerator"]


def __getattr__(name: str) -> Any:
    if name == "CsvReplay":
        from src.offline.csv_replay import CsvReplay

        return CsvReplay
    if name == "ResultVisualizer":
        from src.offline.result_visualizer import ResultVisualizer

        return ResultVisualizer
    if name == "SignalGenerator":
        from src.offline.signal_generator import SignalGenerator

        return SignalGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
