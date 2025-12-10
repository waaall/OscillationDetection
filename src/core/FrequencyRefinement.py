import logging
from typing import Optional, Tuple, Union

import numpy as np


class FrequencyRefinement:
    """频率精细化模块 - 基于最小二乘拟合与一维搜索。"""

    def __init__(self,
                 search_range: Optional[Union[float, Tuple[float, float]]] = None,
                 method: str = "minimize_scalar",
                 tolerance: float = 1e-9,
                 max_iterations: int = 100,
                 step_size: float = 0.0005,
                 retry_on_failure: bool = True,
                 max_grid_points: int = 2000,
                 logger: Optional[logging.Logger] = None):
        """
        :param search_range: 搜索范围（Hz）。float 表示 ±range；(low, high) 表示绝对范围。
        :param method: 'minimize_scalar' (默认) 或 'grid_search'。
        :param tolerance: 频率搜索容差。
        :param max_iterations: 一维搜索最大迭代次数。
        :param step_size: 网格搜索步长（Hz）。
        :param retry_on_failure: 搜索失败时是否扩大一档后重试。
        :param : 网格搜索最大采样点数（防止过密）。
        :param logger: 可选 logger；若为空则使用模块 logger。
        """
        self._search_range = search_range
        self._method = method
        self._tolerance = tolerance
        self._max_iterations = max_iterations
        self._step_size = step_size
        self._retry_on_failure = retry_on_failure
        self._max_grid_points = max_grid_points
        self.logger = logger or logging.getLogger(__name__)
        self._min_samples = 160

    def refine(self,
               signal: np.ndarray,
               sampling_rate: float,
               freq_initial: float,
               return_all_params: bool = False):
        """
        主接口：频率精细化。

        :param signal: 输入信号（单通道）。
        :param sampling_rate: 采样率 (Hz)。
        :param freq_initial: 初始频率估计 (Hz)。
        :param return_all_params: 是否返回 (freq, amp, phase, dc, residual)。
        :return: 频率或完整参数元组；失败时返回 None。
        """
        if signal is None or len(signal) < self._min_samples:
            self.logger.warning("频率精化跳过：数据长度不足（%d/%d）",
                                0 if signal is None else len(signal),
                                self._min_samples)
            return None

        t = np.arange(len(signal), dtype=np.float64) / float(sampling_rate)
        freq_resolution = float(sampling_rate) / len(signal)
        bounds = self._calculate_search_range(freq_initial, freq_resolution)

        result = self._run_search(signal, t, bounds)
        if result is None and self._retry_on_failure:
            expanded_bounds = self._expand_range(freq_initial, bounds)
            self.logger.info("首次精化失败，扩大搜索范围到 [%.6f, %.6f] Hz 重试",
                             expanded_bounds[0], expanded_bounds[1])
            result = self._run_search(signal, t, expanded_bounds)

        if result is None:
            return None

        freq, amp, phase, dc, residual = result
        return result if return_all_params else freq

    def _run_search(self,
                    signal: np.ndarray,
                    t: np.ndarray,
                    bounds: Tuple[float, float]):
        """根据配置选择搜索方法，必要时自动降级。"""
        method = self._method.lower()

        if method == "minimize_scalar":
            result = self._refine_minimize_scalar(signal, t, bounds)
            if result is None:
                self.logger.warning("minimize_scalar 精化失败或 SciPy 不可用，降级为 grid_search")
                result = self._refine_grid_search(signal, t, bounds)
        elif method == "grid_search":
            result = self._refine_grid_search(signal, t, bounds)
        else:
            self.logger.warning("未知方法 %s，使用 grid_search", method)
            result = self._refine_grid_search(signal, t, bounds)

        if result is None:
            return None

        freq, amp, phase, dc, residual = result
        if not self._validate_result(freq, residual):
            return None

        return freq, amp, phase, dc, residual

    def _calculate_search_range(self,
                                freq_initial: float,
                                freq_resolution: float) -> Tuple[float, float]:
        """自适应计算搜索范围，支持软限制。"""
        if isinstance(self._search_range, (tuple, list)) and len(self._search_range) == 2:
            low, high = float(self._search_range[0]), float(self._search_range[1])
            return max(low, 0.0), max(high, low + 1e-6)

        if isinstance(self._search_range, (int, float)):
            base_range = float(self._search_range)
        else:
            base_range = max(5 * freq_resolution, 0.05)

        search_range = min(base_range, 0.5) if 49.0 <= freq_initial <= 51.0 else base_range
        low = max(freq_initial - search_range, 0.0)
        high = freq_initial + search_range

        if high <= low:
            high = low + max(search_range, 1e-6)

        return low, high

    def _expand_range(self,
                      freq_initial: float,
                      bounds: Tuple[float, float],
                      factor: float = 2.0) -> Tuple[float, float]:
        """失败后扩大一档搜索范围。"""
        low, high = bounds
        span = (high - low) * factor / 2.0
        new_low = max(freq_initial - span, 0.0)
        new_high = freq_initial + span
        return new_low, new_high

    def _refine_minimize_scalar(self,
                                signal: np.ndarray,
                                t: np.ndarray,
                                bounds: Tuple[float, float]):
        """使用 SciPy 的 minimize_scalar 做一维优化。"""
        try:
            from scipy.optimize import minimize_scalar
        except ImportError:
            return None

        def objective(freq: float) -> float:
            _, _, _, residual = self._linear_least_squares(signal, t, freq)
            return residual

        res = minimize_scalar(
            objective,
            bounds=bounds,
            method="bounded",
            options={"xatol": self._tolerance, "maxiter": self._max_iterations}
        )

        if not res.success:
            return None

        amp, phase, dc, residual = self._linear_least_squares(signal, t, res.x)
        return res.x, amp, phase, dc, residual

    def _refine_grid_search(self,
                            signal: np.ndarray,
                            t: np.ndarray,
                            bounds: Tuple[float, float]):
        """简单网格搜索 + 二次插值微调。"""
        low, high = bounds
        span = high - low
        if span <= 0:
            return None

        step = self._step_size if self._step_size and self._step_size > 0 else span / 200.0
        points = int(np.ceil(span / step)) + 1
        if points > self._max_grid_points:
            points = self._max_grid_points
            step = span / max(points - 1, 1)

        freqs = np.linspace(low, high, points)
        residuals = np.empty_like(freqs)
        best_idx = None
        best_residual = np.inf
        best_params = (None, None, None)

        for idx, f in enumerate(freqs):
            amp, phase, dc, residual = self._linear_least_squares(signal, t, f)
            residuals[idx] = residual
            if residual < best_residual:
                best_residual = residual
                best_idx = idx
                best_params = (amp, phase, dc)

        if best_idx is None:
            return None

        freq_best = freqs[best_idx]
        amp_best, phase_best, dc_best = best_params

        # 二次插值微调
        if 0 < best_idx < len(freqs) - 1:
            f1, f2, f3 = freqs[best_idx - 1:best_idx + 2]
            r1, r2, r3 = residuals[best_idx - 1:best_idx + 2]
            denom = (r1 - 2 * r2 + r3)
            if abs(denom) > 1e-12:
                delta = 0.5 * (r1 - r3) / denom
                refined_freq = f2 + delta * (f2 - f1)
                if low <= refined_freq <= high:
                    freq_best = refined_freq
                    amp_best, phase_best, dc_best, best_residual = self._linear_least_squares(
                        signal, t, freq_best
                    )

        return freq_best, amp_best, phase_best, dc_best, best_residual

    def _linear_least_squares(self,
                              signal: np.ndarray,
                              t: np.ndarray,
                              freq: float):
        """给定频率，线性最小二乘求解振幅/相位/DC 以及残差。"""
        omega = 2 * np.pi * freq
        sin_comp = np.sin(omega * t)
        cos_comp = np.cos(omega * t)
        H = np.column_stack((sin_comp, cos_comp, np.ones_like(t)))

        theta, *_ = np.linalg.lstsq(H, signal, rcond=None)
        a, b, dc = theta
        amplitude = float(np.hypot(a, b))
        phase = float(np.arctan2(b, a))
        fit = H @ theta
        residual = float(np.sqrt(np.mean((signal - fit) ** 2)))

        return amplitude, phase, float(dc), residual

    def _validate_result(self, freq: float, residual: float) -> bool:
        """简单结果校验。"""
        if not np.isfinite(freq) or not np.isfinite(residual):
            self.logger.warning("精化结果非法：freq=%s, residual=%s", freq, residual)
            return False
        if freq <= 0:
            self.logger.warning("精化结果频率非正：freq=%.6f", freq)
            return False
        return True
