import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging
import os
import sys
from typing import Optional
from pathlib import Path

# 添加项目根目录到Python路径（支持直接运行本文件）
_current_file = Path(__file__).resolve()
_project_root = _current_file.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.core.FFT_analyzer import FFTAnalyzer   # noqa: E402

# FFT 幅值归一化常量 (因为单边FFT需要乘2来补偿负频率部分的能量;直流分量和奈奎斯特频率,不应该乘)
FFT_AMP_NORMAL_FACTOR = 2


class OscillationDetection:
    def __init__(self, csv_file: str, window_size: int = 1000, overlap_ratio: float = 0.5,
                 sampling_rate: float = 1000, threshold: float = 0.3, col_name: str = "值",
                 log_file: Optional[str] = "./log/oscillation_tester.log"):
        """
        :param csv_file: CSV 文件路径，假设有一列 'signal'
        :param window_size: 窗口大小 (点数)
        :param overlap_ratio: 窗口重叠比例 (0~1)，例如 0.5 表示滑动 50%
        :param sampling_rate: 采样率 Hz
        :param threshold: 检测阈值
        :param col_name: CSV文件中的列名
        :param log_file: 日志文件路径
        """
        # 设置日志
        self._setup_logger(log_file)

        # 参数验证
        self._validate_parameters(csv_file, window_size, overlap_ratio, sampling_rate, threshold, col_name)

        # 加载数据
        self._load_data(csv_file, col_name)

        # 初始化参数
        self.window_size = window_size
        self.step_size = int(window_size * (1 - overlap_ratio))
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        # 初始化检测器
        self.detector = FFTAnalyzer(window_size=window_size,
                                    sampling_rate=sampling_rate,
                                    log_file=log_file)

        # 获取频率范围
        self.min_freq = self.detector._min_freq
        self.max_freq = self.detector._max_freq

        # 初始化图形组件
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.line1 = None
        self.line2 = None
        self.trigger_text = None

        self.logger.info("OscillationDetectionTester 初始化完成")

    def _setup_logger(self, log_file: Optional[str]):
        """设置日志系统"""
        self.logger = logging.getLogger("OscillationDetectionTester")
        self.logger.setLevel(logging.INFO)

        # 清除现有的处理器
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # 文件处理器
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(console_formatter)
                self.logger.addHandler(file_handler)
                self.logger.info(f"日志文件设置为: {log_file}")
            except Exception as e:
                self.logger.warning(f"无法创建日志文件 {log_file}: {e}")

    def _validate_parameters(self, csv_file: str, window_size: int, overlap_ratio: float,
                             sampling_rate: float, threshold: float, col_name: str):
        """参数验证"""
        # 检查CSV文件
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV文件不存在: {csv_file}")

        # 检查窗口大小
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size 必须是正整数")

        # 检查重叠比例
        if not isinstance(overlap_ratio, (int, float)) or overlap_ratio <= 0 or overlap_ratio > 1:
            raise ValueError("overlap_ratio 必须是大于0且小于等于1的数值")

        # 检查采样率
        if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
            raise ValueError("sampling_rate 必须是正数")

        # 检查阈值
        if not isinstance(threshold, (int, float)) or threshold < 0:
            raise ValueError("threshold 必须是非负数")

        # 检查列名
        if not isinstance(col_name, str) or not col_name.strip():
            raise ValueError("col_name 必须是非空字符串")

        self.logger.info("参数验证通过")

    def _load_data(self, csv_file: str, col_name: str):
        """加载数据"""
        try:
            self.data = pd.read_csv(csv_file)
            self.logger.info(f"成功加载CSV文件: {csv_file}, 形状: {self.data.shape}")

            if col_name not in self.data.columns:
                available_columns = ", ".join(self.data.columns.tolist())
                raise ValueError(f"CSV文件中不存在列 '{col_name}'。可用列: {available_columns}")

            self.signal = self.data[col_name].values

            # 检查数据有效性
            if len(self.signal) == 0:
                raise ValueError("信号数据为空")

            # 检查是否有足够的数据
            if len(self.signal) < self.window_size if hasattr(self, 'window_size') else len(self.signal) < 10:
                self.logger.warning(f"信号长度 ({len(self.signal)}) 可能不足以进行有效分析")

            # 统计信息
            valid_count = np.count_nonzero(~np.isnan(self.signal))
            nan_count = np.count_nonzero(np.isnan(self.signal))

            self.logger.info(f"信号数据加载完成: 总长度={len(self.signal)}, 有效数据={valid_count}, NaN数量={nan_count}")

            if nan_count > 0:
                self.logger.warning(f"数据中存在 {nan_count} 个NaN值，可能影响分析结果")

        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            raise

    def _setup_canvas(self):
        """设置画布和图形组件"""
        try:
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))

            # 时域图设置
            self.line1, = self.ax1.plot([], [], '#369a62', linewidth=1.5)
            self.ax1.set_xlabel('Time (s)')
            self.ax1.set_ylabel('Amplitude')
            self.ax1.grid(True, alpha=0.3)

            # 频域图设置
            self.line2, = self.ax2.plot([], [], '#844784', linewidth=1.5)
            self.ax2.set_xlabel('Frequency (Hz)')
            self.ax2.set_ylabel('Amplitude')
            self.ax2.grid(True, alpha=0.3)

            # 状态文本
            self.trigger_text = self.ax1.text(0.02, 0.9, "", transform=self.ax1.transAxes,
                                              fontsize=12, color="red", weight="bold",
                                              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

            plt.tight_layout()
            self.logger.info("画布设置完成")

        except Exception as e:
            self.logger.error(f"设置画布失败: {e}")
            raise

    def _update(self, frame_idx):
        """动画更新函数"""
        try:
            start = frame_idx * self.step_size
            end = start + self.window_size

            if end > len(self.signal):
                self.logger.info("数据处理完成")
                return self.line1, self.line2, self.trigger_text

            window_data = self.signal[start:end]

            # 检查窗口数据有效性
            if len(window_data) == 0:
                self.logger.warning(f"窗口 {frame_idx} 数据为空")
                return self.line1, self.line2, self.trigger_text

            # 处理NaN值
            if np.any(np.isnan(window_data)):
                self.logger.warning(f"窗口 {frame_idx} 包含NaN值，使用前向填充")
                window_data = pd.Series(window_data).fillna(method='ffill').fillna(0).values

            # --------- 时域显示 ---------
            t = np.arange(len(window_data)) / self.sampling_rate
            self.line1.set_data(t, window_data)
            self.ax1.set_xlim(0, self.window_size / self.sampling_rate)

            # 安全的ylim设置
            y_min, y_max = np.min(window_data), np.max(window_data)
            if y_min == y_max:
                y_margin = abs(y_min * 0.1) if y_min != 0 else 1
                self.ax1.set_ylim(y_min - y_margin, y_max + y_margin)
            else:
                y_range = y_max - y_min
                self.ax1.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

            # self.ax1.set_title(f"Time: {frame_idx + 1}")

            # --------- 频域显示 ---------
            try:
                fft_vals = np.fft.rfft(window_data)
                fft_freqs = np.fft.rfftfreq(len(window_data), d=1.0 / self.sampling_rate)
                # 幅值归一化 (单边FFT需要乘以归一化因子来补偿负频率部分的能量)
                amplitudes = FFT_AMP_NORMAL_FACTOR * np.abs(fft_vals) / len(window_data)

                # 频率范围过滤
                valid_idx = np.where((fft_freqs >= self.min_freq) & (fft_freqs <= self.max_freq))
                if len(valid_idx[0]) > 0:
                    fft_freqs = fft_freqs[valid_idx]
                    amplitudes = amplitudes[valid_idx]

                    self.line2.set_data(fft_freqs, amplitudes)
                    self.ax2.set_xlim(self.min_freq, self.max_freq)

                    max_amp = np.max(amplitudes)
                    self.ax2.set_ylim(0, max_amp * 1.2 if max_amp > 0 else 1)
                else:
                    self.logger.warning(f"窗口 {frame_idx} 无有效频率数据")

            except Exception as e:
                self.logger.error(f"FFT计算失败 (窗口 {frame_idx}): {e}")

            # self.ax2.set_title("Frequency Domain (FFT)")

            # --------- 振荡检测 ---------
            try:
                result, peak_freq, peak_amp, __ = self.detector.fft_analyze(window_data)
                is_trigger = bool(result and peak_amp is not None and peak_amp >= self.threshold)

                if is_trigger:
                    self.trigger_text.set_text(
                        f"Detected Oscillation!\nFrequency: {peak_freq:.2f}Hz; Amplitude: {peak_amp:.3f}"
                    )
                    self.trigger_text.set_color("red")
                    self.logger.info(
                        f"窗口 {frame_idx}: 检测到振荡 - 频率={peak_freq:.2f}Hz, 幅值={peak_amp:.3f}"
                    )
                else:
                    self.trigger_text.set_text("Normal")
                    self.trigger_text.set_color("green")

            except Exception as e:
                self.logger.error(f"振荡检测失败 (窗口 {frame_idx}): {e}")
                self.trigger_text.set_text("Detection Error")
                self.trigger_text.set_color("orange")

            return self.line1, self.line2, self.trigger_text

        except Exception as e:
            self.logger.error(f"更新动画失败 (窗口 {frame_idx}): {e}")
            return self.line1, self.line2, self.trigger_text

    def run(self, interval: int = 200):
        """
        启动动画
        :param interval: 每帧间隔 (ms)
        """
        try:
            # 验证参数
            if not isinstance(interval, int) or interval <= 0:
                raise ValueError("interval 必须是正整数")

            # 检查数据长度
            total_frames = len(self.signal) // self.step_size
            if total_frames <= 0:
                raise ValueError("数据长度不足，无法生成动画帧")

            self.logger.info(f"开始动画，总帧数: {total_frames}, 帧间隔: {interval}ms")

            # 设置画布
            self._setup_canvas()

            # 创建动画
            ani = animation.FuncAnimation(
                self.fig,
                self._update,
                frames=range(total_frames),
                interval=interval,
                blit=False,
                repeat=False
            )

            self.logger.info("动画启动成功")
            plt.show()

        except Exception as e:
            self.logger.error(f"启动动画失败: {e}")
            raise

    def analyze_static(self, start_window: int = 0, num_windows: int = 10):
        """
        静态分析模式：分析指定数量的窗口
        :param start_window: 起始窗口索引
        :param num_windows: 分析的窗口数量
        """
        try:
            total_frames = len(self.signal) // self.step_size
            end_window = min(start_window + num_windows, total_frames)

            self.logger.info(f"开始静态分析: 窗口 {start_window} 到 {end_window-1}")

            results = []
            for frame_idx in range(start_window, end_window):
                start = frame_idx * self.step_size
                end = start + self.window_size

                if end > len(self.signal):
                    break

                window_data = self.signal[start:end]

                # 处理NaN值
                if np.any(np.isnan(window_data)):
                    window_data = pd.Series(window_data).fillna(method='ffill').fillna(0).values

                # 检测振荡
                success, peak_freq, peak_amp, _ = self.detector.fft_analyze(window_data)
                is_trigger = bool(success and peak_amp is not None and peak_amp >= self.threshold)

                result = {
                    'window': frame_idx,
                    'start_idx': start,
                    'end_idx': end,
                    'is_oscillation': is_trigger,
                    'peak_frequency': peak_freq,
                    'peak_amplitude': peak_amp
                }
                results.append(result)

                if is_trigger:
                    self.logger.info(f"窗口 {frame_idx}: 振荡检测 - 频率={peak_freq:.2f}Hz, 幅值={peak_amp:.3f}")

            self.logger.info(f"静态分析完成，共分析 {len(results)} 个窗口")
            return results

        except Exception as e:
            self.logger.error(f"静态分析失败: {e}")
            raise


def basic_test():
    try:
        tester = OscillationDetection(
            csv_file="../csv-data/data.csv",
            window_size=100,
            overlap_ratio=0.2,
            sampling_rate=1000,
            threshold=5.0
        )

        # 可以选择运行动画或静态分析
        # 动画模式
        tester.run(interval=200)

        # 或者静态分析模式
        # results = tester.analyze_static(start_window=0, num_windows=50)
        # print(f"分析完成，检测到振荡的窗口数: {sum(1 for r in results if r['is_oscillation'])}")

    except Exception as e:
        print(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    basic_test()
