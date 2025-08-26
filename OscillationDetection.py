import os
import sys
import logging
from typing import Tuple, Optional

import numpy as np


class OscillationDetection:
    """
    振荡检测类: 通过DFT/FFT在时域数据中检测振荡频率。
    """

    def __init__(self,
                 window_size: int = 60,
                 sampling_rate: float = 1.0,
                 threshold: float = 0.5,
                 log_file: Optional[str] = None):
        """
        :param window_size: 窗口大小（数据点数），默认 60
        :param sampling_rate: 采样率 (Hz)，默认 1 Hz
        :param threshold: 检测阈值（幅值归一化后）
        :param plot_comparison: 是否绘制对比图
        :param log_file: 日志文件路径（如果为 None，则只输出到控制台）
        """
        # 检查窗口大小
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size 必须是正整数")
        # 检查采样率
        if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
            raise ValueError("sampling_rate 必须是正数")
        # 检查阈值
        if not isinstance(threshold, (int, float)) or threshold < 0:
            raise ValueError("threshold 必须是非负数")

        self._window_size = window_size
        self._sampling_rate = sampling_rate
        self._threshold = threshold

        # 设置日志
        self._setup_logger(log_file)
        # 频率相关参数
        self._set_up_frequency()

    def _set_up_frequency(self):
        nyquist_freq = self._sampling_rate / 2.0
        self._max_freq = min(nyquist_freq * 0.95, 500)  # 留 5% 余量，不超过 500Hz
        self._min_freq = (1.0 / (self._window_size / self._sampling_rate)) * 2
        if self._min_freq >= self._max_freq:
            self.logger.error("最小频率 >= 最大频率, 请检查参数设置")
            return

        self.logger.info("OscillationDetection 初始化完成: "
                         f"window_size={self._window_size}, sampling_rate={self._sampling_rate}, "
                         f"min_freq={self._min_freq:.2f}Hz, max_freq={self._max_freq:.2f}Hz")

    def _setup_logger(self, log_file: Optional[str]):
        """配置日志系统"""
        self.logger = logging.getLogger("OscillationDetection")
        self.logger.setLevel(logging.INFO)

        # 清理旧的 handler
        self.logger.handlers.clear()

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # 文件处理器（可选）
        if log_file is not None:
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                               datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        self.logger.propagate = False  # 避免重复输出

    def detect(self, data: np.ndarray, PLOT: bool = False) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        振荡检测入口函数

        :param data: 输入时域数据 (float 数组)
        :return: (是否检测到, 频率, 幅值)，若未检测到返回 (False, None, None)
        """
        if len(data) < self._window_size:
            self.logger.error("输入数据不足窗口大小，跳过检测")
            return False, None, None
        elif len(data) > self._window_size:
            self.logger.warning("输入数据超过最大窗口大小，进行截断")

        # 截取窗口
        windowed_data = data[-self._window_size:]


        # 执行FFT
        freqs = np.fft.rfftfreq(len(windowed_data), d=1.0 / self._sampling_rate)
        # 幅值归一化 (乘2是因为幅值在频域中被压缩了一半)
        amplitudes = 2 * np.abs(np.fft.rfft(windowed_data)) / len(windowed_data)

        # 限制频率范围
        valid_idx = np.where((freqs >= self._min_freq) & (freqs <= self._max_freq))
        valid_freqs = freqs[valid_idx]
        valid_amplitudes = amplitudes[valid_idx]

        # 边界保护
        if valid_freqs.size == 0:
            self.logger.warning("无有效频率区间，跳过检测")
            return False, None, None

        # 找最大峰值
        peak_idx = np.argmax(valid_amplitudes)
        peak_freq = valid_freqs[peak_idx]
        peak_amp = valid_amplitudes[peak_idx]

        self.logger.debug(f"检测频率范围 {self._min_freq:.2f}-{self._max_freq:.2f}Hz, "
                          f"峰值={peak_amp:.4f} @ {peak_freq:.2f}Hz")

        if PLOT:
            self.__plot_comparison(windowed_data, freqs, amplitudes, "./")

        if peak_amp >= self._threshold:
            self.logger.info(f"检测到振荡: 频率={peak_freq:.2f}Hz, 幅值={peak_amp:.4f}")
            return True, peak_freq, peak_amp
        else:
            return False, None, None

    def __plot_comparison(self, original_data: np.ndarray,
                          freqs: np.ndarray, amplitudes: np.ndarray,
                          output_dir: str):
        """
        绘制原始信号与频域信号对比图
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.error("matplotlib 未安装，无法绘图")
            return

        time = np.arange(len(original_data)) / self._sampling_rate

        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        # 时域信号
        ax[0].plot(time, original_data, '#369a62')
        ax[0].set_title("Original Signal")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Amplitude")
        ax[0].grid(True)
        ax[0].set_xlim(time[0], time[-1])

        # 频域信号
        ax[1].plot(freqs, amplitudes, '#844784')
        ax[1].set_title("Frequency Domain (FFT)")
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_ylabel("Amplitude")
        ax[1].grid(True)
        ax[1].set_xlim(self._min_freq, self._max_freq)

        plt.tight_layout()
        save_path = os.path.join(output_dir, "comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"对比图已保存: {save_path}")


def simple_test():
    # 构造一个 50Hz 正弦波
    fs = 1000
    t = np.arange(0, 1.0, 1/fs)
    signal = np.sin(2 * np.pi * 50 * t)

    # 添加一些噪声使信号更真实
    noise = 0.2 * np.random.randn(len(t))
    signal += noise

    detector = OscillationDetection(window_size=1000,
                                    sampling_rate=fs,
                                    threshold=0.2,
                                    log_file="oscillation.log")

    has_osc, freq, amp = detector.detect(signal, PLOT=True)
    print(f"检测结果: {has_osc}, 频率={freq}, 幅值={amp}")


if __name__ == "__main__":
    simple_test()
