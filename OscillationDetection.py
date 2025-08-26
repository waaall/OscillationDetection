import os
import logging
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # 默认 INFO，可以外部模块配置


class OscillationDetection:
    """
    振荡检测类: 通过DFT/FFT在时域数据中检测振荡频率。
    """

    def __init__(self,
                 window_size: int = 60,
                 sampling_rate: float = 1000.0,
                 threshold: float = 0.5,
                 plot_comparison: bool = False):
        """
        :param window_size: 窗口大小（数据点数），默认 60
        :param sampling_rate: 采样率 (Hz)，默认 1000 Hz
        :param threshold: 检测阈值（幅值归一化后）
        :param plot_comparison: 是否绘制对比图
        """
        self._window_size = window_size
        self._sampling_rate = sampling_rate
        self._threshold = threshold
        self._plot_comparison = plot_comparison

        # 频率相关参数
        self.__nyquist_freq = self._sampling_rate / 2.0
        self.__max_freq = min(self.__nyquist_freq * 0.95, 500)  # 留 5% 余量，不超过 500Hz
        self.__min_freq = 1.0 / (self._window_size / self._sampling_rate)  # 最小可分辨频率

        logger.info("OscillationDetection 初始化完成: "
                    f"window_size={window_size}, sampling_rate={sampling_rate}, "
                    f"min_freq={self.__min_freq:.2f}Hz, max_freq={self.__max_freq:.2f}Hz")

    def detect(self, data: np.ndarray) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        振荡检测入口函数

        :param data: 输入时域数据 (float 数组)
        :return: (是否检测到, 频率, 幅值)，若未检测到返回 (False, None, None)
        """
        if len(data) < self._window_size:
            logger.warning("输入数据不足窗口大小，跳过检测")
            return False, None, None

        # 截取窗口
        windowed_data = data[-self._window_size:]

        # 执行FFT
        fft_vals = np.fft.rfft(windowed_data)
        fft_freqs = np.fft.rfftfreq(len(windowed_data), d=1.0 / self._sampling_rate)

        # 幅值归一化
        amplitudes = np.abs(fft_vals) / len(windowed_data)

        # 限制频率范围
        valid_idx = np.where((fft_freqs >= self.__min_freq) & (fft_freqs <= self.__max_freq))
        valid_freqs = fft_freqs[valid_idx]
        valid_amplitudes = amplitudes[valid_idx]

        # 找最大峰值
        peak_idx = np.argmax(valid_amplitudes)
        peak_freq = valid_freqs[peak_idx]
        peak_amp = valid_amplitudes[peak_idx]

        logger.debug(f"检测频率范围 {self.__min_freq:.2f}-{self.__max_freq:.2f}Hz, "
                     f"峰值={peak_amp:.4f} @ {peak_freq:.2f}Hz")

        if peak_amp >= self.threshold:
            logger.info(f"检测到振荡: 频率={peak_freq:.2f}Hz, 幅值={peak_amp:.4f}")
            return True, peak_freq, peak_amp
        else:
            return False, None, None

    def _plot_comparison(self, original_data: np.ndarray, output_dir: str):
        """
        绘制原始信号与频域信号对比图
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib 未安装，无法绘图")
            return

        time = np.arange(len(original_data)) / self._sampling_rate

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # 时域信号
        ax1.plot(time, original_data, 'b-')
        ax1.set_title("Original Signal")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True)

        # 频域信号
        freqs = np.fft.rfftfreq(len(original_data), d=1.0 / self._sampling_rate)
        amps = np.abs(np.fft.rfft(original_data)) / len(original_data)

        ax2.plot(freqs, amps, 'r-')
        ax2.set_title("Frequency Domain (FFT)")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True)

        plt.tight_layout()
        save_path = os.path.join(output_dir, "comparison.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"对比图已保存: {save_path}")
