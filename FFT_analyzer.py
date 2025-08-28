import os
import sys
import logging
from typing import List, Tuple, Optional, Dict
import numpy as np

from SignalGenerator import SignalGenerator

# FFT 幅值归一化常量 (因为单边FFT需要乘2来补偿负频率部分的能量)
FFT_AMP_NORMAL_FACTOR = 2


class FFTAnalyzer:
    """
    FFT测试类: 可选是否加窗，并支持频率估计优化。
    自动计算有效频率范围、频率分辨率、估计精度等。
    """

    def __init__(self,
                 window_size: int = 800,
                 sampling_rate: int = 10000,
                 log_file: Optional[str] = None):
        """
        :param window_size: 窗口大小（数据点数）
        :param sampling_rate: 采样率 (Hz)
        :param plot_comparison: 是否绘制对比图
        :param log_file: 日志文件路径（如果为 None，则只输出到控制台）
        """
        # safe check
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size 必须是正整数")
        if not isinstance(sampling_rate, (int, float)) or sampling_rate <= 0:
            raise ValueError("sampling_rate 必须是正数")

        self._window_size = window_size
        self._sampling_rate = sampling_rate
        self._log_file = log_file

        self._setup_logger()
        self._setup_frequency_params()

    # 设置 FFT 参数
    def _setup_frequency_params(self):
        self._frequency_resolution = self._sampling_rate / self._window_size  # Δf
        self._nyquist_freq = self._sampling_rate / 2.0
        self._max_freq = min(self._nyquist_freq * 0.95, 500)
        self._min_freq = self._frequency_resolution  # 有效最小频率
        self._freq_range = (self._min_freq, self._max_freq)

        # 简单估计频率精度的方式: Δf / SNR; 发电厂一般较高
        self._snr_db = 80
        self._freq_precision = self._frequency_resolution / (10 ** (self._snr_db / 20))

        self.logger.info("TestFFT 初始化完成: "
                         f"window_size={self._window_size}, sampling_rate={self._sampling_rate}, "
                         f"Δf={self._frequency_resolution:.6f}Hz, "
                         f"min_freq={self._min_freq:.2f}Hz, max_freq={self._max_freq:.2f}Hz, "
                         f"precision≈±{self._freq_precision:.6f}Hz (SNR={self._snr_db}dB)")

    # 设置logger
    def _setup_logger(self):
        self.logger = logging.getLogger("TestFFT")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        if self._log_file is not None:
            file_handler = logging.FileHandler(self._log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                               datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        self.logger.propagate = False

    def generate_test_signal(self,
                             fundamental_freq: float = 50.0,
                             sin_freqs: List[float] = None,
                             sin_amps: List[float] = None,
                             sin_phases: List[float] = None,
                             poly_coeffs: List[float] = None,) -> np.ndarray:
        # 创建信号生成器
        generator = SignalGenerator(
            sampling_rate=self._sampling_rate,
            duration=self._window_size*2,
            noise_level=0.0,
            log_file=self._log_file
        )
            
        # 生成基础谐波信号
        signal = generator.harmonic_wave(fundamental_freq=fundamental_freq)

        # 添加正弦波分量
        if sin_freqs and any(f > 0 for f in sin_freqs):
            sin_signal = generator.sine_wave(
                freqs=sin_freqs,
                amps=sin_amps,
                phases=sin_phases
            )
            signal += sin_signal
            self.logger.info(f"已添加正弦波分量: freqs={sin_freqs}")

        # 添加多项式分量
        if poly_coeffs and any(c != 0 for c in poly_coeffs):
            poly_signal = generator.polynomial(coeffs=poly_coeffs)
            signal += poly_signal
            self.logger.info(f"已添加多项式分量: coeffs={poly_coeffs}")

        return signal

    def fft_analyze(self, signal_data: np.ndarray, PLOT_path: str = None, use_window: bool = False,
                    optimize: bool = True) -> Tuple[bool, Optional[float], Optional[float]]:

        if len(signal_data) < self._window_size:
            self.logger.error("输入数据不足窗口大小，跳过检测")
            return False, None, None
        elif len(signal_data) > self._window_size:
            self.logger.warning("输入数据超过最大窗口大小，进行截断")

        windowed_data = signal_data[-self._window_size:]

        # 窗函数相关(window_correction是加入窗函数后导致能量损失,用于修正幅值)
        window_correction = 1.0
        if use_window:
            window = np.hanning(len(windowed_data))
            windowed_data = windowed_data * window
            window_correction = np.sum(window) / self._window_size

        fft_vals = np.fft.rfft(windowed_data)
        freqs = np.fft.rfftfreq(len(windowed_data), d=1.0 / self._sampling_rate)
        amplitudes = FFT_AMP_NORMAL_FACTOR * np.abs(fft_vals) / (len(windowed_data) * window_correction)

        valid_idx = np.where((freqs >= self._min_freq) & (freqs <= self._max_freq))
        valid_freqs = freqs[valid_idx]
        valid_amps = amplitudes[valid_idx]

        if valid_freqs.size == 0:
            self.logger.warning("无有效频率区间，跳过检测")
            return False, None, None

        peak_idx = np.argmax(valid_amps)
        peak_freq = valid_freqs[peak_idx]
        peak_amp = valid_amps[peak_idx]

        # 插值优化
        if optimize and 1 <= peak_idx < len(valid_amps) - 1:
            alpha = valid_amps[peak_idx - 1]
            beta = valid_amps[peak_idx]
            gamma = valid_amps[peak_idx + 1]
            p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            peak_freq = peak_freq + p * (valid_freqs[1] - valid_freqs[0])
            peak_amp = beta - 0.25 * (alpha - gamma) * p

        if PLOT_path:
            try:
                dir_path = os.path.dirname(PLOT_path)
                if dir_path and not os.path.exists(dir_path):
                    self.logger.warning(f"绘图目录不存在, 将创建: {dir_path}")
                    os.makedirs(dir_path, exist_ok=True)

                self.__plot_comparison(windowed_data, valid_freqs, valid_amps, PLOT_path, peak_freq, peak_amp)
            except Exception as e:
                self.logger.error(f"绘图失败: {e}")

        self.logger.info(f"检测到信号: 频率={peak_freq:.4f}Hz, 幅值={peak_amp:.4f}")
        return True, peak_freq, peak_amp

    def __plot_comparison(self, original_data: np.ndarray,
                          freqs: np.ndarray, amplitudes: np.ndarray,
                          save_path: str, peak_freq: float = None, peak_amp: float = None):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self.logger.error("matplotlib 未安装，无法绘图")
            raise

        time = np.arange(len(original_data)) / self._sampling_rate

        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        ax[0].plot(time, original_data, '#369a62')
        ax[0].set_title("Original Signal")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Amplitude")
        ax[0].grid(True)

        ax[1].plot(freqs, amplitudes, '#844784')
        ax[1].set_title("Frequency Domain (FFT)")
        
        # 在右上角添加参数信息
        info_text = (f"Δf={self._frequency_resolution:.4f}Hz\n"
                    f"Range=[{self._min_freq:.2f}, {self._max_freq:.2f}]Hz\n"
                    f"Precision≈±{self._freq_precision:.4f}Hz")
        ax[1].text(0.98, 0.98, info_text, transform=ax[1].transAxes,
                  fontsize=10, verticalalignment='top', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 标记峰值点
        if peak_freq is not None and peak_amp is not None:
            ax[1].plot(peak_freq, peak_amp, 'ro', markersize=8, label=f'Peak: {peak_freq:.2f}Hz, {peak_amp:.4f}')
            ax[1].legend()
        
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_ylabel("Amplitude")
        ax[1].grid(True)
        ax[1].set_xlim(self._min_freq, self._max_freq)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"对比图已保存: {save_path}")


def simple_test():
    tester = FFTAnalyzer(window_size=800,
                     sampling_rate=10000,
                     log_file="./log/testfft.log")

    signal = tester.generate_test_signal(fundamental_freq=50.02)
    print("---- 不加窗 ----")
    tester.fft_analyze(signal, PLOT_path="./plots/no_window.png", use_window=False, optimize=True)

    print("---- 加窗 ----")
    tester.fft_analyze(signal, PLOT_path="./plots/window.png", use_window=True, optimize=True)


if __name__ == "__main__":
    simple_test()
