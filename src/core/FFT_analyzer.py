import os
import sys
import logging
from typing import List, Tuple, Optional
import numpy as np

from src.core.SignalGenerator import SignalGenerator

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

        self.logger.propagate = True

    def generate_test_signal(self,
                             duration_s: float = None,
                             fundamental_freq: float = 50.0,
                             sin_freqs: List[float] = None,
                             sin_amps: List[float] = None,
                             sin_phases: List[float] = None,
                             poly_coeffs: List[float] = None,) -> np.ndarray:
        # 创建信号生成器
        generator = SignalGenerator(
            sampling_rate=self._sampling_rate,
            duration=duration_s,
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

    def fft_analyze(self, signal_data: np.ndarray, PLOT_path: str = None,
                    use_window: bool = False, refine_frequency: bool = False,
                    IpDFT: bool = True, refine_config: Optional[dict] = None
                    ) -> Tuple[bool, Optional[float], Optional[float], Optional[float]]:
        """
        FFT分析函数，支持相量测量（频率、幅值、相位角）

        :param signal_data: 输入信号数据
        :param PLOT_path: 绘图保存路径
        :param use_window: 是否使用窗函数
        :param IpDFT: 是否使用插值DFT（Jacobsen插值法）
        :param refine_frequency: 是否使用最小二乘精化频率
        :param refine_config: 精化配置
        :return: (success, frequency, amplitude, phase)
        """
        if len(signal_data) < self._window_size:
            self.logger.error("输入数据不足窗口大小，跳过检测")
            return False, None, None, None
        elif len(signal_data) > self._window_size:
            self.logger.warning("输入数据超过最大窗口大小，进行截断")

        windowed_data = signal_data[-self._window_size:]
        N = len(windowed_data)
        # 窗函数相关(window_correction是加入窗函数后导致能量损失,用于修正幅值)
        window = None
        window_correction = 1.0
        if use_window:
            window = np.hanning(N)
            windowed_data = windowed_data * window
            window_correction = np.sum(window) / self._window_size

        fft_vals = np.fft.rfft(windowed_data)
        freqs = np.fft.rfftfreq(N, d=1.0 / self._sampling_rate)
        fft_amps = FFT_AMP_NORMAL_FACTOR * np.abs(fft_vals) / (N * window_correction)

        valid_idx = np.where((freqs >= self._min_freq) & (freqs <= self._max_freq))
        valid_freqs = freqs[valid_idx]
        valid_amps = fft_amps[valid_idx]
        valid_fft_vals = fft_vals[valid_idx]

        if valid_freqs.size == 0:
            self.logger.warning("无有效频率区间，跳过检测")
            return False, None, None, None

        peak_idx = np.argmax(valid_amps)
        peak_freq = valid_freqs[peak_idx]
        peak_amp = valid_amps[peak_idx]
        peak_phase = np.angle(valid_fft_vals[peak_idx])

        # 插值法和相量修正
        if IpDFT and 1 <= peak_idx < len(valid_amps) - 1:
            # 取freq peak 和前后共三点, 计算修正系数delta
            alpha = valid_amps[peak_idx - 1]
            beta = valid_amps[peak_idx]
            gamma = valid_amps[peak_idx + 1]
            delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            peak_freq = peak_freq + delta * (valid_freqs[1] - valid_freqs[0])
            peak_amp = beta - 0.25 * (alpha - gamma) * delta

        # 频率精细化（最小二乘）
        if refine_frequency and peak_freq is not None:
            try:
                from src.core.FrequencyRefinement import FrequencyRefinement

                refiner = FrequencyRefinement(
                    logger=self.logger,
                    **(refine_config or {})
                )

                refined = refiner.refine(
                    windowed_data,
                    self._sampling_rate,
                    peak_freq,
                    return_all_params=True
                )

                if refined is not None:
                    freq_refined, amp_refined, phase_refined, dc_refined, residual = refined
                    self.logger.info(
                        f"频率精化: {peak_freq:.6f}Hz → {freq_refined:.6f}Hz "
                        f"(残差={residual:.6f})"
                    )
                    peak_freq = freq_refined
                    peak_amp = amp_refined
                    peak_phase = phase_refined
                else:
                    self.logger.warning("频率精化失败，保持IpDFT结果")
            except Exception as e:
                self.logger.error(f"频率精化异常: {e}，降级到IpDFT")

        # 绘图
        if PLOT_path:
            try:
                dir_path = os.path.dirname(PLOT_path)
                if dir_path and not os.path.exists(dir_path):
                    self.logger.warning(f"绘图目录不存在, 将创建: {dir_path}")
                    os.makedirs(dir_path, exist_ok=True)

                self.__plot_comparison(windowed_data, valid_freqs, valid_amps,
                                       PLOT_path, peak_freq, peak_amp, peak_phase)
            except Exception as e:
                self.logger.error(f"绘图失败: {e}")

        # 日志输出
        self.logger.debug(f"检测到信号: 频率={peak_freq:.4f}Hz, 幅值={peak_amp:.4f}, "
                          f"相位={np.degrees(peak_phase):.2f}°")

        return True, peak_freq, peak_amp, peak_phase

    def calculate_rocof(self, current_freq: float, previous_freq: float,
                        time_interval: Optional[float] = None) -> float:
        """
        计算频率变化率 (Rate of Change of Frequency)
          :param current_freq: 当前频率 (Hz)
          :param previous_freq: 上一次频率 (Hz)
          :param time_interval: 时间间隔 (s)，默认使用窗口时间
          :return: ROCOF (Hz/s)
        """
        if time_interval is None:
            time_interval = self._window_size / self._sampling_rate

        rocof = (current_freq - previous_freq) / time_interval
        self.logger.info(f"ROCOF: {current_freq:.4f}Hz-{previous_freq:.4f}Hz/{time_interval:.4f}s={rocof:.4f}Hz/s")
        return rocof

    def __plot_comparison(self, original_data: np.ndarray,
                          freqs: np.ndarray, fft_amps: np.ndarray,
                          save_path: str, peak_freq: float = None,
                          peak_amp: float = None, peak_phase: float = None):
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

        # 先画细浅的曲线
        ax[1].plot(freqs, fft_amps, '-', color='#844784', linewidth=1, alpha=0.3)
        # 再画明显的数据点
        ax[1].plot(freqs, fft_amps, 'o', color='#844784', markersize=6,
                   markeredgewidth=1, markeredgecolor='black', alpha=0.8)

        # 设置频率绘图范围
        freq_plot_range = 100  # 可以调整这个值来改变绘图范围
        freq_plot_min = peak_freq - freq_plot_range/2 if peak_freq else self._min_freq
        freq_plot_max = peak_freq + freq_plot_range/2 if peak_freq else self._max_freq

        # 确保绘图范围不超出有效频率范围
        freq_plot_min = max(freq_plot_min, self._min_freq)
        freq_plot_max = min(freq_plot_max, self._max_freq)

        # 根据绘图范围内的点数量来决定标签密度
        plot_range_mask = (freqs >= freq_plot_min) & (freqs <= freq_plot_max)

        # 只为峰值频率点及其左右各3个点添加标签（总共7个点）
        if peak_freq is not None:
            # 找到峰值频率在freqs数组中的索引
            peak_freq_idx = np.argmin(np.abs(freqs - peak_freq))

            # 定义要标记的点的索引范围
            label_indices = range(max(0, peak_freq_idx - 3),
                                  min(len(freqs), peak_freq_idx + 4))

            # 在每个指定点旁边添加数据标签（只有在绘图范围内的才显示）
            for i in label_indices:
                if plot_range_mask[i]:
                    freq, amp = freqs[i], fft_amps[i]
                    ax[1].annotate(f'({freq:.1f}, {amp:.3f})', (freq, amp),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.7)

        ax[1].set_title("Frequency Domain (FFT)")

        # 在右上角添加参数信息
        info_text = (f"Δf={self._frequency_resolution:.4f}Hz\n"
                     f"Precision≈±{self._freq_precision:.4f}Hz"
                     f"\nPhase={np.degrees(peak_phase):.1f}°")

        ax[1].text(0.99, 0.96, info_text, transform=ax[1].transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 标记峰值点
        if peak_freq is not None and peak_amp is not None:
            label_text = f'Estimated Peak: ({peak_freq:.3f}Hz, {peak_amp:.4f})'
            ax[1].plot(peak_freq, peak_amp, 'o', markersize=10,
                       markerfacecolor='none', markeredgecolor='red',
                       markeredgewidth=2, label=label_text)
            ax[1].legend()

        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_ylabel("Amplitude")
        ax[1].grid(True)
        # 使用动态计算的频率范围
        ax[1].set_xlim(freq_plot_min, freq_plot_max)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        self.logger.info(f"对比图已保存: {save_path}")


def simple_test(sampling_rate: int = 4000,
                frequency: float = 50.02,
                window_Ts: int = 8,
                test_refinement: bool = True):
    """
    FFT 精度对比测试：FFT 基线 / IpDFT / 最小二乘精化
    """
    duration_s = 2.0
    window_size = int(sampling_rate / 50 * window_Ts)
    tester = FFTAnalyzer(window_size=window_size,
                         sampling_rate=sampling_rate,
                         log_file="./log/testfft.log")

    # 生成测试信号
    signal = tester.generate_test_signal(duration_s=duration_s,
                                         fundamental_freq=frequency)
    generator = SignalGenerator(sampling_rate=sampling_rate,
                                duration=duration_s)
    sin_signal = generator.sine_wave(freqs=[10], amps=[0.05])
    signal = signal + sin_signal

    print("\n" + "=" * 70)
    print(f"测试配置: 真实频率={frequency:.6f}Hz, 采样率={sampling_rate}Hz, "
          f"窗口={window_size}点")
    print(f"频率分辨率: Δf={sampling_rate / window_size:.4f}Hz")
    print("=" * 70)

    # 方法1: FFT（无窗无插值）
    print("\n【方法1】FFT（基线）")
    success1, freq1, amp1, phase1 = tester.fft_analyze(
        signal, use_window=False, IpDFT=False
    )
    error1 = abs(freq1 - frequency) * 1000  # mHz
    print(f"  频率: {freq1:.6f}Hz, 误差: {error1:.3f}mHz")

    # 方法2: FFT + IpDFT
    print("\n【方法2】FFT + Jacobsen插值")
    success2, freq2, amp2, phase2 = tester.fft_analyze(
        signal, use_window=True, IpDFT=True,
        PLOT_path=f"./plots/{sampling_rate}_{frequency}_{window_Ts}_ipdft.png"
    )
    error2 = abs(freq2 - frequency) * 1000
    improvement2 = error1 / error2 if error2 > 0 else float('inf')
    print(f"  频率: {freq2:.6f}Hz, 误差: {error2:.3f}mHz (精度提升 {improvement2:.1f}x)")

    # 方法3: FFT + IpDFT + 最小二乘精化
    if test_refinement:
        print("\n【方法3】FFT + IpDFT + 最小二乘拟合")
        import time
        start_time = time.time()

        success3, freq3, amp3, phase3 = tester.fft_analyze(
            signal, use_window=True, IpDFT=True,
            refine_frequency=True,
            PLOT_path=f"./plots/{sampling_rate}_{frequency}_{window_Ts}_refined.png"
        )

        elapsed = (time.time() - start_time) * 1000  # ms
        error3 = abs(freq3 - frequency) * 1000
        improvement3 = error1 / error3 if error3 > 0 else float('inf')
        print(f"  频率: {freq3:.6f}Hz, 误差: {error3:.3f}mHz (精度提升 {improvement3:.1f}x)")
        print(f"  耗时: {elapsed:.2f}ms")

    # 对比总结
    print("\n" + "=" * 70)
    print("精度对比总结:")
    print(f"  FFT基线:        {error1:.3f} mHz")
    print(f"  IpDFT插值:      {error2:.3f} mHz  (提升 {improvement2:.1f}x)")
    if test_refinement:
        print(f"  最小二乘精化:   {error3:.3f} mHz  (提升 {improvement3:.1f}x)")
    print("=" * 70)


if __name__ == "__main__":
    simple_test()
