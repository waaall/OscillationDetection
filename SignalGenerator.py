import os
import sys
import logging
from typing import Optional, Sequence, Union
import numpy as np
import pandas as pd


class SignalGenerator:
    def __init__(self,
                 sampling_rate: int = 1000,
                 duration: float = 10.0,
                 noise_level: float = 0.0,
                 seed: Optional[int] = None,
                 log_file: Optional[str] = None):
        """
        可定制化的信号生成器

        :param fs: 采样率 (Hz)
        :param duration: 信号持续时间 (s)
        :param noise_level: 噪声强度 (0~1 比例)
        :param seed: 随机种子(保证可重复性)
        :param log_file: 日志文件路径(可选)
        """
        if sampling_rate <= 0 or duration <= 0:
            raise ValueError("fs 和 duration 必须为正数")
        if not (0 <= noise_level <= 1):
            raise ValueError("noise_level 必须在 [0, 1] 范围内")

        self._sampling_rate = sampling_rate
        self._duration = duration
        self._noise_level = noise_level
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.signal_data = np.arange(0, duration, 1 / sampling_rate)

        # 设置日志
        self._setup_logger(log_file)
        self.logger.info(f"SignalGenerator 初始化: fs={sampling_rate}, duration={duration}, "
                         f"noise_level={noise_level}, seed={seed}")

    def _setup_logger(self, log_file: Optional[str]):
        """配置日志系统"""
        self.logger = logging.getLogger("SignalGenerator")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # 文件处理器(可选)
        if log_file is not None:
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                               datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        self.logger.propagate = False

    def sine_wave(self,
                  freqs: Sequence[float],
                  amps: Optional[Sequence[float]] = None,
                  phases: Optional[Sequence[float]] = None) -> np.ndarray:
        """生成多个正弦波叠加"""
        freqs = np.atleast_1d(freqs)
        amps = np.ones_like(freqs) if amps is None else np.atleast_1d(amps)
        phases = np.zeros_like(freqs) if phases is None else np.atleast_1d(phases)

        if len(freqs) != len(amps) or len(freqs) != len(phases):
            raise ValueError("freqs, amps, phases 长度必须一致")

        signal = np.zeros_like(self.signal_data)
        for f, a, p in zip(freqs, amps, phases):
            signal += a * np.sin(2 * np.pi * f * self.signal_data + p)

        self.logger.info(f"生成正弦波: freqs={freqs}, amps={amps}, phases={phases}")
        return self._add_noise(signal)

    def harmonic_wave(self,
                      fundamental_freq: float = 50.0,  # 默认电网频率50Hz
                      num_harmonics: int = 5,          # 包括基波, 通常分析到5次谐波
                      amps: Optional[Union[Sequence[float], str]] = 'typical',
                      phases: Optional[Sequence[float]] = None) -> np.ndarray:
        """
        生成包含谐波的信号, 特别针对电网信号优化
        
        :param fundamental_freq: 基频 (Hz), 默认50Hz(电网频率)
        :param num_harmonics: 谐波数量(包括基波), 默认5次
        :param amps: 各谐波幅度, 可以是数组或字符串:
                    'typical' = 典型电网谐波幅度 (默认)
                    'linear' = 线性衰减
                    'exponential' = 指数衰减
        :param phases: 各谐波相位(弧度)
        """
        if fundamental_freq <= 0:
            raise ValueError("fundamental_freq 必须为正数")
        if num_harmonics < 1:
            raise ValueError("num_harmonics 必须至少为1")
            
        # 生成谐波频率
        freqs = [fundamental_freq * (i + 1) for i in range(num_harmonics)]
        
        # 处理幅度参数
        if amps is None or amps == 'typical':
            # 典型电网谐波幅度 基波=1.0, 奇次谐波幅度较大, 偶次谐波幅度较小
            typical_amps = {
                1: 1.00,   # 基波
                2: 0.02,   # 2次谐波
                3: 0.04,   # 3次谐波
                4: 0.01,   # 4次谐波
                5: 0.03,   # 5次谐波
            }
            
            amps = []
            for i in range(1, num_harmonics + 1):
                if i in typical_amps:
                    amps.append(typical_amps[i])
                else:
                    # 对于更高次谐波, 使用近似衰减公式
                    amps.append(0.01 / (i/13)**2)
                    
        elif isinstance(amps, str):
            if amps == 'linear':
                amps = [1.0 - i * 0.1 for i in range(num_harmonics)]
                amps = [max(0, a) for a in amps]  # 确保非负
            elif amps == 'exponential':
                amps = [np.exp(-i * 0.5) for i in range(num_harmonics)]
            else:
                raise ValueError("amps 字符串参数只能是 'typical', 'linear' 或 'exponential'")
        elif len(amps) != num_harmonics:
            raise ValueError("amps 长度必须与 num_harmonics 一致")
            
        # 处理相位参数
        if phases is None:
            # 设置典型相位关系：奇次谐波相位接近0, 偶次谐波相位接近π/2
            phases = []
            for i in range(1, num_harmonics + 1):
                if i % 2 == 1:  # 奇次谐波
                    phases.append(0)
                else:  # 偶次谐波
                    phases.append(np.pi/2)
        elif len(phases) != num_harmonics:
            raise ValueError("phases 长度必须与 num_harmonics 一致")
            
        self.logger.info(f"生成电网谐波: 基频={fundamental_freq}Hz, 谐波数={num_harmonics}")
        
        return self.sine_wave(freqs, amps, phases)

    def linear(self, slope: float = 1.0, intercept: float = 0.0) -> np.ndarray:
        """生成直线 y = slope*t + intercept"""
        signal = slope * self.signal_data + intercept
        self.logger.info(f"生成直线: slope={slope}, intercept={intercept}")
        return self._add_noise(signal)

    def polynomial(self, coeffs: Sequence[float] = (1, 0, 0)) -> np.ndarray:
        """生成多项式信号"""
        signal = np.polyval(coeffs, self.signal_data)
        self.logger.info(f"生成多项式: coeffs={coeffs}")
        return self._add_noise(signal)

    def exponential(self, A: float = 1.0, tau: float = 1.0) -> np.ndarray:
        """生成指数衰减/增长"""
        signal = A * np.exp(-self.signal_data / tau)
        self.logger.info(f"生成指数信号: A={A}, tau={tau}")
        return self._add_noise(signal)

    def insert_into_csv(self,
                        csv_path: str,
                        column: str = "signal",
                        start_idx: int = 0,
                        new_signal: Optional[np.ndarray] = None,
                        isCUT: bool = True,
                        save_path: Optional[str] = None) -> pd.DataFrame:
        """
        将模拟信号插入到 CSV 文件中

        :param csv_path: 原始 CSV 文件路径
        :param column: 插入列的名字
        :param start_idx: 插入位置索引
        :param new_signal: 新信号 (numpy 数组)
        :param isCUT: True=替换, False=追加
        :param save_path: 保存路径(如果 None, 则覆盖原文件)
        """
        if new_signal is None or len(new_signal) == 0:
            raise ValueError("new_signal 不能为空")

        if not os.path.exists(csv_path):
            self.logger.warning(f"文件 {csv_path} 不存在, 新建文件")
            df = pd.DataFrame({
                "time": self.signal_data[:len(new_signal)],
                column: new_signal
            })
        else:
            df = pd.read_csv(csv_path)

            if column not in df.columns:
                df[column] = 0.0

            if isCUT:
                end_idx = min(start_idx + len(new_signal), len(df))
                if end_idx - start_idx < len(new_signal):
                    self.logger.warning("新信号长度超过原始数据长度, 仅部分替换")
                df.loc[start_idx:end_idx - 1, column] = new_signal[:end_idx - start_idx]
            else:
                self.logger.info("isCUT=False, 执行追加操作")
                extra_time = np.arange(len(df), len(df) + len(new_signal)) / self._sampling_rate
                df_extra = pd.DataFrame({"time": extra_time, column: new_signal})
                df = pd.concat([df, df_extra], ignore_index=True)

        save_path = save_path or csv_path
        df.to_csv(save_path, index=False)
        self.logger.info(f"信号已写入 {save_path}, 列: {column}, 模式: {'替换' if isCUT else '追加'}")
        return df

    def trim_signal(self, signal: np.ndarray, 
                    n_points: int = 0,
                    frequency: Optional[float] = None,
                    phase_angle: Optional[float] = None,
                    from_end: bool = True) -> np.ndarray:
        """
        裁剪信号的数据点
        
        :param signal: 输入信号
        :param n_points: 要裁剪掉的数据点数量
        :param frequency: 给定频率(Hz), 用于计算基于周期的裁剪点
        :param phase_angle: 给定相位角度(弧度), 用于精确定位裁剪点
        :param from_end: True=从后面裁剪(默认), False=从前面裁剪
        :return: 裁剪后的信号
        """
        if len(signal) == 0:
            raise ValueError("输入信号不能为空")
            
        # 计算要裁剪的点数
        if frequency is not None and phase_angle is not None:
            # 基于频率和相位角度计算裁剪点数
            period_samples = int(self._sampling_rate / frequency)
            phase_offset = int((phase_angle / (2 * np.pi)) * period_samples)
            
            # 简单地使用相位偏移作为裁剪点数
            points_to_trim = phase_offset
            
            self.logger.info(f"基于频率{frequency}Hz和相位{phase_angle}弧度计算裁剪点数: {points_to_trim}")
        else:
            # 使用指定的点数
            points_to_trim = n_points
            
        # 执行裁剪
        if points_to_trim >= len(signal):
            raise ValueError("裁剪的点数不能大于等于信号长度")
            
        if points_to_trim <= 0:
            trimmed_signal = signal.copy()
        else:
            if from_end:
                trimmed_signal = signal[:-points_to_trim]
                self.logger.info(f"从信号后面裁剪 {points_to_trim} 个数据点")
            else:
                trimmed_signal = signal[points_to_trim:]
                self.logger.info(f"从信号前面裁剪 {points_to_trim} 个数据点")
            
        return trimmed_signal

    def _add_noise(self, signal: np.ndarray) -> np.ndarray:
        """添加噪声"""
        if self._noise_level > 0:
            noise = self._noise_level * np.random.randn(len(signal))
            return signal + noise
        return signal


# ===================== 测试 =====================
def test():
    gen = SignalGenerator(sampling_rate=1000, duration=30.0, noise_level=0.1, seed=42)

    sig_sins = gen.sine_wave(freqs=[50, 120], amps=[1, 0.5])
    sig_linear = gen.linear(slope=0.5, intercept=2)
    sig_polynomial = gen.polynomial(coeffs=[1, -2, 1])
    sig_exponential = gen.exponential(A=1, tau=0.5)

    sig = sig_sins + sig_linear + sig_polynomial + sig_exponential

    # 基于频率和相位角度从后面裁剪
    trimmed_sig = gen.trim_signal(sig, frequency=50.0, phase_angle=np.pi/4, from_end=True)
    print(f"基于频率50Hz和相位π/4从后面裁剪后长度: {len(trimmed_sig)}")

    gen.insert_into_csv("./test_data.csv", column="值", start_idx=100,
                        new_signal=trimmed_sig, isCUT=True)

if __name__ == "__main__":
    test()
