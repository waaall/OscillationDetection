import os
import sys
import logging
from typing import Optional, Sequence, Union, List, Mapping, Any, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

PathLikeStr = Union[str, os.PathLike[str]]


class SignalGenerator:
    def __init__(self,
                 sampling_rate: int = 1000,
                 duration: float = 10.0,
                 noise_level: float = 0.0,
                 seed: Optional[int] = None,
                 log_file: Optional[str] = None,
                 start_time: Optional[datetime] = None):
        """
        :param start_time: 起始时间戳(datetime对象), 默认None则使用当前时间
        """
        if sampling_rate <= 0 or duration <= 0:
            raise ValueError("sampling_rate 和 duration 必须为正数")
        if not (0 <= noise_level <= 1):
            raise ValueError("noise_level 必须在 [0, 1] 范围内")

        self._sampling_rate = sampling_rate
        self._duration = duration
        self._noise_level = noise_level
        self._seed = seed
        if seed is not None:
            np.random.seed(seed)

        # 时间戳系统
        if start_time is None:
            self.start_time = datetime.now()
        else:
            self.start_time = start_time

        # 使用“样本数驱动”构造相对时间轴, 避免浮点步长累计误差。
        self._num_samples = self._calculate_num_samples(duration)

        # 保留向后兼容性：秒为单位的浮点数组
        self.signal_data = np.arange(self._num_samples, dtype=float) / self._sampling_rate

        # datetime 时间戳改为懒生成, 只有真正需要时才构造。
        self._timestamps_cache: Optional[List[datetime]] = None

        # 设置日志
        self._setup_logger(log_file)
        self.logger.info(f"SignalGenerator 初始化: fs={sampling_rate}, duration={duration}, "
                         f"noise_level={noise_level}, seed={seed}, "
                         f"start_time={self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def _setup_logger(self, log_file: Optional[str]):
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

        self.logger.propagate = True

    def _calculate_num_samples(self, duration: float) -> int:
        """根据时长和采样率计算采样点数"""
        samples = int(np.ceil(duration * self._sampling_rate - 1e-12))
        return max(1, samples)

    def _generate_timestamps(self,
                             start_idx: int = 0,
                             end_idx: Optional[int] = None,
                             clamp_to_generator: bool = True) -> List[datetime]:
        """
        生成datetime时间戳数组或其中一段
        :return: datetime时间戳列表
        """
        if start_idx < 0:
            raise ValueError("start_idx 不能为负数")

        stop_idx = self._num_samples if end_idx is None else end_idx
        if clamp_to_generator:
            stop_idx = min(stop_idx, self._num_samples)
        if stop_idx < start_idx:
            raise ValueError("end_idx 不能小于 start_idx")

        delta = timedelta(seconds=1 / self._sampling_rate)
        base_time = self.start_time + start_idx * delta
        return [base_time + i * delta for i in range(stop_idx - start_idx)]

    @property
    def timestamps(self) -> List[datetime]:
        """懒生成并缓存完整datetime时间戳数组"""
        if self._timestamps_cache is None:
            self._timestamps_cache = self._generate_timestamps()
        return self._timestamps_cache

    def _format_timestamp_range(self,
                                start_idx: int,
                                end_idx: int,
                                format_str: Optional[str] = None) -> List[str]:
        """按需格式化一段时间戳, 避免无必要地构造整表时间戳"""
        if self._timestamps_cache is not None:
            timestamp_slice = self._timestamps_cache[start_idx:end_idx]
        else:
            timestamp_slice = self._generate_timestamps(start_idx=start_idx, end_idx=end_idx)
        return [self.format_timestamp(ts, format_str) for ts in timestamp_slice]

    def _normalize_insert_signal(self, new_signal: Optional[np.ndarray]) -> np.ndarray:
        """标准化待写入信号, 并校验基础合法性"""
        if new_signal is None:
            raise ValueError("new_signal 不能为空")

        signal_array = np.asarray(new_signal)
        if signal_array.ndim != 1 or signal_array.size == 0:
            raise ValueError("new_signal 必须是一维且非空的数组")

        return signal_array

    def _resolve_write_path(self, path: str, append: bool) -> str:
        """解析实际写入路径。非追加写入默认避免覆盖现有文件"""
        if append or not os.path.exists(path):
            return path

        root, ext = os.path.splitext(path)
        candidate = f"{root}-new{ext}"
        suffix = 2
        while os.path.exists(candidate):
            candidate = f"{root}-new-{suffix}{ext}"
            suffix += 1
        new_path = candidate
        self.logger.warning(f"文件 {path} 已存在, append=False 时改为写入 {new_path}")
        return new_path

    def _build_insert_time_values(self, start_idx: int, count: int, use_datetime: bool,
                                  time_format: Optional[str],
                                  enforce_generator_limit: bool = True) -> Union[np.ndarray, List[str]]:
        """根据模式生成一段 time 列数据"""
        end_idx = start_idx + count
        if enforce_generator_limit and end_idx > self._num_samples:
            raise ValueError("time 写入范围不能超过当前生成器样本数")

        if use_datetime:
            if enforce_generator_limit:
                return self._format_timestamp_range(start_idx, end_idx, time_format)
            timestamps = self._generate_timestamps(
                start_idx=start_idx,
                end_idx=end_idx,
                clamp_to_generator=False,
            )
            return [self.format_timestamp(ts, time_format) for ts in timestamps]
        return np.arange(start_idx, end_idx, dtype=float) / self._sampling_rate

    def _get_auto_time_precision(self) -> str:
        """根据采样间隔自动选择时间戳展示精度"""
        sample_interval_ms = 1000.0 / self._sampling_rate

        # 采样间隔达到 1 秒及以上时, 显示到秒即可。
        if sample_interval_ms >= 1000.0:
            return "seconds"

        # 采样间隔小于 1 秒时, 统一显示到毫秒, 不再继续保留微秒。
        return "milliseconds"

    def format_timestamp(self, dt: datetime,
                         format_str: Optional[str] = None) -> str:
        """
        格式化单个时间戳
        :param dt: datetime对象
        :param format_str: datetime格式字符串；None或'auto'时按采样率自动选择精度
        :return: 格式化的时间字符串
        """
        if format_str is None or format_str == "auto":
            precision = self._get_auto_time_precision()

            if precision == "seconds":
                rounded_dt = dt + timedelta(microseconds=500000)
                return rounded_dt.strftime("%Y-%m-%d %H:%M:%S")

            rounded_dt = dt + timedelta(microseconds=500)
            return rounded_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        return dt.strftime(format_str)

    def _normalize_sine_params(self,
                               freqs: Sequence[float],
                               amps: Optional[Sequence[float]] = None,
                               phases: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """标准化多正弦参数, 便于复用纯载波生成逻辑"""
        freqs_array = np.atleast_1d(np.asarray(freqs, dtype=float))
        if freqs_array.size == 0:
            raise ValueError("freqs 不能为空")

        if amps is None:
            amps_array = np.ones(freqs_array.size, dtype=float)
        else:
            amps_array = np.atleast_1d(np.asarray(amps, dtype=float))

        if phases is None:
            phases_array = np.zeros(freqs_array.size, dtype=float)
        else:
            phases_array = np.atleast_1d(np.asarray(phases, dtype=float))

        if len(freqs_array) != len(amps_array) or len(freqs_array) != len(phases_array):
            raise ValueError("freqs, amps, phases 长度必须一致")

        return freqs_array, amps_array, phases_array

    def _compose_sine_wave(self,
                           freqs: Sequence[float],
                           amps: Optional[Sequence[float]] = None,
                           phases: Optional[Sequence[float]] = None) -> np.ndarray:
        """生成不带噪声的多正弦叠加载波, 供调制和现有接口复用"""
        freqs_array, amps_array, phases_array = self._normalize_sine_params(freqs, amps, phases)

        signal = np.zeros_like(self.signal_data, dtype=float)
        for f, a, p in zip(freqs_array, amps_array, phases_array):
            signal += a * np.sin(2 * np.pi * f * self.signal_data + p)

        return signal

    def _resolve_harmonic_components(self,
                                     fundamental_freq: float = 50.0,
                                     num_harmonics: int = 5,
                                     amps: Optional[Union[Sequence[float], str]] = 'typical',
                                     phases: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """解析谐波频率、幅值和相位, 供普通谐波和调制谐波接口共用"""
        if fundamental_freq <= 0:
            raise ValueError("fundamental_freq 必须为正数")
        if num_harmonics < 1:
            raise ValueError("num_harmonics 必须至少为1")

        # 生成谐波频率
        freqs = np.asarray([fundamental_freq * (i + 1) for i in range(num_harmonics)], dtype=float)

        # 处理幅度参数
        if amps is None or (isinstance(amps, str) and amps == 'typical'):
            # 典型电网谐波幅度 基波=1.0, 奇次谐波幅度较大, 偶次谐波幅度较小
            typical_amps = {
                1: 1.00,   # 基波
                2: 0.02,   # 2次谐波
                3: 0.04,   # 3次谐波
                4: 0.01,   # 4次谐波
                5: 0.03,   # 5次谐波
            }

            amp_values = []
            for i in range(1, num_harmonics + 1):
                if i in typical_amps:
                    amp_values.append(typical_amps[i])
                else:
                    # 对于更高次谐波, 使用近似衰减公式
                    amp_values.append(0.01 / (i/13)**2)
            amps_array = np.asarray(amp_values, dtype=float)
        elif isinstance(amps, str):
            if amps == 'linear':
                amp_values = [max(0.0, 1.0 - i * 0.1) for i in range(num_harmonics)]
                amps_array = np.asarray(amp_values, dtype=float)
            elif amps == 'exponential':
                amps_array = np.asarray([np.exp(-i * 0.5) for i in range(num_harmonics)], dtype=float)
            else:
                raise ValueError("amps 字符串参数只能是 'typical', 'linear' 或 'exponential'")
        else:
            amps_array = np.atleast_1d(np.asarray(amps, dtype=float))
            if len(amps_array) != num_harmonics:
                raise ValueError("amps 长度必须与 num_harmonics 一致")

        # 处理相位参数
        if phases is None:
            # 设置典型相位关系：奇次谐波相位接近0, 偶次谐波相位接近π/2
            phase_values = []
            for i in range(1, num_harmonics + 1):
                if i % 2 == 1:  # 奇次谐波
                    phase_values.append(0.0)
                else:  # 偶次谐波
                    phase_values.append(np.pi / 2)
            phases_array = np.asarray(phase_values, dtype=float)
        else:
            phases_array = np.atleast_1d(np.asarray(phases, dtype=float))
            if len(phases_array) != num_harmonics:
                raise ValueError("phases 长度必须与 num_harmonics 一致")

        return self._normalize_sine_params(freqs, amps_array, phases_array)

    def sine_wave(self,
                  freqs: Sequence[float],
                  amps: Optional[Sequence[float]] = None,
                  phases: Optional[Sequence[float]] = None) -> np.ndarray:
        """生成多个正弦波叠加, 不自动添加噪声"""
        freqs_array, amps_array, phases_array = self._normalize_sine_params(freqs, amps, phases)
        signal = self._compose_sine_wave(freqs_array, amps_array, phases_array)

        self.logger.info(f"生成正弦波: freqs={freqs_array}, amps={amps_array}, phases={phases_array}")
        return signal

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
        freqs_array, amps_array, phases_array = self._resolve_harmonic_components(
            fundamental_freq=fundamental_freq,
            num_harmonics=num_harmonics,
            amps=amps,
            phases=phases,
        )

        self.logger.info(f"生成电网谐波: 基频={fundamental_freq}Hz, 谐波数={num_harmonics}")
        return self.sine_wave(freqs_array, amps_array, phases_array)

    def linear(self, slope: float = 1.0, intercept: float = 0.0) -> np.ndarray:
        """生成直线 y = slope*t + intercept, 不自动添加噪声"""
        signal = slope * self.signal_data + intercept
        self.logger.info(f"生成直线: slope={slope}, intercept={intercept}")
        return signal

    def polynomial(self, coeffs: Sequence[float] = (1, 0, 0)) -> np.ndarray:
        """生成多项式信号, 不自动添加噪声"""
        signal = np.polyval(coeffs, self.signal_data)
        self.logger.info(f"生成多项式: coeffs={coeffs}")
        return signal

    def exponential(self, A: float = 1.0, tau: float = 1.0) -> np.ndarray:
        """生成指数衰减/增长, 不自动添加噪声"""
        signal = A * np.exp(-self.signal_data / tau)
        self.logger.info(f"生成指数信号: A={A}, tau={tau}")
        return signal

    def _duration_to_samples(self, duration: float) -> int:
        """将秒级时长稳定转换为采样点数, 并校验可与采样率对齐"""
        if duration <= 0:
            raise ValueError("segment duration 必须为正数")

        samples_float = duration * self._sampling_rate
        samples = int(round(samples_float))
        if not np.isclose(samples_float, samples, rtol=1e-9, atol=1e-9):
            raise ValueError("segment duration 必须与 sampling_rate 对齐")
        if samples <= 0:
            raise ValueError("segment duration 至少对应 1 个采样点")

        return samples

    def _build_hold_segment(self, value: float, num_samples: int) -> np.ndarray:
        """生成平台段"""
        return np.full(num_samples, float(value), dtype=float)

    def _build_ramp_segment(self,
                            start: float,
                            end: float,
                            num_samples: int,
                            curve: str) -> np.ndarray:
        """生成过渡段, 支持线性和平滑正弦过渡"""
        start_value = float(start)
        end_value = float(end)

        # 单点过渡无法同时覆盖起止端点, 这里约定取终点值, 避免与下一段产生额外跳变。
        if num_samples == 1:
            return np.asarray([end_value], dtype=float)

        if curve == "linear":
            return np.linspace(start_value, end_value, num_samples, dtype=float)
        if curve == "sine":
            progress = np.linspace(0.0, np.pi, num_samples, dtype=float)
            return start_value + (end_value - start_value) * 0.5 * (1.0 - np.cos(progress))

        raise ValueError("ramp curve 只能是 'linear' 或 'sine'")

    def _build_envelope_cycle(self, segments: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """根据分段配置生成单个周期包络"""
        if len(segments) == 0:
            raise ValueError("segments 不能为空")

        cycle_segments = []
        for index, segment in enumerate(segments):
            if not isinstance(segment, Mapping):
                raise ValueError("segment 必须是字典类型")

            segment_type = segment.get("type")
            if segment_type not in {"hold", "ramp"}:
                raise ValueError(f"第 {index} 段的 type 无效: {segment_type}")

            if "duration" not in segment:
                raise ValueError(f"第 {index} 段缺少 duration")
            num_samples = self._duration_to_samples(float(segment["duration"]))

            if segment_type == "hold":
                if "value" not in segment:
                    raise ValueError(f"第 {index} 段缺少 value")
                cycle_segments.append(self._build_hold_segment(float(segment["value"]), num_samples))
                continue

            if "start" not in segment or "end" not in segment:
                raise ValueError(f"第 {index} 段缺少 start 或 end")

            # ramp 未显式指定 curve 时默认按线性插值处理, 便于构造简单过渡。
            curve = str(segment.get("curve", "linear")).lower()
            cycle_segments.append(
                self._build_ramp_segment(
                    start=float(segment["start"]),
                    end=float(segment["end"]),
                    num_samples=num_samples,
                    curve=curve,
                )
            )

        return np.concatenate(cycle_segments).astype(float, copy=False)

    def periodic_envelope(self, segments: Sequence[Mapping[str, Any]]) -> np.ndarray:
        """
        生成可重复的周期包络

        :param segments: 分段配置列表。支持 hold 和 ramp 两种段类型。
        :return: 与当前时长一致的周期包络数组
        """
        cycle = self._build_envelope_cycle(segments)
        target_length = len(self.signal_data)
        repeat_count = int(np.ceil(target_length / len(cycle)))
        envelope = np.tile(cycle, repeat_count)[:target_length]

        self.logger.info(f"生成周期包络: 周期点数={len(cycle)}, 总点数={target_length}")
        return envelope

    def apply_envelope(self, signal: np.ndarray, envelope: np.ndarray) -> np.ndarray:
        """将包络逐点乘到输入信号上"""
        signal_array = np.asarray(signal, dtype=float)
        envelope_array = np.asarray(envelope, dtype=float)

        if len(signal_array) != len(envelope_array):
            raise ValueError("signal 和 envelope 长度必须一致")

        self.logger.info(f"应用包络: 点数={len(signal_array)}")
        return signal_array * envelope_array

    def apply_noise(self, signal: np.ndarray, reference_amp: float = 1.0) -> np.ndarray:
        """
        对已有信号追加噪声

        :param signal: 输入信号
        :param reference_amp: 噪声参考幅值, 最终噪声标准差 = noise_level * reference_amp
        :return: 添加噪声后的信号
        """
        if reference_amp < 0:
            raise ValueError("reference_amp 必须非负")

        signal_array = np.asarray(signal, dtype=float)
        self.logger.info(f"应用噪声: noise_level={self._noise_level}, reference_amp={reference_amp}")
        if self._noise_level > 0:
            noise = self._noise_level * reference_amp * np.random.randn(len(signal_array))
            return signal_array + noise
        return signal_array

    def modulated_sine_wave(self,
                            freqs: Sequence[float],
                            envelope_segments: Sequence[Mapping[str, Any]],
                            amps: Optional[Sequence[float]] = None,
                            phases: Optional[Sequence[float]] = None) -> np.ndarray:
        """
        生成调幅后的多正弦信号, 不自动添加噪声

        :param freqs: 正弦频率列表
        :param envelope_segments: 周期包络分段配置
        :param amps: 各正弦幅值
        :param phases: 各正弦相位
        :return: 调制后的时序信号
        """
        freqs_array, amps_array, phases_array = self._normalize_sine_params(freqs, amps, phases)
        carrier = self._compose_sine_wave(freqs_array, amps_array, phases_array)
        envelope = self.periodic_envelope(envelope_segments)
        modulated_signal = self.apply_envelope(carrier, envelope)

        self.logger.info(
            f"生成调制正弦波: freqs={freqs_array}, amps={amps_array}, phases={phases_array}"
        )
        return modulated_signal

    def modulated_harmonic_wave(self,
                                fundamental_freq: float = 50.0,
                                num_harmonics: int = 5,
                                amps: Optional[Union[Sequence[float], str]] = 'typical',
                                phases: Optional[Sequence[float]] = None,
                                envelope_segments: Optional[Sequence[Mapping[str, Any]]] = None) -> np.ndarray:
        """
        生成调幅后的谐波信号, 不自动添加噪声

        :param fundamental_freq: 基频 (Hz)
        :param num_harmonics: 谐波数量(包括基波)
        :param amps: 各谐波幅度
        :param phases: 各谐波相位
        :param envelope_segments: 周期包络分段配置
        :return: 调制后的谐波信号
        """
        if envelope_segments is None:
            raise ValueError("envelope_segments 不能为空")

        freqs_array, amps_array, phases_array = self._resolve_harmonic_components(
            fundamental_freq=fundamental_freq,
            num_harmonics=num_harmonics,
            amps=amps,
            phases=phases,
        )
        self.logger.info(f"生成调制谐波: 基频={fundamental_freq}Hz, 谐波数={num_harmonics}")
        return self.modulated_sine_wave(
            freqs=freqs_array,
            amps=amps_array,
            phases=phases_array,
            envelope_segments=envelope_segments,
        )

    def insert_into_csv(self,
                        csv_path: PathLikeStr,
                        column: str = "signal",
                        new_signal: Optional[np.ndarray] = None,
                        append: bool = False,
                        save_path: Optional[PathLikeStr] = None,
                        use_datetime: bool = False,
                        time_format: Optional[str] = None) -> pd.DataFrame:
        """
        :param csv_path: CSV 文件路径。append=True 时作为读取源；否则作为默认输出路径
        :param column: 插入列的名字
        :param new_signal: 新信号 (numpy 数组)
        :param append: True=追加到现有文件尾部; False=生成新的结果文件
        :param save_path: 保存路径(如果 None, 则默认使用 csv_path；append=False 且输出重名时自动生成 -new 文件)
        :param use_datetime: 是否使用datetime时间戳(默认False保留兼容性)
        :param time_format: datetime格式字符串；None时按采样率自动选择到秒或毫秒
        """
        csv_path_str = os.fspath(csv_path)
        requested_save_path = os.fspath(save_path) if save_path is not None else csv_path_str
        if not isinstance(append, bool):
            raise ValueError("append 参数必须是 bool 值")
        signal_array = self._normalize_insert_signal(new_signal)

        if not append and len(signal_array) > self._num_samples:
            raise ValueError("new_signal 长度不能超过当前生成器样本数")

        actual_save_path = self._resolve_write_path(requested_save_path, append)

        if not append:
            time_values = self._build_insert_time_values(0, len(signal_array), use_datetime, time_format)
            df = pd.DataFrame({"time": time_values, column: signal_array})
        else:
            file_exists = os.path.exists(csv_path_str)
            if file_exists:
                df = pd.read_csv(csv_path_str)
            else:
                self.logger.warning(f"文件 {csv_path_str} 不存在, 新建文件")
                df = pd.DataFrame()

            write_start = len(df)
            if not file_exists:
                time_values = self._build_insert_time_values(
                    0,
                    len(signal_array),
                    use_datetime,
                    time_format,
                    enforce_generator_limit=False,
                )
                df = pd.DataFrame({"time": time_values, column: signal_array})
            else:
                if column not in df.columns:
                    df[column] = 0.0
                self.logger.info("append=True, 执行追加操作")
                time_values = self._build_insert_time_values(
                    write_start,
                    len(signal_array),
                    use_datetime,
                    time_format,
                    enforce_generator_limit=False,
                )
                df_extra = pd.DataFrame({"time": time_values, column: signal_array})
                df = pd.concat([df, df_extra], ignore_index=True)

        df.to_csv(actual_save_path, index=False)
        self.logger.info(f"信号已写入 {actual_save_path}, 列: {column}, append: {append}, "
                         f"datetime模式: {use_datetime}")
        return df

    def trim_signal(self, signal: np.ndarray,
                    n_points: int = 0,
                    frequency: Optional[float] = None,
                    phase_angle: Optional[float] = None,
                    from_end: bool = True) -> np.ndarray:
        """
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


# ===================== 测试 =====================
def test():
    # 按需求生成 1Hz 采样、1800s 时长的调幅谐波信号。
    gen = SignalGenerator(sampling_rate=1, duration=1800.0, noise_level=0.1, seed=42)

    # 周期包络从低谷平台开始：低谷 50s -> 正弦上升 25s -> 高峰 50s -> 正弦下降 25s。
    envelope_segments = [
        {"type": "hold", "duration": 50, "value": 0.2},
        {"type": "ramp", "duration": 25, "start": 0.2, "end": 1.0, "curve": "sine"},
        {"type": "hold", "duration": 50, "value": 1.0},
        {"type": "ramp", "duration": 25, "start": 1.0, "end": 0.2, "curve": "sine"},
    ]

    # 生成基频 0.1Hz、二次谐波幅值为基波 1/2 的调幅信号, φ1 = φ2 = 0。
    clean_signal = gen.modulated_harmonic_wave(
        fundamental_freq=0.1,
        num_harmonics=2,
        amps=[1.0, 0.5],
        phases=[0.0, 0.0],
        envelope_segments=envelope_segments,
    )

    # 按基波幅值 1.0 添加 10% 比例噪声。
    noisy_signal = gen.apply_noise(clean_signal, reference_amp=1.0)

    print(f"生成信号长度: {len(noisy_signal)}")
    print(f"采样率: {gen._sampling_rate}Hz, 时长: {gen._duration}s")

    gen.insert_into_csv(
        "./real-data.csv",
        column="sig",
        new_signal=noisy_signal,
        append=True,
        use_datetime=True,
    )


if __name__ == "__main__":
    test()
