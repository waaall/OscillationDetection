import os
import sys
import json
import logging
from typing import List, Tuple, Optional, Callable
from datetime import datetime
import numpy as np
import pandas as pd

from FFT_analyzer import FFTAnalyzer


class ConfigLoader:
    """配置文件加载器"""

    @staticmethod
    def load(config_path: str) -> dict:
        """加载JSON配置文件"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config

    @staticmethod
    def create_default(output_path: str):
        """创建默认配置文件"""
        default_config = {
            "input": {
                "csv_path": "csv-data/input.csv",
                "time_column": "Time [s]",
                "signal_column": "signal",
                "time_format": "%Y-%m-%d %H:%M:%S.%f"
            },
            "output": {
                "csv_path": "csv-data/output.csv",
                "frequency_decimals": 3,
                "amplitude_decimals": 4,
                "phase_decimals": 2
            },
            "analysis": {
                "window_duration_ms": 200,
                "step_duration_ms": 100,
                "sampling_rate": 10000,
                "frequency_range": [49.9, 50.1],
                "use_window": True,
                "use_ipdft": True
            },
            "logging": {
                "log_file": "./log/fft_dynamic_analyzer.log",
                "log_level": "INFO"
            }
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2, ensure_ascii=False)


class FFTDynamicAnalyzer:
    """
    动态FFT分析器：使用滑动窗口对时序数据进行连续FFT分析

    核心功能：
    1. 读取带datetime时间戳的CSV文件
    2. 应用滑动窗口策略
    3. 调用FFTAnalyzer进行频率分析
    4. 输出带时间戳的频率/幅值/相位结果
    """

    def __init__(self,
                 config_path: str = None,
                 window_duration_ms: float = 200,
                 step_duration_ms: float = 100,
                 sampling_rate: int = 10000,
                 freq_range: Tuple[float, float] = (49.9, 50.1),
                 use_window: bool = True,
                 use_ipdft: bool = True,
                 log_file: Optional[str] = None):
        """
        初始化动态FFT分析器

        :param config_path: JSON配置文件路径，若提供则覆盖其他参数
        :param window_duration_ms: 分析窗口时长(毫秒)，默认200ms
        :param step_duration_ms: 滑动步长(毫秒)，默认100ms
        :param sampling_rate: 信号采样率(Hz)
        :param freq_range: 关注的频率范围，用于结果过滤
        :param use_window: FFT分析时是否使用汉宁窗
        :param use_ipdft: 是否使用Jacobsen插值法提高频率精度
        :param log_file: 日志文件路径
        """
        # 加载配置文件（如果提供）
        self.config = None
        if config_path:
            self.config = ConfigLoader.load(config_path)
            analysis_cfg = self.config.get('analysis', {})
            logging_cfg = self.config.get('logging', {})

            window_duration_ms = analysis_cfg.get('window_duration_ms', window_duration_ms)
            step_duration_ms = analysis_cfg.get('step_duration_ms', step_duration_ms)
            sampling_rate = analysis_cfg.get('sampling_rate', sampling_rate)
            freq_range = tuple(analysis_cfg.get('frequency_range', freq_range))
            use_window = analysis_cfg.get('use_window', use_window)
            use_ipdft = analysis_cfg.get('use_ipdft', use_ipdft)
            log_file = logging_cfg.get('log_file', log_file)

        # 参数验证
        if window_duration_ms <= 0 or step_duration_ms <= 0:
            raise ValueError("窗口大小和步长必须为正数")
        if sampling_rate <= 0:
            raise ValueError("采样率必须为正数")
        if step_duration_ms > window_duration_ms:
            raise ValueError("步长不能大于窗口大小")

        # 保存配置
        self.window_duration_ms = window_duration_ms
        self.step_duration_ms = step_duration_ms
        self.sampling_rate = sampling_rate
        self.freq_range = freq_range
        self.use_window = use_window
        self.use_ipdft = use_ipdft

        # 计算窗口大小和步长（数据点数）
        self.window_samples = int(sampling_rate * window_duration_ms / 1000)
        self.step_samples = int(sampling_rate * step_duration_ms / 1000)

        # 设置日志
        self._setup_logger(log_file)

        self.logger.info(f"FFTDynamicAnalyzer 初始化完成: "
                        f"窗口={window_duration_ms}ms ({self.window_samples}点), "
                        f"步长={step_duration_ms}ms ({self.step_samples}点), "
                        f"采样率={sampling_rate}Hz, "
                        f"频率范围={freq_range}Hz")

    def _setup_logger(self, log_file: Optional[str]):
        """设置日志系统"""
        self.logger = logging.getLogger("FFTDynamicAnalyzer")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # 文件处理器（可选）
        if log_file is not None:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                              datefmt='%Y-%m-%d %H:%M:%S')
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

        self.logger.propagate = True

    def _parse_timestamp(self, time_str: str) -> datetime:
        """
        灵活解析时间戳，支持多种格式

        :param time_str: 时间字符串
        :return: datetime对象
        """
        # 去除引号
        time_str = time_str.strip().strip('"\'')

        formats = [
            "%Y-%m-%d %H:%M:%S.%f",      # 2025-12-10 10:34:37.998700
            "%Y/%m/%d %H:%M:%S.%f",      # 2025/12/10 10:34:37.998700
            "%Y-%m-%d %H:%M:%S",         # 2025-12-10 10:34:37
            "%Y/%m/%d %H:%M:%S",         # 2025/12/10 10:34:37
        ]

        for fmt in formats:
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue

        raise ValueError(f"无法解析时间戳: {time_str}")

    def _format_timestamp_output(self, dt: datetime) -> str:
        """
        格式化时间戳为输出格式

        输入: datetime(2025, 12, 9, 10, 1, 37, 608123)
        输出: "2025/12/09 10:01:37::608"

        :param dt: datetime对象
        :return: 格式化的时间字符串
        """
        date_str = dt.strftime("%Y/%m/%d %H:%M:%S")
        milliseconds = dt.microsecond // 1000  # 微秒转毫秒
        return f"{date_str}::{milliseconds:03d}"

    def _validate_sampling_rate(self, df: pd.DataFrame) -> float:
        """
        验证实际采样率与配置是否一致

        :param df: 输入DataFrame
        :return: 实际平均采样率
        """
        if len(df) < 2:
            self.logger.warning("数据点太少，无法验证采样率")
            return self.sampling_rate

        # 计算时间间隔
        time_diffs = df['timestamp'].diff().dropna()
        time_diffs_seconds = time_diffs.dt.total_seconds()

        # 统计信息
        mean_interval = time_diffs_seconds.mean()
        std_interval = time_diffs_seconds.std()
        actual_sampling_rate = 1.0 / mean_interval

        # 验证采样率一致性
        expected_interval = 1.0 / self.sampling_rate
        relative_error = abs(mean_interval - expected_interval) / expected_interval

        if relative_error > 0.05:  # 5%误差阈值
            self.logger.warning(
                f"采样率不一致！配置={self.sampling_rate}Hz, "
                f"实际={actual_sampling_rate:.2f}Hz (误差{relative_error*100:.1f}%)"
            )

        # 检测采样不均匀性
        if std_interval / mean_interval > 0.1:  # 10%变异系数阈值
            self.logger.warning(
                f"采样不均匀！时间间隔标准差={std_interval*1000:.3f}ms"
            )

        self.logger.info(f"采样率验证: 实际={actual_sampling_rate:.2f}Hz, "
                        f"间隔均值={mean_interval*1000:.3f}ms")

        return actual_sampling_rate

    def load_csv(self,
                 csv_path: str,
                 time_column: str = None,
                 signal_column: str = None,
                 time_format: str = None) -> pd.DataFrame:
        """
        加载CSV文件并解析时间戳

        :param csv_path: CSV文件路径
        :param time_column: 时间列名（从配置文件读取或使用默认值）
        :param signal_column: 信号数据列名
        :param time_format: 时间戳格式
        :return: DataFrame (columns: ['timestamp', 'signal'])
        """
        # 从配置文件获取参数（如果可用）
        if self.config:
            input_cfg = self.config.get('input', {})
            time_column = time_column or input_cfg.get('time_column', 'Time [s]')
            signal_column = signal_column or input_cfg.get('signal_column', 'signal')
            time_format = time_format or input_cfg.get('time_format', '%Y-%m-%d %H:%M:%S.%f')
        else:
            time_column = time_column or 'time'
            signal_column = signal_column or 'signal'
            time_format = time_format or '%Y-%m-%d %H:%M:%S.%f'

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

        self.logger.info(f"加载CSV文件: {csv_path}")

        # 读取CSV文件
        df = pd.read_csv(csv_path)

        # 检查列是否存在
        if time_column not in df.columns:
            raise ValueError(f"时间列 '{time_column}' 不存在。可用列: {list(df.columns)}")
        if signal_column not in df.columns:
            raise ValueError(f"信号列 '{signal_column}' 不存在。可用列: {list(df.columns)}")

        # 解析时间戳
        self.logger.info("解析时间戳...")
        df['timestamp'] = df[time_column].apply(self._parse_timestamp)
        df['signal'] = df[signal_column].astype(float)

        # 创建干净的DataFrame
        result_df = df[['timestamp', 'signal']].copy()

        # 处理缺失值
        if result_df['signal'].isnull().any():
            null_count = result_df['signal'].isnull().sum()
            self.logger.warning(f"发现 {null_count} 个缺失值，使用前向填充")
            result_df['signal'].fillna(method='ffill', inplace=True)
            result_df['signal'].fillna(method='bfill', inplace=True)  # 处理开头的缺失值

        # 检查时间戳顺序
        if not result_df['timestamp'].is_monotonic_increasing:
            self.logger.warning("时间戳不是递增顺序，进行排序")
            result_df = result_df.sort_values('timestamp').reset_index(drop=True)

        # 验证采样率
        self._validate_sampling_rate(result_df)

        self.logger.info(f"CSV加载完成: {len(result_df)} 个数据点")

        return result_df

    def analyze_dynamic(self,
                       df: pd.DataFrame,
                       progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        动态滑窗FFT分析

        :param df: 输入DataFrame (columns: ['timestamp', 'signal'])
        :param progress_callback: 进度回调函数 callback(current, total)
        :return: 结果DataFrame (columns: ['timestamp', 'frequency', 'amplitude', 'phase'])
        """
        # 检查数据长度
        if len(df) < self.window_samples:
            self.logger.error(
                f"数据长度不足！需要至少 {self.window_samples} 个点，"
                f"实际只有 {len(df)} 个点"
            )
            return pd.DataFrame(columns=['timestamp', 'frequency', 'amplitude', 'phase'])

        # 初始化FFT分析器
        fft_analyzer = FFTAnalyzer(
            window_size=self.window_samples,
            sampling_rate=self.sampling_rate,
            log_file=None  # 不重复记录日志
        )

        # 计算窗口数量
        total_windows = (len(df) - self.window_samples) // self.step_samples + 1
        self.logger.info(f"开始动态FFT分析: {total_windows} 个窗口")

        # 滑动窗口遍历
        results = []
        window_count = 0

        for i in range(0, len(df) - self.window_samples + 1, self.step_samples):
            # 提取窗口数据
            window_data = df['signal'].iloc[i:i+self.window_samples].values
            window_end_time = df['timestamp'].iloc[i + self.window_samples - 1]

            # FFT分析
            success, freq, amp, phase = fft_analyzer.fft_analyze(
                window_data,
                use_window=self.use_window,
                IpDFT=self.use_ipdft
            )

            # 结果过滤（频率范围）
            if success and self.freq_range[0] <= freq <= self.freq_range[1]:
                results.append({
                    'timestamp': window_end_time,
                    'frequency': freq,
                    'amplitude': amp,
                    'phase': phase
                })
                window_count += 1
            elif success:
                self.logger.debug(
                    f"窗口 {window_count}: 频率 {freq:.3f}Hz 超出范围 {self.freq_range}，跳过"
                )

            # 进度回调
            if progress_callback:
                progress_callback(i // self.step_samples + 1, total_windows)

        self.logger.info(f"FFT分析完成: 共 {window_count} 个有效窗口")

        return pd.DataFrame(results)

    def save_results(self,
                    results_df: pd.DataFrame,
                    output_path: str,
                    frequency_decimals: int = None,
                    amplitude_decimals: int = None,
                    phase_decimals: int = None) -> None:
        """
        保存分析结果到CSV

        :param results_df: 结果DataFrame
        :param output_path: 输出文件路径
        :param frequency_decimals: 频率小数位数（从配置读取或使用默认值）
        :param amplitude_decimals: 幅值小数位数
        :param phase_decimals: 相位小数位数
        """
        if len(results_df) == 0:
            self.logger.warning("结果为空，不生成输出文件")
            return

        # 从配置文件获取参数
        if self.config:
            output_cfg = self.config.get('output', {})
            frequency_decimals = frequency_decimals or output_cfg.get('frequency_decimals', 3)
            amplitude_decimals = amplitude_decimals or output_cfg.get('amplitude_decimals', 4)
            phase_decimals = phase_decimals or output_cfg.get('phase_decimals', 2)
        else:
            frequency_decimals = frequency_decimals or 3
            amplitude_decimals = amplitude_decimals or 4
            phase_decimals = phase_decimals or 2

        # 格式化输出DataFrame
        output_df = pd.DataFrame({
            'RX Date/Time': results_df['timestamp'].apply(self._format_timestamp_output),
            '组/A_Freq': results_df['frequency'].round(frequency_decimals),
        })

        # 如果需要包含幅值和相位
        # output_df['Amplitude'] = results_df['amplitude'].round(amplitude_decimals)
        # output_df['Phase_deg'] = np.degrees(results_df['phase']).round(phase_decimals)

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # 保存到CSV
        output_df.to_csv(output_path, index=False)
        self.logger.info(f"结果已保存至: {output_path} ({len(output_df)} 行)")

    def run_pipeline(self,
                    input_csv: str = None,
                    output_csv: str = None,
                    **csv_params) -> pd.DataFrame:
        """
        完整分析流程

        :param input_csv: 输入CSV路径（从配置读取或手动指定）
        :param output_csv: 输出CSV路径
        :param csv_params: load_csv()的额外参数
        :return: 分析结果DataFrame
        """
        # 从配置文件获取路径
        if self.config:
            input_csv = input_csv or self.config['input']['csv_path']
            output_csv = output_csv or self.config['output']['csv_path']

        if not input_csv or not output_csv:
            raise ValueError("必须指定输入和输出CSV路径")

        self.logger.info("=" * 60)
        self.logger.info("开始动态FFT分析流程")
        self.logger.info("=" * 60)

        # 1. 加载CSV
        df = self.load_csv(input_csv, **csv_params)

        # 2. 动态分析
        results_df = self.analyze_dynamic(df)

        # 3. 保存结果
        self.save_results(results_df, output_csv)

        self.logger.info("=" * 60)
        self.logger.info("分析流程完成")
        self.logger.info("=" * 60)

        return results_df


# ===================== 测试 =====================
def test_dynamic_analyzer():
    """测试动态FFT分析器"""
    # 使用配置文件
    analyzer = FFTDynamicAnalyzer(config_path="config_fft_dynamic.json")

    # 运行完整流程
    results = analyzer.run_pipeline()

    print(f"\n分析完成，共 {len(results)} 个窗口结果")
    if len(results) > 0:
        print("\n前5个结果:")
        print(results.head())


if __name__ == "__main__":
    test_dynamic_analyzer()
