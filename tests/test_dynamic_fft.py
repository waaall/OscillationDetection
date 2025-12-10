"""
动态FFT分析系统使用示例

演示如何使用FFTDynamicAnalyzer和SignalGenerator进行动态频率分析
"""

from pathlib import Path
from datetime import datetime
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.FFT_dynamic_analyzer import FFTDynamicAnalyzer
from src.core.SignalGenerator import SignalGenerator


def example1_generate_test_signal():
    """示例1：生成带datetime时间戳的测试信号"""
    print("=" * 60)
    print("示例1：生成测试信号")
    print("=" * 60)

    # 创建信号生成器
    gen = SignalGenerator(
        sampling_rate=10000,      # 10kHz采样率
        duration=2.0,              # 2秒
        start_time=datetime(2025, 12, 10, 10, 0, 0)  # 起始时间
    )

    # 生成50.02Hz的谐波信号
    signal = gen.harmonic_wave(fundamental_freq=50.02)

    # 保存为datetime格式的CSV
    gen.insert_into_csv(
        'csv-data/example_input.csv',
        column='signal',
        new_signal=signal,
        use_datetime=True,
        time_format='%Y-%m-%d %H:%M:%S.%f'
    )

    print(f"\n✓ 测试信号已生成：{len(signal)} 个数据点")
    print("✓ 已保存至: csv-data/example_input.csv")


def example2_analyze_with_config():
    """示例2：使用配置文件进行分析"""
    print("\n" + "=" * 60)
    print("示例2：使用配置文件进行分析")
    print("=" * 60)

    # 使用配置文件
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "src" / "core" / "config_fft_dynamic.json"
    analyzer = FFTDynamicAnalyzer(config_path=str(config_path))

    # 运行完整流程
    results = analyzer.run_pipeline()

    print(f"\n✓ 分析完成：{len(results)} 个窗口")
    if len(results) > 0:
        print(f"✓ 检测频率范围: {results['frequency'].min():.3f} - {results['frequency'].max():.3f} Hz")
        print(f"✓ 平均频率: {results['frequency'].mean():.3f} Hz")


def example3_analyze_programmatically():
    """示例3：编程式配置进行分析"""
    print("\n" + "=" * 60)
    print("示例3：编程式配置进行分析")
    print("=" * 60)

    # 不使用配置文件，直接设置参数
    analyzer = FFTDynamicAnalyzer(
        window_duration_ms=200,    # 200ms窗口
        step_duration_ms=100,       # 100ms步长
        sampling_rate=10000,        # 10kHz采样率
        freq_range=(49.9, 50.1),   # 关注频率范围
        use_window=True,            # 使用汉宁窗
        use_ipdft=True,             # 使用插值DFT
        log_file='./log/example.log'
    )

    # 分步执行
    df = analyzer.load_csv(
        'csv-data/test_dynamic_input.csv',
        time_column='time',
        signal_column='signal'
    )

    results_df = analyzer.analyze_dynamic(df)

    analyzer.save_results(
        results_df,
        'csv-data/example_output.csv'
    )

    print(f"\n✓ 分析完成：{len(results_df)} 个窗口")
    print("✓ 结果已保存至: csv-data/example_output.csv")


def example4_backward_compatibility():
    """示例4：向后兼容性测试"""
    print("\n" + "=" * 60)
    print("示例4：SignalGenerator向后兼容性")
    print("=" * 60)

    # 原有用法（不使用datetime）
    gen_old = SignalGenerator(sampling_rate=1000, duration=1.0)
    signal_old = gen_old.sine_wave(freqs=[50], amps=[1.0])
    print(f"✓ 原有用法正常工作：生成 {len(signal_old)} 个数据点")
    print(f"✓ 时间数组类型：{type(gen_old.signal_data[0])}")

    # 新用法（使用datetime）
    gen_new = SignalGenerator(
        sampling_rate=1000,
        duration=1.0,
        start_time=datetime(2025, 12, 10, 10, 0, 0)
    )
    signal_new = gen_new.sine_wave(freqs=[50], amps=[1.0])
    print(f"\n✓ 新功能正常工作：生成 {len(signal_new)} 个数据点")
    print(f"✓ timestamps类型：{type(gen_new.timestamps[0])}")
    print(f"✓ 同时保留 signal_data (向后兼容): {type(gen_new.signal_data[0])}")


def main():
    """主函数：运行所有示例"""
    print("\n" + "=" * 60)
    print("动态FFT分析系统使用示例")
    print("=" * 60)

    # 运行所有示例
    example1_generate_test_signal()
    example2_analyze_with_config()
    example3_analyze_programmatically()
    example4_backward_compatibility()

    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
