# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于FFT频域分析的振荡检测工具，适用于电网频率监测和信号分析。项目包含三个独立但相关的分析系统：静态FFT分析、动态频率分析和长期振荡检测。

## 核心架构

### 模块依赖关系

```
src/core/SignalGenerator (基础层)
    ↓ 被示例/测试使用
    ├─→ src/core/FFT_analyzer (单次FFT分析/精细化)
    │       ↓
    │   src/OscillationDetection (长期振荡检测)
    │       ↓
    │   tests/test_oscillation_detection.py (开发流程)
    │
    └─→ src/Freq_dynamic_analyzer (动态频率分析，依赖 Zero_Cross_Freq/FrequencyRefinement)
            ↓
        tests/test_dynamic_fft.py (示例/烟囱测试)
```

### 三种分析模式对比

| 模式 | 采样率 | 窗口大小 | 用途 | 输入格式 |
|------|--------|---------|------|---------|
| **FFT_analyzer** | 10kHz | 800点(0.08s) | 静态信号质量分析 | numpy数组 |
| **Freq_dynamic_analyzer** | 可配置(默认10kHz) | 可配置(默认200ms) | 动态频率跟踪 | CSV+datetime时间戳 |
| **OscillationDetection** | 1Hz | 60点(60s) | 长期振荡监测 | CSV |

### 关键设计模式

1. **时间戳系统** (SignalGenerator)
   - `signal_data`: 秒浮点数组（向后兼容）
   - `timestamps`: datetime数组（新功能）
   - `use_datetime` 参数控制输出格式

2. **配置驱动架构**
   - JSON配置文件优先
   - 支持编程式参数覆盖
   - 配置验证和默认值处理

3. **滑动窗口分析**
   - `window_size`: 窗口大小（数据点数）
   - `step_size = window_size * (1 - overlap_ratio)`
   - 窗口结束时刻作为该窗口的时间戳

## 常用命令

### 运行测试和示例

```bash
# 静态FFT分析测试
python -m src.core.FFT_analyzer

# 动态频率分析示例
python tests/test_dynamic_fft.py
# 或使用默认配置跑一遍管线
python -m src.Freq_dynamic_analyzer

# 振荡检测 - 创建配置文件模板
python tests/test_oscillation_detection.py --create-config --config src/oscillate_dev_settings.json

# 振荡检测 - 实时动画模式
python tests/test_oscillation_detection.py --mode animation --config src/oscillate_dev_settings.json

# 振荡检测 - 静态批量分析
python tests/test_oscillation_detection.py --mode static --config src/oscillate_dev_settings.json
```

### 生成测试数据

```python
# 生成带datetime时间戳的测试信号
from src.core.SignalGenerator import SignalGenerator
from datetime import datetime

gen = SignalGenerator(
    sampling_rate=10000,
    duration=1.0,
    start_time=datetime(2025, 12, 10, 10, 0, 0)
)
signal = gen.harmonic_wave(fundamental_freq=50.0)
gen.insert_into_csv(
    "test.csv",
    new_signal=signal,
    use_datetime=True,
    time_format="%Y-%m-%d %H:%M:%S.%f"
)
```

### 动态频率分析

```python
# 使用配置文件
from src.Freq_dynamic_analyzer import FreqDynamicAnalyzer

analyzer = FreqDynamicAnalyzer(config_path="src/config_fft_dynamic.json")
results = analyzer.run_pipeline()

# 编程式配置
analyzer = FreqDynamicAnalyzer(
    window_duration_ms=200,
    step_duration_ms=100,
    sampling_rate=10000,
    freq_range=(49.9, 50.1)
)
df = analyzer.load_csv("input.csv")
results_df = analyzer.analyze_dynamic(df)
analyzer.save_results(results_df, "output.csv")
```

## 代码规范

### 时间戳格式约定

- **输入格式** (微秒精度): `2025-12-10 10:34:37.998700`
- **输出格式** (毫秒精度): `2025/12/10 10:34:37::608`
  - 注意：毫秒部分用 `::` 分隔，而非 `.`

### 时间戳格式化实现

```python
def _format_timestamp_output(self, dt: datetime) -> str:
    date_str = dt.strftime("%Y/%m/%d %H:%M:%S")
    milliseconds = dt.microsecond // 1000
    return f"{date_str}::{milliseconds:03d}"
```

### FFT分析返回值约定

所有FFT分析函数返回格式：
```python
(success: bool, frequency: float, amplitude: float, phase: float)
```
- `phase` 单位为弧度，需要时使用 `np.degrees()` 转换

### 窗口大小计算

```python
# 基于时长计算
window_samples = int(sampling_rate * window_duration_ms / 1000)

# 基于基频倍数计算（用于振荡检测）
window_size = int(sampling_rate / fundamental_freq * window_Ts)
```

## 向后兼容性

### SignalGenerator 重构说明

SignalGenerator 已添加 datetime 支持，但完全向后兼容：

```python
# ✅ 原有代码无需修改
gen = SignalGenerator(sampling_rate=1000, duration=10.0)
signal = gen.sine_wave(freqs=[50])
gen.insert_into_csv("test.csv", new_signal=signal)

# ✅ 新功能：datetime时间戳
gen = SignalGenerator(
    sampling_rate=1000,
    start_time=datetime(2025, 12, 10, 10, 0, 0)
)
gen.insert_into_csv("test.csv", new_signal=signal, use_datetime=True)
```

### 关键点
- `signal_data` 属性保留（秒浮点数组）
- 新增 `timestamps` 属性（datetime数组）
- `insert_into_csv()` 的 `use_datetime` 默认为 `False`

## 配置文件结构

### src/config_fft_dynamic.json (动态频率分析)

```json
{
  "input": {
    "csv_path": "csv-data/clean_200ms_liner_20251210.csv",
    "time_column": "Time [s]",
    "signal_column": "AI 1/U4一次调频动作 [V]",
    "time_format": "%Y-%m-%d %H:%M:%S.%f"
  },
  "output": {
    "csv_path": "csv-data/fft_analysis_results.csv",
    "frequency_decimals": 3,
    "amplitude_decimals": 4,
    "phase_decimals": 2
  },
  "analysis": {
    "window_duration_ms": 200,
    "step_duration_ms": 100,
    "sampling_rate": 10000,
    "frequency_range": [49.9, 50.1],
    "use_window": true,
    "use_ipdft": true,
    "use_zero_crossing": true,
    "zero_cross_config": {
      "window_periods": 6,
      "min_freq_hz": 45.0,
      "max_freq_hz": 65.0,
      "fake_period_ms": 0.0,
      "min_cross_amplitude": 0.0,
      "remove_dc": true,
      "rising_only": true
    },
    "refine_frequency": true,
    "refine_config": {
      "method": "grid_search",
      "search_range": 0.2,
      "step_size": 0.001
    }
  },
  "logging": {
    "log_file": "./log/Freq_dynamic_analyzer.log",
    "log_level": "INFO"
  },
  "description": "动态频率分析配置: 200ms窗口, 100ms步长, 10kHz采样率, 关注50Hz频率"
}
```

### src/oscillate_dev_settings.json (振荡检测)

关键参数：
- `window_size`: 60 (默认，点数)
- `sampling_rate`: 1.0 (Hz，长期监测)
- `overlap_ratio`: 0.5 (50%重叠)
- `threshold`: 振荡判断阈值
- `generate_signal`: true/false (是否生成测试信号)

## 频率分辨率计算

```
频率分辨率(Δf) = sampling_rate / window_size

示例：
- FFT_analyzer: 10000Hz / 800 = 12.5Hz
- Freq_dynamic_analyzer: 10000Hz / 2000 = 5Hz (200ms窗口)
- OscillationDetection: 1Hz / 60 = 0.0167Hz
```

## 插值DFT优化

使用 Jacobsen 三点插值法提高频率精度：

```python
# 当 IpDFT=True 时
delta = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
peak_freq = peak_freq + delta * frequency_resolution
peak_amp = beta - 0.25 * (alpha - gamma) * delta
```

可将频率精度从分辨率（5-12Hz）提高到 0.01Hz 级别。

## 频率精细化（最小二乘拟合）

基于“FFT 粗估 + 最小二乘拟合”，在保持向后兼容的前提下提供 mHz 级频率估计。

```python
from src.core.FFT_analyzer import FFTAnalyzer

analyzer = FFTAnalyzer(window_size=800, sampling_rate=10000)

# 启用精化
success, freq, amp, phase = analyzer.fft_analyze(
    signal,
    use_window=True,
    IpDFT=True,
    refine_frequency=True,
    refine_config={
        "method": "minimize_scalar",  # 默认；无 SciPy 时自动回退到 grid_search
        "search_range": 0.05,         # 可选：覆盖 ±range
        "step_size": 0.001            # grid_search 步长（Hz）
    }
)
```

- **自适应搜索范围**：默认 `range = max(5*Δf, 0.05)`；若初值在 49–51Hz，软限制为 `min(range, 0.5)`。精化失败时自动扩大一档后重试。
- **模型**：给定频率下，线性最小二乘解 `y = a*sin(ωt) + b*cos(ωt) + dc`，再做一维频率搜索（Brent/网格+二次插值）。
- **适用场景**：单频 + DC，SNR≥30dB，数据长度≥0.16s（推荐 ≥0.5s）。实时循环默认关闭，仅按需开启。

## 日志系统

所有模块使用统一的日志模式：
- **控制台**: INFO级别，简洁格式 `[%(levelname)s] %(message)s`
- **文件**: INFO级别，完整格式 `[%(asctime)s] %(message)s`
- 日志目录自动创建（`./log/`）

## 项目特定规范

基于用户的全局 CLAUDE.md 规范：

1. **模块化设计**: 避免硬编码，使用配置文件或参数传递
2. **中文注释**: 代码关键步骤使用简洁的中文注释
3. **假设先行**: 信息不足时先列出假设再实现
4. **避免生成文档**: 代码修改不主动生成总结性文档文件

## 边界情况处理

### Freq_dynamic_analyzer 边界处理

| 情况 | 处理方式 |
|------|---------|
| 数据长度不足窗口大小 | 返回空结果并记录错误 |
| 采样率不均匀（>5%误差） | 警告日志，继续分析 |
| 时间戳乱序 | 自动排序并警告 |
| 缺失值 | 前向填充 `fillna(method='ffill')` |
| 频率超出范围 | 跳过该窗口结果 |

## 文件组织

```
src/                # 核心代码与配置 (config_fft_dynamic.json, oscillate_dev_settings.json)
src/core/           # FFT_analyzer、SignalGenerator、Zero_Cross_Freq、FrequencyRefinement
src/Freq_dynamic_analyzer.py     # 动态频率分析入口
src/OscillationDetection.py      # 振荡检测核心
src/com/modbus-dcs.py            # Modbus RTU 原型
tests/              # 示例/测试脚本 (test_dynamic_fft.py 等)
csv-data/           # CSV数据目录
log/                # 日志目录（自动创建）
plots/              # 图表输出目录
docs/               # 说明文档
```

## 性能考虑

- **大文件处理**: 405454行数据约需 5秒，内存使用 < 10MB
- **窗口优化**: 使用向量化操作避免Python循环
- **内存优化**: 分块处理大文件（如需要）

## 测试数据生成模式

用于生成不同类型的测试信号：

```python
# 正弦波叠加
signal = gen.sine_wave(freqs=[50, 120], amps=[1, 0.5])

# 谐波信号（电网典型）
signal = gen.harmonic_wave(
    fundamental_freq=50.0,
    num_harmonics=5,
    amps='typical'  # 或 'linear', 'exponential'
)

# 多项式 + 指数
signal = gen.polynomial(coeffs=[1, -2, 1])
signal += gen.exponential(A=1, tau=0.5)
```
