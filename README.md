# OscillationDetection

基于FFT频域分析的振荡检测工具，适用于频率监测和信号分析。

## 项目结构

```
├── SignalGenerator.py          # 信号生成器（基础依赖）
├── FFT_analyzer.py            # FFT频域分析工具
├── FFT_dynamic_analyzer.py    # 动态FFT分析器
├── test_oscillation_detection.py  # 振荡检测测试框架
├── modbus-dcs.py              # Modbus RTU通信模块（待集成）
├── OscillationDetection.py    # 振荡检测核心算法
├── config_fft_dynamic.json    # 动态FFT分析配置文件
├── example_dynamic_fft.py     # 动态FFT使用示例
└── oscillate_dev_settings.json   # 配置文件模板
```

## 核心功能

### 1. FFT频域分析器 (`FFT_analyzer.py`)

**用途**: 单次FFT分析，适用于信号质量评估和频率成分分析

**特点**:

- 高精度频率检测（默认10kHz采样率）
- 支持电网谐波分析（50Hz基频+5次谐波）
- 窗函数处理和插值优化
- 可视化频域图表

**使用示例**:

```python
from FFT_analyzer import FFTAnalyzer

# 创建分析器
analyzer = FFTAnalyzer(window_size=800, sampling_rate=10000)

# 生成测试信号
signal = analyzer.generate_test_signal(fundamental_freq=50.02)

# FFT分析
detected, freq, amp = analyzer.fft_analyze(signal, PLOT_path="result.png")
```

### 2. 振荡检测测试框架 (`test_oscillation_detection.py`)

**用途**: 连续监测和振荡检测，适用于实时系统监控

**特点**:

- 滑窗检测（默认1Hz采样率，60点窗口）
- 配置文件驱动
- 动画实时显示/静态批量分析
- 振荡阈值判断

**使用示例**:

```bash
# 创建配置文件模板
python test_oscillation_detection.py --create-config

# 实时动画监测
python test_oscillation_detection.py --mode animation

# 静态批量分析
python test_oscillation_detection.py --mode static
```

### 3. 动态FFT分析器 (`FFT_dynamic_analyzer.py`) **[新增]**

**用途**: 对带datetime时间戳的CSV文件进行连续滑窗FFT分析，适用于动态频率变化分析

**特点**:

- 支持datetime时间戳输入输出
- 滑动窗口连续分析（默认200ms窗口，100ms步长）
- 可配置采样率、窗口大小、频率范围
- 自动验证采样率和处理边界情况
- 完整的JSON配置文件支持

**输入CSV格式**:
```csv
2025-12-10 10:34:37.998700,137.83313
2025-12-10 10:34:37.998800,136.77002
```

**输出CSV格式**:
```csv
RX Date/Time,组/A_Freq
2025/12/09 10:01:37::608,50.000
2025/12/09 10:01:37::722,50.000
```

**使用示例**:

```python
from FFT_dynamic_analyzer import FFTDynamicAnalyzer

# 方式1: 使用配置文件（推荐）
analyzer = FFTDynamicAnalyzer(config_path="config_fft_dynamic.json")
results = analyzer.run_pipeline()

# 方式2: 编程式配置
analyzer = FFTDynamicAnalyzer(
    window_duration_ms=200,
    step_duration_ms=100,
    sampling_rate=10000,
    freq_range=(49.9, 50.1)
)
df = analyzer.load_csv("input.csv")
results_df = analyzer.analyze_dynamic(df)
analyzer.save_results(results_df, "output.csv")
```

**运行示例脚本**:
```bash
python example_dynamic_fft.py
```

## 配置说明

### FFT分析器配置

- `sampling_rate`: 10000Hz (高精度分析)
- `window_size`: 800点 (0.08秒窗口)
- 频率范围: 12.5Hz - 500Hz

### 振荡检测配置

- `sampling_rate`: 1Hz (长期监测)
- `window_size`: 60点 (1分钟窗口)
- `threshold`: 振荡判断阈值
- `overlap_ratio`: 窗口重叠比例

### 动态FFT分析器配置 **[新增]**

- `window_duration_ms`: 200ms (分析窗口大小)
- `step_duration_ms`: 100ms (滑动步长)
- `sampling_rate`: 10000Hz (可配置采样率)
- `frequency_range`: [49.9, 50.1]Hz (关注频率范围)
- `use_window`: true (使用Hanning窗)
- `use_ipdft`: true (使用插值DFT提高精度)

配置文件示例 (`config_fft_dynamic.json`):
```json
{
  "analysis": {
    "window_duration_ms": 200,
    "step_duration_ms": 100,
    "sampling_rate": 10000,
    "frequency_range": [49.9, 50.1]
  }
}
```

## 应用场景对比

| 功能               | FFT分析器     | 动态FFT分析器 **[新增]** | 振荡检测        |
| ------------------ | ------------- | ----------------------- | --------------- |
| **目标**     | 信号质量分析  | 动态频率跟踪            | 振荡监测预警    |
| **分析方式** | 单次分析      | 滑窗连续分析            | 连续监测        |
| **采样率**   | 10kHz         | 可配置（默认10kHz）     | 1Hz             |
| **窗口大小** | 800点(0.08s)  | 可配置（默认200ms）     | 60点(60s)       |
| **输入格式** | numpy数组     | CSV+datetime时间戳      | CSV             |
| **输出**     | 频谱图 + 峰值 | 时序频率CSV             | 振荡判断 + 趋势 |
| **适用场景** | 静态信号分析  | 动态频率变化分析        | 长期振荡监测    |

## 依赖库

```bash
pip install numpy pandas matplotlib logging
```

## 通信模块 (待集成)

```bash
pip install pymodbus pymysql
```

`modbus-dcs.py` 实现Modbus RTU从站功能，用于从DCS系统获取实时数据：

- 支持MySQL点位配置
- 串口通信 (RTU协议)
- 浮点数解析 (IEEE 754)

## 快速开始

1. **FFT分析**: 直接运行 `python FFT_analyzer.py`
2. **动态FFT分析** **[新增]**: 运行 `python example_dynamic_fft.py` 查看完整示例
3. **振荡检测**: 运行 `python test_oscillation_detection.py --create-config` 后编辑配置文件
4. **信号生成**: 使用 `SignalGenerator` 类创建测试信号（现已支持datetime时间戳）

## 输出示例

- **FFT分析**: 生成频域图表，标注峰值频率和幅值
- **动态FFT分析** **[新增]**: 生成时序频率CSV文件，格式为 `RX Date/Time,组/A_Freq`
- **振荡检测**: 输出检测结果和振荡统计信息
- **日志**: 详细的分析过程和参数信息

## 更新日志

### v2.0 - 动态FFT分析系统 (2025-12-10)

**新增功能**:
- ✅ `FFT_dynamic_analyzer.py`: 动态FFT分析器，支持滑动窗口连续分析
- ✅ `SignalGenerator.py`: 添加datetime时间戳支持（完全向后兼容）
- ✅ `config_fft_dynamic.json`: 模块化JSON配置系统
- ✅ `example_dynamic_fft.py`: 完整使用示例脚本

**核心特性**:
- datetime时间戳输入输出（微秒输入 → 毫秒输出）
- 可配置窗口大小和滑动步长（默认200ms/100ms）
- 自动采样率验证和边界情况处理
- 支持多种时间戳格式自动识别
- 频率范围过滤（适用于50Hz电网信号）

**向后兼容性**: 所有原有功能保持不变，现有代码无需修改即可运行
