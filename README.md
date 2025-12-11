# OscillationDetection

基于FFT频域分析的振荡检测工具，适用于频率监测和信号分析。

## 项目结构

```
├── src/
│   ├── __init__.py
│   ├── FFT_analyzer.py                 # 入口脚本(调用 core 版本)
│   ├── oscillate_dev_settings.json     # 振荡检测配置
│   └── core/
│       ├── FFT_analyzer.py             # 单次FFT分析
│       ├── FFT_dynamic_analyzer.py     # 动态频率分析器
│       ├── FrequencyRefinement.py      # 频率精细化
│       ├── OscillationDetection.py     # 振荡检测核心
│       ├── SignalGenerator.py          # 信号生成器
│       └── config_fft_dynamic.json     # 动态频率配置
├── tests/
│   ├── test_dynamic_fft.py             # 动态频率示例/烟囱测试
│   ├── test_frequency_refinement.py    # 频率精细化单元测试
│   └── test_oscillation_detection.py   # 振荡检测开发流程
├── csv-data/                           # 输入/输出CSV样例
├── plots/                              # 绘图输出目录
├── log/                                # 日志输出目录
├── docs/                               # 说明文档
├── AGENTS.md / CLAUDE.md / README.md   # 项目文档
└── modbus-dcs.py                       # Modbus RTU 原型（独立）
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
from src.core.FFT_analyzer import FFTAnalyzer

# 创建分析器
analyzer = FFTAnalyzer(window_size=800, sampling_rate=10000)

# 生成测试信号
signal = analyzer.generate_test_signal(fundamental_freq=50.02)

# FFT分析
detected, freq, amp = analyzer.fft_analyze(signal, PLOT_path="result.png")
```

#### 1.1 频率精细化（最小二乘拟合）

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

#### 1.2 过零检测（Zero-Crossing）

**用途**: 基于信号过零点的频率估计方法，适用于高信噪比单频信号，计算速度快于FFT。

**特点**:
- 直接计算信号零交叉点间隔（无需FFT变换）
- 适用场景：SNR ≥ 40dB，单频或准单频信号（如电网50Hz）
- 计算速度优势：对于长窗口，ZC比FFT+精化快约2-5倍
- 支持DC去除、仅上升沿检测等配置选项

**使用示例**:

```python
from src.core.FFT_dynamic_analyzer import FFTDynamicAnalyzer

# 方式1: 配置文件启用ZC
analyzer = FFTDynamicAnalyzer(config_path="config.json")
# 在 config.json 中设置 "use_zero_crossing": true

# 方式2: 初始化时启用ZC
analyzer = FFTDynamicAnalyzer(
    use_zero_crossing=True,
    zero_cross_config={
        "window_periods": 6,         # 窗口包含周期数
        "min_freq_hz": 45.0,          # 最小有效频率
        "max_freq_hz": 65.0,          # 最大有效频率
        "remove_dc": True,            # 去除DC分量
        "rising_only": True           # 仅检测上升沿
    }
)

# 方式3: 运行时动态切换（优先级最高）
results = analyzer.analyze_dynamic(df, use_zero_crossing=True)
```

**算法优先级**（从高到低）:
1. `analyze_dynamic(use_zero_crossing=True/False)` - 运行时参数（最高优先级）
2. `config.json` 中的 `"use_zero_crossing"` - 配置文件
3. `FFTDynamicAnalyzer(use_zero_crossing=...)` - 初始化参数（默认 `False`）

**ZC vs FFT对比**:

| 特性 | 过零检测(ZC) | FFT分析 |
|------|-------------|---------|
| 计算速度 | 快（O(n)） | 中等（O(n log n)） |
| 适用信噪比 | ≥40dB | ≥30dB |
| 适用信号 | 单频/准单频 | 多频/复杂信号 |
| 频率精度 | 高（取决于采样率） | 高（取决于窗口大小+精化） |
| 谐波抑制 | 差 | 好（可分离谐波） |

### 2. 振荡检测测试框架 (`tests/test_oscillation_detection.py`)

**用途**: 连续监测和振荡检测，适用于实时系统监控

**特点**:

- 滑窗检测（默认1Hz采样率，60点窗口）
- 配置文件驱动
- 动画实时显示/静态批量分析
- 振荡阈值判断

**使用示例**:

```bash
# 创建配置文件模板
python tests/test_oscillation_detection.py --create-config --config src/oscillate_dev_settings.json

# 实时动画监测
python tests/test_oscillation_detection.py --mode animation --config src/oscillate_dev_settings.json

# 静态批量分析
python tests/test_oscillation_detection.py --mode static --config src/oscillate_dev_settings.json
```

### 3. 动态频率分析器 (`src/core/FFT_dynamic_analyzer.py`) **[新增]**

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
from src.core.FFT_dynamic_analyzer import FFTDynamicAnalyzer

# 方式1: 使用配置文件（推荐）
analyzer = FFTDynamicAnalyzer(config_path="src/core/config_fft_dynamic.json")
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
python tests/test_dynamic_fft.py          # 运行示例流程
python -m src.core.FFT_dynamic_analyzer   # 使用默认配置跑一遍管线
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

### 动态频率分析器配置 **[新增]**

- `window_duration_ms`: 200ms (分析窗口大小)
- `step_duration_ms`: 100ms (滑动步长)
- `sampling_rate`: 10000Hz (可配置采样率)
- `frequency_range`: [49.9, 50.1]Hz (关注频率范围)
- `use_window`: true (使用Hanning窗)
- `use_ipdft`: true (使用插值DFT提高精度)
- `use_zero_crossing`: false (频率估计算法：false=FFT, true=过零检测)
- `zero_cross_config`: {...} (过零检测参数，仅当use_zero_crossing=true时生效)

配置文件示例 (`src/core/config_fft_dynamic.json`):
```json
{
  "analysis": {
    "window_duration_ms": 200,
    "step_duration_ms": 100,
    "sampling_rate": 10000,
    "frequency_range": [49.9, 50.1],
    "use_zero_crossing": false,
    "zero_cross_config": {
      "window_periods": 6,
      "min_freq_hz": 45.0,
      "max_freq_hz": 65.0,
      "remove_dc": true,
      "rising_only": true
    }
  }
}
```

**注**: 将 `use_zero_crossing` 设为 `true` 可启用过零检测算法替代FFT

## 应用场景对比

| 功能               | FFT分析器     | 动态频率分析器 **[新增]** | 振荡检测        |
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

1. **FFT分析**: 运行 `python -m src.FFT_analyzer`
2. **动态频率分析** **[新增]**: 运行 `python tests/test_dynamic_fft.py` 查看完整示例，或 `python -m src.core.FFT_dynamic_analyzer` 用配置跑管线
3. **振荡检测**: 运行 `python tests/test_oscillation_detection.py --create-config --config src/oscillate_dev_settings.json` 后编辑配置文件
4. **信号生成**: 使用 `from src.core.SignalGenerator import SignalGenerator` 创建测试信号（现已支持datetime时间戳）

## 输出示例

- **FFT分析**: 生成频域图表，标注峰值频率和幅值
- **动态频率分析** **[新增]**: 生成时序频率CSV文件，格式为 `RX Date/Time,组/A_Freq`
- **振荡检测**: 输出检测结果和振荡统计信息
- **日志**: 详细的分析过程和参数信息

## 更新日志

### v2.0 - 动态频率分析系统 (2025-12-10)

**新增功能**:
- ✅ `src/core/FFT_dynamic_analyzer.py`: 动态频率分析器，支持滑动窗口连续分析
- ✅ `src/core/SignalGenerator.py`: 添加datetime时间戳支持（完全向后兼容）
- ✅ `src/core/config_fft_dynamic.json`: 模块化JSON配置系统
- ✅ `tests/test_dynamic_fft.py`: 完整使用示例脚本
- ✅ 核心模块集中到 `src/core/`，测试迁移到 `tests/`

**核心特性**:
- datetime时间戳输入输出（微秒输入 → 毫秒输出）
- 可配置窗口大小和滑动步长（默认200ms/100ms）
- 自动采样率验证和边界情况处理
- 支持多种时间戳格式自动识别
- 频率范围过滤（适用于50Hz电网信号）

**向后兼容性**: 所有原有功能保持不变，现有代码无需修改即可运行
