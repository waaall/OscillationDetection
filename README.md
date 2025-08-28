# OscillationDetection

基于FFT频域分析的振荡检测工具，适用于频率监测和信号分析。

## 项目结构

```
├── SignalGenerator.py          # 信号生成器（基础依赖）
├── FFT_analyzer.py            # FFT频域分析工具
├── test_oscillation_detection.py  # 振荡检测测试框架
├── modbus-dcs.py              # Modbus RTU通信模块（待集成）
├── OscillationDetection.py    # 振荡检测核心算法
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

## 应用场景对比

| 功能               | FFT分析器     | 振荡检测        |
| ------------------ | ------------- | --------------- |
| **目标**     | 信号质量分析  | 振荡监测预警    |
| **分析方式** | 单次分析      | 连续监测        |
| **输出**     | 频谱图 + 峰值 | 振荡判断 + 趋势 |

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
2. **振荡检测**: 运行 `python test_oscillation_detection.py --create-config` 后编辑配置文件
3. **信号生成**: 使用 `SignalGenerator` 类创建测试信号

## 输出示例

- **FFT分析**: 生成频域图表，标注峰值频率和幅值
- **振荡检测**: 输出检测结果和振荡统计信息
- **日志**: 详细的分析过程和参数信息
