# OscillationDetection

低频工艺信号振荡检测工具。当前版本以 FFT 频带检测为唯一核心算法，目标是为后续 FastAPI 服务化提供稳定的离线回放链路和统一输出模型。

## 当前结构

```text
src/
  core/fft_analyzer.py          # 单窗口 FFT 频带检测核心
  core/detection_pipeline.py    # 时间戳/采样率/窗口切分/状态映射
  offline/csv_replay.py         # 离线 CSV 回放与结果落盘
  offline/csv_replay.default.json # 离线回放默认配置
  offline/result_visualizer.py  # 离线结果可视化
  offline/signal_generator.py   # 离线测试信号生成工具
  api/                          # FastAPI 协议层预留
tests/
  test_detection_pipeline.py    # 核心流程单元测试
  test_csv_replay.py            # CSV 回放集成测试
  test_oscillation_detection.py # 可视化输出测试
```

## 设计边界

- 检测对象：电力/工艺信号中的低频振荡。
- 检测目标：异常报警，不追求高精度频率估计。
- 核心算法输入：等间隔采样数组 + `sampling_rate_hz`。
- 核心算法处理：去均值、Hann 窗、`rFFT`、单边幅值归一化、全频峰值和目标频带峰值提取。
- 报警规则：目标频带内最大谱峰幅值 `>= amplitude_threshold`。
- 窗口规则：`window_duration_s >= 3 / target_freq_range_hz[0]`。

## 统一输出字段

每个窗口都会输出一条记录，不再像旧版那样跳过无效窗口。

```text
window_id,status,reason,start_index,end_index,start_time,end_time,sample_count,sampling_rate_hz,dominant_freq_hz,dominant_period_s,peak_amplitude,overall_peak_freq_hz,overall_peak_amplitude,threshold,target_band_low_hz,target_band_high_hz
```

- `status`: `ok` / `alarm` / `out_of_band` / `insufficient_data` / `invalid_input`
- `reason`: `peak_above_threshold` / `peak_below_threshold` / `dominant_peak_outside_band` / `not_enough_samples` / `timestamp_parse_failed` / `sampling_context_missing` / `resample_failed` / `empty_signal_after_cleanup` / `invalid_frequency_range`

## 配置示例

`src/offline/csv_replay.default.json`

```json
{
  "input": {
    "csv_path": "csv-data/input.csv",
    "time_column": "timestamp",
    "value_column": "signal",
    "timestamp_format": "%Y-%m-%d %H:%M:%S.%f",
    "has_timestamp": true
  },
  "analysis": {
    "sampling_rate_hz": 1.0,
    "target_freq_range_hz": [0.0167, 0.2],
    "window_duration_s": 180.0,
    "amplitude_threshold": 0.5,
    "allow_resample": false
  },
  "replay": {
    "step_duration_s": 1.0
  },
  "output": {
    "result_csv_path": "csv-data/output.csv",
    "include_plot": true,
    "plot_dir": "plots"
  }
}
```

## 使用方式

运行离线 CSV 回放：

```bash
python -m src.offline.csv_replay
```

运行测试：

```bash
python -m pytest tests/test_detection_pipeline.py tests/test_csv_replay.py tests/test_oscillation_detection.py
```
