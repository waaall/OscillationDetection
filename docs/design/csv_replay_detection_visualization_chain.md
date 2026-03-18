# CSV 回放、检测与绘图链路说明

## 1. 目的

本文档说明离线分析链路中几类核心脚本和模块之间的数据流转关系，重点回答以下问题：

- 原始 CSV 是如何进入检测流程的
- `csv-data/output.csv` 中的字段是如何计算出来的
- `plots/dominant_frequency.png` 和 `plots/window_0_spectrum.png` 分别画的是什么
- 为什么某些图和某些字段看起来“不一致”
- 在什么条件下会跳过绘图或只画部分图

本文档覆盖的主要模块：

- `src/offline/filter_by_sampling_interval.py`
- `src/offline/csv_replay.py`
- `src/core/detection_pipeline.py`
- `src/core/fft_analyzer.py`
- `src/offline/result_visualizer.py`


## 2. 总体链路

### 2.1 原始数据进入检测之前

如果原始 CSV 的时间戳存在缺口、乱序或不规则采样，推荐先使用：

- `src/offline/filter_by_sampling_interval.py`

它的职责是：

- 检查时间戳是否满足目标采样间隔
- 筛选出规则采样的数据段
- 默认只保留最长连续规则片段，避免后续检测因为片段断点报 `invalid_input`

该脚本是前置清洗工具，不参与 FFT 检测和绘图。


### 2.2 离线回放主流程

离线分析主入口是：

- `python src/offline/csv_replay.py`

对应代码在 `src/offline/csv_replay.py` 的 `main()` 和 `run_pipeline()`。

主流程如下：

1. 从 `src/offline/csv_replay.default.json` 读取配置
2. `load_csv()` 读取原始 CSV
3. `analyze_dynamic()` 提取信号列和时间戳列
4. `DetectionPipeline.analyze_samples()` 对数据做预处理、滑窗、FFT、状态判定
5. `results_to_frame()` 将嵌套结果展平为标准 DataFrame
6. `save_results()` 写出 `csv-data/output.csv`
7. `generate_plots()` 调用 `ResultVisualizer` 生成图像


## 3. 模块职责分层

### 3.1 `filter_by_sampling_interval.py`

职责：

- 针对原始 CSV 做时间戳规则性筛选
- 输出规则采样的 CSV

不负责：

- FFT
- 振荡判定
- 状态分类
- 绘图


### 3.2 `csv_replay.py`

职责：

- 读取配置
- 读取 CSV
- 调用核心检测流程
- 保存标准输出表
- 调用绘图器生成 PNG

它是离线层编排器，不做 FFT 本身。


### 3.3 `detection_pipeline.py`

职责：

- 输入清洗
- 时间戳解析
- 采样率推断与校验
- 可选重采样
- 滑窗切片
- 调用 FFT 分析器
- 将 FFT 结果映射为业务状态

如果预处理失败，会直接返回一条：

- `status = invalid_input`

这时不会进入窗口 FFT 分析，也不会有频谱调试数据。


### 3.4 `fft_analyzer.py`

职责：

- 对单个窗口做 FFT
- 计算全频段峰值
- 计算目标频带内峰值
- 可选返回完整频谱数组

它只关心“单窗口、均匀采样”的纯算法问题，不处理 CSV、滑窗和文件输出。


### 3.5 `result_visualizer.py`

职责：

- 基于 `results_df` 和 `raw_results` 绘制静态 PNG

当前它不是动画系统，也不会默认为每个窗口都生成独立图像。


## 4. `output.csv` 是怎么来的

`output.csv` 的每一行对应一个分析窗口。

这些结果来自：

1. `DetectionPipeline.analyze_samples()` 的窗口遍历
2. 每个窗口调用一次 `FFTAnalyzer.analyze()`
3. `DetectionPipeline._result_from_metrics()` 组装结果
4. `CsvReplay.results_to_frame()` 展平为表格

关键字段含义：

- `window_id`
  - 第几个窗口
- `status`
  - 该窗口的状态，例如 `ok`、`alarm`、`out_of_band`
- `reason`
  - 该状态的直接原因
- `dominant_freq_hz`
  - 目标频带内主峰频率
- `peak_amplitude`
  - 目标频带内主峰幅值
- `overall_peak_freq_hz`
  - 全频段最大峰频率
- `overall_peak_amplitude`
  - 全频段最大峰幅值

这里最容易混淆的是：

- `dominant_*` 表示目标频带内峰值
- `overall_*` 表示全频段峰值

这两组值不一定相同。


## 5. `dominant_frequency.png` 是怎么来的

`plots/dominant_frequency.png` 来自：

- `ResultVisualizer.plot_summary()`

它不是原始波形图，而是“窗口级结果总览图”。

图中三张子图分别对应：

### 5.1 第一张子图

- 横轴：`window_id`
- 纵轴：`dominant_freq_hz`

也就是每个窗口在目标频带内检测到的主峰频率。


### 5.2 第二张子图

- 横轴：`window_id`
- 纵轴：`peak_amplitude`
- 同时叠加一条阈值线 `threshold`

也就是每个窗口在目标频带内的峰值幅值。


### 5.3 第三张子图

- 横轴：`window_id`
- 纵轴：状态编码后的数值

它用于展示状态随窗口推进的变化。


## 6. `window_0_spectrum.png` 是怎么来的

`plots/window_0_spectrum.png` 来自：

- `ResultVisualizer.plot_window_spectrum(window_id=0, ...)`

当前 `csv_replay.py` 支持按步长批量输出窗口图：

- `output.window_plot_stride = 0`
  - 只画第 `0` 个窗口，兼容旧行为
- `output.window_plot_stride = 1`
  - 每个窗口都画
- `output.window_plot_stride = n`
  - 画 `0, n, 2n, 3n...` 这些窗口

当前图像内容分两部分：

### 6.1 上半部分

- 当前窗口的实际时域数据
- 横轴是窗口内相对时间 `window_time_s`
- 纵轴是这个窗口内的原始数值 `window_values`


### 6.2 下半部分

- 当前窗口对应的 FFT 频谱
- 横轴是频率
- 纵轴是频谱幅值
- 半透明色带表示目标频带范围
- 会额外标记两个峰值点：
  - `Overall Peak`
  - `Band Peak`

并在点旁边直接标注：

- `(频率Hz, 幅值)`


## 7. 为什么 `window_0_spectrum.png` 和 `dominant_frequency.png` 有时看起来对不上

这通常不是画图错误，而是两张图默认关注的指标不同。

### 7.1 频谱图看的可能是全频段最高峰

在 `window_0_spectrum.png` 的频谱里，肉眼最显眼的最高点通常对应：

- `overall_peak_freq_hz`
- `overall_peak_amplitude`


### 7.2 总览图第一、第二张子图画的是目标频带内峰值

`dominant_frequency.png` 的前两张子图使用的是：

- `dominant_freq_hz`
- `peak_amplitude`

也就是目标频带内主峰，而不是全频段主峰。


### 7.3 当全频段最高峰落在目标频带外，就会出现“看起来不一致”

例如：

- 频谱图最高点是 `0.016667 Hz`
- 但目标频带下限配置是 `0.0167 Hz`

这时因为 FFT bin 频率略小于配置下限，最高点会被排除在目标频带外：

- `overall_peak_*` 仍然记录这个最高点
- `dominant_*` 会退到目标频带内的下一个峰

于是就会出现：

- 频谱图最高点看起来像 `(0.016667, 0.035)`
- 但 summary 或 `output.csv` 中显示的是 `(0.033333, 0.024)`

这是频带边界判定带来的结果，不是两张图在画不同批次的数据。


## 8. 什么时候会不画图，或者只画部分图

### 8.1 完全不画图

如果以下任一条件不满足，`generate_plots()` 会直接返回：

- `include_plot = true`
- `last_results` 非空


### 8.2 只画总览图，不画窗口频谱图

`window_0_spectrum.png` 依赖每个窗口结果中的 `debug` 字段。

而 `debug` 只有在真正进入窗口 FFT 分析时才存在。

如果结果是以下情况，就不会有 `debug`：

- `invalid_input`
- `insufficient_data`

这时可能仍然能画 summary，但不会画窗口频谱图。


## 9. 当前绘图能力边界

当前实现的边界如下：

- 是静态 PNG，不是动画
- 默认只输出一张 summary 图
- 默认只输出一个窗口频谱图，即 `window_0_spectrum.png`
- 可以通过 `output.window_plot_stride` 控制窗口图输出步长

如果后续需要增强，常见方向有：

- 为所有窗口生成 `window_{id}_spectrum.png`
- 生成 GIF 或 MP4 动画
- 在 summary 图里同时绘制 `overall_peak_*` 和 `dominant_*`
- 在窗口图中叠加带内判定边界和阈值线


## 10. 典型排查顺序

当离线结果和图像看起来不一致时，建议按以下顺序排查：

1. 先看 `csv-data/output.csv`
   - 确认是 `dominant_*` 还是 `overall_*`
2. 再看 `plots/window_0_spectrum.png`
   - 区分最高峰是全频段峰还是目标频带峰
3. 再看目标频带配置
   - 特别关注 `target_band_low_hz` 是否刚好卡在 FFT bin 边界
4. 再看输入时间戳是否规则
   - 不规则采样可能在进入窗口 FFT 前就被拦截
5. 必要时先用 `filter_by_sampling_interval.py` 清洗数据


## 11. 建议

对于当前项目，比较稳妥的离线分析使用方式是：

1. 原始 CSV 如果采样不规则，先用 `filter_by_sampling_interval.py`
2. 再运行 `csv_replay.py`
3. 先看 `output.csv`
4. 再结合 `dominant_frequency.png` 和 `window_0_spectrum.png` 做解释

如果关注的是“目标频带内振荡”，应优先关注：

- `dominant_freq_hz`
- `peak_amplitude`

如果关注的是“窗口里最强的实际频率成分”，应优先关注：

- `overall_peak_freq_hz`
- `overall_peak_amplitude`
