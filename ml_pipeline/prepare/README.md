# 对称设计标签准备

历史纵剖图中的八点坐标作为原始观测保留。预处理先以 `B—C` 定义桥轴，将五节苗外斜弦方向固定为与对应三节苗斜弦平行，再把厚度造成的抬高量分离为法向偏移。第一版智能设计只学习修正后左右比例均值：

- `design_target_alpha = (observed_alpha_left + observed_alpha_right) / 2`
- `design_target_beta = (observed_beta_left + observed_beta_right) / 2`

修正后的外节点满足：

```text
Q1 = A + h_left × n + alpha_left × (B - A)
Q4 = D + h_right × n + alpha_right × (C - D)
```

其中 `n` 为桥轴法向，`h_left/right` 作为厚度与测绘偏移指标保留，不进入第一版节点位置模型。这样桥面坡度、图纸旋转和木材厚度不会污染 `alpha`，设计端斜弦角始终由三节苗几何确定。

历史桥因地形、墩台沉降、构件尺寸、施工和测绘误差出现的左右不对称属于有效观测，不作为标注错误，也不会仅因不对称超过阈值而排除。输出通过 `has_observed_asymmetry`、`observed_asymmetry_flags`、`alpha_asymmetry` 和 `beta_asymmetry` 保留诊断信息；训练标签仍使用左右均值，以保证智能设计输出中心对称结构。只有比例越出几何可行域、缺点、跨信息非法或人工标为低质量时才排除。

标注工具支持结构化图纸缺陷 `annotation.defects`（可多选：`node_unclear` 节点不可辨识、`member_missing` 构件缺失/遮挡、`geometry_conflict` 几何冲突、`drawing_distortion` 图纸变形/比例失真、`scale_unknown` 比例/尺寸信息缺失、`repair_reconstruction` 维修/重建痕迹）。预处理输出 `defects`（分号连接）与 `has_defects`；存在任一缺陷的样本保留在准备数据中，但以 `defects` 原因排除出标准五节苗训练，不依赖人工把质量降为 low。

若 `Q2—Q3` 沿 `B—C` 的投影长度不超过中弦长度的 1%，视为“五节苗无平弦、两根内斜弦直接相接”的特殊构造。记录和八点坐标仍保留，并写入 `inner_flat_chord_ratio`、`has_zero_inner_flat_chord` 与 `scope_exclusion_reasons=zero_inner_flat_chord`，但不进入当前标准五节苗模型训练。该规则与历史左右不对称无关。

标注中的 `span.clear_span_m` 保存当前跨净跨（米），作为绝对尺度；`bridge.span_m` 作为旧格式兼容字段。节点位置仍使用无量纲比例表达。旧 JSON 没有跨长时仍可整理，输出 CSV 的 `span_m` 留空，后续补录即可。

多跨桥按“一跨一个 JSON”整理。预处理输出 `span_count`、`span_index`、`span_position_normalized`、`is_edge_span` 和唯一的 `sample_key`。同一桥各跨共享 `bridge_id`；有迁建、重建或共享测绘来源时，再共享 `bridge.source_group_id`。输出中的 `split_group_key` 优先采用 `source_group_id`，否则采用桥梁 ID，建模划分必须按该字段分组，禁止同一桥的不同跨进入不同数据折。

三节苗设计几何采用知识约束：左斜弦水平投影、中间平弦、右斜弦水平投影各占净跨 `1/3`。由 `A/B/C/D` 水平化后计算 `three_miao_rise_span_ratio=f/L`、`three_miao_rise_m` 和 `three_miao_design_chord_angle_deg=atan(3f/L)`。矢跨比和斜弦角是确定性变换，不应作为两个独立信息源同时输入模型；历史图纸中三段投影偏差仅作为测绘诊断，设计特征按三等分约束生成。

若连续换图标注造成桥名沿用，可先干运行检查，再应用修复。脚本以 `source_image.file_name` 为准恢复桥名、补充分组 ID、规范化单跨/多跨文件名；应用时会在数据目录旁创建带时间戳的完整备份：

```powershell
python -m ml_pipeline.prepare.repair_bridge_metadata C:\path\to\数据集
python -m ml_pipeline.prepare.repair_bridge_metadata C:\path\to\数据集 --apply
```

缺失净跨可通过历史桥梁数据库的原图文件名精确匹配后补齐。脚本先读取公开的桥梁清单和详情，把 `source_image.file_name` 与词条 `drawings[].url` 的文件名做完全匹配；已有记录若只保存了一个跨长字段，也会同步补齐另一个。默认只预览，`--apply` 时同时写入 `span.clear_span_m` 和兼容字段 `bridge.span_m`，并在数据目录旁创建完整备份和 `span_backfill_audit.csv` 审计表：

```powershell
python -m ml_pipeline.prepare.backfill_wbridge_spans C:\path\to\数据集
python -m ml_pipeline.prepare.backfill_wbridge_spans C:\path\to\数据集 --apply
```

结构化 `max_span_length` 为空的记录不会自动猜值；当前仅飞云桥词条正文明确记载“拱跨约19米”，以显式约值覆盖并在审计表的 `source_field` 中标记为 `description_approximate`。任何原图未唯一匹配或缺少可用跨长的记录都会令应用命令失败，避免静默误填。

批量整理标注：

```powershell
python -m ml_pipeline.prepare.symmetric_targets `
  --input-dir D:\下载 `
  --pattern "*纵剖_five_miao_nodes.json" `
  --output D:\下载\five_miao_pilot_symmetric_targets.csv
```

输出同时包含：桥/跨级样本键、原始八点坐标、左右实测比例、对称设计标签、左右不对称度、中段轴线角度和法向投影偏移。法向投影偏移在第一版中仅用于分析，不自动判定样本无效。

## v7 三层目标预处理

经原图复核确认，高残差样本的点位没有错误，差异可能来自地形、墩台沉降、施工尺寸、腐蚀和测量等历史作用。为避免把服役后的现状形态直接当成新桥规范设计，本轮将目标拆成三层：

- 原始观测层：保留左右比例、八点坐标和不对称诊断；
- 历史对称代理层：`historical_observation_alpha/beta` 沿用左右均值，仅允许研究历史样本规律；
- 专家规范目标层：`normative_design_alpha/beta` 默认留空，必须同时具有批准状态、复核人和证据说明才可进入规范目标训练。

22 条已复核高残差记录写为 `confirmed_valid_historical_variation`，全部保留，且不把总体可能成因强行归因到某一座桥。生成分层数据和专家复核模板：

```powershell
python -m ml_pipeline.prepare.layered_targets `
  --input-dir <数据集目录> `
  --output-dir <新输出目录> `
  --historical-review-file <高误差复核清单.csv>
```

如有专家目标，在模板中填写规范 `alpha/beta`、`approved`、复核人和证据说明，再通过 `--normative-override-file` 导入。程序会拒绝未知样本、非法比例和缺少审计信息的批准记录。历史代理与规范目标使用独立字段，导入专家值不会覆盖历史观测。
