# 五节苗节点比例 Pilot 训练

第一版仅使用完整且可在设计时取得的 `span_m`，分别预测中心对称设计目标 `alpha`、`beta`。训练比较固定规则、训练折中位数、Ridge 残差回归和 Huber 残差回归。

- 外层：按 `split_group_key` 执行 Leave-One-Group-Out；
- 内层：5 折 GroupKFold 调参；
- 主选择指标：按真实 `bridge_key` 计算的桥级宏平均 MAE，避免多跨桥权重过高；`split_group_key` 只负责隔离同桥/共享测绘谱系，并另报分组宏 MAE；
- 若桥级宏 MAE 与最优值相差不超过 `0.002`，优先选择更简单模型，避免把小样本抖动误判为优势；
- `needs_review=true`（几何越界等实质错误）或 `quality=low` 的标注不参与训练；历史结构左右不对称只记录为诊断信息，经左右均值对称化后继续参与训练；
- 五节苗平弦归一化长度不超过 1%、两根内斜弦直接相接的特殊构造保留在准备数据中，但以 `zero_inner_flat_chord` 原因排除，不参与当前标准桥型训练；
- 产物状态固定为 `pilot_not_for_production`，经专家复核和外部验证前不得直接上线。

使用算法服务的 Python 环境运行：

```powershell
$python = "C:\Users\24982\.conda\envs\bridge_dvadmin3\python.exe"
& $python -m ml_pipeline.train.pilot `
  --input-dir "C:\Users\24982\Desktop\数据集" `
  --output-dir "C:\Users\24982\Desktop\机器学习结果\five_miao_pilot_v1"
```

输出包括训练清单、数据哈希、准备数据、排除记录、OOF 预测、指标、调参历史、图表和两个 joblib 模型产物。

扩充几何和跨位特征后运行 v5 独立留出训练：

```powershell
$python = "C:\Users\24982\.conda\envs\bridge_dvadmin3\python.exe"
& $python -m ml_pipeline.train.feature_ablation `
  --input-dir "C:\Users\24982\Desktop\数据集" `
  --output-dir "C:\Users\24982\Desktop\机器学习结果\five_miao_pilot_v5_independent_holdout"
```

v5 比较净跨、矢跨比、绝对矢高交互和多跨布局组合。斜弦角由 `atan(3f/L)` 唯一确定，只导出用于解释，不与矢跨比同时作为独立特征。特征、模型和超参数只在约 80% 的开发分组中选择，其余约 20% `split_group_key` 作为独立内部留出集，仅在选择完成后评估一次。该留出仍来自同一网站数据源，不得写成外部验证。

输出新增 `data_partition.csv`、`independent_holdout_predictions.csv` 和 `independent_holdout_metrics.json`。最终 joblib 产物以全量合格记录重拟合，但保留开发集选择指标、独立留出指标、训练范围和 Pilot 状态，供算法服务做受控加载与规则回退。

## v6 小样本算法改进

v6 冻结 v5 的 `data_partition.csv`，避免因为重新随机划分而制造不可比的指标。开发集先用分组 5×3 折筛选 Ridge、Huber、ElasticNet、样条 Ridge、RBF-SVR、Gaussian Process 和浅层 Extra Trees，再对筛选候选与 v5 基线执行 Leave-One-Group-Out 确认。只有桥级宏 MAE 至少改善 `0.002`，并且绝对误差 q90 不恶化超过 `0.005` 时，才允许替换基线；独立内部留出集不参与该决定。

```powershell
python -m ml_pipeline.train.model_improvement `
  --input-dir "C:\Users\24982\Desktop\数据集" `
  --output-dir "C:\Users\24982\Desktop\机器学习结果\five_miao_pilot_v6_small_sample" `
  --partition-file "C:\Users\24982\Desktop\机器学习结果\five_miao_pilot_v5_independent_holdout\data_partition.csv"
```

本轮七类模型均未达到稳定替换门槛，因此 `alpha` 继续采用净跨 Ridge，`beta` 继续采用净跨+矢跨比 Ridge。这个结果说明当前主要瓶颈是桥型、构造和标注一致性等信息不足，而不是回归器容量不足。v6 产物新增标准化 k 近邻局部适用域描述；服务端在输入仍位于最小/最大范围内但远离训练样本时返回 `sparse_training_region` 提示，并提供基于开发集分组 OOF 绝对误差 q90 的预测区间。

## v7 历史代理抗噪声基线

v7 保留全部 22 条已确认有效的高残差历史观测，在冻结的 v5 分区上比较 Ridge、Huber 和中位数分位回归。训练目标明确标记为 `historical_symmetric_proxy_not_expert_normative`，不把历史服役差异解释为规范设计目标。筛选、调参和替换决定只使用开发集，冻结留出集不参与选模。

```powershell
python -m ml_pipeline.train.noise_aware_baseline `
  --input-dir <数据集目录> `
  --output-dir <新结果目录> `
  --partition-file <v5目录\data_partition.csv> `
  --historical-review-file <高误差复核清单.csv>
```

本轮 Huber 和分位回归均未达到预设的稳定替换门槛，`alpha`、`beta` 继续保留 Ridge；冻结留出桥级宏 MAE 分别为 `0.09333`、`0.02991`。研究产物状态固定为 `research_proxy_not_for_deployment`，并故意不满足算法服务加载契约。只有专家批准的规范目标达到可训练数量后，才能另建面向新桥设计的模型。

## v8 外节点两类结构门控实验

根据构造观察，把外节点分为三节苗斜弦前半区 `alpha<0.5` 与后半区 `alpha>=0.5`。v8 先用设计时可取得的总体参数判别结构类型，再分别调用前半区和后半区的独立回归器。分类器在预测时禁止读取真实 `alpha`；数据继续使用冻结 v5 分区和 `split_group_key` 分组验证，并同时报告自动门控结果和“类型已知”的诊断上限。

```powershell
python -m ml_pipeline.train.structure_gated_alpha `
  --input-dir <数据集目录> `
  --output-dir <新结果目录> `
  --partition-file <v5或v6目录\data_partition.csv> `
  --baseline-dir <v6结果目录>
```

只有自动门控的开发集 LOGO 桥级宏 MAE 至少改善 `0.002`，且 q90 不恶化超过 `0.005`，才允许进入受控接入评审。类型已知时的改善只能证明“双专家”有价值，不能替代前置分类器的端到端验证。

## v9 设计人员给定模式的条件双专家

v9 不再把自动分类器作为主流程。在线合同仍要求设计人员预先选择 `front_half` 或 `back_half`（现有字段也可传 `uncommitted`；默认仍是后半区）。历史训练标签不再用均值 α 在 0.5 处一刀切：`STRUCTURE_BOUNDARY_BAND`（±0.03）内为 `uncommitted`。贴边预测用夹在带内的训练中位数/0.5，**不用 v6**：本地 2026-08-25 复跑中远济门控已是 uncommitted，但 v6 把 0.503 预测成 0.622。带外未提交（该次 25−22=3）对两个半区专家做未截断插值。已提交后半区咏归式向下收缩与 2/3 取较高值；岚下前半区低尾不混合、不删样本。学习器仍是 Ridge/v9 双专家。产物保持 `research_designer_mode_not_for_deployment`。

```powershell
python -m ml_pipeline.train.design_mode_alpha `
  --input-dir <数据集目录> `
  --output-dir <新结果目录> `
  --partition-file <v5或v6目录\data_partition.csv> `
  --baseline-dir <v6结果目录>
```

开发集先在每个**已提交**半区内分别比较分区中位数、Ridge 和 Huber，再以 `split_group_key` LOGO 生成条件预测。对照包括固定规则、原始 v6 单模型、按已提交半区截断的 v6，以及仅使用类型先验的分区训练中位数；同时报告前后半区等权宏 MAE。另对边界带记录和真正左右跨区记录做排除敏感性分析。

只有条件双专家在开发集上相对最强对照的模式宏 MAE和总体桥级宏 MAE均至少改善 `0.002`、总体 q90 不恶化超过 `0.005`，且前后半区各自的 MAE/q90 均不明显劣于各自最强对照，才进入系统接入评审。冻结内部留出不参与选择，且因已在多轮研究中查看，不得称为全新外部验证。

本地复跑冻结内部留出（云端 VM 通常没有 103 条 JSON 与 v6 产物）后，比较 `independent_holdout_predictions.csv` 的总体 MAE，以及：

- 远济：`derived_design_mode` 应为 `uncommitted`，`predicted_alpha` 应接近 0.5（夹在 ±0.03 内），**不应**再是 v6 的 ~0.622。2026-08-25 本地目录 `five_miao_pilot_v9_boundary_high_tail_20260825` 曾报 0.622，复跑必须用**新的空输出目录**。
- 咏归：`derived_design_mode` 应为 `back_half`，`predicted_alpha` 应高于修复前专家收缩值 ~0.597，并至少不差于固定规则 2/3≈0.667（真值均值 0.748）。主体后半区行不应被抬到 ~0.75。
- 岚下：`derived_design_mode` 应为 `front_half`，`predicted_alpha` 仍应接近前半区专家/主体（约 0.43），不得被拉到 ~0.23；该样本保留。

## v10 显式模式部分共享回归

v10 只研究 `alpha`，不改 `beta` 和线上服务。在设计模式已知的条件下比较四种复杂度：前后半区中位数（M0）、模式截距不同但共享跨径斜率（M1）、允许模式斜率不同但对斜率差施加强惩罚的部分共享模型（M2），以及 v9 完全独立专家（M3）。

```powershell
python -m ml_pipeline.train.partial_pooling_alpha `
  --input-dir <数据集目录> `
  --output-dir <新结果目录> `
  --partition-file <v5目录\data_partition.csv>
```

超参数只在开发集 `GroupKFold(5)` 中筛选，再用按 `split_group_key` 的 LOGO 确认。当前 102 条合格样本的结果为：M1/M2 开发集模式宏 MAE 均约 `0.05166`，v9 M3 为 `0.05163`；M2 自动选择 `interaction_penalty=1000`，实质上收缩到共享斜率。按 `0.002` 容差选择更简单的 M1，但它未达到相对 M0/M3 的实质替换门槛，因此不生成部署 joblib，也不替换线上 v9。21 条历史留出集上的 M1 模式宏 MAE 为 `0.05929`，仅作描述性复核。

输出包括逐模型开发集 LOGO/历史留出指标、逐样本预测、模式分组 conformal 区间、分组无泄漏审计、模型筛选记录、PNG 对比图和 Markdown 报告。

## v11 倾斜角消融

v11 检查 `three_miao_design_chord_angle_deg` 是否能改善显式模式 alpha。倾斜角按 `atan(3 × rise/span)` 生成，与矢跨比是一一对应的确定性变换；实验比较“净跨”“角度”“净跨+角度”“净跨+矢跨比”，但代码禁止角度与矢跨比同时进入同一个模型。

```powershell
python -m ml_pipeline.train.angle_ablation_alpha `
  --input-dir <数据集目录> `
  --output-dir <新结果目录> `
  --partition-file <v5目录\data_partition.csv>
```

开发集 LOGO 模式宏 MAE 分别为：共享净跨 `0.05166`、共享角度 `0.05722`、净跨+角度 `0.05254`、净跨+矢跨比 `0.05260`、v9 独立专家 `0.05163`。相对共享净跨，净跨+角度的桥级配对 MAE 平均增加 `0.00074`，95% bootstrap CI `[0.00043,0.00111]`，只在 `10/76` 个桥组上更优；角度没有通过替换门槛。最终可追溯结果目录为 `five_miao_pilot_v11_angle_ablation_alpha_v2`，不生成部署产物。

## 原始标签与完整预处理的严格配对对照

该实验冻结 102 条标准样本、`split_group_key`、v5 开发/内部留出分区、特征、Ridge 模型和超参数，只改变训练标签构造。原始标签基线分别拟合左右观测并取预测均值；完整流程拟合平行角度修正后的对称设计目标。范围过滤和净跨元数据保持不变，否则样本总体发生变化，不能再称为严格配对。

```powershell
python -m ml_pipeline.train.preprocessing_strict_control `
  --data-dir <数据集目录> `
  --partition <v5目录\data_partition.csv> `
  --output-dir <新结果目录>

python -m ml_pipeline.train.preprocessing_strict_control_plots `
  --result-dir <新结果目录>
```

输出 `strict_control_metrics.json`、逐样本预测 CSV、Markdown 报告及 PNG/PDF 图。开发集和独立内部留出中，完整预处理使 alpha 桥级宏 MAE 分别由 `0.07569` 降至 `0.07258`、由 `0.09649` 降至 `0.09333`；配对 bootstrap 95% 区间仍跨 0，因此应表述为“小幅一致改善但统计证据有限”。beta 在相同线性模型下数值等价，这是左右分别拟合后取均值与直接拟合均值的代数结果，不能声称精度提升。共同评估目标仍是预处理生成的设计代理而不是独立专家真值，因此该对照主要验证标签构造一致性，不能单独证明恢复了历史原始设计意图。
