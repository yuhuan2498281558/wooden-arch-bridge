# 五节苗斜弦节点小样本预测：特殊场景方法调研

**核查日期：** 2026-08-23  
**适用范围：** 当前以设计参数预测 `alpha/beta`、再解析重建 `Q1—Q4` 的表格回归任务；图像方法只作为后续标注辅助。  
**来源口径：** 仅采用原始论文、作者代码和官方文档。

## 1. 结论先行

当前首选方案不是换更大的回归器或改做端到端图像网络，而是：

1. 保留设计人员显式给出的 `front_half/back_half`，不再让模型从净跨、矢跨比等弱特征自动猜模式；
2. 在两个模式之间做**部分共享/收缩**：优先比较“模式截距 + 共享斜率”“向共享斜率收缩的模式斜率”和现有两套独立 Ridge；
3. 用桥梁/测绘谱系级折外残差做 **group-aware conformal** 校准，输出区间、适用域和拒识，而不是只输出点值；
4. 把高残差拆成结构异质性、历史不对称、标注含糊和适用域稀疏四类，不能统一加权，更不能删除已复核的真实样本；
5. 后续若要从图纸自动取点，采用“结构线分割/矢量化 → 线段—交点图 → 已知拓扑匹配”的混合流水线，只用于半自动标注和补充结构特征。

仓库现有证据支持这一排序：102 条标准样本中，`alpha` v6 留出 MAE 为 `0.09333`，明显难于 `beta=0.02991`；v8 在模式已知时开发集 LOGO MAE 可到 `0.05299`，但自动门控平衡准确率仅 `0.68834`、端到端 MAE 为 `0.08457`；v9 显式模式 MAE 为 `0.05296`，但只比模式中位数 `0.05459` 小幅改善且边界敏感。因此瓶颈主要是模式/目标语义和有效结构特征，不是模型容量。

## 2. 当前问题应怎样建模

### 2.1 显式模式优于自动门控

均方误差回归会逼近条件均值；当同一输入对应前、后半区两种合理位置时，平均值可能落在两种构造之间。Bishop 的[混合密度网络原始报告](https://www.microsoft.com/en-us/research/publication/mixture-density-networks/)正是为多值连续映射提出条件分布建模；[Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)则通过门控把不同子任务交给局部专家。

但本项目在线设计模式已经由设计人员给定，且 v8 证明当前特征不足以稳定自动门控，所以无需上高容量 MDN：

- 在线合同继续使用显式 `front_half/back_half`；
- v8 自动门控只保留论文消融；
- 若模式未给出，默认策略、拒识或同时展示两个方案都比静默取跨模式均值更可控；
- 历史样本的模式由目标 `alpha` 派生，只能用于条件回归研究，不能称为独立专家标签或自动结构识别。

### 2.2 两个专家不要完全独立，先试部分共享

Stan 官方层级模型文档把完整合并和各组完全独立之间的折中称为 partial pooling，组数据越少，收缩越能稳定估计；同时也提示组数很少时层级方差难估，需要强先验或更简单参数化（[Stan User's Guide：hierarchical regression](https://mc-stan.org/docs/2_28/stan-users-guide/hierarchical-logistic-regression.html)、[小单元与层级先验说明](https://mc-stan.org/docs/2_28/stan-users-guide/multilevel-regression-and-poststratification.html)）。本项目只有两个模式，不宜直接堆复杂贝叶斯层级。

建议在冻结 v5 分区上依次比较：

| 候选 | 形式 | 作用 |
|---|---|---|
| M0 | 两个模式各自训练中位数 | 强基线 |
| M1 | 模式截距 + 共享净跨斜率 | 最大共享、最低方差 |
| M2 | 模式截距 + 向公共值收缩的模式斜率 | 推荐的 partial-pooling 近似 |
| M3 | 两套独立 Ridge（当前 v9） | 无共享对照 |

M2 可用带 L2 收缩的模式交互项实现，不必先引入 MCMC。仍以开发集嵌套 LOGO 选择；现有冻结留出已经在多轮研究中查看，只能作描述性复验，不能再参与选择，正式替换需依赖新的外部桥组。除桥级宏 MAE 外，必须同时报告模式宏 MAE、q90、米制节点误差和边界样本敏感性。

### 2.3 目标语义比算法更重要

当前 `design_target_alpha/beta` 是历史左右观测的对称均值，不是专家恢复的原始设计意图；规范设计目标批准数仍为 0。已有 22 条高残差样本经原图复核无误，说明残差不能再简单归因于错标。

下一批标注预算应优先用于小规模、高信息量的**专家规范目标层**，而不是继续给同一历史代理换模型。建议覆盖：

- 前/后半区及 `alpha≈0.5` 边界；
- 当前折外残差和局部稀疏度最高的桥；
- 不同外/内斜弦排布、零平弦与标准平弦构造；
- 不同测绘谱系、地区、年代及图纸质量；
- 能在新桥设计时取得的结构字段，而非仅在历史图纸上事后可见的字段。

## 3. 不确定性、难例与拒识

### 3.1 区分四种“难”

| 类型 | 可用证据 | 处理方式 |
|---|---|---|
| 可学习的少数构型 | 显式模式、可复现结构字段 | 分层/条件模型，模式平衡评估 |
| 标注含糊 | 双人复标分歧、线条遮挡/模糊 | 概率似然或降权，返回大区间 |
| 历史真实扰动 | 左右不对称、沉降/施工等可能影响 | 保留原观测，不冒充标注噪声 |
| 适用域稀疏 | kNN 距离、训练范围、专家分歧 | OOD 提示、拒识或规则回退 |

Kendall 与 Gal 区分了观测噪声的 aleatoric uncertainty 和数据不足导致的 epistemic uncertainty，并给出输入相关噪声似然（[NeurIPS 2017 原文](https://proceedings.neurips.cc/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html)）。LUVLi 进一步联合预测关键点位置和协方差，用于识别定位失败（[CVPR 2020 论文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kumar_LUVLi_Face_Alignment_Estimating_Landmarks_Location_Uncertainty_and_Visibility_Likelihood_CVPR_2020_paper.pdf)、[作者代码](https://github.com/merlresearch/LUVLi)）。对本项目而言，只有复标分歧和图纸清晰度可直接进入“标注噪声”；左右历史差异必须继续单独保存。

### 3.2 难例加权只能放在细化阶段

Cascaded Pyramid Network 的 Online Hard Keypoint Mining 只在 RefineNet 使用时有效；同时用于全局阶段反而下降（[CVPR 2018 论文](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Cascaded_Pyramid_Network_CVPR_2018_paper.pdf)、[作者代码](https://github.com/chenyilun95/tf-cpn)）。对应到当前任务：

- 基础损失必须覆盖全部训练样本；
- 若要强化 `alpha`，只在模式内残差细化或辅助损失中使用；
- 权重只能由训练折内信息产生，不能读取留出/测试残差；
- 已复核高残差样本不能因为难而删除，也不能无限放大权重；
- 先用模式平衡采样和稳健损失，再判断难例辅助项是否带来桥级 q90 改善。

### 3.3 conformal 必须按桥/谱系分组

普通 conformal 依赖样本交换性；同桥多跨或同源测绘记录逐行校准会高估有效样本量。Dunn、Wasserman 与 Ramdas 针对两层分组数据提出 CDF pooling 和按组抽样方法，明确指出组间分布不同会破坏普通交换性（[原始论文](https://arxiv.org/abs/1809.07441)、[复现实验代码](https://github.com/RobinMDunn/ConformalTwoLayer)）。

建议：

1. 用 `split_group_key` 的 LOGO/分组 OOF 预测生成校准残差；
2. 多跨组若要求整桥同时可靠，使用组内最大绝对误差作为该组分数；若只要求随机一跨覆盖，可每组抽一条并重复抽样；
3. 在独立桥组上报告 90% 区间的实际覆盖率和平均宽度；
4. 全局桥组覆盖是主结果，前/后模式条件覆盖只在校准组数足够时报告；
5. conformal 不能解决地域、年代或图纸来源漂移，仍需训练范围、局部稀疏和外部桥验证。

关键点 conformal 已有直接先例：Yang 与 Pavone 把关键点热图校准为圆/椭圆预测集，再把不确定性传播到几何结果（[CVPR 2023 论文](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_Object_Pose_Estimation_With_Statistical_Guarantees_Conformal_Keypoint_Detection_and_CVPR_2023_paper.pdf)、[作者代码](https://github.com/NVlabs/ConformalKeypoint)）。本项目可更简单地把 `alpha/beta` 区间解析传播到 Q1—Q4 和杆件长度区间。

## 4. 图网络、线段/交点与合成数据：只作后续辅助

### 4.1 推荐混合流水线，不推荐端到端替代

技术图纸像素稀疏，尺寸线、文字和构件线相互干扰。工程图原始工作采用“栅格矢量化 → 按连通/距离建图 → GCN 分类组件”，而非直接从整图输出最终工程参数（[Component Segmentation of Engineering Drawings，2023](https://doi.org/10.1016/j.compind.2023.103885)、[作者预印本](https://arxiv.org/abs/2212.00290)）。P&ID 数字化也把检测与沿线图搜索组合起来恢复连接关系（[CVPRW 2020 原文](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w8/Mani_Automatic_Digitization_of_Engineering_Diagrams_Using_Deep_Learning_and_Graph_CVPRW_2020_paper.pdf)）。

最接近本场景的输电塔设计图工作先分割结构掩膜，再融合节点热图和骨架热图提取节点及连线；论文同时报告通用 L-CNN/HAWP 在复杂交叉结构中会产生冗余节点或错误连线，并承认图纸风格迁移会失败（[原始论文及 DOI](https://doi.org/10.32604/cmc.2024.059094)、[出版方 PDF](https://file.techscience.com/files/cmc/2025/TSP_CMC-82-2/TSP_CMC_59094/TSP_CMC_59094.pdf)）。

因此后续半自动标注可试：

```text
结构区域去文字/尺寸线
  → 线段与交点候选
  → 构造节点—杆件图
  → 用已知 A/B/C/D/Q1—Q4 拓扑匹配
  → 共线、平行、顺序、对称/非对称语义检查
  → 人工确认后写回标注
```

候选工具可比较 L-CNN（[论文](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhou_End-to-End_Wireframe_Parsing_ICCV_2019_paper.pdf)、[代码](https://github.com/zhou13/lcnn)）、PPGNet 的点对图表示（[论文](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_PPGNet_Learning_Point-Pair_Graph_for_Line_Segment_Detection_CVPR_2019_paper.pdf)、[代码](https://github.com/svip-lab/PPGNet)）和 HAWP 的线段—交点联合提议（[论文](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xue_Holistically-Attracted_Wireframe_Parsing_CVPR_2020_paper.pdf)、[代码](https://github.com/cherubicXN/hawp)）。它们都不是在历史木拱桥测绘图上验证的，必须先做 10—15 张小试验。

### 4.2 合成数据能解决图像域，不会凭空恢复设计规律

Domain Randomization 通过随机纹理、光照、相机和场景参数使仅合成数据训练的定位器迁移到真实图像（[Tobin 等，IROS 2017](https://arxiv.org/abs/1703.06907)）；DeepLabCut 则证明预训练迁移和少量多样标注可降低关键点标注量（[Nature Neuroscience 2018](https://www.nature.com/articles/s41593-018-0209-y)、[官方代码](https://github.com/DeepLabCut/DeepLabCut)）。

本项目可用确定性绘图器随机化跨度、矢跨比、模式、`alpha/beta`、线宽、裁切、旋转、扫描模糊、JPEG 噪声、尺寸文字遮挡和缺线，自动生成精确节点/杆件标签；用途应限于线段、交点、结构掩膜或标注工具预训练。

不能用人为设定的合成 `alpha` 分布训练“设计参数 → alpha”后再声称学到了古桥规律，因为标签规律本身就是生成器假设。真实桥组必须作为最终测试，合成图也不能进入主效果指标。

## 5. 建议的最小实验包

1. 冻结现有 102 条样本、v5 开发/留出划分和 `split_group_key`。
2. 只对 `alpha` 比较 M0—M3；`beta` 保持 v6，避免把已解决问题复杂化。
3. 主指标：桥级宏 MAE；安全指标：q90、米制 Q1/Q4 误差、非法几何率、模式边界敏感性。
4. 用开发集 group-OOF 残差校准 90% 区间，在冻结留出只评覆盖率和区间宽度。
5. 对高误差分层报告：模式、`|alpha-0.5|`、测绘谱系、图纸质量、历史不对称、局部稀疏度。
6. 新增特征必须在设计时可获得，并逐项做分组消融；后验从目标或原图读出的字段不得偷偷进入在线特征。
7. 只有 M1/M2 在开发集嵌套 LOGO 超过模式中位数与 v9、且 q90 不恶化，才冻结候选；现有冻结留出仅作描述性复验，正式替换还必须在新的外部桥组稳定复现。
8. 图像辅助另立试验：先在 10—15 张图上测结构线召回、拓扑正确率和投影后 `alpha/beta` 误差，不与表格回归混成一个结论。

## 6. 明确不建议

- 不建议在 102 条表格样本上训练高容量 GNN、Transformer、MDN 或深度集成作为主模型；
- 不建议重新启用自动模式门控作为在线默认；
- 不建议把 `alpha≈0.5` 边界样本硬切后只报有利子集；
- 不建议把左右历史不对称当作标注方差，或把高残差当作错标；
- 不建议用逐行随机划分、逐行 conformal 或同桥跨折；
- 不建议用合成目标替代真实历史/专家目标；
- 不建议端到端从测绘图直接替代当前 `alpha/beta` 参数化主链。

## 7. 局限

没有发现直接针对五节苗木拱桥牛头节点预测的公开基准。线框、人体/物体关键点、输电塔和工程图论文提供的是可迁移方法证据，不是本项目效果保证。当前样本主要来自历史现状测绘，尚无专家规范设计真值；任何精度改善首先只能解释为历史代理预测改善。前/后半区只有两个组，也限制了复杂层级模型和模式条件 conformal 的稳定性。最终工程结论仍需外部桥组和专家复核。
