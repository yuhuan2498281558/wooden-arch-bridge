# 实现说明

> 保存稳定的架构事实和确认过的技术决策；具体代码始终是最终事实来源。

## 架构摘要

- `web/`：Vue 3 + TypeScript + Vite + Element Plus 前端，包含桥梁设计页面和路由集成。
- `backend/`：Django 4.2 + Django REST Framework 后端，承担平台、认证和业务管理。
- `bridge_algorithm_service/`：独立 FastAPI 算法服务，提供预测、绘图、分析、优化、Excel 导出、BIMFACE 代理和标注工具页面。
- `ml_pipeline/`：标注元数据修复、对称目标预处理和 Pilot 训练流水线。
- `docker-compose.bridge.yml`：编排 Web、Django、算法服务和 Redis。

当前算法产品边界是三节苗、五节苗骨架生成：输出两类构件的长度、根径和五节苗牛头节点位置。矢跨比和根径属于导师论文既有能力，本轮新增模型只负责五节苗节点位置。桥面、横向联系、榫卯细部、栏杆、廊屋及其他构件只可作为信息库资料，不进入当前预测合同、训练目标或绘图验收范围。

## 重要目录与入口

- `bridge_algorithm_service/main.py`：FastAPI `app`；主要入口包括 `/predict`、`/visualize`、`/analyze`、`/optimize`、`/export/excel`、`/annotation`。
- `bridge_algorithm_service/design_parameter_model.py`：导师两阶段模型运行适配器；校验产物哈希，执行 SSA-XGBoost 矢跨比预测和 CF-BPNN 五类根径矩阵推理。
- `bridge_algorithm_service/mentor_models/`：从用户备份转换出的受控运行产物与清单；不在在线请求中反序列化旧 pickle/MAT。
- `bridge_algorithm_service/service_life.py`：解析 Weibull–Gamma 贝叶斯寿命筛选模型；支持方案先验、情景因子和在役年限右删失更新。
- `bridge_algorithm_service/structural_check.py`：方案阶段三节苗/五节苗承重包络验算；按平面系分担桥面荷载，逐构件做压应力与欧拉屈曲双控，并给出三档腐朽深度截面寿命代理。
- `bridge_algorithm_service/safety_integration.py`：把结构包络 DCR、情景化腐朽截面寿命与 Weibull–Gamma 贝叶斯寿命连成一条分析链；综合评分与风险提示由此导出。


- `bridge_algorithm_service/node_geometry.py`：比例测量、平行斜弦拟合、节点重建、可行域和兼容载荷。
- `bridge_algorithm_service/node_model.py`：受控加载 `alpha/beta` Pilot 产物，构建设计时特征，执行训练范围检查、预测合法性检查和整组规则回退。
- `bridge_algorithm_service/back_half_high_tail.py`：已提交后半区咏归式高尾的保守 2/3 混合；不改 Ridge，不新增在线模式。
- `bridge_algorithm_service/front_half_low_tail.py`：岚下前半区低尾的调查结论（不混合）；v9 前半区专家落在主体，照搬咏归触发会破坏约 0.45 的前半区。
- `bridge_algorithm_service/annotation_tool/index.html`：单页本地八点标注与复核工具。
- `ml_pipeline/prepare/symmetric_targets.py`：JSON → 训练行及中心对称设计标签。
- `ml_pipeline/prepare/repair_bridge_metadata.py`：桥名、来源分组和文件名修复。
- `ml_pipeline/prepare/backfill_wbridge_spans.py`：按网站详情原图文件名精确匹配，干运行/备份后补齐并同步净跨字段，输出来源审计表。
- `ml_pipeline/prepare/layered_targets.py`：分离原始历史观测、历史对称代理和专家规范设计目标，导入高误差复核结论并生成专家目标模板。
- `ml_pipeline/train/pilot.py`：固定规则、中位数、Ridge、Huber 的分组嵌套评估与模型产物生成。
- `ml_pipeline/train/noise_aware_baseline.py`：在冻结分区上比较 Ridge、Huber 和中位数分位回归的历史代理研究基线；产物禁止部署。
- `ml_pipeline/train/structure_gated_alpha.py`：v8 外节点前/后半区门控实验；冻结分区中比较 Logistic 门控与两套独立专家，分别报告自动门控和已知类型诊断上限。
- `bridge_algorithm_service/tests/`：节点几何、预处理、训练和服务集成测试。
- `docs/五节苗牛头节点预测研发与论文路线.md`：研究、数据、算法、集成和论文路线的权威专项文档。

## 核心数据流

```text
历史纵剖图
  → 浏览器标注 A/B/C/D/Q1/Q2/Q3/Q4
  → 一跨一个 JSON（保留原始像素坐标、桥/跨/谱系信息）
  → 平行角度修正，分离外斜弦法向抬高
  → 左右观测比例 alpha_L/R、beta_L/R
  → 左右均值 design_target_alpha/beta
  → 按 split_group_key 分组训练与验证
  → 模型输出 alpha/beta
  → 训练范围与可行域检查后，从三节苗 A/B/C/D 重建对称 Q1—Q4
  → 绘图；模型未启用、缺失、契约不符、OOD 或预测非法时整组回退固定规则
```

当前骨架组合流为：导师两阶段模型提供矢跨比和三节苗/五节苗五类根径；净跨与矢跨比确定三节苗 `A—D`；节点 Pilot 给出 `alpha/beta` 并重建 `Q1—Q4`；构件长度由最终几何确定性计算；最后组合三节苗、五节苗骨架图和可追溯结果。其他构件不进入这一数据流。

结构图纸输出规范（Rev 3.0）：采用 A3 标准幅面（3200×1800），主图呈现纯净实体双线轮廓与正向方形牛头节点木块（无构件内中心线），尺寸全部基于中心线基准测量标注，统一使用 Ø 直径符号（取推荐范围中值）；右侧面板标准化为 5 个工程分区（设计输入、模型输出、确定性几何 Q1-Q4 坐标与 BOM 明细表、醒目适用域与回退状态卡片、非模型示意说明与国标图签）。

导师模型采用兼容适配而非覆盖旧主程序：导入工具把旧 XGBoost pickle 转为原生 Booster 文件，把 CF-BPNN MAT 转为数值权重包，并在清单中记录哈希和原始契约。在线主路径只加载转换后的受控产物；旧规则公式保留为显式回退并返回 `model_status=fallback` 及原因。这样既复现备份数值，又不把旧 pickle 直接放进服务热路径。

## 不变量与兼容性要求

- 节点顺序固定为 `A,B,C,D,Q1,Q2,Q3,Q4`；比例满足 `0<alpha<1`、`0<beta<0.5`。
- 设计重建：`Q1=A+alpha(B-A)`、`Q2=B+beta(C-B)`、`Q3=C+beta(B-C)`、`Q4=D+alpha(C-D)`。
- 历史左右观测不强制对称；设计目标和设计输出必须中心对称。
- 外斜弦角度跟随三节苗对应斜弦；法向抬高独立保存，不混入 `alpha`。
- 三节苗设计坐标固定三段纵向投影各占净跨 `1/3`；斜弦角由 `atan(3f/L)` 从矢跨比派生，不与矢跨比重复输入模型。
- 多跨桥每跨一条记录；`sample_key` 唯一，`split_group_key` 隔离同桥和共享测绘谱系。
- 保留旧结果兼容：模型不可用或结果缺失时仍可用 `alpha=2/3`、`beta=1/4` 规则回退。
- 导师模型被关闭、产物缺失或推理失败时，矢跨比和五类根径也可使用旧规则回退；不得把回退结果标注成 SSA-XGBoost/CF-BPNN 输出。
- 模型产物必须记录版本、目标定义、特征名、训练范围、分组数量和交叉验证指标。

## 已确认实现决策

- 当前智能设计不是完整桥梁或施工图自动设计，只复现和按需生成三节苗、五节苗骨架；本轮需要新增闭环的结构预测结果只有五节苗牛头节点位置。
- 小论文贡献边界固定为五节苗节点数据集、`alpha/beta` 参数化、知识约束小样本预测、误差/不确定性和系统接入；不重复声明导师论文已经完成的矢跨比、根径预测为本轮贡献。
- 俯视图和现场照片适合进入桥梁信息库并辅助解释构造，但当前不新建横向构件、层位、榫卯和三维碰撞预测模块。
- 历史代理节点模型可以受控用于“传统骨架智慧复现型 Pilot”，专家规范目标层保留为后续校准机制，而非当前系统联调的阻塞条件；响应必须继续标注模型状态、数据语义、适用域和回退原因。
- 第一阶段不做端到端图像关键点网络；人工标注图纸，模型只预测两个无量纲比例。
- 预处理目标为 `parallel-angle-fixed-left-right-mean-v2`。
- 历史不对称通过诊断字段保留，不设置为 `needs_review`；只有实质数据错误或低质量才排除。
- 22 条 v6 高误差样本已逐图人工复核并确认标注无误；大残差记录为有效历史现状观测，不因模型误差而删除、降质或移动节点。地形、沉降、构件尺寸/施工误差、腐蚀和测量误差只作为总体可能机制，缺少逐桥证据时不写入具体原因标签。
- 当前 `design_target_alpha/beta` 是左右历史观测均值形成的对称化代理目标，不等于已经恢复的原始设计意图。后续若用于新桥规范化设计，应新增可追溯的专家设计目标层，或以稳健噪声模型显式分离历史服役偏差；不得把历史变形量直接复制到新桥。
- 外节点两类结构以 `alpha<0.5`（前半区）和 `alpha>=0.5`（后半区）定义**已提交**半区的几何截断。自动派生标签使用 `STRUCTURE_BOUNDARY_BAND`（`AMBIGUITY_BAND=0.03`）：两侧都明显离开 0.5 才提交半区；贴边（远济均值 0.503、22/103 条 |α-0.5|<0.03）为 `uncommitted`，评估走 v6 连续预测。门控分类器在训练折外预测时不得读取真实 `alpha`；两类专家必须各自只在本类**已提交**训练数据上拟合，并继续按 `split_group_key` 隔离。
- v7 将该口径落实为三层字段：原始观测保持不变，`historical_observation_alpha/beta` 作为研究代理，`normative_design_alpha/beta` 仅接受同时具备批准状态、复核人和证据的专家覆盖。高残差复核结论不会改变训练资格，也不写入未经证实的逐桥成因。
- `Q2—Q3` 沿 `B—C` 的归一化投影长度不超过 `0.01` 时，标记为 `zero_inner_flat_chord` 特殊构造；保留原始观测但排除出标准五节苗训练范围，不将其判为标注错误。
- 训练外层采用 Leave-One-Group-Out，内层采用 GroupKFold；主指标为桥级宏平均 MAE。
- 候选模型差异在 `0.002` 容差内时优先更简单模型。
- 当前 v3 Pilot 使用 `span_m` 单特征；59 条原始记录中 58 条、52 个独立分组参训，零平弦特殊构造排除 1 条；`alpha/beta` 均选 Ridge，但不得直接视为生产模型。
- v4 特征消融显示：`alpha` 保留净跨单特征，`beta` 采用净跨加矢跨比；绝对矢高交互及多跨布局未获得稳定增益。两者仍选 Ridge，并继续标记为 Pilot。
- v5 将内层调参指标改为真实 `bridge_key` 宏 MAE，另报 `split_group_key` 宏 MAE；特征/模型只在开发集选择，保留 20 个隔离分组作独立内部留出。103 条中 102 条、96 组参训；`alpha` 仍选净跨 Ridge，留出桥级宏 MAE `0.09333`，`beta` 仍选净跨+矢跨比 Ridge，留出桥级宏 MAE `0.02991`。该留出不是外部验证，产物状态仍为 `pilot_not_for_production`。
- v6 冻结 v5 分区，在开发集分组筛选 Ridge、Huber、ElasticNet、样条 Ridge、RBF-SVR、Gaussian Process 和浅层 Extra Trees，再用 LOGO 比较筛选候选与 v5 Ridge。替换必须同时满足桥级宏 MAE 至少改善 `0.002` 和 q90 不恶化超过 `0.005`；本轮没有候选通过，两个目标均保留 Ridge。独立内部留出不参与选择，结果与 v5 一致。
- v6 模型产物记录标准化 k 近邻局部适用域；服务端继续对硬范围越界执行整组规则回退，对范围内但远离训练点的输入返回 `sparse_training_region` 复核提示，并按开发集分组 OOF q90 给出截断到可行域的经验预测区间。
- v7 在冻结 v5 分区上以历史代理比较 Ridge、Huber 和中位数分位回归；抗噪声候选未达到 `0.002` 的稳定改善门槛，两个目标保留 Ridge，冻结留出桥级宏 MAE 为 `0.09333/0.02991`。产物标记 `research_proxy_not_for_deployment` 并故意不满足服务加载契约，防止把历史现状模型误接为新桥设计模型。
- v8 先用 Logistic 判断外节点前/后半区，再由两套独立 Ridge 预测 `alpha`。类型已知时开发集 LOGO 桥级宏 MAE 为 `0.05299`，但自动门控平衡准确率仅 `0.68834`，端到端 MAE/q90 为 `0.08457/0.18278`，未通过相对 v6 的替换门槛；研究产物状态为 `research_structure_gate_not_for_deployment`。
- v9 取消自动门控和新增人工重标：历史类型按边界带派生（贴边不切后半区），不声称为独立专家标注；在线由设计人员在自然语言交互区选择 `front_half/back_half`，未传值统一默认 `back_half`，自然语言文本中的明确模式覆盖控件值。前后半区分别选择净跨 Ridge；已提交后半区若专家相对训练中位数明显向下收缩（咏归式），则与 2/3 规则取较高值，不另拟合高尾专家、不改 Ridge 类。岚下前半区低尾（2026-08-25 重标均值 0.232）为孤立留出点，v9 专家约 0.427 落在前半区主体（中位数约 0.449），不混合、不删样本。开发集 LOGO 桥级宏 MAE `0.05296`、模式宏 MAE `0.05163`、q90 `0.10638`；只允许受控 Pilot 接入。派生门控实现于 `derived_structure_mode`（`structure_gated_alpha.py`），高尾混合实现于 `back_half_high_tail.py`。
- v9 通过独立环境开关只覆盖 alpha，beta 保持 v6；v9 未启用、契约不符或越出适用域时回退 v6/规则，并记录 `design_mode.applied=false`。若回退 alpha 仍位于所选半区可生成带回退标识的方案图，否则拒绝出图。
- `ml_pipeline/train/partial_pooling_alpha.py` 是 v10 离线研究：在显式模式条件下用对称编码联合拟合模式截距、公共跨径斜率和受惩罚的模式×跨径交互，并与分区中位数、完全共享斜率和 v9 独立专家在冻结分区上比较。开发集筛选采用 `GroupKFold(5)`，确认采用 `split_group_key` LOGO；当前 M2 的交互惩罚选到 `1000`，退化为近似共享斜率，且未达到相对 v9 的 `0.002` 替换门槛。该脚本只写研究指标、预测、分组 conformal 区间和审计，不写部署 joblib，不改变线上 v9/beta。
- `ml_pipeline/train/angle_ablation_alpha.py` 是 v11 倾斜角消融：共享模式截距下分别比较净跨、`atan(3f/L)` 倾斜角、净跨+倾斜角和净跨+矢跨比，并显式拒绝把角度与矢跨比同时输入。开发集模式宏 MAE 为 `0.05166/0.05722/0.05254/0.05260`；净跨+角度相对净跨的桥级配对 MAE 增加 `0.00074`（95% bootstrap CI `[0.00043,0.00111]`），未通过替换门槛。实验只输出离线指标、预测、配对审计和图表，不改线上模型。
- `ml_pipeline/train/direct_coordinate_baseline.py` 是研究对照而非部署候选：在冻结 v5 分区上直接回归 Q1—Q4 的原始像素、图像归一化坐标和桥轴局部坐标，并与当前 `alpha/beta` 参数化统一换算为节点米制误差。必须同时报告历史观测复现和对称设计代理两个终点；论文主比较使用最强桥轴局部坐标基线，原始像素仅展示图纸坐标不可迁移性。冻结留出中局部坐标与知识参数化的历史复现差异区间跨 0，而知识参数化对对称设计代理减少 `0.2046 m`（95% CI `[0.1366,0.2744]`）并把几何警告率从 `95.2%` 降为 0；对称设计代理仍不是专家真值。直接坐标会复制非对称/偏轴，不进入在线预测合同。
- 导师参数模型通过 `BRIDGE_DESIGN_MODEL_ENABLED` 控制，默认启用并从 `bridge_algorithm_service/mentor_models` 读取；`/health` 暴露阶段名称、版本和可用状态。
- 节点模型通过 `BRIDGE_NODE_MODEL_ENABLED` 与 `BRIDGE_NODE_MODEL_DIR` 启用；本地启动脚本发现审核后的本地产物时自动启用，生产 Compose 默认关闭并要求显式只读挂载。`/health` 暴露可用状态，`/predict` 返回模型版本、特征、留出指标、训练范围状态或回退原因。
- 标注工具只在本地浏览器读取图像；载入 JSON 后必须匹配 `source_image.file_name`，不同底图不得复用。
- 桥梁设计工作台的标签页与 `/bridge/model3d|drawing|params|results|safety|optimize` 路径双向同步；程序化跳转也必须更新路由，使侧栏选中态以路由为唯一来源。
- 图纸、分析和 Excel 必须使用同一组不可变 `design_inputs + result` 预测快照；表单变更后要求重新预测，后端再次校验快照。显式选择的前/后半区、结果 `design_mode.applied` 和 alpha 半区必须一致，否则拒绝出图。
- 图纸采用主图优先的精简版式：只保留结构模式、净跨、矢高、alpha/beta、适用域/回退/Pilot状态和简化图签；详细输入、坐标、构件长度、根径、版本和区间在后续结果界面展示。图纸不使用静默默认规格，B/C 工程命名保留“三节苗拱顶牛头节点”。
- 寿命 V1 使用可解析 Weibull–Gamma 生存模型，不在请求热路径运行 MCMC；已服役年限作为右删失证据，输出中位数、90% 先验/后验预测区间和未来 10 年存续概率。
- 当前寿命数值先验及暴露、维护、防腐因子均为待校准的方案筛选假设；木材中文俗名不形成物种寿命排序，跨度、桥宽、矢高和未标定的定性状态不进入腐朽寿命模型。
- 寿命模型依据与后续校准路线见 `docs/research/bayesian-service-life-open-source.md`；取得失效/右删失巡检数据后再升级分层 PyMC/Stan 离线校准。
- 自然语言入口仅用确定性规则完成 `_extract_params → _normalize_params`，随后进入与结构化请求相同的模型主链 `_predict`；不启动或调用 Ollama。
  - 2026-08-16：`_extract_params` 显式提取 rise（矢高/拱高/起水高度/rise）和“三节苗/五节苗 N 系”计数；n1/n2 口径为“系”，一系三节苗含斜弦 2 根、平弦 1 根，一系五节苗含外/内斜弦各 2 根、平弦 1 根，图纸数量按 `n1、2n1、2n2、2n2、n2` 展开；图纸去警示红（三节苗绿色系、长度深灰、外推提示琥珀色）；`start_bridge_algorithm.ps1` 必须保持 UTF-8 with BOM，Windows PowerShell 5.1 下无 BOM 会使节点模型自动启用条件失效。
  - 算法镜像依赖补齐 `scikit-learn==1.6.1`、`joblib==1.4.2`，否则 `node_model.py` 在镜像内无法导入；`.gitignore` 忽略 `.env/.env.local/scratch/`，部署前清理根目录抓取临时文件。
  - 生产部署修正：算法 Dockerfile 内置阿里云 apt 源；`REDIS_URL` 只保留 `redis://:quote(pw)@host:6379`，DB 编号由缓存/Celery 各自追加；`requirements.bridge.txt` 显式加 `django-celery-results==2.5.1`；`bridge-celery` 编排加 `C_FORCE_ROOT=true`。服务器 `/opt/bridge-dvadmin3` 与本地按同一套配置同步。
  - 前端工作台跳转统一走 `goToTab`（先 `router.replace` 再切 tab），PNG 图纸单独存 sessionStorage；`/bridge/*` 七个路由虽复用同一组件，但跨路由会重新挂载，恢复状态时不得清空已生成图纸，避免发送参数后出现“跳图纸—空白—再跳一次”。
  - n1/n2 改为“系”不改变任何模型数值：`service_life.py` 的贝叶斯寿命模型不接收 n1/n2，安全分析 `analyze` 只用 span、rise_span 和根径结果；优化模块仍按原 n2 数值计算 density/redundancy，不受单位标签影响。若未来把总构件数作为贝叶斯因子输入，三节苗应传 `3×n1`，五节苗应传 `5×n2`。

- 2026-08-14 review 修复后新增/变更的决策：
- 标注 span 字段（count/index/clear_span_m）为 `None` 时沿 `span → bridge → dimensions` 链回退（`_first()` 辅助）；空白 `source_group_id` 回退 `bridge_key`，不得坍缩为空分组键。
- 网站净跨补齐以 `source_image.file_name == drawings[].url` 文件名完全一致为身份依据，不按桥名模糊写入；同名桥、别名、迁建/原构因此由原图词条消歧。应用前必须零未解决，应用时完整备份并生成审计 CSV。
  - `load_training_rows` 强制校验：同一 `bridge_key` 只允许一个 `split_group_key`、`sample_key` 全局唯一；缺 `span_m` 以 `missing_span_m` 软排除该行，不再中断整个训练。
  - 生产环境 `SECRET_KEY` 必须经环境变量注入（DEBUG=False 时拒绝 `django-insecure-` 前缀默认值）；compose 对 `SECRET_KEY`/`DATABASE_PASSWORD`/`REDIS_PASSWORD` 采用 fail-closed 强制注入。
  - nginx 已补 `/sse/`、`/ws/` 反代；`/chat` 未识别 span/width 关键词时返回 `need_params`（前端已有对应分支）。
  - `/bridge/*` 七个工作台路径在 `route.ts` 静态注册（isHide 不占侧栏）作为菜单缺失时的兜底。
  - 完整修复清单与部署注意事项见 `docs/review-fix-log-2026-08-14.md`。
  - 前端工作台外壳按既有版本恢复为白色侧栏、白色顶栏、浅灰内容区和单一柔和蓝色选中态；展开侧栏显示业务系统名称，菜单使用 15px/600 字重，侧栏、横向布局及登录页不再渲染 DVAdmin Logo 和文字，主题缓存版本升级时仅清理视觉配置，避免旧配色回灌。

## 方法学 P0 修复状态

- 已修复内层调参的样本级 MAE 偏差：划分仍按 `split_group_key` 防止谱系泄漏，评分改按真实 `bridge_key` 宏平均。
- 已同时输出 `bridge_macro_mae` 与 `split_group_macro_mae`，兼容字段 `group_macro_mae` 从 v5 起明确等于真实桥级指标。
- 已将约 20% 隔离分组固定为独立内部留出；特征、模型和超参数只在开发集选择，留出集仅在选择完成后评估一次。
- 仍需独立外部数据验证；v5 的内部留出不能写成外部泛化结论。
