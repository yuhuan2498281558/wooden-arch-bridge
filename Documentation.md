# 环境与操作文档

## 技术栈与版本

- 前端：Vue 3、TypeScript 4.9、Vite 5、Element Plus；Node.js 要求 `>=16`，npm 要求 `>=7`。
- 平台后端：Python、Django 4.2.14、Django REST Framework 3.15.2、django-celery-results 2.5.1（必须显式安装，否则 Celery 结果表无迁移）。
- 算法服务：FastAPI 0.115.6、Uvicorn 0.30.3、Pillow、OpenPyXL、NumPy 2.2.6、XGBoost 3.2.0、scikit-learn 1.6.1、joblib 1.4.2（算法镜像必须包含后两项才能导入 `node_model.py`）。
- 机器学习：导师矢跨比模型在线使用 XGBoost；导师根径模型按 CF-BPNN 权重矩阵直接推理；五节苗节点 Pilot 使用 scikit-learn/joblib；离线研发另使用 SciPy、Matplotlib。
- 部署：Docker Compose；Web、Django、FastAPI 和 Redis 四个服务。

## 权威文档

- 平台基础安装：`README.zh.md`。
- 标注工具：`bridge_algorithm_service/annotation_tool/README.md`。
- 对称预处理：`ml_pipeline/prepare/README.md`。
- Pilot 训练：`ml_pipeline/train/README.md`。
- 研究与集成路线：`docs/五节苗牛头节点预测研发与论文路线.md`。

## 环境前置条件

- 本机算法服务启动脚本依赖已配置的 Conda Python 环境；脚本当前含机器专属解释器路径，迁移机器时需修改。
- 前端依赖位于 `web/`，使用 npm/yarn；不要无理由重装或覆盖现有锁文件。
- Docker 部署前从 `.env.bridge.example` 创建私有 `.env` 并填写真实值，绝不提交真实密码或密钥。

## 本地运行

算法服务（已验证入口）：

```powershell
.\start_bridge_algorithm.ps1
```

- 健康检查：`http://127.0.0.1:8001/health`
- 标注工具：`http://127.0.0.1:8001/annotation`
- `start_bridge_algorithm.ps1` 必须保存为 UTF-8 with BOM；Windows PowerShell 5.1 对无 BOM 的 UTF-8 脚本按 ANSI 解析，会导致节点模型自动启用条件不生效（已实测修复）。

前端（依据 `web/package.json`）：

```powershell
Set-Location web
npm run dev
```

平台后端的初始化和运行方式以 `README.zh.md` 为准；本项目存在本地/服务器差异，执行迁移前应检查当前数据库配置。
当前本地前端开发环境的 `VITE_API_URL` 指向 `http://127.0.0.1:8002`，联调时可使用：

```powershell
& "C:\Users\24982\.conda\envs\bridge_dvadmin3\python.exe" backend/manage.py runserver 127.0.0.1:8002 --noreload
```

## 测试与检查

算法服务、节点几何和训练流水线测试（2026-08-23 已验证）：

```powershell
python -m pytest bridge_algorithm_service/tests -q
```

最近结果：`111 passed`（含自然语言矢高解析、“系”计数、结构承重验算、安全分析联动、Excel 命名与版式、直接坐标编码/解码、显式模式部分共享回归及倾斜角消融用例）。

标注工具脚本语法检查：提取 `index.html` 的 `<script>` 内容后运行 `node --check`；修改交互后还需浏览器实测。

桥名元数据检查/修复：

```powershell
python -m ml_pipeline.prepare.repair_bridge_metadata <数据集目录>
python -m ml_pipeline.prepare.repair_bridge_metadata <数据集目录> --apply
```

第二条会修改文件并创建备份，先运行第一条干检查。

网站净跨精确匹配补齐/同步：

```powershell
python -m ml_pipeline.prepare.backfill_wbridge_spans <数据集目录>
python -m ml_pipeline.prepare.backfill_wbridge_spans <数据集目录> --apply
```

第一条必须确认 `unresolved=0` 后再应用。应用命令会在数据目录旁创建完整 JSON 备份和 `span_backfill_audit.csv`；重复干运行应显示 `matched=0 existing=<总记录数> unresolved=0`。

## 预处理与训练

生成对称目标 CSV：

```powershell
python -m ml_pipeline.prepare.symmetric_targets `
  --input-dir <数据集目录> `
  --output <输出CSV>
```

运行 Pilot（输出目录必须不存在或为空）：

```powershell
<Python解释器> -m ml_pipeline.train.pilot `
  --input-dir <数据集目录> `
  --output-dir <新结果目录>
```

当前已生成的权威内部验证版本为 `five-miao-node-pilot-v5-independent-holdout`：103 条完成预处理，102 条、96 个 `split_group_key` 参与训练，合龙桥第 2 跨以 `zero_inner_flat_chord` 排除。开发集 81 条/76 组，独立内部留出集 21 条/20 组；该留出仍来自同一网站数据源，不属于外部验证。预处理标识为 `parallel-angle-fixed-left-right-mean-v2`，结构范围策略为 `exclude-zero-inner-flat-chord-v1`；具体机器路径不写入共享记忆。

v5 选择结果：`alpha` 使用净跨 Ridge，独立留出桥级宏 MAE `0.09333`；`beta` 使用净跨+矢跨比 Ridge，独立留出桥级宏 MAE `0.02991`。相同留出集上的固定规则分别为 `0.13737`、`0.05768`。产物继续标记 `pilot_not_for_production`。

v6 使用 `ml_pipeline.train.model_improvement` 冻结 v5 的 `data_partition.csv`，比较 Ridge、Huber、ElasticNet、样条 Ridge、RBF-SVR、Gaussian Process 和浅层 Extra Trees。筛选与 LOGO 确认均只使用开发集；本轮没有复杂候选达到 MAE/q90 双替换门槛，因此两个目标保留 Ridge，留出指标保持 `alpha=0.09333`、`beta=0.02991`。权威产物目录名为 `five_miao_pilot_v6_small_sample`，状态仍为 `pilot_not_for_production`。

v7 新增 `ml_pipeline.prepare.layered_targets`，把原始历史观测、左右均值历史代理和专家规范设计目标拆成独立字段。当前 103 条记录中，102 条满足历史代理研究训练条件，22 条高残差记录经原图复核后标为有效历史差异并保留；专家规范目标批准数为 0。`normative_target_review_template.csv` 是后续专家补录入口，批准记录必须具有合法比例、复核人和证据说明。

`ml_pipeline.train.noise_aware_baseline` 在冻结 v5 分区上比较 Ridge、Huber 与中位数分位回归。抗噪声候选没有达到预设替换门槛，两个目标仍保留 Ridge，冻结留出桥级宏 MAE 保持 `alpha=0.09333`、`beta=0.02991`。v7 只研究历史对称代理，产物状态为 `research_proxy_not_for_deployment`，不接入算法服务。

v8 外节点两类结构实验：

```powershell
python -m ml_pipeline.train.structure_gated_alpha `
  --input-dir <数据集目录> `
  --output-dir <新结果目录> `
  --partition-file <v6目录\data_partition.csv> `
  --baseline-dir <v6结果目录>
```

当前结果目录为 `five_miao_pilot_v8_structure_gated_alpha`。前半区 35 条、后半区 67 条；类型已知时双专家开发集 LOGO 桥级宏 MAE 为 `0.05299`，但自动门控仅为 `0.08457`，未通过替换门槛。产物固定标记 `research_structure_gate_not_for_deployment`，不作为默认在线路由。

v9 设计人员给定模式实验：

```powershell
python -m ml_pipeline.train.design_mode_alpha `
  --input-dir <数据集目录> `
  --output-dir <新结果目录> `
  --partition-file <v6目录\data_partition.csv> `
  --baseline-dir <v6结果目录>
```

v9 不新增人工重标。自动派生用 `STRUCTURE_BOUNDARY_BAND`（±0.03）：贴边（远济 0.503 / 0.506/0.500）为 `uncommitted`，评估走 v6 连续预测；在线仍由设计人员选 `front_half/back_half`。已提交后半区若专家相对训练中位数明显向下收缩（咏归式 ~0.597 vs ~0.622），则与 2/3 取较高值；α≥0.70 仅 7 条，不另拟合专家、不换 Ridge。岚下 2026-08-25 重标均值 0.232，v9 前半区专家约 0.427 落在主体，不混合、不删样本。产物仍为 `research_designer_mode_not_for_deployment`。云端无 JSON 时本地复跑后看留出 CSV 的远济、咏归和岚下行。

v10 显式模式部分共享实验使用 `ml_pipeline.train.partial_pooling_alpha`：冻结 v5 分区，比较分区中位数、共享跨径斜率、受惩罚的模式斜率差和 v9 独立专家。开发集 LOGO 的模式宏 MAE 分别为 `0.05459/0.05166/0.05166/0.05163`，q90 分别为 `0.11002/0.10135/0.10146/0.10638`。部分共享模型选择 `shared_penalty=10`、`interaction_penalty=1000`，表明现有样本不支持稳定的模式特异斜率；按容差优先选择更简单的共享斜率 M1，但未达到相对 v9 的 `0.002` 替换门槛。历史留出 M1 模式宏 MAE 为 `0.05929`，因该留出已反复查看，仅作描述。v10 不生成部署产物，线上 v9 和 beta 均保持不变。

v11 倾斜角消融使用 `ml_pipeline.train.angle_ablation_alpha`：倾斜角由 `atan(3f/L)` 确定，角度与矢跨比禁止同时入模。开发集 LOGO 模式宏 MAE 为共享净跨 `0.05166`、共享角度 `0.05722`、净跨+角度 `0.05254`、净跨+矢跨比 `0.05260`、v9 独立专家 `0.05163`。净跨+角度相对净跨的桥级配对 MAE 增加 `0.00074`（95% bootstrap CI `[0.00043,0.00111]`），只在 `10/76` 个桥组上更优；角度没有通过替换门槛。历史留出仅作描述，最终可追溯结果目录名为 `five_miao_pilot_v11_angle_ablation_alpha_v2`，不生成部署产物。

原始标签与完整预处理严格配对实验使用 `ml_pipeline.train.preprocessing_strict_control`：冻结 102 条标准样本、v5 分区、`split_group_key`、特征和 Ridge(alpha=1e-8)，原始标签基线分别拟合左右观测再取预测均值，完整流程拟合平行角度修正后的对称设计目标。alpha 开发集 LOGO 桥级宏 MAE `0.07569→0.07258`（降低 4.1%），内部留出 `0.09649→0.09333`（降低 3.3%）；两处配对 bootstrap 95% 区间均跨 0，应描述为小幅一致改善而非统计显著。beta 为 `0.03034→0.03034`、`0.02991→0.02991`，在线性 Ridge 下数值等价。完整指标、逐样本预测和图位于 `preprocessing_strict_control_v1` 结果目录；论文图片备选说明见 `docs/research/paper-figure-candidates-20260820.md`。

原始像素/坐标直接预测对照使用：

```powershell
python -m ml_pipeline.train.direct_coordinate_baseline `
  --data-dir <数据集目录> `
  --partition <v5目录\data_partition.csv> `
  --output-dir <新结果目录>
```

该实验冻结样本和 v5 分区，以同一 Ridge 比较原始像素、图像宽高归一化坐标、桥轴局部坐标和知识参数化 `alpha/beta`；统一输出历史观测复现、对称设计代理、米制节点误差和几何警告率。当前结果目录为 `scratch/direct_coordinate_baseline_v2`，仅用于研究对照，不生成在线模型产物。

节点模型受控接入使用两个环境变量：

- `BRIDGE_NODE_MODEL_ENABLED=true`：显式启用；默认关闭并继续使用固定规则。
- `BRIDGE_NODE_MODEL_DIR=<产物目录>`：本地启动时指定同时包含 `alpha_model.joblib`、`beta_model.joblib` 的目录。
- `BRIDGE_NODE_DESIGN_MODE_MODEL_ENABLED=true`：受控启用 v9 设计模式 alpha；生产默认关闭。
- `BRIDGE_NODE_DESIGN_MODE_MODEL_DIR=<产物目录>`：目录须包含 `alpha_design_mode_model.joblib`，并继续依赖上面的 v6 beta。

Compose 部署通过 `BRIDGE_NODE_MODEL_HOST_DIR` 指定宿主产物目录并只读挂载到容器 `/app/model_artifacts`。v6 缺失、契约不符、输入越出训练范围或预测非法时，`/predict` 自动整组回退固定规则；v9 不可用时仅回退 v6。v6 对最小/最大范围内但局部样本稀疏的输入返回复核提示和 90% 经验误差区间；v9 也按所选模式返回独立区间和局部适用域。`/health` 的 `node_model.design_mode_model` 显示启用和可用状态。仅加载本项目离线训练产生并经过审核的受信任 joblib 产物。

导师参数模型默认启用：

- `BRIDGE_DESIGN_MODEL_ENABLED=true`：启用 SSA-XGBoost + CF-BPNN 主链。
- `BRIDGE_DESIGN_MODEL_DIR=<目录>`：可选覆盖运行产物目录；默认使用 `bridge_algorithm_service/mentor_models`。

运行产物由 `tools/import_mentor_design_models.py` 从受信任的旧版 `bridge-design.zip` 一次性转换。在线服务加载 XGBoost 原生模型和 NumPy 数值权重包，不直接加载旧 pickle/MAT。`manifest.json` 记录输入输出契约、哈希和原有指标；模型不可用时接口返回带明确原因的规则回退。旧阶段一记录的独立测试 `R²=-0.38868`、`MAE=0.02074`，接入完成不等于已经完成外部泛化验证。

## 构建与部署

前端构建（依据 `web/package.json`）：

```powershell
Set-Location web
npm run build
```

2026-08-16 部署前复验构建通过（`✓ built in 54.03s`）；当前仍有项目既有的 Sass 旧 API/除法弃用提示和大 chunk 警告，不影响产物生成。

桥梁系统 Docker 编排：

```powershell
docker compose -f docker-compose.bridge.yml up --build
```

部署会读取私有环境变量；执行前检查端口、数据库、Redis、CORS 和 BIMFACE 配置。
- 当前 `8.155.172.84` 服务器 `.env` 已显式启用 v6/v9 Pilot；仓库默认值仍是 `false`，其他新环境需自行显式开启。

服务器同步实测注意：
- 服务器 `.env` 必须先有 `SECRET_KEY`，否则新版 compose 拒绝构建。
- 算法镜像已在 Dockerfile 内把 Debian apt 源切到 `mirrors.aliyun.com`，避免云主机拉取 `fonts-noto-cjk` 极慢。
- `REDIS_URL` 不得自带 `/db`；由 `django_redis`/Celery 配置分别追加缓存库和 broker 库，密码必须 `urllib.parse.quote`。
- `bridge-celery` 需要 `C_FORCE_ROOT=true`（编排已配置），并显式安装 `django-celery-results`。

## 已知环境问题

- 工作区常有未提交改动，避免覆盖无关修改。记忆文件保持 UTF-8。Pilot 结果目录必须为空，重复训练用新版本目录。

## 交付文档

- 推荐交付：`docs/delivery/中国木拱廊桥智能设计系统_系统技术与使用说明_V1.1_精排版.docx`；V1.0 仅作版式对照。可维护源稿 `docs/delivery/wood-arch-bridge-system-report-current.md` 已是导师模型主链口径，精排 Word 待下一版同步。节点 Pilot、未标定寿命参数和待配置 BIMFACE 能力继续按边界项说明。
