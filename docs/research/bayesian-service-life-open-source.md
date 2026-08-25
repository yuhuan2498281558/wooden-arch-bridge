# 木结构/木桥贝叶斯使用寿命估计：开源实现与可落地方案

> 调研日期：2026-08-13  
> 范围：方案阶段的木结构/木桥耐久性寿命表达；当前接口仅稳定拥有 `span`、`width`、`rise`、`wood_type` 等少量字段。本文不讨论结构安全鉴定，也不把耐久性寿命等同于承载能力失效时间。

## 结论先行

当前系统不能从现有输入诚实地算出某座木桥“预计使用 38 年”。现有 `analyze()` 返回固定的 `prior_life=35`、`estimated_life=38` 和 `ci_90=[28,48]`，没有似然、观测数据或不确定性传播，因而不是一次贝叶斯更新（见 [`bridge_algorithm_service/main.py`](../../bridge_algorithm_service/main.py#L610-L616)）。

建议分两阶段落地：

1. **现在：先验预测/情景评估。** 使用有版本、有出处、经专家确认的参考寿命分布，输出“先验预测中位数、90% 先验预测区间、各时间点存活概率”，并显式列出缺失因素。没有桥梁观测时，后验应等于先验，不能把一个人为加减后的点值称为“贝叶斯估计”。
2. **有数据后：贝叶斯 Weibull 生存模型。** 用失败时间和仍在服役样本的右删失时间建立似然；再按材料、暴露、构造和维护分层。Stan 官方生存模型说明了 Weibull 模型以及右删失样本通过生存函数进入似然的方式，[PyMC 官方示例](https://www.pymc.io/projects/examples/en/stable/survival_analysis/bayes_param_survival.html)也给出了同类实现。

木材腐朽模型的一手研究把木材含水率与温度作为关键直接变量。23 个欧洲现场点的研究发现，木材含水率与木温形成的组合剂量能支撑寿命预测，而宏观气候指数与腐朽进展相关性较差（[Brischke & Rapp, 2008，论文元数据和摘要](https://research.uni-hannover.de/en/publications/dose-response-relationships-between-wood-moisture-content-wood-te/)）。后续设计框架同样使用逐日木材含水率、温度、暴露剂量和材料抵抗剂量（[Niklewski et al., 2021](https://www.mdpi.com/1999-4907/12/6/721)；[Brischke et al., 2021](https://www.mdpi.com/1999-4907/12/5/576)）。因此，仅凭跨度、桥宽和矢高不应虚构腐朽寿命修正。

## 一、当前输入能说明什么

| 输入 | 对耐久性寿命的可用性 | 建议 |
|---|---|---|
| `wood_type` | 可能影响材料抵抗，但中文俗名不足以唯一确定植物学种、心材/边材、处理状态和材料批次 | 只用来选择“材料先验组”；映射不完整时退回未知材料，不做武断排序 |
| `span`、`width`、`rise` | 可用于结构几何和荷载模型；现阶段没有经过验证的路径把它们直接映射到腐朽寿命 | V1 寿命模型中不使用，避免伪相关 |
| 构件尺寸/节点几何 | 可能通过表面积、端面暴露、积水及剩余截面影响耐久与极限状态，但当前接口没有这些机制变量 | 后续在有暴露/构造证据和结构极限状态模型时再加入 |
| `result` 中的构件直径 | 可作为后续“腐朽深度达到容许截面损失”的阈值输入，但当前没有腐朽速率与极限状态校核 | 不应直接乘一个经验系数变成年数 |

当前 `wood_type` 选项也有语义风险：

- “杉木”通常可能指 `Cunninghamia lanceolata`，但心材、边材及处理状态不明。一项针对该树种的试验将其归为“稍耐久”，但这是试件腐朽/耐候研究，不能直接转换成整桥年限（[原始研究](https://www.mdpi.com/1999-4907/11/12/1326)）。
- 马尾松 `Pinus massoniana` 的实验研究报告了对褐腐菌的快速质量损失，且试验材料来自单株、特定菌种与土块试验；这同样不能直接推出桥梁寿命（[原始研究](https://doi.org/10.1016/j.carbpol.2022.119242)）。
- “柏木”没有植物学名称，可能对应不同属种；未明确前不得给独立数值先验。
- “混合木材”还涉及构件级材料组合。系统寿命通常受最脆弱关键构件和可更换策略影响，不能简单取三个树种寿命的平均值。

## 二、推荐的模型定义

### 2.1 先定义终点

“使用寿命”必须绑定一个可观测终点。推荐 V1 明确为：

> 在给定材料、暴露、构造与维护情景下，直到因真菌腐朽达到预先定义的维修/更换阈值的时间。

不应称为“桥梁结构失效时间”，除非已经建立荷载效应、初始抗力、腐朽导致的截面/强度退化和系统可靠度模型。木材腐朽会降低强度，但从质量损失或腐朽等级到系统失效仍需要结构模型；USDA《Wood Handbook》分别讨论木材生物劣化、结构分析和木桥使用，可作为工程边界参考（[USDA 官方入口及分章](https://research.fs.usda.gov/fpl/wood-handbook)）。

### 2.2 V1：先验预测情景，而不是假后验

首选一个可解析的 Weibull–Gamma 模型。固定代表老化趋势的 Weibull 形状 `k>1`，对正的风险率 `lambda` 设置 Gamma 先验（这里 `b` 是 rate，不是 scale）：

```text
T | lambda,k ~ Weibull(hazard = lambda*k*t^(k-1))
lambda ~ Gamma(shape=a, rate=b)
S(t | lambda,k) = exp(-lambda*t^k)
S_prior(t) = (b/(b+t^k))^a
q_p = [b*((1-p)^(-1/a)-1)]^(1/k)
```

它有三个优点：Weibull 能表达随时间上升的失效风险；对 `lambda` 积分后自然传播了风险率不确定性；所有运行时结果均可解析计算，不需在线 MCMC。Weibull 的公式、危险率与删失似然可直接核对 [Stan 官方生存模型文档](https://mc-stan.org/docs/stan-users-guide/survival.html)及其[文档源代码](https://github.com/stan-dev/docs/blob/master/src/stan-users-guide/survival.qmd)。

参数不应由程序员拍脑袋填写。每个“终点 × 暴露情景 × 材料组”至少由可追溯资料或结构/木材专家给出两个分位数，再数值求解 `a,b`；若专家先确定 `a,k` 与参考中位数 `m50`，则：

```text
b = m50^k / (2^(1/a)-1)
```

这种录入方式比直接写“均值 35 年、上下 10 年”更容易审计。`k` 控制风险随时间变化的形状，`a` 控制先验异质性，二者必须有专家/数据依据并做敏感性分析。若只有一个参考寿命点，则必须对形状和离散度另行给出处；没有第二条证据时应返回“无法给出数值区间”。ISO 15686-8 提供参考寿命数据选择、格式化和因子法估计的框架，但不替项目提供具体修正因子（[ISO 官方标准页](https://www.iso.org/standard/39070.html)）。

推荐将情景拆开显示，不在输入缺失时用一组隐藏权重混成一个看似精确的数字：

- `sheltered_dry`：遮蔽、排水良好、无土壤/水体接触；
- `exposed_wet`：露天雨淋，但不长期接触土壤或水；
- `ground_or_water_contact`：接地或临水关键构件；
- `unknown`：关键暴露未知，输出各情景范围，不输出单一推荐年限。

木材服务寿命的剂量—抵抗框架把暴露剂量 `D_Ed` 与材料抵抗剂量 `D_Rd` 分开；材料抵抗又受湿润能力和固有抗腐性共同影响（[Meyer–Veltrup 模型验证论文](https://www.mdpi.com/1999-4907/12/5/576)）。因此场景拆分有物理依据，也比把 `span` 塞进回归式更可解释。

### 2.3 V2：有历史寿命数据后的贝叶斯更新

V1 的 Weibull–Gamma 模型可先用闭式公式更新。若有 `d` 个达到终点的样本，以及所有失效或右删失样本的观测时长 `t_i`：

```text
a_post = a + d
b_post = b + sum(t_i^k)
lambda | data ~ Gamma(a_post, b_post)
S_post(t) = (b_post/(b_post+t^k))^a_post
```

仍在服役的桥不增加 `d`，但其观测年限会进入 `sum(t_i^k)`；这正是右删失证据。该闭式模型适合数据很少时的首个真实更新版本，也便于单元测试。它仍假设所有样本共享固定 `k` 且条件同质；材料、暴露和维护出现系统差异后，应升级到下述分层模型。

对桥/构件 `i`：

```text
log(lambda_i) = beta0
              + beta_material[material_i]
              + beta_exposure[exposure_i]
              + beta_detail[detail_i]
              + beta_maintenance[maintenance_i]
              + group_effect[bridge_or_region_i]
T_i ~ Weibull(k, lambda_i)
```

似然必须保留删失信息：

- 已达到终点：贡献密度 `f(t_i)`；
- 随访到 `c_i` 仍未达到终点（右删失）：贡献 `S(c_i)`；
- 两次巡检之间首次发现达到终点（区间删失）：贡献 `S(l_i)-S(u_i)`。

[Stan 官方文档](https://mc-stan.org/docs/stan-users-guide/survival.html)明确说明未失效样本通过互补累积分布/生存函数进入似然；[PyMC 参数生存分析示例](https://www.pymc.io/projects/examples/en/stable/survival_analysis/bayes_param_survival.html)及其 [GitHub notebook](https://github.com/pymc-devs/pymc-examples/blob/main/examples/survival_analysis/bayes_param_survival.ipynb)可作为 Python 实现起点。不能丢弃仍在服役的老桥，否则寿命样本会产生严重选择偏差。

### 2.4 V3：有巡检或传感数据后的退化更新

若以后获得腐朽深度、质量损失、剩余截面或含水率时序，可考虑：

1. 用逐日木材含水率和木温累计暴露剂量；
2. 用材料抵抗剂量表示树种、心/边材、湿润能力和防腐处理；
3. 用单调 Gamma 过程表示不可逆退化，并在巡检后更新退化率；
4. 当退化达到维修/更换阈值时得到剩余寿命分布。

Gamma 过程已被用于老化桥梁随机抗力退化（[开放获取原始论文](https://doi.org/10.1016/j.jtte.2018.11.001)），木结构领域也已有结合空间腐朽和巡检证据的动态贝叶斯网络（[原始论文](https://doi.org/10.1016/j.engstruct.2020.110301)）。两者都需要历史服务或巡检证据；在当前“新方案 + 无观测”接口上直接引入会增加复杂度，却不会凭空增加信息。

## 三、先验和退化因子如何落地

### 3.1 必须新增或确认的字段

要从“先验情景”升级成个体化寿命估计，至少需要：

| 类别 | 字段示例 | 原因 |
|---|---|---|
| 终点 | `endpoint_id`、维修/更换阈值 | 决定“寿命”到底是什么 |
| 材料 | 植物学种、心/边材、等级、批次、胶合/实木 | 中文俗名不足以确定抵抗性 |
| 处理 | 防腐剂、保持量/渗透、改性、涂层 | 处理可显著改变材料抵抗 |
| 暴露 | 地点/气候文件、雨淋、遮蔽、距地/水、朝向 | 决定湿润和干燥循环 |
| 构造 | 端面暴露、排水、积水缝隙、金属连接处滞水 | 决定局部材料气候 |
| 维护 | 检查周期、重涂/修缮策略 | 改变达到终点的时间 |
| 更新证据 | 建成年、巡检日期、状态/腐朽深度、测量误差 | 构造删失或退化似然 |

木材水分、温度和暴露的重要性可由 USDA 官方《Wood Handbook》和现场剂量研究交叉核实（[USDA](https://research.fs.usda.gov/fpl/wood-handbook)；[Brischke & Rapp, 2008](https://doi.org/10.1007/s00226-008-0191-8)）。模型评估论文还指出，性能化木材寿命设计通常需要把吸湿/湿热模型与腐朽剂量模型组合（[Niklewski et al., 2021](https://www.mdpi.com/1999-4907/12/6/721)）。

### 3.2 当前字段的建议处理

- `wood_type=default` 或“混合木材”：`material_evidence=unknown`；不返回物种特定的单值。
- `wood_type=杉木/马尾松/柏木`：先做词汇映射校验；若缺植物学种、心/边材或处理状态，仍应标记 `material_detail_missing=true`。
- `span/width/rise`：保留在响应的输入快照中，但 V1 的 `factors_used` 不包含它们。
- 不从 `overall_score`、`beta_proxy` 或启发式“腐朽风险”等级反向生成寿命；这些量目前不是现场证据。

### 3.3 数值先验的治理

建议把数值先验放到独立、版本化配置，不写死在接口函数中。每条记录至少包含：

```json
{
  "prior_id": "timber-decay-repair-threshold-v1",
  "endpoint_id": "fungal_decay_repair_threshold",
  "material_group": "...",
  "exposure_scenario": "...",
  "q05_years": null,
  "median_years": null,
  "source_urls": [],
  "elicited_by": [],
  "approved_at": null,
  "applicability": "...",
  "status": "draft"
}
```

本次检索没有找到能直接把“杉木/马尾松/柏木 + 跨度/桥宽/矢高”映射成整桥年限的开放校准数据。现有物种试验可以说明相对抗腐或试件质量损失，却不能作为整桥绝对年限表。因此本文不伪造三个树种的年限数字；数值必须由适用地区、终点和暴露条件一致的数据或专家正式给定。

可用 [WoodSolutions 官方《Timber Service Life Design Guide》](https://www.woodsolutions.com.au/system/files/2025-07/WS%20TDG%2005%20Timber%20Service%20Life%20Design%2002-21.pdf)及 [FWPA TimberLife 技术手册索引](https://fwpa.com.au/report/technical-manuals-for-timber-service-life-design-guide/)校核情景量级和字段设计。它们同时考虑气候、耐久等级、心/边材、防腐处理、构造与维护，且建立于澳大利亚材料和暴露数据；只能用于先验情景与量级校核，不能直接移植成中国木桥的树种年限。

## 四、不确定性传播与输出契约

### 4.1 V1 不需要在线 MCMC

若 `k`、`lambda` 由已审批先验配置给定，Weibull 的中位数、分位数和 `S(t)` 都可解析计算；若参数本身也有不确定性，可离线抽样后保存固定、可复现的先验预测分位数。不要在每个 FastAPI 请求中运行 MCMC。

有校准数据后，可离线用 PyMC 或 Stan 拟合，再把后验抽样/汇总作为版本化模型产物部署。PyMC 为 Apache-2.0 开源项目（[官方 GitHub 仓库](https://github.com/pymc-devs/pymc)与[许可证](https://github.com/pymc-devs/pymc/blob/main/LICENSE)）；Stan/CmdStan 为 BSD 系列许可，[CmdStanPy](https://github.com/stan-dev/cmdstanpy)提供 Python 接口；[ArviZ](https://github.com/arviz-devs/arviz)可用于诊断和预测检查。

### 4.2 推荐响应

```json
{
  "status": "prior_only",
  "endpoint": "fungal_decay_repair_threshold",
  "method": "weibull_prior_predictive",
  "model_version": "service-life-prior-v1",
  "scenario": "unknown_exposure",
  "median_years": null,
  "prediction_interval_90_years": null,
  "survival_probability": {"10": null, "20": null, "30": null, "50": null},
  "evidence_used": [],
  "factors_used": ["wood_type"],
  "factors_not_used": ["span", "width", "rise"],
  "missing_factors": ["endpoint_threshold", "exposure", "heartwood_or_sapwood", "treatment", "detailing", "maintenance"],
  "assumptions": [],
  "warning": "仅为方案阶段先验情景，不是结构安全鉴定或个体桥梁后验寿命。"
}
```

有经审批的完整先验时，`null` 才替换为数值。术语应使用“90% 先验预测区间”或“90% 后验预测区间”，不再写“90% 置信区间”。建议同时显示多个时间点的存活概率；单独显示均值会掩盖右偏分布与尾部风险。

### 4.3 解释要求

前端至少显示：

- 终点定义；
- 结果是 `prior_only` 还是 `posterior_updated`；
- 使用了哪些证据以及没有使用哪些几何字段；
- 关键缺失因素和当前情景假设；
- 中位数和预测区间，而不是“先验 35、估计 38”两个来源不明的点值；
- “方案筛选用途，不替代材料检测、耐久性设计或结构安全鉴定”。

## 五、验证与验收建议

### 5.1 V1 先验预测

1. **配置完整性：** 每个可返回数值的先验必须有终点、适用情景、至少两个分位数、来源、批准人与版本。
2. **数学性质：** `q05 < median < q95`；`S(0)=1`；`S(t)` 单调不增；所有概率在 `[0,1]`。
3. **语义测试：** 改变 `span/width/rise` 不得改变 V1 腐朽寿命；选择未知/混合材料不得冒充物种精确结果。
4. **先验预测检查：** 由木材耐久与桥梁维护专家检查模拟寿命、短期失败概率及长尾是否合理；不通过则修改先验并记录版本，而不是修改展示文案。
5. **敏感性：** 分别报告材料、暴露、构造、维护先验变化对结果的影响，避免把主要不确定性藏在一个总分中。

### 5.2 有观测数据后的模型

1. 按桥梁/地区/测绘或维护谱系分组交叉验证，禁止同一桥的多个构件跨训练与测试折。
2. 保留右删失和区间删失；报告不同删失机制下的敏感性。
3. 评价指定时间点的校准、预测区间覆盖率、时间依赖 Brier 分数/综合 Brier 分数，而不只报告相关系数或 C-index。[scikit-survival 官方指南](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html)说明了删失数据下的 C-index、动态 AUC 和 Brier 分数及其局限。
4. 使用后验预测检查；ArviZ 提供 [`plot_ppc`](https://python.arviz.org/en/stable/api/generated/arviz.plot_ppc.html)等工具。
5. 检查 MCMC 收敛、有效样本量和 Monte Carlo 误差；ArviZ 的 [`rhat`](https://python.arviz.org/en/stable/api/generated/arviz.rhat.html)文档同时指向 ESS 与 MCSE。
6. 与三个基线比较：不分组的 Kaplan–Meier、仅材料组的先验/生存模型、现行参考寿命规则。复杂模型只有在外部或严格分组验证中显著改善校准才升级。
7. 在中国桥梁、目标树种、实际构造和气候条件上做外部验证；欧洲试件/建筑构件模型不能未经校准直接作为中国木拱桥定量结论。

## 六、开源项目取舍

| 项目 | 可复用内容 | 当前阶段取舍 |
|---|---|---|
| [PyMC](https://github.com/pymc-devs/pymc) / [官方生存 notebook](https://github.com/pymc-devs/pymc-examples/blob/main/examples/survival_analysis/bayes_param_survival.ipynb) | Bayesian Weibull/AFT、删失似然、后验预测 | 推荐用于离线校准；不放在线请求热路径 |
| [Stan 生存模型源码](https://github.com/stan-dev/docs/blob/master/src/stan-users-guide/survival.qmd) / [CmdStanPy](https://github.com/stan-dev/cmdstanpy) | 明确的密度、CCDF 删失项、成熟采样器 | 可作独立复核实现；团队更熟 Python 时 PyMC 成本更低 |
| [ArviZ](https://github.com/arviz-devs/arviz) | R-hat、ESS、MCSE、先验/后验预测检查 | 推荐作为离线验收工具 |
| [scikit-survival](https://github.com/sebp/scikit-survival) | 删失数据的 Brier、动态 AUC、C-index | 推荐用于模型外部验证，不承担贝叶斯拟合 |

检索中未找到同时满足“公开许可、可审查源码、针对中国杉木/马尾松/柏木木桥、包含气候/构造暴露并有外部验证”的即插即用 GitHub 模型。应复用通用贝叶斯生存工具和已发表的木材剂量—抵抗结构，而不是复制来源不明的 GitHub 年限公式。

## 七、建议实施顺序

1. 删除固定 `35/38/[28,48]` 逻辑，先把接口改成 `prior_only`，无批准先验时返回 `insufficient_evidence`。
2. 明确一个耐久性终点并建立版本化先验注册表；请木材与桥梁专家给出适用地区/暴露情景的分位数，而不是单一均值。
3. V1 用解析 Weibull 计算中位数、90% 先验预测区间和 `S(10/20/30/50)`；几何参数不进入腐朽寿命。
4. 前端新增暴露、材料细节、处理和维护字段；关键字段缺失时展示情景范围。
5. 建立构件级寿命/巡检数据表，保留未失效样本和测量误差。
6. 离线用 PyMC/Stan 拟合分层 Weibull；通过分组外部验证后再把后验产物接入 FastAPI。
7. 只有在获得含水率/温度/腐朽深度时序后，才升级到剂量—抵抗、Gamma 过程或动态贝叶斯网络。

## 主要一手来源

- [ISO 15686-8:2008 官方页：参考使用寿命与因子法](https://www.iso.org/standard/39070.html)
- [USDA Forest Products Laboratory：Wood Handbook 官方入口](https://research.fs.usda.gov/fpl/wood-handbook)
- [USDA FPL：Limiting Conditions for Decay in Wood Systems](https://www.fpl.fs.usda.gov/documnts/pdf2002/morri02a.pdf)
- [USDA Forest Service：Timber Bridge Inspection Manual](https://www.fpl.fs.usda.gov/documnts/misc/em7700_8--entire-publication.pdf)
- [WoodSolutions：Timber Service Life Design Guide](https://www.woodsolutions.com.au/system/files/2025-07/WS%20TDG%2005%20Timber%20Service%20Life%20Design%2002-21.pdf)
- [FWPA：TimberLife 技术手册索引](https://fwpa.com.au/report/technical-manuals-for-timber-service-life-design-guide/)
- [国家标准信息公共服务平台：GB/T 13942.2-2009 木材天然耐久性试验方法](https://std.samr.gov.cn/gb/search/gbDetailed?id=71F772D7CAB4D3A7E05397BE0A0AB82A)
- [Brischke & Rapp (2008)：木材含水率、木温和腐朽剂量—响应](https://doi.org/10.1007/s00226-008-0191-8)
- [Viitanen et al. (2010)：可嵌入湿热模型的木材腐朽模型](https://doi.org/10.1007/s00107-010-0450-x)
- [Niklewski et al. (2021)：木材湿度模型与腐朽模型评价](https://www.mdpi.com/1999-4907/12/6/721)
- [Brischke et al. (2021)：Meyer–Veltrup 材料抵抗模型验证](https://www.mdpi.com/1999-4907/12/5/576)
- [Meyer-Veltrup et al. (2017)：湿润能力与固有耐久性的组合预测](https://doi.org/10.1007/s00226-017-0893-x)
- [Tran, Bastidas-Arteaga & Aoues (2020)：木结构腐朽的动态贝叶斯网络和巡检更新](https://doi.org/10.1016/j.engstruct.2020.110301)
- [Wang et al. (2019)：桥梁 Gamma 过程退化与贝叶斯更新](https://doi.org/10.1016/j.jtte.2018.11.001)
- [Stan 官方生存模型](https://mc-stan.org/docs/stan-users-guide/survival.html)
- [PyMC 官方 Bayesian 参数生存分析](https://www.pymc.io/projects/examples/en/stable/survival_analysis/bayes_param_survival.html)
