# 五节苗节点模型产物目录

本目录仅作为算法服务的只读挂载入口，不提交训练生成的二进制模型。

受控启用时放置同一训练版本的两个受信任文件：

- `alpha_model.joblib`
- `beta_model.joblib`

设置 `BRIDGE_NODE_MODEL_ENABLED=true` 后，算法服务从 `BRIDGE_NODE_MODEL_DIR` 加载。任何缺文件、契约不匹配、输入越出训练范围或预测非法都会整组回退到固定规则。v6 产物还会返回局部训练密度、稀疏区域提示和 90% 经验误差区间；稀疏提示不强制回退，但前端应要求人工复核。

v9 设计模式实验另使用：

- `alpha_design_mode_model.joblib`

设置 `BRIDGE_NODE_DESIGN_MODE_MODEL_ENABLED=true` 与 `BRIDGE_NODE_DESIGN_MODE_MODEL_DIR=<本目录>` 后，仅当请求显式传入 `outer_node_structure_type=front_half|back_half` 时，v9 对应 alpha 专家才覆盖 v6 alpha；beta 始终来自已通过检查的 v6 模型。v9 加载或适用域检查失败时只回退到 v6，不会破坏原有节点预测链。

joblib 属于可执行反序列化格式，只能放置本项目离线训练并完成审核的产物。
