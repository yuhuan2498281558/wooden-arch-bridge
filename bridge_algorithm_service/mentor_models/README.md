# 导师两阶段设计参数模型

本目录存放运行时模型资产：

- `stage1_xgboost.ubj`：从旧系统的 `stage1_xgboost.pkl` 转换得到的 XGBoost 原生模型；
- `stage2_cfbpnn_weights.npz`：从 `CF-BPNN.mat` 原样提取的级联前向网络权重和偏置；
- `manifest.json`：特征顺序、归一化范围、输出范围、模型指标、来源文件哈希和运行时资产哈希。

运行服务不会反序列化旧 pickle，也不需要 MATLAB。重新导入时使用：

```powershell
python tools/import_mentor_design_models.py `
  --bridge-design-zip D:\qq\bridge-design.zip `
  --ssa-xgboost-zip D:\qq\SSA-XGBoost.zip `
  --output-dir bridge_algorithm_service\mentor_models `
  --force
```

旧压缩包同时包含原始 MATLAB SSA 和后期 Python 候选搜索脚本；现有模型产物保持原值回放，不能仅凭文件名断言其训练时具体采用了哪一个搜索实现。
