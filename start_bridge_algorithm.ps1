$ErrorActionPreference = "Stop"
$root = if (-not [string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $PSScriptRoot
} else {
    (Get-Location).Path
}
$root = [string]$root
if ([string]::IsNullOrWhiteSpace($root)) {
    throw "Unable to resolve the bridge project root"
}
Set-Location -LiteralPath $root

# 本地联调时仅在 alpha/beta 两份受信任产物同时存在的情况下启用节点 Pilot。
# 显式设置的环境变量优先，生产 Compose 仍保持默认关闭。
$nodeModelDir = "$root\bridge_algorithm_service\model_artifacts"
$alphaModel = "$nodeModelDir\alpha_model.joblib"
$betaModel = "$nodeModelDir\beta_model.joblib"
if ([string]::IsNullOrWhiteSpace($env:BRIDGE_NODE_MODEL_ENABLED) -and
    (Test-Path -LiteralPath $alphaModel) -and
    (Test-Path -LiteralPath $betaModel)) {
    $env:BRIDGE_NODE_MODEL_ENABLED = "true"
    $env:BRIDGE_NODE_MODEL_DIR = $nodeModelDir
}

$designModeModel = "$nodeModelDir\alpha_design_mode_model.joblib"
if ([string]::IsNullOrWhiteSpace($env:BRIDGE_NODE_DESIGN_MODE_MODEL_ENABLED) -and
    (Test-Path -LiteralPath $designModeModel)) {
    $env:BRIDGE_NODE_DESIGN_MODE_MODEL_ENABLED = "true"
    $env:BRIDGE_NODE_DESIGN_MODE_MODEL_DIR = $nodeModelDir
}

if ([string]::IsNullOrWhiteSpace($env:BIMFACE_APP_KEY)) {
    $env:BIMFACE_APP_KEY = "e93qcJ1SsXWkFQzZUKJjJpoHyXDzFpuw"
}
if ([string]::IsNullOrWhiteSpace($env:BIMFACE_APP_SECRET)) {
    $env:BIMFACE_APP_SECRET = "BioDbtAyOakfPIFBH2eIhfE8ynvr5siC"
}
if ([string]::IsNullOrWhiteSpace($env:BIMFACE_FILE_ID)) {
    $env:BIMFACE_FILE_ID = "10000996606458"
}

& "C:\Users\24982\.conda\envs\bridge_dvadmin3\python.exe" -m uvicorn bridge_algorithm_service.main:app --host 127.0.0.1 --port 8001 --reload
