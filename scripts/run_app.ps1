[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$StreamlitArguments
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Application = Join-Path $RepositoryRoot "app\streamlit_app.py"
$EnvironmentFile = Join-Path $RepositoryRoot ".env"
if (-not $env:UV_CACHE_DIR) {
    $env:UV_CACHE_DIR = Join-Path $RepositoryRoot ".uv-cache"
}
if (-not $env:YOLO_CONFIG_DIR) {
    $env:YOLO_CONFIG_DIR = Join-Path $RepositoryRoot ".runtime\ultralytics"
}
$env:YOLO_AUTOINSTALL = "false"
$env:YOLO_OFFLINE = "true"
New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Run scripts/bootstrap.ps1 after installing uv."
}
if (-not (Test-Path -LiteralPath $Application -PathType Leaf)) {
    throw "Streamlit entry point not found: $Application"
}

Push-Location -LiteralPath $RepositoryRoot
try {
    $UvArguments = @("run", "--frozen")
    if (Test-Path -LiteralPath $EnvironmentFile -PathType Leaf) {
        $UvArguments += @("--env-file", $EnvironmentFile)
    }
    & uv @UvArguments streamlit run $Application --server.headless true @StreamlitArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Streamlit exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
