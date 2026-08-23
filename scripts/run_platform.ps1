[CmdletBinding()]
param(
    [string]$BindAddress,
    [ValidateRange(1, 65535)]
    [int]$ListenPort,
    [switch]$NoSync,
    [switch]$NoEnvFile
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$ServiceProject = Join-Path $RepositoryRoot "services\control_api"
$EnvironmentFile = Join-Path $RepositoryRoot ".env"
$RuntimeDirectory = Join-Path $RepositoryRoot ".runtime\platform"

if (-not $env:UV_CACHE_DIR) {
    $env:UV_CACHE_DIR = Join-Path $RepositoryRoot ".uv-cache"
}
if ($BindAddress) {
    $env:CONTROL_API_HOST = $BindAddress
}
if ($ListenPort) {
    $env:CONTROL_API_PORT = $ListenPort.ToString([System.Globalization.CultureInfo]::InvariantCulture)
}

function Invoke-CheckedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$CommandArguments
    )

    & $Executable @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Executable $($CommandArguments -join ' ')"
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Run scripts/bootstrap_platform.ps1 after installing uv."
}
if (-not (Test-Path -LiteralPath (Join-Path $ServiceProject "uv.lock") -PathType Leaf)) {
    throw "Control API lockfile not found: $ServiceProject"
}

New-Item -ItemType Directory -Force -Path $RuntimeDirectory | Out-Null

Push-Location -LiteralPath $RepositoryRoot
try {
    if (-not $NoSync) {
        Write-Host "Synchronizing the locked control API environment..."
        Invoke-CheckedCommand uv sync --project $ServiceProject --locked
    }

    $UvArguments = @("run", "--project", $ServiceProject, "--frozen")
    if (-not $NoEnvFile -and (Test-Path -LiteralPath $EnvironmentFile -PathType Leaf)) {
        $UvArguments += @("--env-file", $EnvironmentFile)
    }
    $UvArguments += @("python", "-m", "control_api")

    $DisplayAddress = if ($env:CONTROL_API_HOST) { $env:CONTROL_API_HOST } else { "127.0.0.1" }
    $DisplayPort = if ($env:CONTROL_API_PORT) { $env:CONTROL_API_PORT } else { "8000" }
    Write-Host "Starting Campus Access at http://${DisplayAddress}:${DisplayPort}/"
    Invoke-CheckedCommand uv @UvArguments
}
finally {
    Pop-Location
}
