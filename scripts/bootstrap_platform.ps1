[CmdletBinding()]
param(
    [string]$PythonPath,
    [switch]$NoHooks,
    [switch]$SkipChecks
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
if (-not $env:UV_CACHE_DIR) {
    $env:UV_CACHE_DIR = Join-Path $RepositoryRoot ".uv-cache"
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
    throw "uv is required. Install it from https://docs.astral.sh/uv/ and run this script again."
}

Push-Location -LiteralPath $RepositoryRoot
try {
    if (-not $PythonPath) {
        $PythonPath = (& uv python find 3.12 2>$null | Select-Object -First 1)
        if ($LASTEXITCODE -ne 0 -or -not $PythonPath) {
            Write-Host "Python 3.12 was not found; installing a managed interpreter with uv..."
            Invoke-CheckedCommand uv python install 3.12
            $PythonPath = (& uv python find 3.12 | Select-Object -First 1)
        }
    }

    Write-Host "Preparing the computer-vision environment with Python 3.12..."
    Invoke-CheckedCommand uv sync --python $PythonPath --locked

    Write-Host "Preparing the lightweight control-plane environment..."
    Invoke-CheckedCommand uv sync --project services/control_api --python $PythonPath --locked

    if (-not $NoHooks -and (Test-Path -LiteralPath ".pre-commit-config.yaml")) {
        Write-Host "Installing pre-commit and pre-push hooks..."
        Invoke-CheckedCommand uv run --frozen pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push
    }

    if (-not $SkipChecks) {
        Write-Host "Running platform diagnostics..."
        Invoke-CheckedCommand uv run --frozen python scripts/platform_doctor.py
        Invoke-CheckedCommand uv run --frozen python scripts/doctor.py --manifest-only
    }
    Write-Host "Campus platform bootstrap completed successfully."
}
finally {
    Pop-Location
}
