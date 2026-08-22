[CmdletBinding()]
param(
    [switch]$RuntimeOnly,
    [switch]$AllGroups,
    [switch]$NoHooks
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
if ($RuntimeOnly -and $AllGroups) {
    throw "-RuntimeOnly and -AllGroups cannot be used together."
}

Push-Location -LiteralPath $RepositoryRoot
try {
    Write-Host "Checking for Python 3.12..."
    & uv python find 3.12 *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Python 3.12 was not found; installing a managed interpreter with uv..."
        Invoke-CheckedCommand uv python install 3.12
    }

    $SyncArguments = @("sync", "--locked")
    if ($RuntimeOnly) {
        $SyncArguments += "--no-dev"
    }
    elseif ($AllGroups) {
        $SyncArguments += "--all-groups"
    }

    Write-Host "Creating or updating the locked environment..."
    Invoke-CheckedCommand uv @SyncArguments

    if (-not $RuntimeOnly -and -not $NoHooks -and (Test-Path -LiteralPath ".pre-commit-config.yaml")) {
        Write-Host "Installing pre-commit and pre-push hooks..."
        Invoke-CheckedCommand uv run --frozen pre-commit install --install-hooks --hook-type pre-commit --hook-type pre-push
    }

    Write-Host "Running environment diagnostics..."
    Invoke-CheckedCommand uv run --frozen python scripts/doctor.py
    Write-Host "Bootstrap completed successfully."
}
finally {
    Pop-Location
}
