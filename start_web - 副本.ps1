param(
    [Alias("Host")]
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$InstallDeps,
    [switch]$BuildFrontend,
    [switch]$DevFrontend
)

$ErrorActionPreference = "Stop"

function Get-CommandPath {
    param([string]$Name)
    try {
        return (Get-Command $Name -ErrorAction Stop).Source
    } catch {
        return $null
    }
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SystemPython = "D:\python3.11\python.exe"
$NpmCmd = Get-CommandPath "npm.cmd"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[EzYOLO] .venv not found, creating virtual environment..."
    if (-not (Test-Path $SystemPython)) {
        $SystemPython = Get-CommandPath "python"
    }
    if (-not $SystemPython) {
        throw "Python not found. Please install Python 3.11+ first."
    }
    & $SystemPython -m venv ".venv"
}

if (-not (Test-Path $VenvPython)) {
    throw "Failed to create virtual environment at .venv"
}

if ($InstallDeps) {
    Write-Host "[EzYOLO] Installing Python dependencies..."
    & $VenvPython -m pip install -r "requirements.txt"
}

if ($BuildFrontend -or $DevFrontend) {
    if (-not $NpmCmd) {
        throw "npm.cmd not found. Please install Node.js first."
    }

    Push-Location (Join-Path $ProjectRoot "toolbox-frontend")
    try {
        Write-Host "[EzYOLO] Installing frontend dependencies..."
        & $NpmCmd ci

        if ($BuildFrontend) {
            Write-Host "[EzYOLO] Building frontend..."
            & $NpmCmd run build
        }

        if ($DevFrontend) {
            Write-Host "[EzYOLO] Starting frontend dev server in a new window..."
            Start-Process powershell -ArgumentList @(
                "-NoProfile",
                "-ExecutionPolicy", "Bypass",
                "-Command",
                "Set-Location '$((Join-Path $ProjectRoot "toolbox-frontend").Replace("'", "''"))'; npm.cmd run dev"
            )
        }
    } finally {
        Pop-Location
    }
}

Write-Host "[EzYOLO] Starting backend at http://$BindHost`:$Port ..."
Write-Host "[EzYOLO] Open in browser: http://$BindHost`:$Port"
& $VenvPython "backend\app\main.py" --host $BindHost --port $Port
