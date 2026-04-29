param(
    [Alias("Host")]
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$InstallDeps,
    [switch]$NoBrowser
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
$FrontendDir = Join-Path $ProjectRoot "toolbox-frontend"
$Url = "http://$BindHost`:$Port"

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

if (-not $NpmCmd) {
    throw "npm.cmd not found. Please install Node.js first."
}

if (-not (Test-Path $FrontendDir)) {
    throw "Frontend directory not found: $FrontendDir"
}

if ($InstallDeps) {
    Write-Host "[EzYOLO] Installing Python dependencies..."
    & $VenvPython -m pip install -r "requirements.txt"

    Push-Location $FrontendDir
    try {
        Write-Host "[EzYOLO] Installing frontend dependencies..."
        & $NpmCmd ci
    } finally {
        Pop-Location
    }
}

Write-Host "[EzYOLO] Starting frontend dev server in a new window..."
Start-Process powershell -ArgumentList @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-Command",
    "Set-Location '$($FrontendDir.Replace("'", "''"))'; npm.cmd run dev"
)

if (-not $NoBrowser) {
    Write-Host "[EzYOLO] Opening browser: $Url"
    Start-Process $Url
}

Write-Host "[EzYOLO] Starting backend at $Url ..."
& $VenvPython "backend\app\main.py" --host $BindHost --port $Port
