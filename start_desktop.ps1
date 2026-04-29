param(
    [switch]$InstallDeps,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AppArgs
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

Write-Host "[EzYOLO] Launching desktop app..."
& $VenvPython "main.py" @AppArgs

