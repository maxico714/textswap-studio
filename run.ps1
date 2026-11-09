$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Run .\setup.ps1 first." -ForegroundColor Yellow
    exit 1
}
& .\.venv\Scripts\python.exe app.py
