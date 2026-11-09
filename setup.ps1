$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "TextSwap Studio setup" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv with Python 3.12..."
    py -3.12 -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host ""
$tess = $null
foreach ($candidate in @(
    "C:\Program Files\Tesseract-OCR\tesseract.exe",
    "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "$env:LOCALAPPDATA\Programs\Tesseract-OCR\tesseract.exe"
)) {
    if (Test-Path $candidate) { $tess = $candidate; break }
}
if (-not $tess) { $tess = (Get-Command tesseract -ErrorAction SilentlyContinue).Source }
if ($tess) {
    Write-Host "Tesseract found: $tess" -ForegroundColor Green
    & $tess --version | Select-Object -First 1
} else {
    Write-Host "Tesseract NOT found." -ForegroundColor Yellow
    Write-Host "Install it from https://github.com/UB-Mannheim/tesseract/wiki, then add it to PATH."
    Write-Host "Hindi OCR needs the 'hin' language pack in the Tesseract installer."
}
Write-Host ""
Write-Host "Next: .\run.ps1  ->  http://127.0.0.1:5000" -ForegroundColor Cyan
