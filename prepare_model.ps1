<#
.SYNOPSIS
    One-time step: builds the models the Windows builds ship with, in models\.

.DESCRIPTION
    - ggml-telugu-small-q8_0.bin (speech to text): downloads vasista22/whisper-telugu-small (~1 GB),
      converts it to whisper.cpp's format and quantises it to 8 bits (~250 MB).
    - nllb-200-600m-int8\ (English translation): downloads facebook/nllb-200-distilled-600M
      (~2.5 GB) and converts it to CTranslate2 with 8-bit weights (~620 MB).
    Needs 64-bit Python; it creates its own .venv-model with PyTorch (CPU), transformers and
    CTranslate2. Both build.ps1 -Arch x64 and -Arch x86 then use the result. Models that already
    exist are skipped. Intermediate files go to build\model and can be deleted afterwards.

.PARAMETER Python
    64-bit Python command for the model venv. Default: "py -3.13" if available, else "python".
#>
param([string]$Python = "")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. (Join-Path $PSScriptRoot "scripts\build-common.ps1")

if ((Test-Path "models\ggml-telugu-small-q8_0.bin") -and (Test-Path "models\nllb-200-600m-int8\model.bin")) {
    Write-Host "Both models already exist in models\; delete one to rebuild it." -ForegroundColor Green
    return
}

$VenvPython = Join-Path $PSScriptRoot ".venv-model\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    if (-not $Python) {
        $Python = if (Get-Command py -ErrorAction SilentlyContinue) { "py -3.13" } else { "python" }
    }
    $parts = @($Python -split " ")
    Invoke-Checked $parts[0] (@($parts | Select-Object -Skip 1) + @("-m", "venv", ".venv-model"))
}
$bits = & $VenvPython -c "import struct; print(struct.calcsize('P') * 8)"
if ($bits -ne "64") { throw ".venv-model must use 64-bit Python (PyTorch has no 32-bit build)." }
Invoke-Checked $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $VenvPython @("-m", "pip", "install", "-r", "requirements-model.txt")

# whisper-quantize comes with the 64-bit whisper.cpp tools.
$tools = Get-WhisperCpp -Arch "x64" -Destination (Join-Path $PSScriptRoot "build\model\whisper-x64") -AllTools
Invoke-Checked $VenvPython @("scripts\prepare_model.py", (Join-Path $tools "whisper-quantize.exe"))
