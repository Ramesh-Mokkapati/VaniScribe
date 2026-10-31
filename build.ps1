<#
.SYNOPSIS
    Builds dist\VaniScribe-<arch>.exe and the release package dist\VaniScribe-<arch>.zip for x64 or x86.

.DESCRIPTION
    1. Creates/updates a virtual environment (.venv for x64, .venv-x86 for x86) and installs requirements.
    2. Downloads into vendor\<arch>\bin, unless already there:
       - FFmpeg (MP3 conversion and audio decoding)
       - a JavaScript runtime yt-dlp needs for YouTube: Deno for x64, Node.js 22 LTS for x86
         (Deno has no 32-bit build)
       - whisper.cpp's whisper-cli (offline speech-to-text)
    3. Runs PyInstaller with ytmp3.spec.
    4. Packages the exe with models\ (from prepare_model.ps1: the Telugu speech model and the
       English translation model) into a zip.

    The same script is used by the GitHub Actions pipeline.

.PARAMETER NoModel
    Package without the speech model (download-only; transcription reports the model missing).

.PARAMETER Arch
    x64 (default) or x86 (for 32-bit Windows).

.PARAMETER Python
    Python command used to create the venv; it must match -Arch. Defaults: "py -3.13" for x64;
    for x86, "py -3.13-32" if installed, otherwise a private 32-bit Python is downloaded into tools\.

.PARAMETER Clean
    Delete this architecture's build output and re-download its vendored tools.
#>
param(
    [ValidateSet("x64", "x86")]
    [string]$Arch = "x64",
    [string]$Python = "",
    [switch]$Clean,
    [switch]$NoModel
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # Invoke-WebRequest is very slow with the progress bar on
Set-Location $PSScriptRoot

$FfmpegUrl = @{
    x64 = "https://github.com/yt-dlp/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
    x86 = "https://github.com/yt-dlp/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win32-gpl.zip"
}[$Arch]
$DenoUrl = "https://github.com/denoland/deno/releases/latest/download/deno-x86_64-pc-windows-msvc.zip"
$NodeMajor = "v22"                          # last Node.js LTS line with 32-bit Windows builds
$JsRuntime = if ($Arch -eq "x64") { "deno.exe" } else { "node.exe" }
$PythonX86Version = "3.13.17"               # nuget "pythonx86" package, used when no 32-bit Python is installed
$VendorBin = Join-Path $PSScriptRoot "vendor\$Arch\bin"
$Venv = if ($Arch -eq "x64") { ".venv" } else { ".venv-$Arch" }

. (Join-Path $PSScriptRoot "scripts\build-common.ps1")

function Get-DefaultPython {
    if ($Arch -eq "x64") {
        if (Get-Command py -ErrorAction SilentlyContinue) { return "py -3.13" } else { return "python" }
    }
    if ((Get-Command py -ErrorAction SilentlyContinue) -and ((py -0 2>$null) -match "3\.13-32")) { return "py -3.13-32" }
    $exe = Join-Path $PSScriptRoot "tools\python-x86\tools\python.exe"
    if (-not (Test-Path $exe)) {
        Write-Host "Downloading 32-bit Python $PythonX86Version..."
        $pkg = Join-Path ([IO.Path]::GetTempPath()) "pythonx86-$PythonX86Version.zip"
        Get-File "https://www.nuget.org/api/v2/package/pythonx86/$PythonX86Version" $pkg
        Expand-Archive $pkg -DestinationPath (Join-Path $PSScriptRoot "tools\python-x86") -Force
        Remove-Item $pkg
    }
    return $exe
}

if ($Clean) {
    Remove-Item -Recurse -Force "build\$Arch", "dist\VaniScribe-$Arch.exe", "vendor\$Arch" -ErrorAction SilentlyContinue
}

# --- 1. Python environment -------------------------------------------------
$VenvPython = Join-Path $PSScriptRoot "$Venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    if (-not $Python) { $Python = Get-DefaultPython }
    Write-Host "Creating $Venv with: $Python"
    $parts = @($Python -split " ")
    Invoke-Checked $parts[0] (@($parts | Select-Object -Skip 1) + @("-m", "venv", $Venv))
}
$bits = & $VenvPython -c "import struct; print(struct.calcsize('P') * 8)"
if (($Arch -eq "x64" -and $bits -ne "64") -or ($Arch -eq "x86" -and $bits -ne "32")) {
    throw "$Venv uses $bits-bit Python, which does not match -Arch $Arch. Delete $Venv or pass a matching -Python."
}
Invoke-Checked $VenvPython @("-m", "pip", "install", "--upgrade", "pip")
Invoke-Checked $VenvPython @("-m", "pip", "install", "-r", "requirements.txt", "-r", "requirements-build.txt")
if ($Arch -eq "x64") {
    Invoke-Checked $VenvPython @("-m", "pip", "install", "-r", "requirements-ctranslate2.txt")   # CTranslate2 (no 32-bit build)
}

# --- 2. Vendored tools -----------------------------------------------------
New-Item -ItemType Directory -Force $VendorBin | Out-Null
$Temp = Join-Path ([IO.Path]::GetTempPath()) ("ytmp3-build-" + [guid]::NewGuid())
New-Item -ItemType Directory -Force $Temp | Out-Null
try {
    if (-not (Test-Path (Join-Path $VendorBin "ffmpeg.exe"))) {
        Write-Host "Downloading FFmpeg ($Arch)..."
        $zip = Join-Path $Temp "ffmpeg.zip"
        Get-File $FfmpegUrl $zip
        Expand-Archive $zip -DestinationPath (Join-Path $Temp "ffmpeg")
        Get-ChildItem (Join-Path $Temp "ffmpeg") -Recurse -Include ffmpeg.exe, ffprobe.exe |
            Copy-Item -Destination $VendorBin
        Get-ChildItem (Join-Path $Temp "ffmpeg") -Recurse -Filter LICENSE.txt | Select-Object -First 1 |
            Copy-Item -Destination (Join-Path $VendorBin "FFMPEG-LICENSE.txt")
    }
    if (-not (Test-Path (Join-Path $VendorBin $JsRuntime))) {
        $zip = Join-Path $Temp "js.zip"
        if ($Arch -eq "x64") {
            Write-Host "Downloading Deno..."
            Get-File $DenoUrl $zip
            Expand-Archive $zip -DestinationPath $VendorBin -Force
        } else {
            # Assign first: Windows PowerShell 5.1 pipes a JSON array from Invoke-RestMethod as one object.
            $releases = Invoke-RestMethod "https://nodejs.org/dist/index.json"
            $version = ($releases | Where-Object { $_.version -like "$NodeMajor.*" -and $_.files -contains "win-x86-zip" } |
                Select-Object -First 1).version
            Write-Host "Downloading Node.js $version (x86)..."
            Get-File "https://nodejs.org/dist/$version/node-$version-win-x86.zip" $zip
            Expand-Archive $zip -DestinationPath (Join-Path $Temp "node")
            Get-ChildItem (Join-Path $Temp "node") -Recurse -Filter node.exe | Copy-Item -Destination $VendorBin
        }
    }
} finally {
    Remove-Item -Recurse -Force $Temp -ErrorAction SilentlyContinue
}
Get-WhisperCpp -Arch $Arch -Destination $VendorBin | Out-Null   # speech-to-text engine
foreach ($tool in "ffmpeg.exe", "ffprobe.exe", $JsRuntime, "whisper-cli.exe") {
    if (-not (Test-Path (Join-Path $VendorBin $tool))) { throw "Missing vendor\$Arch\bin\$tool" }
}

# --- 3. Build the exe ------------------------------------------------------
$env:VANISCRIBE_ARCH = $Arch
Invoke-Checked $VenvPython @("-m", "PyInstaller", "--noconfirm", "--clean", "--workpath", "build\$Arch", "ytmp3.spec")

$exe = Join-Path $PSScriptRoot "dist\VaniScribe-$Arch.exe"
if (-not (Test-Path $exe)) { throw "Build finished but $exe was not produced" }
$sizeMb = [math]::Round((Get-Item $exe).Length / 1MB, 1)
Write-Host "Built $exe ($sizeMb MB)" -ForegroundColor Green

# --- 4. Package: the exe plus the speech model next to it, as one zip -------
# The model stays outside the exe so it isn't unpacked on every launch.
$Model = "models\ggml-telugu-small-q8_0.bin"
if ($NoModel) {
    Write-Host "Skipping the speech model (-NoModel): the Transcribe tab will report it missing." -ForegroundColor Yellow
} elseif (-not (Test-Path $Model)) {
    throw "$Model not found. Run .\prepare_model.ps1 once first, or pass -NoModel for a download-only package."
} elseif (-not (Test-Path "models\nllb-200-600m-int8\model.bin")) {
    Write-Host "models\nllb-200-600m-int8 not found: packaging without English translation." -ForegroundColor Yellow
}
$Package = Join-Path $PSScriptRoot "dist\VaniScribe-$Arch"
Remove-Item -Recurse -Force $Package, "$Package.zip" -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force (Join-Path $Package "models") | Out-Null
Copy-Item $exe $Package
if (-not $NoModel) {
    Copy-Item "models\*" (Join-Path $Package "models") -Recurse
}
Compress-Archive -Path $Package -DestinationPath "$Package.zip"
$zipMb = [math]::Round((Get-Item "$Package.zip").Length / 1MB, 1)
Write-Host "Packaged $Package.zip ($zipMb MB): VaniScribe-$Arch.exe + models\" -ForegroundColor Green
