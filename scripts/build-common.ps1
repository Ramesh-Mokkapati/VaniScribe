# Helpers shared by build.ps1 and prepare_model.ps1 (dot-source this file).

# whisper.cpp v1.9.5. Its Windows binaries are published under the matching "b" build tag.
$WhisperCppBuild = "b5454"

function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    # Tools like PyInstaller log to stderr; Windows PowerShell would turn that into a terminating
    # error under "Stop", so rely on the exit code instead.
    $ErrorActionPreference = "Continue"
    & $Exe @Arguments 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Arguments -join ' ') failed with exit code $LASTEXITCODE" }
}

function Get-File {
    param([string]$Url, [string]$OutFile)
    # curl.exe ships with Windows 10+ and resumes interrupted transfers; Invoke-WebRequest can
    # silently leave a truncated file on a flaky connection. A connection that stays open but
    # stops sending (under 10 KB/s for 2 minutes) is treated as failed, then resumed.
    if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
        Invoke-Checked "curl.exe" @("-sSL", "--fail", "--retry", "10", "--retry-delay", "5",
                                    "--retry-all-errors", "--connect-timeout", "30",
                                    "--speed-limit", "10240", "--speed-time", "120",
                                    "-C", "-", "-o", $OutFile, $Url)
    } else {
        Invoke-WebRequest $Url -OutFile $OutFile
    }
}

function Get-WhisperCpp {
    <# Puts whisper.cpp's Windows binaries for $Arch into $Destination and returns that folder.
       By default only what the app needs (whisper-cli plus its DLLs); -AllTools keeps everything. #>
    param([ValidateSet("x64", "x86")][string]$Arch, [string]$Destination, [switch]$AllTools)
    if (Test-Path (Join-Path $Destination "whisper-cli.exe")) {
        if (-not $AllTools -or (Test-Path (Join-Path $Destination "whisper-quantize.exe"))) { return $Destination }
    }
    $asset = if ($Arch -eq "x64") { "whisper-bin-x64.zip" } else { "whisper-bin-Win32.zip" }
    $tmp = Join-Path ([IO.Path]::GetTempPath()) ("whispercpp-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Force $tmp, $Destination | Out-Null
    try {
        Write-Host "Downloading whisper.cpp $WhisperCppBuild ($Arch)..."
        Get-File "https://github.com/ggml-org/whisper.cpp/releases/download/$WhisperCppBuild/$asset" (Join-Path $tmp $asset)
        Expand-Archive (Join-Path $tmp $asset) -DestinationPath (Join-Path $tmp "x")
        $bin = (Get-ChildItem (Join-Path $tmp "x") -Recurse -Filter whisper-cli.exe | Select-Object -First 1).DirectoryName
        $files = if ($AllTools) { Get-ChildItem $bin -File } else {
            Get-ChildItem $bin -File | Where-Object { $_.Name -eq "whisper-cli.exe" -or $_.Name -eq "whisper.dll" -or $_.Name -like "ggml*.dll" }
        }
        $files | Copy-Item -Destination $Destination
    } finally {
        Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
    }
    return $Destination
}
