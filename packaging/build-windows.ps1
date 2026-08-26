param(
    [string]$Python = "python",
    [string]$FfmpegDirectory = "",
    [switch]$SkipDependencyInstall,
    [string]$TauriCli = ""
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Desktop = Join-Path $Root "desktop"
$Resources = Join-Path $Desktop "src-tauri\resources"
$BackendResources = Join-Path $Resources "backend"
$FfmpegResources = Join-Path $Resources "ffmpeg"
$LicenseResources = Join-Path $Resources "licenses"
$RequiredFiles = @(
    (Join-Path $Root "desktop_backend.py"),
    (Join-Path $Root "Amiri-Regular.ttf"),
    (Join-Path $Desktop "src-tauri\icons\icon.ico")
)

foreach ($requiredFile in $RequiredFiles) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required packaging input is missing: $requiredFile"
    }
}

function Test-NativeFfmpegDirectory {
    param([string]$Directory)

    if (-not $Directory) { return $false }
    $ffmpeg = Join-Path $Directory "ffmpeg.exe"
    $ffprobe = Join-Path $Directory "ffprobe.exe"
    foreach ($executable in @($ffmpeg, $ffprobe)) {
        if (-not (Test-Path -LiteralPath $executable -PathType Leaf)) { return $false }
        $versionInfo = (Get-Item -LiteralPath $executable).VersionInfo
        $identity = "$($versionInfo.ProductName) $($versionInfo.FileDescription)"
        if ($identity -match "ShimGen|Chocolatey Shim") { return $false }
    }

    & $ffprobe -version *> $null
    return $LASTEXITCODE -eq 0
}

function Find-NativeFfmpegDirectory {
    param([string]$RequestedDirectory)

    $candidates = @()
    if ($RequestedDirectory) {
        $candidates += $RequestedDirectory
    } else {
        foreach ($commandName in @("ffmpeg", "ffprobe")) {
            $command = Get-Command $commandName -ErrorAction SilentlyContinue
            if ($command -and $command.Source) {
                $candidates += Split-Path $command.Source
            }
        }

        $chocolateyRoot = if ($env:ChocolateyInstall) { $env:ChocolateyInstall } else { Join-Path $env:ProgramData "chocolatey" }
        $chocolateyTools = Join-Path $chocolateyRoot "lib\ffmpeg\tools"
        if (Test-Path -LiteralPath $chocolateyTools -PathType Container) {
            $candidates += Get-ChildItem -LiteralPath $chocolateyTools -Recurse -File -Filter "ffprobe.exe" |
                ForEach-Object { $_.DirectoryName }
        }

        $candidates += "C:\ffmpeg"
        $candidates += "C:\ffmpeg\bin"
    }

    foreach ($candidate in $candidates | Where-Object { $_ } | Select-Object -Unique) {
        if (Test-NativeFfmpegDirectory $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    throw "Could not locate native FFmpeg and FFprobe binaries. Chocolatey shim launchers are not valid packaging inputs."
}

$FfmpegDirectory = Find-NativeFfmpegDirectory $FfmpegDirectory

if ($SkipDependencyInstall) {
    & $Python -c "import PyInstaller; print(PyInstaller.__version__)"
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller is unavailable in the selected existing Python environment."
    }
} else {
    & $Python -m pip install -e "$Root[align,desktop-build]"
    if ($LASTEXITCODE -ne 0) {
        throw "Python packaging dependencies could not be installed."
    }
}
& $Python -m PyInstaller --noconfirm --clean --distpath (Join-Path $Root "dist") --workpath (Join-Path $Root "build") (Join-Path $Root "packaging\SubGenBackend.spec")
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller backend build failed."
}

if (Test-Path $BackendResources) { Remove-Item -LiteralPath $BackendResources -Recurse -Force }
New-Item -ItemType Directory -Path $BackendResources -Force | Out-Null
Copy-Item -Path (Join-Path $Root "dist\SubGenBackend\*") -Destination $BackendResources -Recurse -Force

$BackendExecutable = Join-Path $BackendResources "SubGenBackend.exe"
$WhisperXSmokeReport = Join-Path $Root "build\whisperx-frozen-smoke.json"
if (Test-Path -LiteralPath $WhisperXSmokeReport) {
    Remove-Item -LiteralPath $WhisperXSmokeReport -Force
}
$SmokeProcess = Start-Process `
    -FilePath $BackendExecutable `
    -ArgumentList @("--self-test-whisperx", "--self-test-report", $WhisperXSmokeReport) `
    -WindowStyle Hidden `
    -Wait `
    -PassThru
$SmokeExitCode = $SmokeProcess.ExitCode
if (-not (Test-Path -LiteralPath $WhisperXSmokeReport -PathType Leaf)) {
    throw "Frozen WhisperX smoke test did not produce its diagnostic report."
}
$SmokeReport = Get-Content -LiteralPath $WhisperXSmokeReport -Raw | ConvertFrom-Json
if ($SmokeExitCode -ne 0 -or -not $SmokeReport.ok) {
    Write-Host (Get-Content -LiteralPath $WhisperXSmokeReport -Raw)
    throw "Frozen WhisperX/Wav2Vec2 import smoke test failed. Installer creation was stopped."
}
Write-Host "Frozen WhisperX smoke test passed with Transformers $($SmokeReport.transformers_version)."

$ffmpegExecutable = Join-Path $FfmpegDirectory "ffmpeg.exe"
$ffprobeExecutable = Join-Path $FfmpegDirectory "ffprobe.exe"
foreach ($requiredFile in @($ffmpegExecutable, $ffprobeExecutable)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required FFmpeg binary is missing: $requiredFile"
    }
}
if (Test-Path $FfmpegResources) { Remove-Item -LiteralPath $FfmpegResources -Recurse -Force }
New-Item -ItemType Directory -Path $FfmpegResources -Force | Out-Null
foreach ($requiredFile in @($ffmpegExecutable, $ffprobeExecutable)) {
    Copy-Item -LiteralPath $requiredFile -Destination $FfmpegResources -Force
}
if (-not (Test-NativeFfmpegDirectory $FfmpegResources)) {
    throw "Packaged FFmpeg validation failed in $FfmpegResources"
}
Write-Host "Bundling native FFmpeg runtime from $FfmpegDirectory"
New-Item -ItemType Directory -Path $LicenseResources -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $Root "packaging\THIRD_PARTY_NOTICES.txt") -Destination $LicenseResources -Force
Copy-Item -LiteralPath (Join-Path $Root "web\vendor\LUCIDE_LICENSE.txt") -Destination $LicenseResources -Force

Push-Location $Desktop
try {
    if ($SkipDependencyInstall) {
        if (-not $TauriCli) {
            $TauriCli = Join-Path $Desktop "node_modules\@tauri-apps\cli\tauri.js"
        }
        if (-not (Test-Path -LiteralPath $TauriCli -PathType Leaf)) {
            throw "The existing Tauri CLI was not found at $TauriCli."
        }
        & node $TauriCli build --bundles nsis
    } else {
        npm ci
        if ($LASTEXITCODE -ne 0) {
            throw "Desktop dependencies could not be installed."
        }
        npm run build
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Tauri desktop build failed."
    }
} finally {
    Pop-Location
}

$TauriConfig = Get-Content -LiteralPath (Join-Path $Desktop "src-tauri\tauri.conf.json") -Raw | ConvertFrom-Json
$Version = [string]$TauriConfig.version
$BundleDirectory = Join-Path $Desktop "src-tauri\target\release\bundle\nsis"
$Installer = Get-ChildItem -LiteralPath $BundleDirectory -File -Filter "SubGen_${Version}_x64-setup.exe" |
    Sort-Object LastWriteTimeUtc -Descending |
    Select-Object -First 1
if (-not $Installer) {
    throw "Tauri completed without the expected SubGen $Version NSIS installer."
}
$InstallerHash = (Get-FileHash -LiteralPath $Installer.FullName -Algorithm SHA256).Hash
$SourceCommit = (& git -C $Root rev-parse HEAD).Trim()
$SourceRemote = (& git -C $Root remote get-url origin).Trim()
$TrackedChanges = @(& git -C $Root status --porcelain --untracked-files=no)
$BuildManifest = [ordered]@{
    schema = "subgen_windows_installer_build_v1"
    product_version = $Version
    built_at_utc = [DateTime]::UtcNow.ToString("o")
    source_commit = $SourceCommit
    source_remote = $SourceRemote
    source_tracked_worktree_clean = ($TrackedChanges.Count -eq 0)
    dependency_install_skipped = [bool]$SkipDependencyInstall
    python = ((& $Python --version 2>&1) -join " ").Trim()
    pyinstaller = ((& $Python -m PyInstaller --version 2>&1) -join " ").Trim()
    node = ((& node --version 2>&1) -join " ").Trim()
    rustc = ((& rustc --version 2>&1) -join " ").Trim()
    tauri_cli = if ($TauriCli) { (Resolve-Path -LiteralPath $TauriCli).Path } else { "npm run build" }
    installer_name = $Installer.Name
    installer_size_bytes = $Installer.Length
    installer_sha256 = $InstallerHash
}
$BuildManifestPath = [IO.Path]::ChangeExtension($Installer.FullName, ".build.json")
$BuildManifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $BuildManifestPath -Encoding UTF8
Write-Host "Installer: $($Installer.FullName)"
Write-Host "Build manifest: $BuildManifestPath"
Write-Host "SHA-256: $InstallerHash"
