[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceDirectory,

    [Parameter(Mandatory = $true)]
    [string]$ConfigurationDirectory,

    [Parameter(Mandatory = $true)]
    [string]$GitBashExecutable,

    [Parameter(Mandatory = $true)]
    [string]$CompilerExecutable,

    [Parameter(Mandatory = $true)]
    [string]$CompilerVersion,

    [string]$VisualStudioEnvironmentScript,

    [Parameter(Mandatory = $true)]
    [string]$VulkanHeadersDirectory,

    [Parameter(Mandatory = $true)]
    [string]$NasmExecutable,
    [Parameter(Mandatory = $true)]
    [string]$BackendArgumentsFile,
    [string]$BackendConfigurationIdentity
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# MSVC here names the Windows ABI; the compiler can also be clang-cl.
$scriptHash = (Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash
$backendArguments = @(Get-Content -LiteralPath $BackendArgumentsFile | Where-Object { $_ })
$backendHash = (Get-FileHash -LiteralPath $BackendArgumentsFile -Algorithm SHA256).Hash
$configurationIdentity = "ffmpeg-9.0.1-msvc-abi-$CompilerExecutable-$CompilerVersion-$scriptHash-$backendHash-$BackendConfigurationIdentity-v16"
$configurationStamp = Join-Path $ConfigurationDirectory "crossvmaf-configure-stamp.txt"
$configurationHeader = Join-Path $ConfigurationDirectory "config.h"
$configurationAssemblyHeader = Join-Path $ConfigurationDirectory "config.asm"

if ((Test-Path -LiteralPath $configurationStamp -PathType Leaf) -and
    (Test-Path -LiteralPath $configurationHeader -PathType Leaf) -and
    (Test-Path -LiteralPath $configurationAssemblyHeader -PathType Leaf)) {
    $installedIdentity = Get-Content -LiteralPath $configurationStamp -Raw
    if ($installedIdentity -eq $configurationIdentity) {
        exit 0
    }
}

if (-not (Test-Path -LiteralPath $GitBashExecutable -PathType Leaf)) {
    throw "Git Bash is required to configure bundled FFmpeg sources: $GitBashExecutable"
}
if ($VisualStudioEnvironmentScript -and
    -not (Test-Path -LiteralPath $VisualStudioEnvironmentScript -PathType Leaf)) {
    throw "Visual Studio x64 environment script was not found: $VisualStudioEnvironmentScript"
}
if (-not (Test-Path -LiteralPath $CompilerExecutable -PathType Leaf)) {
    throw "The CMake-selected compiler was not found: $CompilerExecutable"
}
if (-not (Test-Path -LiteralPath $VulkanHeadersDirectory -PathType Container)) {
    throw "Pinned Vulkan headers directory was not found: $VulkanHeadersDirectory"
}
if (-not (Test-Path -LiteralPath $NasmExecutable -PathType Leaf)) {
    throw "NASM is required for bundled FFmpeg x86-64 assembly: $NasmExecutable"
}

if (Test-Path -LiteralPath $ConfigurationDirectory) {
    $resolvedConfiguration = [IO.Path]::GetFullPath($ConfigurationDirectory)
    if ((Split-Path -Leaf $resolvedConfiguration) -ne 'crossvmaf-ffmpeg-config' -or
        (Split-Path -Leaf (Split-Path -Parent $resolvedConfiguration)) -ne '_deps') {
        throw "Unexpected FFmpeg generated configuration path: $resolvedConfiguration"
    }
    Remove-Item -LiteralPath $resolvedConfiguration -Recurse -Force
}
New-Item -ItemType Directory -Path $ConfigurationDirectory | Out-Null

if ($VisualStudioEnvironmentScript) {
    $environmentCommand = 'call "{0}" >nul && set' -f $VisualStudioEnvironmentScript
    $environmentLines = & $env:ComSpec /d /s /c $environmentCommand
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to initialize the Visual Studio x64 compiler environment."
    }

    foreach ($environmentLine in $environmentLines) {
        if ($environmentLine -match '^([^=][^=]*)=(.*)$') {
            [Environment]::SetEnvironmentVariable(
                $Matches[1],
                $Matches[2],
                [EnvironmentVariableTarget]::Process)
        }
    }
}
$nasmDirectory = Split-Path -Parent $NasmExecutable
$compilerDirectory = Split-Path -Parent $CompilerExecutable
$compilerName = Split-Path -Leaf $CompilerExecutable
$env:Path = "$compilerDirectory;$nasmDirectory;$env:Path"

function ConvertTo-MsysPath([string]$WindowsPath) {
    $fullPath = [System.IO.Path]::GetFullPath($WindowsPath)
    if ($fullPath -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Cannot convert path to Git Bash syntax: $fullPath"
    }
    $driveLetter = $Matches[1].ToLowerInvariant()
    $pathRemainder = $Matches[2] -replace '\\', '/'
    return "/$driveLetter/$pathRemainder"
}

$configureScript = (ConvertTo-MsysPath $SourceDirectory) + "/configure"
$vulkanIncludeDirectory = ConvertTo-MsysPath $VulkanHeadersDirectory
$configureArguments = @(
    "--toolchain=msvc",
    "--cc=$compilerName",
    "--cxx=$compilerName",
    "--disable-programs",
    "--enable-shared",
    "--disable-static",
    "--disable-symver",
    "--disable-doc",
    "--disable-debug",
    "--disable-gpl",
    "--disable-version3",
    "--disable-nonfree",
    "--disable-autodetect",
    "--disable-libaom",
    "--disable-network",
    "--disable-avdevice",
    "--disable-avfilter",
    "--disable-swscale",
    "--disable-swresample",
    "--disable-encoders",
    "--disable-muxers",
    "--disable-filters",
    "--disable-devices",
    "--disable-everything",
    "--enable-vulkan",
    "--enable-avcodec",
    "--enable-avformat",
    "--enable-avutil",
    "--enable-decoder=h264,hevc,av1,vp8,vp9,mpeg1video,mpeg2video,mpeg4,vc1,wmv3,flv,mjpeg",
    "--enable-demuxer=h264,hevc,av1,obu,ivf,m4v,mpegvideo,mpegps,mpegts,vc1,mjpeg,mpjpeg,mov,matroska,avi,asf,flv,dv",
    "--enable-parser=h264,hevc,av1,vp8,vp9,mpegvideo,mpeg4video,vc1,mjpeg",
    "--enable-protocol=file",
    "--disable-inline-asm",
    "--x86asmexe=nasm",
    "--extra-cflags=-I$vulkanIncludeDirectory"
)

$configureArguments += $backendArguments

if ($compilerName -eq 'clang-cl.exe') {
    # Use the LLVM tools beside clang-cl. FFmpeg's MSVC defaults otherwise
    # invoke link/lib from PATH and require a developer-shell environment.
    $configureArguments += @('--ld=lld-link.exe', '--host-ld=lld-link.exe',
        '--ar=llvm-lib.exe', '--nm=llvm-nm.exe', '--windres=llvm-rc.exe')
}

Push-Location $ConfigurationDirectory
try {
    & $GitBashExecutable $configureScript @configureArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Bundled FFmpeg configuration failed with exit code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}

Set-Content -LiteralPath $configurationStamp `
    -Value $configurationIdentity `
    -NoNewline
