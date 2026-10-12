$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$packageRoot = Join-Path $env:GITHUB_WORKSPACE 'build/windows-release/packages'
$markers = @(Get-ChildItem -LiteralPath $packageRoot -Filter last-package.txt -Recurse -File)
if ($markers.Count -ne 1) {
    throw 'Expected exactly one verified runtime package from this build.'
}
$package = (Get-Content -LiteralPath $markers[0].FullName -Raw).Trim()
$package = (Resolve-Path -LiteralPath $package).Path
$resolvedRoot = (Resolve-Path -LiteralPath $packageRoot).Path + [IO.Path]::DirectorySeparatorChar
if (-not $package.StartsWith($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The package marker points outside the current build.'
}

$manifest = Get-Content -LiteralPath (Join-Path $package 'build-manifest.json') -Raw | ConvertFrom-Json
$revision = git rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw 'Cannot identify the release commit.' }
if ($manifest.configuration -ne 'Release' -or $manifest.dirty -or $manifest.revision -ne $revision) {
    throw 'The package must come from a clean Release build of the checked-out tag.'
}
foreach ($required in @('crossvmaf.exe', 'crossvmaf-studio.exe')) {
    if ($required -notin $manifest.files.path) { throw "The runtime package is missing $required." }
}
if (-not ($manifest.files.path -like '*.dll') -or -not ($manifest.files.path -like 'licenses/*')) {
    throw 'The runtime package must include its DLLs and license notices.'
}

# These commands exercise the packaged CLI and DLL loader without requiring a GPU.
$cli = Join-Path $package 'crossvmaf.exe'
& $cli --help
if ($LASTEXITCODE -ne 0) { throw 'The packaged CLI cannot start.' }
$models = @(& $cli --list-models)
if ($LASTEXITCODE -ne 0 -or $models.Count -ne 8) { throw 'The packaged CLI cannot list all eight models.' }

$assets = Join-Path $env:GITHUB_WORKSPACE 'build/release-assets'
New-Item -ItemType Directory -Path $assets | Out-Null
$archiveName = "CrossVMAF-$env:RELEASE_TAG-windows-x64.zip"
$archivePath = Join-Path $assets $archiveName
[IO.Compression.ZipFile]::CreateFromDirectory($package, $archivePath)

# Verify the archive itself, not just the loose files that were passed to the packager.
$archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    foreach ($record in $manifest.files) {
        $entry = $archive.GetEntry($record.path)
        if ($null -eq $entry -or $entry.Length -ne $record.bytes) {
            throw "Missing or truncated archive entry: $($record.path)"
        }
        $stream = $entry.Open()
        try {
            $hash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($stream)).ToLowerInvariant()
            if ($hash -cne $record.sha256) { throw "Archive checksum mismatch: $($record.path)" }
        } finally {
            $stream.Dispose()
        }
    }
} finally {
    $archive.Dispose()
}

$checksum = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
"$checksum  $archiveName" | Set-Content -LiteralPath (Join-Path $assets 'SHA256SUMS.txt') -Encoding utf8NoBOM
@"
Windows x64 precompiled Release build from commit $revision.

Download **$archiveName** and extract the entire ZIP into one directory. Run
**crossvmaf-studio.exe** for the desktop app, or **crossvmaf.exe** for the CLI.
Keep the bundled DLLs, licenses directory, and build-manifest.json with the applications.
SHA256SUMS.txt contains the archive checksum; build-manifest.json records individual file hashes.

Requires the Microsoft Visual C++ x64 runtime and a compatible Vulkan GPU driver.
Linux support is incomplete; this release includes Windows binaries only. This is
an early version and Windows compatibility issues may occur. Please report problems
through this repository's Issues page.

The corresponding source is available in the source archives for this tag,
including the bundled FFmpeg source and dependency build configuration.
"@ | Set-Content -LiteralPath (Join-Path $assets 'release-notes.md') -Encoding utf8NoBOM
