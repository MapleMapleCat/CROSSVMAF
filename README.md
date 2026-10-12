# CrossVMAF

**English** | [简体中文](README.zh-CN.md)

**Vulkan-accelerated video quality evaluation for Windows and Linux.**

> [!IMPORTANT]
> This version is still in early development. Linux support is incomplete, and
> the Windows version may have compatibility issues. Please report any problems
> through this repository's GitHub Issues.

CrossVMAF compares a reference video with a distorted video using an independent
C++20 implementation of the VMAF 1.0.16 feature and model pipeline. Use the
**Studio desktop app**, the **command-line tool**, or the **C++ streaming API**.
Feature evaluation runs on a Vulkan GPU; final model prediction runs on the CPU.

CrossVMAF does not modify or link against Netflix libvmaf and is not an official
Netflix product.

[Quick start](#quick-start) · [Build](#build) · [Studio](#studio) ·
[Usage](#usage) · [C++ API](#c-api) · [License](#license)

## Highlights

- **Vulkan compute:** evaluate video quality on compatible GPUs across vendors.
- **Desktop workflow:** queue evaluations, compare results, inspect quality
  curves, and export JSON, CSV, or customer PDF reports.
- **Automatic model selection:** choose from eight bundled models based on the
  reference video's decoded dimensions and frame rate, with CLI overrides.
- **Hardware decoding:** select Intel QSV, NVIDIA NVDEC, Vulkan Video, or Linux
  VA-API where supported, with software fallback in automatic mode.
- **CPU verification:** compare GPU results with an independent CPU reference
  pass using `--verify-cpu`.

## Quick start

For Windows x64, download `CrossVMAF-<tag>-windows-x64.zip` from this repository's
**Releases** page and extract the entire archive. Launch `crossvmaf-studio.exe`
for the desktop app or run `crossvmaf.exe --help` for the CLI. Keep the bundled
DLLs and `licenses/` directory together. The Microsoft Visual C++ x64 runtime
and a compatible Vulkan GPU driver must be installed.

[Build from source](#build), then run the CLI from the repository root. This
example uses a Windows Release build in PowerShell 7:

```powershell
$CrossVmaf = ".\build\windows-release\apps\crossvmaf\Release\crossvmaf.exe"

& $CrossVmaf --list-devices
& $CrossVmaf --self-test

& $CrossVmaf `
  --reference ".\reference.mp4" `
  --distorted ".\distorted.mp4" `
  --output ".\result.json"
```

The defaults select the compute device, decoder, and model automatically, pool
scores using the mean, and write JSON. Omit `--output` to write to standard
output, or add `--format csv` for CSV.

Prefer a graphical interface? Launch Studio after the same build:

```powershell
.\build\windows-release\apps\studio\Release\crossvmaf-studio.exe
```

On Linux, the CLI is `build/linux-clang/apps/crossvmaf/Release/crossvmaf` and
accepts the same options.

> [!IMPORTANT]
> Inputs must have matching decoded dimensions, pixel format, and bit depth,
> with matching frame counts and compatible timestamps. CrossVMAF does not
> resize or convert color space. See [input requirements](#input-requirements).

## Build

Native builds support **Windows and Linux on x86-64**. The codec integration
does not currently support macOS, ARM, or cross-compilation.

### Requirements

| Dependency | Requirement |
| --- | --- |
| Build tools | CMake 3.28+ and Ninja |
| Compiler | LLVM/Clang with C++20 and C17 support; `clang-cl` on Windows, `clang`/`clang++` on Linux; include `llvm-readobj` for ABI checks |
| Python | Python 3.10+, with Meson and Ninja available to the dav1d build |
| Studio | Rust 1.92+ and Cargo; Slint runtime and UI compiler are built from pinned sources |
| Source tools | Git, NASM, and network access for the first configure/build |
| Runtime | Vulkan Loader and a compatible Vulkan 1.1+ compute GPU |
| Windows | Visual Studio or Build Tools with **Desktop development with C++**, Windows SDK, and Git for Windows / Git Bash |
| Linux | POSIX shell; `libva`, `libva-drm`, `libdrm` development packages and `pkg-config` for the default QSV build |

Install Meson for the Python interpreter used by CMake:

```sh
python -m pip install meson ninja
```

Vulkan Video decoding additionally needs Vulkan 1.3 and the relevant decode
extensions. QSV, NVDEC, and VA-API have their own device and driver requirements;
see [decode backends](#decode-backends).

### Windows — PowerShell 7

Install LLVM in its standard `%ProgramFiles%/LLVM` location and use the Rust MSVC
toolchain for Studio. Run from an ordinary PowerShell 7 terminal with CMake,
Ninja, Python, NASM, Git, and Rust available on `PATH`:

```powershell
cmake --preset windows-release
cmake --build --preset windows-release --parallel
```

The presets select LLVM tools and discover the Microsoft SDK and C++ runtime.
No Developer PowerShell initialization or WSL is needed. Use `windows-debug`
in both commands for a Debug build.

| Application | Release executable |
| --- | --- |
| CLI | `build/windows-release/apps/crossvmaf/Release/crossvmaf.exe` |
| Studio | `build/windows-release/apps/studio/Release/crossvmaf-studio.exe` |

A full build also refreshes `out/crossvmaf-studio.exe` and its runtime libraries
for convenient local launching.

### Linux — native shell

```sh
cmake -S . -B build/linux-clang -G "Ninja Multi-Config" \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CONFIGURATION_TYPES='Debug;Release' \
  -DFETCHCONTENT_BASE_DIR="$PWD/.cache/fetchcontent" \
  -DCROSSVMAF_BUILD_TOOLS=ON \
  -DCROSSVMAF_ENABLE_QSV=ON -DCROSSVMAF_ENABLE_NVDEC=ON
cmake --build build/linux-clang --config Release --parallel
```

For Debug, use `--config Debug` when building.
VA-API is off by default; add `-DCROSSVMAF_ENABLE_VAAPI=ON` to enable it.

<details>
<summary>Build options, custom toolchains, and existing build directories</summary>

- **CLI-only build:** add `-DCROSSVMAF_BUILD_STUDIO=OFF` to the configure command
  to omit Studio and its Rust dependency.
- **Custom LLVM location:** set `CROSSVMAF_LLVM_ROOT` in the `environment` of an
  inherited configure preset in the ignored `CMakeUserPresets.json`. Point your
  build presets at that configure preset too.
- **Windows compiler migration:** run `cmake --fresh --preset windows-release`
  (or `windows-debug`) when switching an existing build to these LLVM presets.
  Reapply local `-D` options. Do not reuse an MSVC CMake cache.
- **Linux generator changes:** use a new build directory when switching from
  single-config Ninja to Ninja Multi-Config. GCC can also be selected in a
  separate native build with `-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++`.
- **Dependency cache:** downloaded sources live in `.cache/fetchcontent` and are
  reused across build trees. FFmpeg's complete, unmodified source is vendored in
  `third_party/ffmpeg`; no external FFmpeg executable is required.
- **Generated assets:** the build compiles, validates, audits, and embeds runtime
  shaders, and verifies and embeds the bundled models.

Dependency versions are pinned in [`cmake/Dependencies.cmake`](cmake/Dependencies.cmake),
[`cmake/DecodeBackends.cmake`](cmake/DecodeBackends.cmake),
[`cmake/FFmpeg.cmake`](cmake/FFmpeg.cmake), and
[`cmake/Dav1d.cmake`](cmake/Dav1d.cmake).

</details>

### Runtime package

Create a fresh, dependency-checked Windows Release package with:

```powershell
cmake --build --preset windows-release --target crossvmaf_package --parallel
```

Each invocation creates a new directory under
`build/windows-release/packages/Windows-AMD64/Release`, including runtime
libraries, license notices, and a `build-manifest.json` with binary hashes,
build configuration, and runtime prerequisites. Packaging uses canonical build
outputs, independently of `out/`. Keep runtime libraries and the `licenses/`
directory with the applications when distributing binaries.

## Studio

Studio provides a desktop workflow for selecting inputs, queuing evaluations,
comparing results, and reviewing saved reports. The interface supports English
and Simplified Chinese, with light and dark themes.

- **Evaluation settings:** choose compute and decode devices, backend, transfer
  policy, decoder threads, and buffered frames. Defaults use automatic decoding,
  automatic threads, and four buffered frames per input.
- **Queue behavior:** evaluation settings are captured when a task is enqueued.
  The queue pauses on failure by default.
- **Reports:** export JSON or CSV for analysis, or choose **Customer report (PDF)**
  for an A4 report with the pooled score, quality curve, score distribution,
  review segments, and evaluation details.

PDF reports use the active interface language and embed their fonts. Saved
evaluations can be exported without the source videos, a browser, a printer
driver, or a Python runtime. Studio selects the viewing model automatically;
use the CLI when you need an explicit model override.

## Usage

### Automatic model selection

The CLI and Studio select a model from the reference video's actual decoded
evaluation dimensions and frame rate:

| Reference video | Automatic choice |
| --- | --- |
| Long edge ≥ 3840 px **or** short edge ≥ 2160 px | 4K / 1.5H model |
| Smaller dimensions, including 1440p | 1080p / 3H model |
| Known constant frame rate ≥ 50 fps | Matching HFR variant |
| Lower, unknown, or variable frame rate | Standard variant |

These thresholds are CrossVMAF's policy. Input videos are not resized, and model
selection does not simulate a different display. Resolutions and rates outside
the calibrated presets, such as 8K or 120 fps, are approximations.

List all eight bundled variants or override the automatic choice:

```powershell
& $CrossVmaf --list-models
& $CrossVmaf --reference reference.mp4 --distorted distorted.mp4 `
  --model vmaf_v1.0.16_3d0h_2160 --output result.json
```

The selected model is used for CPU verification and recorded in reports.
Models are embedded at build time; custom models and runtime parameter overrides
are not supported. [Model provenance and hashes](models/README.md).

### Decode backends

VMAF feature evaluation always uses the selected Vulkan compute device, including
when video decoding runs in software. With default settings, automatic decoding
tries eligible backends in this order:

| Compute GPU | Windows | Linux |
| --- | --- | --- |
| Intel | QSV → Vulkan Video → software | QSV → VA-API → software |
| NVIDIA | NVDEC → Vulkan Video → software | NVDEC → VA-API → software |
| AMD or other vendor | Vulkan Video → software | VA-API → software |

Candidates must pass build, platform, device, codec, and driver checks. Automatic
mode tries GPU transfer and can reopen both inputs in software during startup.
VA-API must be enabled at build time to participate in Linux selection.

- `--decode hardware` requires hardware decoding for both inputs, with no software
  fallback. Automatic transfer in this mode also permits eligible host paths.
- `--decode-backend NAME` pins a backend: use it or fail.
- `--decode-transfer gpu` forbids decoded-pixel downloads; `host` requests host
  transfer. A separate decoder adapter requires host transfer.

Hardware support varies by GPU and driver. The tested Intel Arc 130T has a known
Vulkan Main10 surface-reuse failure, and the Vulkan automatic driver exclusion
remains in place. Use `--list-decode-backends` and `--list-decode-devices` to
inspect available backends and devices. Successful compilation alone does not
establish hardware conformance.

### CLI options

Run `& $CrossVmaf --help` for the command-line reference.

| Option | Purpose / default |
| --- | --- |
| `--reference PATH` | Reference video; required |
| `--distorted PATH` | Distorted video; required |
| `--model NAME` | Force a bundled model; default: automatic |
| `--device auto\|UUID` | Vulkan compute device; default: `auto` |
| `--pool mean\|min\|harmonic` | Sequence pooling; default: `mean` |
| `--format json\|csv` | Output format; default: `json` |
| `--output PATH` | Write to a file; default: standard output |
| `--verify-cpu` | Run the independent CPU verification pass |

<details>
<summary>Decoder controls, diagnostics, and device discovery</summary>

| Option | Purpose / default |
| --- | --- |
| `--decode auto\|hardware\|software` | Decode mode; default: `auto` |
| `--decode-backend auto\|vulkan\|qsv\|nvdec\|vaapi` | Backend; default: `auto`; named backends are strict |
| `--decode-device same\|ID` | Decoder device; default: same physical GPU as compute |
| `--decode-transfer auto\|gpu\|host` | Frame transfer policy; default: `auto` |
| `--decoded-frame-queue N` | Buffered frames per source: `1..64`; default: `4` |
| `--decoder-threads N` | Threads per source: `0..64`; default: `0` (automatic) |
| `--diagnostic-speed-chroma` | Add per-frame U/V diagnostics to JSON output |
| `--list-devices` | List compatible Vulkan compute devices |
| `--list-models` | List the eight bundled models |
| `--list-decode-backends` | List compiled decode backends |
| `--list-decode-devices` | List decoder device IDs; optionally filter with `--decode-backend NAME` |
| `--self-test [device-uuid]` | Run the device self-test |

Automatic software AV1 decoding shares a host-wide dav1d worker budget between
the two inputs, capped at 16 workers per source. Native FFmpeg codecs retain their
own automatic thread policy. Hardware backends may interpret thread counts
differently.

</details>

### CPU verification

```powershell
& $CrossVmaf --reference reference.mkv --distorted distorted.mkv `
  --verify-cpu --output verified.json
```

Verification switches Speed Chroma to exact math, decodes both inputs again,
and compares the Vulkan results with the independent CPU reference pass. Use it
for correctness checks; it adds work and is not a throughput benchmark.

### Input requirements

| Category | Supported inputs |
| --- | --- |
| Codecs | H.264, HEVC, AV1, VP8, VP9, MPEG-1/2, MPEG-4 Part 2, VC-1/WMV3, FLV1, MJPEG |
| Containers | MP4/MOV (including fragmented MP4 and CMAF-compatible files), Matroska/WebM, IVF, MPEG-PS/TS, AVI, ASF, FLV, multipart JPEG, and enabled codecs' common elementary streams |
| Software-decoded frames | Planar little-endian YUV 4:2:0, 4:2:2, or 4:4:4; 8-, 10-, 12-, or 16-bit samples |
| Native Vulkan Video surfaces | NV12 (8-bit 4:2:0) or P010 (10-bit 4:2:0) |
| Input access | Local files only |

AV1 software decoding uses dav1d. The FFmpeg integration includes no network
protocols, input devices, encoders, muxers, filters, or `libswscale`. RAW/Y4M is
not a production input path.

Prepare inputs with matching decoded descriptors. CrossVMAF performs no resizing,
rotation, deinterlacing, color-space conversion, or range conversion. Unsupported
layouts and mid-stream resolution, pixel-format, or bit-depth changes fail.

Frames are paired in decoded display order. When both inputs provide timestamps,
their timelines are normalized to the first decoded frame and must agree within
the pairing tolerance. Timestamps on only one input, or different final frame
counts, fail explicitly.

## C++ API

Include [`crossvmaf/crossvmaf.hpp`](include/crossvmaf/crossvmaf.hpp) and open an
`EvaluationStream` with reference and distorted file paths. `next()` returns
ordered batches; after it returns an empty batch, `finish()` drains the remaining
GPU and temporal results.

<details>
<summary>Minimal streaming example</summary>

```cpp
#include <crossvmaf/crossvmaf.hpp>

#include <iostream>
#include <utility>
#include <vector>

int main() {
    crossvmaf::StreamOptions options;
    options.reference_path = "reference.mp4";
    options.distorted_path = "distorted.mp4";

    auto opened = crossvmaf::EvaluationStream::open(std::move(options));
    if (!opened) {
        std::cerr << opened.error().message << '\n';
        return 1;
    }

    auto stream = std::move(opened).value();
    std::vector<crossvmaf::FrameResult> results;
    for (;;) {
        auto batch = stream->next();
        if (!batch) {
            std::cerr << batch.error().message << '\n';
            return 2;
        }
        if (batch.value().empty()) break;
        results.insert(results.end(), batch.value().begin(), batch.value().end());
    }

    auto tail = stream->finish();
    if (!tail) {
        std::cerr << tail.error().message << '\n';
        return 3;
    }
    results.insert(results.end(), tail.value().begin(), tail.value().end());
    return 0;
}
```

`frame_count_hint()` is optional because containers may not report an exact
count. `decoded_frame_count()` becomes final after input exhaustion. The
[public streaming API](include/crossvmaf/stream.hpp) exposes stream controls,
cancellation, metadata, and execution statistics.

</details>

## License

CrossVMAF is licensed under [BSD-2-Clause-Patent](LICENSE). Dependencies and assets
retain their own terms; see the [third-party license files](third_party/licenses/)
and license notices staged with the applications.

- **FFmpeg:** the selected build is LGPL-2.1-or-later, with GPL and nonfree
  components disabled. It is linked as a replaceable `crossvmaf-ffmpeg` shared
  library built from the complete, unmodified vendored source tree.
- **dav1d:** BSD-2-Clause, statically linked into the FFmpeg shared library.
- **Studio:** Slint has separate license terms; the project's BSD license does
  not cover the complete Studio dependency stack.

Keep the staged `licenses/` directory with distributed applications. Provide the
corresponding FFmpeg source and CrossVMAF build integration for the exact release,
including the dependency notices and the notices for Studio's transitive
dependencies and bundled assets.
