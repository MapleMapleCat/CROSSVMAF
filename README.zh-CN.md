# CrossVMAF

[English](README.md) | **简体中文**

**面向 Windows 和 Linux 的 Vulkan 加速视频质量评估工具。**

> [!IMPORTANT]
> 当前版本仍处于早期开发阶段，Linux 支持尚不完善，Windows 版本也可能存在兼容性问题。
> 如遇问题，欢迎通过本仓库的 GitHub Issues 反馈。

CrossVMAF 使用独立的 C++20 实现，按照 VMAF 1.0.16 的特征与模型计算流程，
比较参考视频和失真视频的质量。你可以使用 **Studio 桌面应用**、**命令行工具**
或 **C++ 流式 API**。特征计算在 Vulkan GPU 上执行，最终的模型预测在 CPU 上完成。

CrossVMAF 不修改或链接 Netflix libvmaf，也不是 Netflix 官方产品。

[快速上手](#快速上手) · [构建](#构建) · [Studio](#studio) ·
[使用说明](#使用说明) · [C++ API](#c-api) · [许可证](#许可证)

## 主要功能

- **Vulkan 计算：** 在不同厂商的兼容 GPU 上进行视频质量评估。
- **桌面工作流：** 将评估任务加入队列、比较结果、查看质量曲线，并导出 JSON、CSV 或客户 PDF 报告。
- **自动选择模型：** 根据参考视频解码后的尺寸和帧率，从八个内置模型中自动选择；命令行支持手动指定。
- **硬件解码：** 在支持的设备上使用 Intel QSV、NVIDIA NVDEC、Vulkan Video 或 Linux VA-API；自动模式支持回退到软件解码。
- **CPU 验证：** 使用 `--verify-cpu`，将 GPU 结果与独立 CPU 参考实现的计算结果进行比较。

## 快速上手

Windows x64 用户可以从本仓库的 **Releases** 页面下载
`CrossVMAF-<tag>-windows-x64.zip`，完整解压后运行 `crossvmaf-studio.exe`
启动桌面应用，或运行 `crossvmaf.exe --help` 查看命令行用法。
请将附带的 DLL 和 `licenses/` 目录与程序放在一起。
系统需要安装 Microsoft Visual C++ x64 运行时和兼容的 Vulkan GPU 驱动。

[从源码构建](#构建)后，在仓库根目录运行命令行工具。
以下示例使用 Windows Release 构建和 PowerShell 7：

```powershell
$CrossVmaf = ".\build\windows-release\apps\crossvmaf\Release\crossvmaf.exe"

& $CrossVmaf --list-devices
& $CrossVmaf --self-test

& $CrossVmaf `
  --reference ".\reference.mp4" `
  --distorted ".\distorted.mp4" `
  --output ".\result.json"
```

默认情况下，程序自动选择计算设备、解码器和模型，以均值汇总序列分数，并输出 JSON。
省略 `--output` 可写入标准输出；添加 `--format csv` 可改为输出 CSV。

如果偏好图形界面，可以在同一次构建后启动 Studio：

```powershell
.\build\windows-release\apps\studio\Release\crossvmaf-studio.exe
```

Linux 命令行程序位于 `build/linux-clang/apps/crossvmaf/Release/crossvmaf`，
使用相同的命令行选项。

> [!IMPORTANT]
> 两个输入解码后的尺寸、像素格式和位深必须一致，帧数必须相同，时间戳也必须兼容。
> CrossVMAF 不会缩放图像或转换色彩空间。详见[输入要求](#输入要求)。

## 构建

原生构建支持 **x86-64 平台的 Windows 和 Linux**。
当前的视频编解码集成尚不支持 macOS、ARM 或交叉编译。

### 环境要求

| 依赖 | 要求 |
| --- | --- |
| 构建工具 | CMake 3.28+ 和 Ninja |
| 编译器 | 支持 C++20 和 C17 的 LLVM/Clang；Windows 使用 `clang-cl`，Linux 使用 `clang`/`clang++`；需包含用于 ABI 检查的 `llvm-readobj` |
| Python | Python 3.10+，且 dav1d 构建环境中可用 Meson 和 Ninja |
| Studio | Rust 1.92+ 和 Cargo；Slint 运行时及 UI 编译器从固定版本源码构建 |
| 源码工具 | Git、NASM，以及首次配置和构建时的网络连接 |
| 运行环境 | Vulkan Loader 和兼容 Vulkan 1.1+ 的计算 GPU |
| Windows | Visual Studio 或 Build Tools，安装 **使用 C++ 的桌面开发**（Desktop development with C++）工作负载、Windows SDK，以及 Git for Windows / Git Bash |
| Linux | POSIX Shell；默认启用的 QSV 构建还需要 `libva`、`libva-drm`、`libdrm` 开发包和 `pkg-config` |

为 CMake 使用的 Python 解释器安装 Meson：

```sh
python -m pip install meson ninja
```

Vulkan Video 解码还需要 Vulkan 1.3 和相应的解码扩展。
QSV、NVDEC 和 VA-API 各有设备及驱动要求，详见[解码后端](#解码后端)。

### Windows — PowerShell 7

将 LLVM 安装到默认的 `%ProgramFiles%/LLVM` 路径，构建 Studio 时使用 Rust MSVC 工具链。
在普通 PowerShell 7 终端中执行以下命令，并确保 CMake、Ninja、Python、NASM、Git 和 Rust
均可通过 `PATH` 访问：

```powershell
cmake --preset windows-release
cmake --build --preset windows-release --parallel
```

预设会选择 LLVM 工具，并发现已安装的 Microsoft SDK 和 C++ 运行时。
无需初始化 Developer PowerShell，也无需 WSL。
如需 Debug 构建，将两条命令中的预设名称都替换为 `windows-debug`。

| 应用 | Release 可执行文件 |
| --- | --- |
| CLI | `build/windows-release/apps/crossvmaf/Release/crossvmaf.exe` |
| Studio | `build/windows-release/apps/studio/Release/crossvmaf-studio.exe` |

完整构建还会更新 `out/crossvmaf-studio.exe` 及其运行库，方便在本地启动。

### Linux — 原生 Shell

```sh
cmake -S . -B build/linux-clang -G "Ninja Multi-Config" \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CONFIGURATION_TYPES='Debug;Release' \
  -DFETCHCONTENT_BASE_DIR="$PWD/.cache/fetchcontent" \
  -DCROSSVMAF_BUILD_TOOLS=ON \
  -DCROSSVMAF_ENABLE_QSV=ON -DCROSSVMAF_ENABLE_NVDEC=ON
cmake --build build/linux-clang --config Release --parallel
```

如需 Debug 构建，构建时使用 `--config Debug`。
VA-API 默认关闭，可在配置命令中添加 `-DCROSSVMAF_ENABLE_VAAPI=ON` 启用。

<details>
<summary>构建选项、自定义工具链与已有构建目录</summary>

- **仅构建命令行工具：** 在配置命令中添加 `-DCROSSVMAF_BUILD_STUDIO=OFF`，
  即可排除 Studio 及其 Rust 依赖。
- **自定义 LLVM 路径：** 在被 Git 忽略的 `CMakeUserPresets.json` 中创建继承的配置预设，
  并在其 `environment` 中设置 `CROSSVMAF_LLVM_ROOT`。相应的构建预设也应指向该配置预设。
- **Windows 编译器迁移：** 将已有构建切换到这些 LLVM 预设时，执行
  `cmake --fresh --preset windows-release`（或 `windows-debug`），并重新传入本地 `-D` 选项。
  不要复用 MSVC 的 CMake 缓存。
- **Linux 生成器变更：** 从单配置 Ninja 切换到 Ninja Multi-Config 时，使用新的构建目录。
  也可以在独立的原生构建目录中通过 `-DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++` 选择 GCC。
- **依赖缓存：** 下载的源码位于 `.cache/fetchcontent`，可供不同构建目录复用。
  完整且未经修改的 FFmpeg 源码随仓库存放在 `third_party/ffmpeg`，无需外部 FFmpeg 可执行文件。
- **生成资源：** 构建过程会编译、验证、审计并嵌入运行时着色器，同时校验并嵌入内置模型。

依赖版本固定在 [`cmake/Dependencies.cmake`](cmake/Dependencies.cmake)、
[`cmake/DecodeBackends.cmake`](cmake/DecodeBackends.cmake)、
[`cmake/FFmpeg.cmake`](cmake/FFmpeg.cmake) 和
[`cmake/Dav1d.cmake`](cmake/Dav1d.cmake) 中。

</details>

### 运行包

使用以下命令生成新的 Windows Release 运行包，并检查其依赖：

```powershell
cmake --build --preset windows-release --target crossvmaf_package --parallel
```

每次调用都会在 `build/windows-release/packages/Windows-AMD64/Release` 下创建新目录，
包含运行库、许可证声明，以及记录二进制哈希、构建配置和运行前提的 `build-manifest.json`。
打包使用各目标的正式构建产物，不依赖 `out/` 中的文件。
分发二进制程序时，应将运行库和 `licenses/` 目录一并保留。

## Studio

Studio 提供选择输入、排队评估、比较结果和查看已保存报告的桌面工作流。
界面支持英语和简体中文，并提供浅色与深色主题。

- **评估设置：** 选择计算设备、解码设备、后端、传输策略、解码线程数和缓冲帧数。
  默认使用自动解码、自动线程分配，每个输入缓冲四帧。
- **队列行为：** 任务加入队列时会保存当时的评估设置。默认在任务失败时暂停队列。
- **报告：** 导出 JSON 或 CSV 用于分析，或选择 **客户报告（PDF）**（Customer report (PDF)），
  生成包含汇总分数、质量曲线、分数分布、待检查片段和评估详情的 A4 报告。

PDF 报告使用当前界面语言，并嵌入字体。导出已保存的评估结果时，
无需原始视频、浏览器、打印机驱动或 Python 运行环境。
Studio 会自动选择观看条件对应的模型；如需手动指定模型，请使用命令行工具。

## 使用说明

### 自动选择模型

命令行工具和 Studio 根据参考视频实际解码后的评估尺寸及帧率选择模型：

| 参考视频 | 自动选择 |
| --- | --- |
| 长边 ≥ 3840 像素，**或**短边 ≥ 2160 像素 | 4K / 1.5H 模型 |
| 更小的尺寸，包括 1440p | 1080p / 3H 模型 |
| 已知且恒定的帧率 ≥ 50 fps | 对应的 HFR 高帧率版本 |
| 更低、未知或可变的帧率 | 标准版本 |

这些阈值是 CrossVMAF 的选择策略。程序不会缩放输入视频，选择模型也不会模拟不同的显示设备。
对于超出模型标定预设的分辨率或帧率，例如 8K 或 120 fps，评估属于近似使用。

列出全部八个内置模型，或手动覆盖自动选择结果：

```powershell
& $CrossVmaf --list-models
& $CrossVmaf --reference reference.mp4 --distorted distorted.mp4 `
  --model vmaf_v1.0.16_3d0h_2160 --output result.json
```

所选模型会用于 CPU 验证，并记录在报告中。
模型在构建时嵌入，不支持自定义模型或运行时参数覆盖。
详见[模型来源与哈希](models/README.md)。

### 解码后端

VMAF 特征计算始终使用所选的 Vulkan 计算设备，即使视频采用软件解码也是如此。
默认设置下，自动解码会按以下顺序尝试符合条件的后端：

| 计算 GPU | Windows | Linux |
| --- | --- | --- |
| Intel | QSV → Vulkan Video → 软件解码 | QSV → VA-API → 软件解码 |
| NVIDIA | NVDEC → Vulkan Video → 软件解码 | NVDEC → VA-API → 软件解码 |
| AMD 或其他厂商 | Vulkan Video → 软件解码 | VA-API → 软件解码 |

候选后端必须通过构建配置、平台、设备、编解码器和驱动检查。
自动模式会尝试 GPU 传输，并可在启动阶段以软件解码重新打开两个输入。
VA-API 必须在构建时启用，才会参与 Linux 上的自动选择。

- `--decode hardware` 要求两个输入都使用硬件解码，不回退到软件解码。
  该模式下的自动传输也允许使用符合条件的主机内存传输路径。
- `--decode-backend NAME` 固定使用指定后端，无法使用时任务失败。
- `--decode-transfer gpu` 禁止将解码后的像素下载到主机内存；`host` 请求主机内存传输。
  使用与计算 GPU 不同的解码设备时，需要采用主机内存传输。

硬件支持情况取决于 GPU 和驱动。已测试的 Intel Arc 130T 存在已知的 Vulkan Main10
解码表面复用失败问题，自动 Vulkan 选择仍保留相应的驱动排除规则。
使用 `--list-decode-backends` 和 `--list-decode-devices` 查看可用后端及设备。
成功编译本身不能证明硬件路径的正确性。

### 命令行选项

运行 `& $CrossVmaf --help` 查看命令行参考。

| 选项 | 用途 / 默认值 |
| --- | --- |
| `--reference PATH` | 参考视频，必填 |
| `--distorted PATH` | 失真视频，必填 |
| `--model NAME` | 强制使用指定内置模型；默认自动选择 |
| `--device auto\|UUID` | Vulkan 计算设备；默认 `auto` |
| `--pool mean\|min\|harmonic` | 序列分数汇总方式；默认 `mean` |
| `--format json\|csv` | 输出格式；默认 `json` |
| `--output PATH` | 写入指定文件；默认写入标准输出 |
| `--verify-cpu` | 执行独立的 CPU 验证流程 |

<details>
<summary>解码控制、诊断与设备发现</summary>

| 选项 | 用途 / 默认值 |
| --- | --- |
| `--decode auto\|hardware\|software` | 解码模式；默认 `auto` |
| `--decode-backend auto\|vulkan\|qsv\|nvdec\|vaapi` | 解码后端；默认 `auto`；明确指定后端后不自动更换 |
| `--decode-device same\|ID` | 解码设备；默认与计算使用同一块物理 GPU |
| `--decode-transfer auto\|gpu\|host` | 帧传输策略；默认 `auto` |
| `--decoded-frame-queue N` | 每个输入的缓冲帧数：`1..64`；默认 `4` |
| `--decoder-threads N` | 每个输入的线程数：`0..64`；默认 `0`，表示自动分配 |
| `--diagnostic-speed-chroma` | 在 JSON 输出中添加逐帧 U/V 诊断信息 |
| `--list-devices` | 列出兼容的 Vulkan 计算设备 |
| `--list-models` | 列出八个内置模型 |
| `--list-decode-backends` | 列出已编译的解码后端 |
| `--list-decode-devices` | 列出解码设备 ID；可用 `--decode-backend NAME` 筛选 |
| `--self-test [device-uuid]` | 运行设备自检 |

AV1 自动软件解码会让两个输入共享主机级 dav1d 工作线程预算，每个输入最多使用 16 个工作线程。
FFmpeg 原生编解码器保留各自的自动线程策略。硬件后端对线程数的解释可能不同。

</details>

### CPU 验证

```powershell
& $CrossVmaf --reference reference.mkv --distorted distorted.mkv `
  --verify-cpu --output verified.json
```

验证模式会将 Speed Chroma 切换为精确计算，重新解码两个输入，
并将 Vulkan 结果与独立 CPU 参考实现的结果逐项比较。
该功能用于正确性检查，会增加计算工作量，不适合作为吞吐量基准测试。

### 输入要求

| 类别 | 支持的输入 |
| --- | --- |
| 编解码器 | H.264、HEVC、AV1、VP8、VP9、MPEG-1/2、MPEG-4 Part 2、VC-1/WMV3、FLV1、MJPEG |
| 容器 | MP4/MOV（包括分片 MP4 和兼容 CMAF 的文件）、Matroska/WebM、IVF、MPEG-PS/TS、AVI、ASF、FLV、multipart JPEG，以及已启用编解码器的常见基本码流格式 |
| 软件解码帧 | 小端序平面 YUV 4:2:0、4:2:2 或 4:4:4；8、10、12 或 16 位采样 |
| 原生 Vulkan Video 解码表面 | NV12（8 位 4:2:0）或 P010（10 位 4:2:0） |
| 输入访问 | 仅支持本地文件 |

AV1 软件解码使用 dav1d。FFmpeg 集成不包含网络协议、输入设备、编码器、封装器、滤镜或 `libswscale`。
RAW/Y4M 不属于正式支持的输入路径。

请准备解码后帧描述信息一致的输入。CrossVMAF 不进行缩放、旋转、去隔行、色彩空间转换或范围转换。
不支持的像素布局，以及码流中途发生的分辨率、像素格式或位深变化，都会导致任务失败。

帧按解码后的显示顺序配对。如果两个输入都提供时间戳，会分别以各自的首个解码帧为起点归一化时间轴，
对应帧的时间戳必须在配对容差范围内一致。
只有一个输入具有时间戳，或两个输入的最终帧数不同，都会明确报错。

## C++ API

包含 [`crossvmaf/crossvmaf.hpp`](include/crossvmaf/crossvmaf.hpp)，
并使用参考视频和失真视频的文件路径打开 `EvaluationStream`。
`next()` 按顺序返回结果批次；当它返回空批次后，调用 `finish()`
取回剩余的 GPU 计算结果及时间相关结果。

<details>
<summary>最小流式示例</summary>

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

容器未必能提供精确帧数，因此 `frame_count_hint()` 的返回值是可选的。
`decoded_frame_count()` 在输入耗尽后成为最终计数。
[公共流式 API](include/crossvmaf/stream.hpp) 提供流控制、取消、元数据和执行统计接口。

</details>

## 许可证

CrossVMAF 使用 [BSD-2-Clause-Patent](LICENSE) 许可证。
依赖和资源保留各自的许可条款，详见[第三方许可证文件](third_party/licenses/)
以及随应用程序提供的许可证声明。

- **FFmpeg：** 所选构建采用 LGPL-2.1-or-later，禁用 GPL 和 nonfree 组件。
  程序动态链接可替换的 `crossvmaf-ffmpeg` 共享库，该库由随仓库提供的完整、未经修改的源码树构建。
- **dav1d：** 使用 BSD-2-Clause 许可证，静态链接到 FFmpeg 共享库中。
- **Studio：** Slint 具有独立的许可条款；本项目的 BSD 许可证不涵盖 Studio 的完整依赖栈。

分发应用程序时，请保留构建时收集的 `licenses/` 目录。
应提供与具体发布版本对应的 FFmpeg 源码和 CrossVMAF 构建集成，
并包含依赖声明，以及 Studio 间接依赖和内置资源的相关声明。
