"""Collect and validate a fresh runtime package from explicit CMake target paths."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ffmpeg_abi", ROOT / "cmake/ffmpeg/abi/abi.py")
abi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(abi)

# Explicit platform prerequisites. A DLL on the developer's PATH is not evidence
# that it belongs to Windows or that it will be available in a distributed package.
WINDOWS_DLLS = set("""advapi32.dll bcrypt.dll bcryptprimitives.dll cfgmgr32.dll
combase.dll comctl32.dll comdlg32.dll crypt32.dll d3d11.dll d3d12.dll dcomp.dll
dwrite.dll dwmapi.dll dxgi.dll gdi32.dll imm32.dll kernel32.dll ntdll.dll ole32.dll
oleaut32.dll opengl32.dll propsys.dll rpcrt4.dll secur32.dll setupapi.dll shell32.dll
shlwapi.dll uiautomationcore.dll user32.dll userenv.dll usp10.dll uxtheme.dll
version.dll winmm.dll winspool.drv ws2_32.dll powrprof.dll hid.dll""".split())


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def is_binary(path: Path) -> bool:
    with path.open("rb") as stream:
        magic = stream.read(4)
    return magic[:2] == b"MZ" or magic == b"\x7fELF"


def prerequisite(name: str, windows: bool) -> str | None:
    if windows:
        if name in WINDOWS_DLLS or name.startswith(("api-ms-win-", "ext-ms-win-")):
            return "Windows"
        if re.fullmatch(r"(?:msvcp140(?:_[a-z_]+)?|vcruntime140(?:_1)?|concrt140|ucrtbase)d?\.dll", name):
            return "Microsoft C/C++ runtime (Debug builds require the development runtime)"
    elif re.fullmatch(r"(?:lib(?:c|m|dl|pthread|rt)\.so\.[0-9]+|libstdc\+\+\.so\.6|"
                      r"libgcc_s\.so\.1|ld-linux[^/]*\.so\.[0-9]+|"
                      r"lib(?:va|va-drm|drm)\.so\.[0-9]+)", name):
        return "Linux system runtime"
    return None


def validate_dependencies(files: dict[str, Path], reader: str) -> dict[str, str]:
    external: dict[str, str] = {}
    for path in files.values():
        if not is_binary(path):
            continue
        windows = abi.binary_format(path) == "pe"
        if windows:
            imports = abi.pe_imports(reader, path)
        else:
            output = abi.inspect(reader, path, "--needed-libs")
            match = re.search(r"NeededLibraries \[\n(.*?)\n\]", output, re.S)
            if not match:
                raise ValueError(f"Cannot read runtime dependencies of {path}")
            imports = [line.strip() for line in match[1].splitlines() if line.strip()]
        for dependency in imports:
            key = dependency.lower() if windows else dependency
            if key in files:
                continue
            category = prerequisite(key, windows)
            if category is None:
                raise ValueError(f"{path.name}: unbundled runtime dependency: {dependency}")
            external[key] = category
    return external


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--config", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--reader", required=True)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--abi-major", required=True, type=int)
    parser.add_argument("--ffmpeg-name", required=True)
    parser.add_argument("--backend", action="append", default=[])
    args = parser.parse_args()
    # Each attempt gets an empty directory. No cleanup or reuse of older releases.
    args.output_root.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc)
    destination = Path(tempfile.mkdtemp(prefix=created.strftime("%Y%m%dT%H%M%SZ-"),
                                       dir=args.output_root)).resolve()
    files: dict[str, Path] = {}
    records = []
    try:
        for line in args.inputs.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            source_name, relative_name = line.split("\t")
            source = Path(source_name)
            target = (destination / relative_name).resolve()
            if not target.is_relative_to(destination) or target == destination:
                raise ValueError(f"Invalid package path: {relative_name}")
            identity = relative_name.lower() if os.name == "nt" else relative_name
            source_hash = digest(source)
            if identity in files:
                if digest(files[identity]) != source_hash:
                    raise ValueError(f"Conflicting package inputs: {relative_name}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if digest(target) != source_hash:
                raise ValueError(f"Input changed during packaging: {source}")
            files[identity] = target
            records.append({"path": relative_name, "sha256": source_hash, "bytes": target.stat().st_size})

        ffmpeg = files[args.ffmpeg_name.lower() if os.name == "nt" else args.ffmpeg_name]
        abi.check_exports(args.contract, args.reader, ffmpeg)
        for path in files.values():
            if is_binary(path):
                abi.check_imports(args.contract, args.reader, path, args.ffmpeg_name)
        external = validate_dependencies(files, args.reader)
        revision = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(args.source), "status", "--porcelain"], text=True)
        backends = dict(item.split("=", 1) for item in args.backend)
        driver_dependencies = {"vulkan": "system Vulkan loader and compute driver"}
        if backends.get("amf") == "ON":
            driver_dependencies["amf"] = "AMD AMF runtime from the Radeon driver, loaded on demand"
        if backends.get("nvdec") == "ON":
            driver_dependencies["nvdec"] = "NVIDIA CUDA and decode driver libraries, loaded on demand"
        if backends.get("qsv") == "ON":
            driver_dependencies["qsv"] = "Intel GPU implementation, selected by the bundled VPL dispatcher"
        manifest = {"schema": "crossvmaf.runtime-package.v1", "created_utc": created.isoformat(),
                    "revision": revision, "dirty": bool(status),
                    "configuration": args.config, "compiler": args.compiler, "decode_backends": backends,
                    "ffmpeg_abi_major": args.abi_major, "abi_contract_sha256": digest(args.contract),
                    "external_dependencies": external, "dynamic_driver_dependencies": driver_dependencies,
                    "files": sorted(records, key=lambda item: item["path"])}
        (destination / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (args.output_root / "last-package.txt").write_text(str(destination) + "\n", encoding="utf-8")
        print(f"Runtime package verified: {destination}")
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Incomplete package at {destination}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
