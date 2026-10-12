"""Generate explicit linker inputs and audit the linked FFmpeg ABI.

Only exports.txt defines the contract. Binary inspection never updates it.
Uses the Clang toolchain's llvm-readobj, with no third-party Python modules.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys


def load_contract(path: Path) -> list[str]:
    names = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        name = line.partition("#")[0].strip()
        if not name:
            continue
        if not re.fullmatch(r"(?:av_|avcodec_|avformat_|avutil_)[A-Za-z0-9_]+", name):
            raise ValueError(f"{path}:{number}: not an explicit public function: {name!r}")
        if name in names:
            raise ValueError(f"{path}:{number}: duplicate export: {name}")
        names.append(name)
    if not names:
        raise ValueError(f"{path}: empty ABI contract")
    if names != sorted(names):
        raise ValueError(f"{path}: keep function names sorted")
    return names


def write_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8", newline="\n")


def generate(contract: Path, output: Path, library: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", library):
        raise ValueError("invalid DLL library name")
    names = load_contract(contract)
    write_changed(output / "exports.def", f'LIBRARY "{library}"\nEXPORTS\n' +
                  "".join(f"    {name}\n" for name in names))
    write_changed(output / "exports.map", "{\n    global:\n" +
                  "".join(f"        {name};\n" for name in names) +
                  "    local: *;\n};\n")


def inspect(reader: str, binary: Path, *flags: str) -> str:
    return subprocess.run([reader, *flags, str(binary)], check=True,
                          capture_output=True, text=True, encoding="utf-8").stdout


def field(block: str, name: str) -> str:
    found = re.search(rf"^\s*{re.escape(name)}: (.*)$", block, re.M)
    if not found:
        raise ValueError(f"llvm-readobj output missing {name}")
    return found[1].strip()


def binary_format(binary: Path) -> str:
    with binary.open("rb") as stream:
        magic = stream.read(4)
    if magic[:2] == b"MZ":
        return "pe"
    if magic == b"\x7fELF":
        return "elf"
    raise ValueError(f"unsupported binary format: {binary}")


def exported_functions(reader: str, binary: Path) -> set[str]:
    functions: set[str] = set()
    if binary_format(binary) == "pe":
        output = inspect(reader, binary, "--sections", "--coff-exports")
        executable_ranges = []
        for block in re.findall(r"^  Section \{\n(.*?)^  \}", output, re.M | re.S):
            if "IMAGE_SCN_MEM_EXECUTE" in block:
                start = int(field(block, "VirtualAddress"), 0)
                size = max(int(field(block, "VirtualSize"), 0),
                           int(field(block, "RawDataSize"), 0))
                executable_ranges.append((start, start + size))
        for block in re.findall(r"^Export \{\n(.*?)^\}", output, re.M | re.S):
            name = field(block, "Name")
            if not re.search(r"^\s*RVA:", block, re.M):
                raise ValueError(f"{binary}: forwarded or non-address export: {name!r}")
            address = int(field(block, "RVA"), 0)
            if not name or not any(lo <= address < hi for lo, hi in executable_ranges):
                raise ValueError(f"{binary}: ordinal-only, data or forwarded export: {name!r}")
            if name in functions:
                raise ValueError(f"{binary}: duplicate export: {name}")
            functions.add(name)
    else:
        output = inspect(reader, binary, "--dyn-symbols")
        for block in re.findall(r"^  Symbol \{\n(.*?)^  \}", output, re.M | re.S):
            binding = field(block, "Binding").split()[0]
            section = field(block, "Section").split()[0]
            if binding not in ("Global", "Weak") or section == "Undefined":
                continue
            name = field(block, "Name").split(" (")[0]
            if field(block, "Type").split()[0] != "Function":
                raise ValueError(f"{binary}: exported non-function: {name}")
            if name in functions:
                raise ValueError(f"{binary}: duplicate export: {name}")
            functions.add(name)
    if not functions:
        raise ValueError(f"{binary}: no function exports found")
    return functions


def check_exports(contract: Path, reader: str, binary: Path) -> None:
    expected = set(load_contract(contract))
    actual = exported_functions(reader, binary)
    missing, extra = sorted(expected - actual), sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"{binary}: ABI mismatch; missing={missing}; unexpected={extra}")


def pe_imports(reader: str, binary: Path) -> dict[str, list[str]]:
    output = inspect(reader, binary, "--coff-imports")
    result: dict[str, list[str]] = {}
    for block in re.findall(r"^(?:Import|DelayImport) \{\n(.*?)^\}", output, re.M | re.S):
        name = field(block, "Name").lower()
        result.setdefault(name, []).extend(
            symbol.split(" (")[0].strip()
            for symbol in re.findall(r"^\s*Symbol: (.*)$", block, re.M))
    return result


def check_imports(contract: Path, reader: str, binary: Path, library: str) -> None:
    expected = set(load_contract(contract))
    if binary_format(binary) == "pe":
        for name, symbols in pe_imports(reader, binary).items():
            if name.startswith("crossvmaf-ffmpeg"):
                if name != library.lower():
                    raise ValueError(f"{binary}: unexpected FFmpeg DLL: {name}")
                unexpected = sorted(set(symbols) - expected)
                if unexpected:
                    raise ValueError(f"{binary}: imports outside ABI: {unexpected}")
    else:
        output = inspect(reader, binary, "--dyn-symbols")
        for block in re.findall(r"^  Symbol \{\n(.*?)^  \}", output, re.M | re.S):
            if field(block, "Section").split()[0] != "Undefined":
                continue
            name = field(block, "Name").split(" (")[0].split("@")[0]
            if name.startswith(("av_", "avcodec_", "avformat_", "avutil_", "avpriv_", "ff_")):
                if name not in expected:
                    raise ValueError(f"{binary}: imports outside ABI: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--contract", type=Path, required=True)
    gen.add_argument("--output", type=Path, required=True)
    gen.add_argument("--library", required=True)
    for action in ("check-exports", "check-imports"):
        check = sub.add_parser(action)
        check.add_argument("--contract", type=Path, required=True)
        check.add_argument("--reader", required=True)
        check.add_argument("--binary", type=Path, required=True)
        if action == "check-imports":
            check.add_argument("--library", required=True)
    args = parser.parse_args()
    try:
        if args.action == "generate":
            generate(args.contract, args.output, args.library)
        elif args.action == "check-exports":
            check_exports(args.contract, args.reader, args.binary)
            print(f"FFmpeg ABI verified: {len(load_contract(args.contract))} function exports")
        else:
            check_imports(args.contract, args.reader, args.binary, args.library)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"FFmpeg ABI error: {error}", file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            print(error.stderr, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
