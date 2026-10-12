"""Build upstream dav1d with the parent CMake compiler and configuration."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


def load_compiler_environment(vcvars: str | None) -> dict[str, str]:
    environment = dict(os.environ)
    if vcvars:
        command_interpreter = environment.get("COMSPEC", "cmd.exe")
        # cmd.exe does not understand list2cmdline's backslash-escaped quotes.
        environment_command = (
            f'"{command_interpreter}" /d /s /c "call "{vcvars}" >nul && set"')
        completed = subprocess.run(
            environment_command, capture_output=True, text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                "Failed to initialize the MSVC environment:\n"
                + completed.stdout + completed.stderr)
        for line in completed.stdout.splitlines():
            name, separator, value = line.partition("=")
            if separator and name:
                # Windows environment names are case-insensitive.
                for existing_name in list(environment):
                    if existing_name.casefold() == name.casefold():
                        del environment[existing_name]
                environment[name.upper()] = value
    return environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--configuration", required=True,
                        choices=["Debug", "Release", "RelWithDebInfo", "MinSizeRel"])
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--nasm", required=True, type=Path)
    parser.add_argument("--vcvars")
    parser.add_argument("--runtime", choices=[
        "MultiThreaded", "MultiThreadedDebug",
        "MultiThreadedDLL", "MultiThreadedDebugDLL"])
    arguments = parser.parse_args()

    environment = load_compiler_environment(arguments.vcvars)
    search_directories = [str(Path(arguments.compiler).parent), str(arguments.nasm.parent)]
    if importlib.util.find_spec("ninja") is not None:
        import ninja

        search_directories.append(ninja.BIN_DIR)
    environment["PATH"] = os.pathsep.join(
        [*search_directories, environment.get("PATH", "")])
    if shutil.which("ninja", path=environment["PATH"]) is None:
        raise RuntimeError(
            f'Ninja is required: "{sys.executable}" -m pip install ninja')

    arguments.build.mkdir(parents=True, exist_ok=True)
    native_file = arguments.build / "crossvmaf-native.ini"
    # Native files avoid shell parsing of compiler paths containing spaces.
    native_contents = (
        "[binaries]\n"
        f"c = '{Path(arguments.compiler).as_posix()}'\n"
        f"nasm = '{arguments.nasm.as_posix()}'\n"
    )
    if not native_file.exists() or native_file.read_text() != native_contents:
        native_file.write_text(native_contents, encoding="utf-8")

    build_types = {
        "Debug": "debug", "Release": "release",
        "RelWithDebInfo": "debugoptimized", "MinSizeRel": "minsize",
    }
    meson_command = [sys.executable, "-m", "mesonbuild.mesonmain"]
    setup_command = [
        *meson_command, "setup", str(arguments.build), str(arguments.source),
        "--backend=ninja", "--native-file", str(native_file),
        f"--buildtype={build_types[arguments.configuration]}",
        "--default-library=static", "--wrap-mode=nodownload",
        "-Denable_asm=true", "-Dbitdepths=8,16", "-Db_staticpic=true",
        "-Denable_tools=false", "-Denable_tests=false",
        "-Denable_examples=false", "-Denable_docs=false",
    ]
    if arguments.runtime:
        runtime_types = {
            "MultiThreaded": "mt", "MultiThreadedDebug": "mtd",
            "MultiThreadedDLL": "md", "MultiThreadedDebugDLL": "mdd",
        }
        setup_command.append(f"-Db_vscrt={runtime_types[arguments.runtime]}")
    if (arguments.build / "meson-private" / "coredata.dat").exists():
        setup_command.append("--reconfigure")
    subprocess.run(setup_command, env=environment, check=True)
    subprocess.run(
        [*meson_command, "compile", "-C", str(arguments.build)],
        env=environment, check=True,
    )


if __name__ == "__main__":
    main()
