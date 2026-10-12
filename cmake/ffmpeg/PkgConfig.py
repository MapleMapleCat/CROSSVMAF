"""pkg-config overlay for CMake-built pinned VPL and NVIDIA headers."""
import json
import re
import subprocess
import sys

packages = json.load(open(sys.argv[1], encoding="utf-8"))
arguments = sys.argv[2:]
queries = [arg for arg in arguments if not arg.startswith("-")]
known = []
for query in queries:
    match = re.fullmatch(r"([\w.+-]+)(?:\s*(>=|<=|=|>|<)\s*([\d.]+))?", query)
    if not match or match[1] not in packages:
        system = packages.get("_system", "")
        if system and not system.endswith("-NOTFOUND"):
            sys.exit(subprocess.call([system, *arguments]))
        sys.exit(1)
    package = packages[match[1]]
    if match[2]:
        version = lambda s: tuple(map(int, s.split(".")))
        a, b = version(package["version"]), version(match[3])
        if not {">=": a >= b, "<=": a <= b, "=": a == b, ">": a > b, "<": a < b}[match[2]]:
            sys.exit(1)
    known.append(package)
if not known:
    sys.exit(0 if "--version" in arguments else 1)
if "--cflags" in arguments:
    print(" ".join(p["cflags"] for p in known))
if "--libs" in arguments:
    print(" ".join(p["libs"] for p in known))
if "--modversion" in arguments:
    print(known[0]["version"])
