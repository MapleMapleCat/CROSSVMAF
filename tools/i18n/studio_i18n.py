#!/usr/bin/env python3
"""Maintain CrossVMAF Studio's gettext catalogs.

The Studio UI has two translation call sites that share one catalog per
language:

* ``@tr(...)`` in ``apps/studio/ui/**/*.slint``, bundled by slint-compiler;
* ``tr(...)``, ``trc(...)`` and ``trn(...)`` from ``i18n/translator.hpp`` in
  ``apps/studio/src/**/*.{cpp,hpp}``, compiled into C++ tables by
  ``generate-cpp``.

Commands::

    update         re-extract messages, rewrite the .pot and merge every .po
    check          fail when the .pot or any .po is stale or incomplete
    init LANG      start a catalog for a new language from the .pot
    generate-cpp   emit the C++ lookup tables (used by the build)

Message ids are the English source text. The layout of apps/studio/i18n is
the one slint-compiler expects: ``<lang>/LC_MESSAGES/crossvmaf_studio.po``.
See docs/STUDIO_I18N.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
STUDIO = REPOSITORY / "apps" / "studio"
CATALOG_DIRECTORY = STUDIO / "i18n"
DOMAIN = "crossvmaf_studio"
TEMPLATE = CATALOG_DIRECTORY / f"{DOMAIN}.pot"

# Plural rules for languages a catalog may be started with; others need
# --plural-forms.
KNOWN_PLURAL_FORMS = {
    "zh": "nplurals=1; plural=0;",
    "ja": "nplurals=1; plural=0;",
    "ko": "nplurals=1; plural=0;",
    "de": "nplurals=2; plural=(n != 1);",
    "es": "nplurals=2; plural=(n != 1);",
    "fr": "nplurals=2; plural=(n > 1);",
}

Key = tuple[str, str, str]  # (msgctxt, msgid, msgid_plural)


class CatalogError(Exception):
    pass


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY).as_posix()
    except ValueError:
        return path.as_posix()


@dataclasses.dataclass
class Message:
    context: str
    source: str
    plural: str = ""
    references: list[str] = dataclasses.field(default_factory=list)
    translations: list[str] = dataclasses.field(default_factory=list)
    comments: list[str] = dataclasses.field(default_factory=list)
    flags: list[str] = dataclasses.field(default_factory=list)

    @property
    def key(self) -> Key:
        return (self.context, self.source, self.plural)

    @property
    def fuzzy(self) -> bool:
        return "fuzzy" in self.flags


@dataclasses.dataclass
class Catalog:
    header: dict[str, str]
    header_comments: list[str]
    messages: list[Message]

    def plural_count(self) -> int:
        match = re.search(r"nplurals\s*=\s*(\d+)", self.header.get("Plural-Forms", ""))
        return int(match.group(1)) if match else 2

    def plural_expression(self) -> str:
        match = re.search(r"plural\s*=\s*([^;]+);?", self.header.get("Plural-Forms", ""))
        return match.group(1).strip() if match else "n != 1"


# ---------------------------------------------------------------- extraction


class Scanner:
    """Minimal lexer shared by the Slint and C++ extractors."""

    def __init__(self, path: Path, text: str) -> None:
        self.path = path
        self.text = text
        self.position = 0

    def error(self, message: str) -> CatalogError:
        return CatalogError(f"{self.location()}: {message}")

    def location(self) -> str:
        line = self.text.count("\n", 0, self.position) + 1
        return f"{display_path(self.path)}:{line}"

    def peek(self, count: int = 1) -> str:
        return self.text[self.position:self.position + count]

    def skip_space(self) -> None:
        while self.position < len(self.text):
            if self.text[self.position].isspace():
                self.position += 1
            elif self.peek(2) == "//":
                end = self.text.find("\n", self.position)
                self.position = len(self.text) if end < 0 else end
            elif self.peek(2) == "/*":
                end = self.text.find("*/", self.position + 2)
                if end < 0:
                    raise self.error("unterminated comment")
                self.position = end + 2
            else:
                return

    def skip_string(self) -> None:
        quote = self.text[self.position]
        self.position += 1
        while self.position < len(self.text):
            character = self.text[self.position]
            if character == "\\":
                self.position += 2
                continue
            self.position += 1
            if character == quote:
                return
        raise self.error("unterminated literal")

    def skip_raw_string(self) -> None:
        # R"delimiter( ... )delimiter"
        open_paren = self.text.index("(", self.position)
        delimiter = self.text[self.position + 2:open_paren]
        end = self.text.find(f"){delimiter}\"", open_paren)
        if end < 0:
            raise self.error("unterminated raw string")
        self.position = end + len(delimiter) + 2


def unescape_slint(scanner: Scanner, body: str) -> str:
    result = []
    index = 0
    while index < len(body):
        character = body[index]
        if character != "\\":
            result.append(character)
            index += 1
            continue
        escape = body[index + 1:index + 2]
        if escape in ('"', "\\"):
            result.append(escape)
            index += 2
        elif escape == "n":
            result.append("\n")
            index += 2
        elif escape == "u" and body[index + 2:index + 3] == "{":
            end = body.index("}", index)
            result.append(chr(int(body[index + 3:end], 16)))
            index = end + 1
        elif escape == "{":
            raise scanner.error("@tr strings cannot interpolate; pass arguments instead")
        else:
            raise scanner.error(f"unsupported escape \\{escape}")
    return "".join(result)


def unescape_cpp(scanner: Scanner, body: str) -> str:
    output = bytearray()
    index = 0
    simple = {"n": b"\n", "t": b"\t", "\\": b"\\", '"': b'"', "'": b"'", "?": b"?"}
    while index < len(body):
        character = body[index]
        if character != "\\":
            output += character.encode("utf-8")
            index += 1
            continue
        escape = body[index + 1:index + 2]
        if escape in simple:
            output += simple[escape]
            index += 2
        elif escape == "x":
            match = re.match(r"[0-9A-Fa-f]+", body[index + 2:])
            if not match:
                raise scanner.error("invalid \\x escape")
            output.append(int(match.group(0), 16) & 0xFF)
            index += 2 + len(match.group(0))
        elif escape.isdigit():
            match = re.match(r"[0-7]{1,3}", body[index + 1:])
            output.append(int(match.group(0), 8))
            index += 1 + len(match.group(0))
        else:
            raise scanner.error(f"unsupported escape \\{escape}")
    return output.decode("utf-8")


def read_slint_literal(scanner: Scanner) -> str | None:
    scanner.skip_space()
    if scanner.peek() != '"':
        return None
    start = scanner.position
    scanner.skip_string()
    return unescape_slint(scanner, scanner.text[start + 1:scanner.position - 1])


def read_cpp_literal(scanner: Scanner) -> str | None:
    """Reads one string literal plus any adjacent ones it concatenates with."""
    parts = []
    while True:
        scanner.skip_space()
        if scanner.peek() != '"':
            break
        start = scanner.position
        scanner.skip_string()
        parts.append(unescape_cpp(scanner, scanner.text[start + 1:scanner.position - 1]))
    return "".join(parts) if parts else None


def extract_slint(path: Path) -> list[Message]:
    scanner = Scanner(path, path.read_text(encoding="utf-8"))
    messages = []
    while scanner.position < len(scanner.text):
        scanner.skip_space()
        if scanner.position >= len(scanner.text):
            break
        if scanner.peek() == '"':
            scanner.skip_string()
            continue
        if scanner.peek(4) != "@tr(":
            scanner.position += 1
            continue
        reference = scanner.location()
        scanner.position += 4
        first = read_slint_literal(scanner)
        if first is None:
            raise scanner.error("@tr needs a string literal")
        context, source, plural = "", first, ""
        scanner.skip_space()
        if scanner.peek(2) == "=>":
            scanner.position += 2
            context, source = first, read_slint_literal(scanner)
            if source is None:
                raise scanner.error("@tr context must be followed by a string literal")
            scanner.skip_space()
        if scanner.peek() == "|":
            scanner.position += 1
            plural = read_slint_literal(scanner)
            if plural is None:
                raise scanner.error("@tr plural form must be a string literal")
        messages.append(Message(context, source, plural, [reference]))
    return messages


CPP_CALL = re.compile(r"\b(trc|trn|tr)\s*\(")


def extract_cpp(path: Path) -> list[Message]:
    scanner = Scanner(path, path.read_text(encoding="utf-8"))
    messages = []
    text = scanner.text
    while scanner.position < len(text):
        character = text[scanner.position]
        if scanner.peek(2) in ("//", "/*"):
            scanner.skip_space()
            continue
        if character == "R" and scanner.peek(2) == 'R"':
            scanner.skip_raw_string()
            continue
        if character == "'" and scanner.position > 0 and text[scanner.position - 1].isalnum():
            scanner.position += 1  # Digit separator, as in 1'000'000.
            continue
        if character in "\"'":
            scanner.skip_string()
            continue
        match = CPP_CALL.match(text, scanner.position)
        if not match or (scanner.position > 0 and (text[scanner.position - 1].isalnum() or
                                                   text[scanner.position - 1] in "_.>")):
            scanner.position += 1
            continue
        function = match.group(1)
        reference = scanner.location()
        scanner.position = match.end()
        first = read_cpp_literal(scanner)
        if first is None:
            continue  # A declaration or a forwarding call, not a message.
        if function == "tr":
            messages.append(Message("", first, "", [reference]))
            continue
        scanner.skip_space()
        if scanner.peek() != ",":
            raise scanner.error(f"{function} needs two string literals")
        scanner.position += 1
        second = read_cpp_literal(scanner)
        if second is None:
            raise scanner.error(f"{function} needs two string literals")
        if function == "trc":
            messages.append(Message(first, second, "", [reference]))
        else:
            messages.append(Message("", first, second, [reference]))
    return messages


def source_files() -> list[Path]:
    slint = sorted((STUDIO / "ui").rglob("*.slint"))
    cpp = sorted(p for p in (STUDIO / "src").rglob("*") if p.suffix in (".cpp", ".hpp"))
    return slint + cpp


def extract() -> list[Message]:
    merged: dict[Key, Message] = {}
    for path in source_files():
        found = extract_slint(path) if path.suffix == ".slint" else extract_cpp(path)
        relative = display_path(path)
        for message in found:
            validate_placeholders(message.references[0], message.source, message.plural)
            existing = merged.get(message.key)
            if existing is None:
                message.references = [relative]
                merged[message.key] = message
            elif relative not in existing.references:
                existing.references.append(relative)
    return list(merged.values())


# ---------------------------------------------------------------- placeholders

PLACEHOLDER = re.compile(r"\{\{|\}\}|\{([^{}]*)\}")


def placeholders(text: str) -> list[str]:
    return sorted(m.group(1) for m in PLACEHOLDER.finditer(text) if m.group(0) not in ("{{", "}}"))


def validate_placeholders(where: str, source: str, plural: str) -> None:
    for text in filter(None, (source, plural)):
        found = placeholders(text)
        if "" in found and any(p.isdigit() for p in found):
            raise CatalogError(f"{where}: {text!r} mixes '{{}}' and positional placeholders")
        for name in found:
            if name not in ("", "n") and not name.isdigit():
                raise CatalogError(f"{where}: {text!r} has invalid placeholder {{{name}}}")


def comparable_placeholders(text: str) -> list[str]:
    # A plural translation may drop {n}: some languages spell out "one".
    return [p for p in placeholders(text) if p != "n"]


# ---------------------------------------------------------------- .po files


def po_escape(text: str) -> str:
    return (text.replace("\\", "\\\\").replace('"', '\\"')
            .replace("\n", "\\n").replace("\t", "\\t"))


def po_unescape(text: str) -> str:
    return re.sub(r'\\(["\\nt])',
                  lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), text)


def parse_po(path: Path) -> Catalog:
    messages: list[Message] = []
    header: dict[str, str] | None = None
    header_comments: list[str] = []
    pending: dict[str, object] = {}
    field: str | None = None

    def flush() -> None:
        nonlocal header, pending, field
        if "msgid" not in pending:
            pending, field = {}, None
            return
        source = pending["msgid"]
        context = pending.get("msgctxt", "")
        translations = pending.get("msgstr", [])
        if source == "" and context == "" and header is None:
            header = {}
            for line in (translations[0] if translations else "").split("\n"):
                if ":" in line:
                    name, value = line.split(":", 1)
                    header[name.strip()] = value.strip()
            header_comments.extend(pending.get("comments", []))
        else:
            messages.append(Message(
                context, source, pending.get("msgid_plural", ""),
                pending.get("references", []), translations,
                pending.get("comments", []), pending.get("flags", [])))
        pending, field = {}, None

    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.startswith("#~"):
            continue  # Obsolete entries are dropped by update.
        if line.startswith("#"):
            if field is not None:
                flush()
            if line.startswith("#:"):
                pending.setdefault("references", []).extend(line[2:].split())
            elif line.startswith("#,"):
                pending.setdefault("flags", []).extend(
                    f.strip() for f in line[2:].split(","))
            elif not line.startswith("#."):
                pending.setdefault("comments", []).append(raw)
            continue
        match = re.match(r'^(msgctxt|msgid_plural|msgid|msgstr(?:\[(\d+)\])?)\s+"(.*)"$', line)
        if match:
            keyword, index, value = match.group(1), match.group(2), po_unescape(match.group(3))
            if keyword in ("msgctxt", "msgid") and field in ("msgstr", "msgstr[]"):
                flush()
            if keyword.startswith("msgstr"):
                translations = pending.setdefault("msgstr", [])
                slot = int(index) if index is not None else 0
                while len(translations) <= slot:
                    translations.append("")
                translations[slot] = value
                field = "msgstr[]"
                pending["_slot"] = slot
            else:
                pending[keyword] = value
                field = keyword
            continue
        if line.startswith('"') and line.endswith('"') and field is not None:
            value = po_unescape(line[1:-1])
            if field == "msgstr[]":
                pending["msgstr"][pending["_slot"]] += value
            else:
                pending[field] += value
            continue
        raise CatalogError(f"{path}:{number}: cannot parse {raw!r}")
    flush()
    if header is None:
        raise CatalogError(f"{path}: missing header entry")
    return Catalog(header, header_comments, messages)


def write_po(path: Path, catalog: Catalog, template: bool = False) -> None:
    lines = list(catalog.header_comments)
    lines += ['msgid ""', 'msgstr ""']
    lines += [f'"{po_escape(f"{name}: {value}")}\\n"' for name, value in catalog.header.items()]
    plural_count = catalog.plural_count()
    for message in catalog.messages:
        lines.append("")
        lines += message.comments
        for reference in message.references:
            lines.append(f"#: {reference}")
        if message.flags:
            lines.append("#, " + ", ".join(message.flags))
        if message.context:
            lines.append(f'msgctxt "{po_escape(message.context)}"')
        lines.append(f'msgid "{po_escape(message.source)}"')
        if message.plural:
            lines.append(f'msgid_plural "{po_escape(message.plural)}"')
            forms = 2 if template else plural_count
            for index in range(forms):
                value = "" if template else (message.translations[index]
                                             if index < len(message.translations) else "")
                lines.append(f'msgstr[{index}] "{po_escape(value)}"')
        else:
            value = "" if template or not message.translations else message.translations[0]
            lines.append(f'msgstr "{po_escape(value)}"')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def translation_catalogs() -> list[Path]:
    return sorted(CATALOG_DIRECTORY.glob(f"*/LC_MESSAGES/{DOMAIN}.po"))


def template_catalog(messages: list[Message]) -> Catalog:
    return Catalog(
        {
            "Project-Id-Version": "CrossVMAF Studio",
            "MIME-Version": "1.0",
            "Content-Type": "text/plain; charset=UTF-8",
            "Content-Transfer-Encoding": "8bit",
        },
        ["# Translation template for CrossVMAF Studio.",
         "# Generated by tools/i18n/studio_i18n.py update; do not edit."],
        messages)


def merge(template: list[Message], catalog: Catalog) -> tuple[Catalog, list[Message]]:
    existing = {message.key: message for message in catalog.messages}
    merged = []
    for message in template:
        previous = existing.pop(message.key, None)
        merged.append(Message(
            message.context, message.source, message.plural, list(message.references),
            list(previous.translations) if previous else [],
            list(previous.comments) if previous else [],
            list(previous.flags) if previous else []))
    return Catalog(catalog.header, catalog.header_comments, merged), list(existing.values())


# ---------------------------------------------------------------- commands


def command_update(_: argparse.Namespace) -> int:
    messages = extract()
    write_po(TEMPLATE, template_catalog(messages), template=True)
    print(f"{display_path(TEMPLATE)}: {len(messages)} messages")
    for path in translation_catalogs():
        merged, removed = merge(messages, parse_po(path))
        write_po(path, merged)
        missing = sum(1 for m in merged.messages if not is_translated(merged, m))
        print(f"{display_path(path)}: {missing} untranslated")
        for message in removed:
            print(f"  removed: {message.source!r}")
    return 0


def is_translated(catalog: Catalog, message: Message) -> bool:
    if message.fuzzy:
        return False
    forms = catalog.plural_count() if message.plural else 1
    return len(message.translations) >= forms and all(message.translations[:forms])


def command_check(_: argparse.Namespace) -> int:
    problems: list[str] = []
    messages = extract()
    expected = {message.key for message in messages}
    if not TEMPLATE.exists():
        problems.append(f"{TEMPLATE}: missing; run update")
    else:
        template = {message.key for message in parse_po(TEMPLATE).messages}
        if template != expected:
            problems.append(
                f"{display_path(TEMPLATE)} is stale "
                f"({len(expected - template)} new, {len(template - expected)} removed); run update")
    catalogs = translation_catalogs()
    if not catalogs:
        problems.append("no translation catalogs found")
    for path in catalogs:
        name = display_path(path)
        catalog = parse_po(path)
        if "Plural-Forms" not in catalog.header:
            problems.append(f"{name}: header lacks Plural-Forms")
        plural_expression_to_cpp(catalog.plural_expression(), name)
        present = {message.key for message in catalog.messages}
        for key in sorted(expected - present):
            problems.append(f"{name}: missing {key[1]!r}; run update")
        for key in sorted(present - expected):
            problems.append(f"{name}: obsolete {key[1]!r}; run update")
        for message in catalog.messages:
            if message.key not in expected:
                continue
            if not is_translated(catalog, message):
                problems.append(f"{name}: untranslated {message.source!r}")
                continue
            source = comparable_placeholders(message.plural or message.source)
            for translation in message.translations:
                if comparable_placeholders(translation) != source:
                    problems.append(
                        f"{name}: placeholders of {translation!r} do not match {message.source!r}")
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        return 1
    print(f"{len(messages)} messages; {len(catalogs)} complete catalog(s)")
    return 0


def command_init(arguments: argparse.Namespace) -> int:
    language = arguments.language
    path = CATALOG_DIRECTORY / language / "LC_MESSAGES" / f"{DOMAIN}.po"
    if path.exists():
        raise CatalogError(f"{path} already exists")
    plural_forms = arguments.plural_forms or KNOWN_PLURAL_FORMS.get(language.split("_")[0])
    if plural_forms is None:
        raise CatalogError(f"pass --plural-forms for {language}")
    header = dict(template_catalog([]).header)
    header["Language"] = language
    header["Plural-Forms"] = plural_forms
    catalog = Catalog(header, [f"# CrossVMAF Studio translation: {language}."], [])
    merged, _ = merge(extract(), catalog)
    write_po(path, merged)
    print(f"created {display_path(path)}; "
          "register the language in apps/studio/src/i18n/language.cpp")
    return 0


PLURAL_EXPRESSION = re.compile(r"^[n0-9\s()?:%<>=!&|]+$")


def plural_expression_to_cpp(expression: str, where: str) -> str:
    if not PLURAL_EXPRESSION.match(expression):
        raise CatalogError(f"{where}: unsupported Plural-Forms expression {expression!r}")
    return expression


def cpp_string(text: str) -> str:
    data = text.encode("utf-8")
    output = []
    for index, byte in enumerate(data):
        character = chr(byte)
        if character in '"\\' or byte < 0x20 or byte >= 0x7F:
            output.append(f"\\{byte:03o}")
            # MSVC C4125 (decimal digit terminates octal escape): an octal
            # escape followed by an ASCII digit — "\2102", "\24593" — warns,
            # and /WX turns it into an error. Split the literal so the digit
            # starts a fresh one; adjacent literals concatenate to the same
            # bytes.
            if index + 1 < len(data) and 0x30 <= data[index + 1] <= 0x39:
                output.append('" "')
        else:
            output.append(character)
    return '"' + "".join(output) + '"'


def command_generate_cpp(arguments: argparse.Namespace) -> int:
    lines = [
        "// Generated by tools/i18n/studio_i18n.py generate-cpp. Do not edit.",
        "// Source: apps/studio/i18n/*/LC_MESSAGES/crossvmaf_studio.po",
        "",
        "namespace crossvmaf::studio::i18n::catalog {",
    ]
    names = []
    for path in sorted(arguments.catalogs):
        catalog = parse_po(Path(path))
        name = Path(path).parent.parent.name
        if not re.match(r"^[A-Za-z]{2,3}(_[A-Za-z0-9]+)?$", name):
            raise CatalogError(f"{path}: unexpected catalog directory {name!r}")
        names.append(name)
        expression = plural_expression_to_cpp(catalog.plural_expression(), path)
        entries = [m for m in catalog.messages if is_translated(catalog, m)]
        entries.sort(key=lambda m: tuple(part.encode("utf-8") for part in m.key))
        lines += [
            "",
            f"[[nodiscard]] constexpr std::size_t {name}_plural_form(std::uint64_t n) noexcept {{",
            "    static_cast<void>(n);",
            f"    return static_cast<std::size_t>({expression});",
            "}",
            "",
            f"inline constexpr std::array<Entry, {len(entries)}> {name}_entries{{{{",
        ]
        for message in entries:
            forms = ", ".join(cpp_string(t) for t in message.translations)
            lines.append(f"    {{{cpp_string(message.context)}, {cpp_string(message.source)}, "
                         f"{cpp_string(message.plural)}, {{{forms}}}}},")
        lines.append("}};")
    lines += [
        "",
        f"inline constexpr std::array<Catalog, {len(names)}> generated_catalogs{{{{",
        *[f"    {{{cpp_string(name)}, &{name}_plural_form, {name}_entries}}," for name in names],
        "}};",
        "",
        "}  // namespace crossvmaf::studio::i18n::catalog",
        "",
    ]
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("update", help="re-extract and merge catalogs").set_defaults(run=command_update)
    commands.add_parser("check", help="verify catalogs are current and complete").set_defaults(run=command_check)
    init = commands.add_parser("init", help="start a catalog for a new language")
    init.add_argument("language", help="gettext locale name, for example ja or pt_BR")
    init.add_argument("--plural-forms", help='for example "nplurals=2; plural=(n != 1);"')
    init.set_defaults(run=command_init)
    generate = commands.add_parser("generate-cpp", help="emit C++ tables")
    generate.add_argument("--output", required=True)
    generate.add_argument("catalogs", nargs="*")
    generate.set_defaults(run=command_generate_cpp)
    arguments = parser.parse_args()
    try:
        return arguments.run(arguments)
    except CatalogError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
