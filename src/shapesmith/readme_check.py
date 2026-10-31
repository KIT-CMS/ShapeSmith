"""The `shapesmith` commands a text names (a README of the core or of an analysis) checked against the CLI: every named
command exists and takes every option given to it.

A command is `shapesmith <command> ...` in a code block or an inline code span, up to the next command, a comment
(`#`), `&&`, `;` or the end of the line or span; an inline code span may also start with the command itself followed
by an option (`datacards --variant s0`), unless it is a call of the combine tool (` -M `).
"""
from __future__ import annotations

import re

import typer.main

COMMAND = re.compile(r"\bshapesmith\s+([a-z][a-z-]*)")
OPTION = re.compile(r"(?<![\w./-])(--?[A-Za-z][\w-]*)")
INLINE = re.compile(r"`([^`\n]+)`")
FENCE = re.compile(r"^```[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
END = re.compile(r"#|&&|;")


def cli_options(app) -> dict[str, set[str]]:
    """Command -> its options (every spelling, e.g. --config and -c, --systematics and --no-systematics)."""
    group = typer.main.get_command(app)
    return {name: {opt for param in command.params for opt in (*param.opts, *param.secondary_opts)} for name, command in group.commands.items()}


def _segments(text: str) -> list[tuple[str, str]]:
    """(command, the text of its arguments) of every `shapesmith <command>` in `text` (one line or span)."""
    matches = list(COMMAND.finditer(text))
    result = []
    for match, following in zip(matches, matches[1:] + [None]):
        arguments = text[match.end(): following.start() if following else len(text)]
        result.append((match.group(1), END.split(arguments, maxsplit=1)[0]))
    return result


def mentions(readme: str, commands: set[str]) -> list[tuple[str, str, str]]:
    """(command, its options as one string, the line or span it was found in) of every command the text names."""
    found = []
    for block in FENCE.findall(readme):
        for line in block.splitlines():
            found += [(command, arguments, line.strip()) for command, arguments in _segments(line)]
    prose = FENCE.sub("", readme)
    for span in INLINE.findall(prose):
        segments = _segments(span)
        first = span.split()[0] if span.split() else ""
        if not segments and first in commands and " -M " not in span and OPTION.search(span[len(first):]):
            segments = [(first, span[len(first):])]
        found += [(command, arguments, span) for command, arguments in segments]
    return found


def problems(readme: str, app) -> list[str]:
    """The commands of the text that the CLI lacks and the options their command does not take."""
    options = cli_options(app)
    result = []
    for command, arguments, where in mentions(readme, set(options)):
        if command not in options:
            result.append(f"unknown command `shapesmith {command}` in: {where}")
            continue
        unknown = [option for option in OPTION.findall(arguments) if option not in options[command]]
        result += [f"`shapesmith {command}` has no option {option}, in: {where}" for option in unknown]
    return result
