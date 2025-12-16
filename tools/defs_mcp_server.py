#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025  Rami Ardati

import ast
from pathlib import Path
from typing import Literal, List
from typing_extensions import TypedDict

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("code-outline", json_response=True)


Kind = Literal["class", "def", "async def"]
Scope = Literal["top", "all"]


class SymbolRange(TypedDict):
    path: str
    kind: Kind
    name: str
    qualname: str
    start_line: int
    end_line: int
    signature: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _start_line_including_decorators(node: ast.AST) -> int:
    linenos = [getattr(node, "lineno", 1)]
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        for d in decorators:
            d_lineno = getattr(d, "lineno", None)
            if d_lineno is not None:
                linenos.append(d_lineno)
    return min(linenos)


def _kind(node: ast.AST) -> Kind:
    if isinstance(node, ast.ClassDef):
        return "class"
    if isinstance(node, ast.AsyncFunctionDef):
        return "async def"
    return "def"


def _collect_symbols(path: Path, scope: Scope) -> List[SymbolRange]:
    source = _read_text(path)
    lines = source.splitlines()
    tree = ast.parse(source, filename=str(path))

    symbols: List[SymbolRange] = []

    def add_node(node: ast.AST, parents: List[str]) -> None:
        name = getattr(node, "name", "<unknown>")
        qualname = ".".join(parents + [name]) if parents else name

        start = _start_line_including_decorators(node)
        end = getattr(node, "end_lineno", None) or getattr(node, "lineno", start) or start

        # "signature" = the `def/class ...` line (not decorator line)
        sig_line_no = getattr(node, "lineno", start)
        signature = lines[sig_line_no - 1].rstrip() if 1 <= sig_line_no <= len(lines) else ""

        symbols.append(
            {
                "path": str(path),
                "kind": _kind(node),
                "name": name,
                "qualname": qualname,
                "start_line": int(start),
                "end_line": int(end),
                "signature": signature,
            }
        )

    def walk(node: ast.AST, parents: List[str]) -> None:
        for child in ast.iter_child_nodes(node):
            is_sym = isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            if not is_sym:
                if scope == "all":
                    walk(child, parents)
                continue

            add_node(child, parents)

            if scope == "all":
                # include methods + nested defs; qualname is built by pushing current symbol name
                child_name = getattr(child, "name", "<unknown>")
                walk(child, parents + [child_name])

    if scope == "top":
        # top-level only = only Module.body symbols, no recursion
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                add_node(node, [])
    else:
        walk(tree, [])

    symbols.sort(key=lambda s: (s["start_line"], s["end_line"], s["qualname"]))
    return symbols


@mcp.tool()
def list_defs(path: str, scope: Scope = "top") -> List[SymbolRange]:
    """
    Return (start_line, end_line) for defs/classes.

    :param path: Absolute or relative path to a Python file.
    :type path: str
    :param scope: "top" for top-level only, "all" to include methods/nested defs.
    :type scope: Literal["top", "all"]
    :return: List of symbols with line ranges.
    :rtype: List[SymbolRange]
    """
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    return _collect_symbols(file_path.resolve(), scope)


@mcp.tool()
def read_lines(path: str, start_line: int, end_line: int, with_line_numbers: bool = True) -> str:
    """
    Read a slice of a file so Copilot can fetch only what it needs.

    :param path: Absolute or relative path to a file.
    :type path: str
    :param start_line: 1-based start line (inclusive).
    :type start_line: int
    :param end_line: 1-based end line (inclusive).
    :type end_line: int
    :param with_line_numbers: Prefix each line with "NNNN: ".
    :type with_line_numbers: bool
    :return: The requested slice of text.
    :rtype: str
    """
    if start_line < 1 or end_line < start_line:
        raise ValueError("Invalid line range")

    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    lines = _read_text(file_path.resolve()).splitlines(True)  # keep newlines
    n = len(lines)

    start_idx = max(0, start_line - 1)
    end_idx = min(n, end_line)

    # guardrail against accidental huge pulls
    max_lines = 4000
    if (end_idx - start_idx) > max_lines:
        raise ValueError(f"Requested too many lines: {end_idx - start_idx} (max {max_lines})")

    chunk = lines[start_idx:end_idx]
    if not with_line_numbers:
        return "".join(chunk)

    out = []
    width = len(str(end_idx))
    for i, l in enumerate(chunk, start=start_line):
        out.append(f"{i:0{width}d}: {l}")
    return "".join(out)


if __name__ == "__main__":
    mcp.run(transport="stdio")
