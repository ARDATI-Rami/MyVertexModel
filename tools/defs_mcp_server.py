# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025  Rami Ardati
#!/usr/bin/env python3
import re
from pathlib import Path
from typing import Literal, TypedDict, List

from mcp.server.fastmcp import FastMCP

# Name as it should appear to Copilot
mcp = FastMCP("code-outline", json_response=True)


class Symbol(TypedDict):
    path: str
    line: int
    kind: Literal["class", "def"]
    name: str
    signature: str
    raw: str


# Regex close to your grep:
#   ^(class|def) NAME ...
_DEFINITION_RE = re.compile(
    r"^(?P<kind>class|def)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)[^\n]*"
)


def _scan_file(path: Path) -> List[Symbol]:
    symbols: List[Symbol] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, start=1):
            match = _DEFINITION_RE.match(line)
            if not match:
                continue

            kind = match.group("kind")
            name = match.group("name")
            signature = match.group(0).rstrip("\n")

            symbols.append(
                {
                    "path": str(path),
                    "line": lineno,
                    "kind": kind,
                    "name": name,
                    "signature": signature,
                    "raw": line.rstrip("\n"),
                }
            )

    return symbols


@mcp.tool()
def list_defs(path: str) -> List[Symbol]:
    """
    List top-level `class` and `def` lines in a file with line numbers.

    The `path` can be absolute, or relative to the current working directory
    (your workspace root if started from VS Code).
    """
    file_path = Path(path).expanduser()

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    return _scan_file(file_path.resolve())


if __name__ == "__main__":
    # For Copilot / VS Code, stdio is the simplest
    mcp.run(transport="stdio")
