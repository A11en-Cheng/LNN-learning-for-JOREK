"""Minimal Fortran namelist reader for the JOREK run files.

Only what the parameter channel needs is supported: scalar assignments,
indexed scalar assignments (``D_perp(3) = 0.13d0``), range array constructors
(``ZK_i_perp(1:5) = (/ ... /)``), quoted strings, Fortran ``d``-exponent
literals and ``!`` inline comments.  Block markers (``&name`` / ``/``),
``*``-prefixed header lines and unparseable junk are ignored.  Keys are
normalised to lower case because Fortran namelists are case-insensitive.
"""

from __future__ import annotations

import re
from typing import Any


_ARRAY_CONSTRUCTOR_RE = re.compile(r"\(\s*/\s*(.*?)\s*/\)", re.S)
_INDEXED_LHS_RE = re.compile(r"([a-zA-Z_]\w*)\s*\(\s*(\d+)\s*(?::\s*(\d+)\s*)?\)")
_ASSIGN_RE = re.compile(
    r"([a-zA-Z_]\w*(?:\s*\(\s*\d+\s*(?::\s*\d+\s*)?\))?)\s*=\s*(.+)"
)
_PLAIN_NUMBER_RE = re.compile(
    r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdDqQ][+-]?\d+)?"
)


def _to_float(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e").replace("Q", "E").replace("q", "e"))


def _parse_value(text: str) -> Any:
    text = text.strip()
    if len(text) >= 2 and text[0] in {"'", '"'} and text[-1] == text[0]:
        return text[1:-1].strip()
    constructor = _ARRAY_CONSTRUCTOR_RE.search(text)
    if constructor is not None:
        return [_to_float(item) for item in constructor.group(1).split(",") if item.strip()]
    if _PLAIN_NUMBER_RE.fullmatch(text):
        return _to_float(text)
    return text


def _store_array(values: dict[str, list], name: str, start: int, end: int | None, payload: Any) -> None:
    array = values.setdefault(name, [])
    if isinstance(payload, list):
        stop = end if end is not None else start + len(payload) - 1
        while len(array) < stop:
            array.append(0.0)
        for offset, item in enumerate(payload, start=start - 1):
            if offset >= len(array):
                array.append(item)
            else:
                array[offset] = item
        return
    while len(array) < start:
        array.append(0.0)
    array[start - 1] = payload


def parse_namelist(text: str) -> dict[str, Any]:
    """Parse a Fortran namelist body into a lower-case key dictionary."""
    values: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("!", 1)[0].strip()
        if not line or line.startswith(("&", "/", "*")):
            continue
        constructors: list[str] = []

        def stash(match: re.Match) -> str:
            constructors.append(match.group(0))
            return f"\x00C{len(constructors) - 1}\x00"

        masked = _ARRAY_CONSTRUCTOR_RE.sub(stash, line)
        for part in masked.split(","):
            part = re.sub(
                r"\x00C(\d+)\x00",
                lambda match: constructors[int(match.group(1))],
                part,
            ).strip()
            if not part:
                continue
            assignment = _ASSIGN_RE.search(part)
            if assignment is None:
                continue
            lhs, rhs = assignment.group(1).strip(), assignment.group(2).strip()
            indexed = _INDEXED_LHS_RE.fullmatch(lhs)
            if indexed is not None:
                name = indexed.group(1).lower()
                start = int(indexed.group(2))
                end = int(indexed.group(3)) if indexed.group(3) else None
                _store_array(values, name, start, end, _parse_value(rhs))
            else:
                values[lhs.lower()] = _parse_value(rhs)
    return values
