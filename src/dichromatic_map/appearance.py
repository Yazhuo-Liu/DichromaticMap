"""Qt-free grain colours and unique, portable layer marker identifiers."""

from __future__ import annotations

import re

from .crystal import MAX_LAYERS

DEFAULT_GRAIN_COLORS = ("#1677d2", "#e35d35")
BASE_SYMBOLS = ("o", "d", "t", "s", "p", "h", "star", "+", "x", "t1", "t2", "t3")
_ALL_SYMBOLS = frozenset(BASE_SYMBOLS) | {
    f"number:{number}" for number in range(1, MAX_LAYERS + 1)
}


def _layer_count(count: int) -> int:
    if type(count) is not int or not 1 <= count <= MAX_LAYERS:
        raise ValueError(f"Layer count must be an integer between 1 and {MAX_LAYERS}")
    return count


def default_layer_symbols(count: int) -> list[str]:
    """Keep established shapes, then use distinct numbered circle markers.

    A marker such as ``number:13`` is a stable identifier, not a layer binding:
    users may assign it to any layer and retain it after changing geometry.
    """
    count = _layer_count(count)
    return list(BASE_SYMBOLS[:count]) + [
        f"number:{number}" for number in range(len(BASE_SYMBOLS) + 1, count + 1)
    ]


def available_layer_symbols(count: int) -> list[str]:
    """Return a compact choice palette with spare markers at every layer count.

    Existing assignments outside this palette remain valid and should also be
    offered by the UI. The global numbered-marker range is 1 through MAX_LAYERS.
    """
    count = _layer_count(count)
    return list(BASE_SYMBOLS) + [
        f"number:{number}" for number in range(1, max(len(BASE_SYMBOLS), count) + 1)
    ]


def validate_appearance(colors, symbols, count: int) -> tuple[list[str], list[str]]:
    """Validate and copy colours and markers, normalizing hex colours to lowercase."""
    count = _layer_count(count)
    if not isinstance(colors, (list, tuple)) or len(colors) != 2:
        raise ValueError("Two grain colors are required")
    if any(not isinstance(color, str) or re.fullmatch(r"#[0-9a-fA-F]{6}", color) is None
           for color in colors):
        raise ValueError("Grain colors must use #RRGGBB hexadecimal notation")
    if not isinstance(symbols, (list, tuple)) or len(symbols) != count:
        raise ValueError("Layer symbols must match the geometry layer count")
    if any(not isinstance(symbol, str) or symbol not in _ALL_SYMBOLS for symbol in symbols):
        raise ValueError("Unknown layer symbol")
    if len(set(symbols)) != count:
        raise ValueError("Each layer must have a unique symbol")
    return [color.lower() for color in colors], list(symbols)


def resize_layer_symbols(existing, count: int) -> list[str]:
    """Preserve valid retained assignments, filling new layers without duplicates.

    Numbered identifiers remain valid after shrinking the layer count. Invalid
    or duplicate entries are replaced with unused defaults.
    """
    count = _layer_count(count)
    result = [None] * count
    used = set()
    for index, symbol in enumerate(existing[:count]):
        if isinstance(symbol, str) and symbol in _ALL_SYMBOLS and symbol not in used:
            result[index] = symbol
            used.add(symbol)
    unused = iter(symbol for symbol in default_layer_symbols(count) if symbol not in used)
    for index, symbol in enumerate(result):
        if symbol is None:
            result[index] = next(unused)
    return result
