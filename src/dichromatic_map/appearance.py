"""Qt-free grain colours, unique marker identifiers and per-layer size scales."""

from __future__ import annotations

import math
from numbers import Real
import re

from .crystal import MAX_LAYERS

DEFAULT_GRAIN_COLORS = ("#1677d2", "#e35d35")
DEFAULT_LAYER_SIZE_SCALE = 1.0
MIN_LAYER_SIZE_SCALE = 0.25
MAX_LAYER_SIZE_SCALE = 4.0
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


def default_layer_size_scales(count: int) -> list[float]:
    """Retain the established marker diameter for every layer by default."""
    return [DEFAULT_LAYER_SIZE_SCALE] * _layer_count(count)


def _valid_size_scale(value) -> bool:
    return (
        isinstance(value, Real) and not isinstance(value, bool)
        and MIN_LAYER_SIZE_SCALE <= value <= MAX_LAYER_SIZE_SCALE
        and math.isfinite(value)
    )


def validate_layer_size_scales(scales, count: int) -> list[float]:
    """Validate and copy finite diameter multipliers, one per geometry layer."""
    count = _layer_count(count)
    if not isinstance(scales, (list, tuple)) or len(scales) != count:
        raise ValueError("Layer size scales must match the geometry layer count")
    if any(not _valid_size_scale(scale) for scale in scales):
        raise ValueError(
            "Layer size scales must be finite numbers between "
            f"{MIN_LAYER_SIZE_SCALE:g} and {MAX_LAYER_SIZE_SCALE:g}"
        )
    return [float(scale) for scale in scales]


def resize_layer_size_scales(existing, count: int) -> list[float]:
    """Retain valid layer sizes, using the original diameter for new layers."""
    result = default_layer_size_scales(count)
    for index, scale in enumerate(existing[:count]):
        if _valid_size_scale(scale):
            result[index] = float(scale)
    return result


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
