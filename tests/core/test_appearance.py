"""Portable appearance preferences stay unique as the layer count changes."""

import pytest

from dichromatic_map.appearance import (
    BASE_SYMBOLS, DEFAULT_GRAIN_COLORS, available_layer_symbols,
    default_layer_symbols, resize_layer_symbols, validate_appearance,
)
from dichromatic_map.state import PatternParameters, PatternState


@pytest.mark.parametrize("count", [1, 2, 3, 12, 13, 38, 256])
def test_defaults_preserve_shapes_and_assign_distinct_markers_to_every_layer(count):
    symbols = default_layer_symbols(count)
    assert len(symbols) == len(set(symbols)) == count
    assert symbols[:12] == list(BASE_SYMBOLS[:count])
    assert symbols[12:] == [f"number:{number}" for number in range(13, count + 1)]
    assert validate_appearance(DEFAULT_GRAIN_COLORS, symbols, count)[1] == symbols
    palette = available_layer_symbols(count)
    assert set(symbols) <= set(palette)
    assert len(set(palette) - set(symbols)) >= 12


def test_resizing_preserves_custom_assignments_and_avoids_new_duplicates():
    # Retaining a marker otherwise used by a later default must reserve it first.
    original = ["d", "number:13", "star", "o"]
    expanded = resize_layer_symbols(original, 38)
    assert expanded[:4] == original
    assert len(expanded) == len(set(expanded)) == 38
    validate_appearance(DEFAULT_GRAIN_COLORS, expanded, 38)
    assert original == ["d", "number:13", "star", "o"]
    assert resize_layer_symbols(expanded, 2) == original[:2]


def test_high_numbered_assignment_survives_smaller_geometry():
    before = ["number:256", "number:38", "t"]
    after = resize_layer_symbols(before, 2)
    assert after == before[:2]
    assert set(after).isdisjoint(available_layer_symbols(2))
    assert validate_appearance(DEFAULT_GRAIN_COLORS, after, 2)[1] == after


def test_resizing_repairs_invalid_entries_without_replacing_valid_later_assignments():
    assert resize_layer_symbols(["d", "d", "o", "bad", None], 5) == ["d", "t", "o", "s", "p"]
    assert resize_layer_symbols([], 256) == default_layer_symbols(256)


def test_color_normalization_and_independent_preference_lists():
    colors = ["#AAbbFF", "#AAbbFF"]  # Only layer symbols must be distinct.
    symbols = ["x", "t3"]
    normalized_colors, normalized_symbols = validate_appearance(colors, symbols, 2)
    assert normalized_colors == ["#aabbff", "#aabbff"]
    normalized_colors[0] = "#000000"
    normalized_symbols[0] = "o"
    assert colors == ["#AAbbFF", "#AAbbFF"] and symbols == ["x", "t3"]
    first = PatternState(PatternParameters())
    second = PatternState(PatternParameters())
    assert first.grain_colors == list(DEFAULT_GRAIN_COLORS)
    assert first.layer_symbols == ["o", "d"]
    first.grain_colors[0] = "#000000"
    first.layer_symbols[0] = "x"
    assert second.grain_colors == list(DEFAULT_GRAIN_COLORS)
    assert second.layer_symbols == ["o", "d"]


@pytest.mark.parametrize("count", [0, -1, 257, True, 2.0, "2", None])
def test_invalid_layer_counts_are_rejected(count):
    for helper in (default_layer_symbols, available_layer_symbols):
        with pytest.raises(ValueError, match="Layer count"):
            helper(count)
    with pytest.raises(ValueError, match="Layer count"):
        resize_layer_symbols([], count)


@pytest.mark.parametrize("colors", [
    None, "#123456", [], ["#123456"], ["#123456"] * 3,
    ["red", "#123456"], ["#123", "#123456"], ["#12345678", "#123456"],
    [" #123456", "#123456"], ["#gggggg", "#123456"], [True, "#123456"],
])
def test_invalid_colors_are_rejected(colors):
    with pytest.raises(ValueError, match="color"):
        validate_appearance(colors, ["o", "d"], 2)


@pytest.mark.parametrize("symbols", [
    None, "od", [], ["o"], ["o", "d", "t"], ["o", "o"],
    ["o", "unknown"], ["o", True], ["o", []], ["o", "number:0"],
    ["o", "number:257"], ["o", "number:01"], ["o", "number:1.0"],
])
def test_invalid_symbol_assignments_are_rejected(symbols):
    with pytest.raises(ValueError, match="symbol"):
        validate_appearance(DEFAULT_GRAIN_COLORS, symbols, 2)
