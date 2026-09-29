"""Do not assemble a release with missing or stale binaries."""

import hashlib
from pathlib import Path
import runpy

import pytest

tomllib = pytest.importorskip("tomllib", reason="Release tools require Python 3.12+")
ROOT = Path(__file__).resolve().parents[2]
write_checksums = runpy.run_path(str(ROOT / "scripts" / "release_checksums.py"))["write_checksums"]
pytestmark = pytest.mark.packaging


def asset_names():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    return [
        f"dichromatic_map-{version}-py3-none-any.whl", f"dichromatic_map-{version}.tar.gz",
        f"DichromaticMap-{version}-linux-x86_64.tar.gz",
        *(f"DichromaticMap-{version}-{target}.zip" for target in (
            "windows-x86_64", "macos-arm64", "macos-x86_64",
        )),
    ]


def test_complete_release_checksums_and_repeat(tmp_path):
    (tmp_path / "older-release").mkdir()
    for index, name in enumerate(asset_names()):
        (tmp_path / name).write_bytes(bytes([index]) * 200)
    checksum_file = write_checksums(tmp_path, complete=True)
    lines = checksum_file.read_text().splitlines()
    assert len(lines) == 6
    for line in lines:
        digest, name = line.split("  ", 1)
        assert digest == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    write_checksums(tmp_path, complete=True)
    assert checksum_file.read_text().splitlines() == lines


def test_partial_build_is_allowed_but_not_a_complete_release(tmp_path):
    (tmp_path / asset_names()[0]).write_bytes(b"wheel")
    write_checksums(tmp_path)
    with pytest.raises(ValueError, match="Unexpected or missing"):
        write_checksums(tmp_path, complete=True)


@pytest.mark.parametrize("unexpected", [None, "dichromatic_map-0.0.0.tar.gz", "unrelated.zip"])
def test_empty_or_stale_assets_are_rejected(tmp_path, unexpected):
    if unexpected is not None:
        (tmp_path / unexpected).write_bytes(b"stale")
    with pytest.raises(ValueError, match="Unexpected or missing"):
        write_checksums(tmp_path)
