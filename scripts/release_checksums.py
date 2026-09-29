"""Generate checksums; --complete also checks the native release matrix."""

import argparse
import hashlib
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def write_checksums(directory, *, complete=False):
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    expected = {
        f"dichromatic_map-{version}-py3-none-any.whl",
        f"dichromatic_map-{version}.tar.gz",
        f"DichromaticMap-{version}-linux-x86_64.tar.gz",
        *(f"DichromaticMap-{version}-{platform}.zip" for platform in (
            "windows-x86_64", "macos-arm64", "macos-x86_64",
        )),
    }
    files = sorted(path for path in directory.iterdir()
                   if path.is_file() and path.name != "SHA256SUMS.txt")
    names = {path.name for path in files}
    if not files or names - expected or (complete and names != expected):
        raise ValueError(f"Unexpected or missing release assets: found {sorted(names)}, expected {sorted(expected)}")
    lines = []
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        lines.append(f"{digest.hexdigest()}  {path.name}\n")
    target = directory / "SHA256SUMS.txt"
    target.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {target} ({len(files)} assets)")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--complete", action="store_true")
    args = parser.parse_args()
    write_checksums(args.directory, complete=args.complete)
