"""Extract a native archive and test it outside the checkout, without PYTHONPATH."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def check_archive(archive):
    archive = Path(archive).resolve()
    with tempfile.TemporaryDirectory(prefix="dmap-executable-check-") as directory:
        outside = Path(directory)
        unpacked = outside / "unpacked"
        unpacked.mkdir()
        if sys.platform == "darwin":
            # Python's zip extraction does not restore framework symbolic links.
            subprocess.run(["ditto", "-x", "-k", str(archive), str(unpacked)], check=True)
            matches = list(unpacked.glob("*/DichromaticMap.app/Contents/MacOS/DichromaticMap"))
        else:
            shutil.unpack_archive(str(archive), str(unpacked))
            name = "DichromaticMap.exe" if sys.platform == "win32" else "DichromaticMap"
            matches = list(unpacked.glob("*/" + name))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one executable in {archive.name}, found {matches}")
        environment = dict(os.environ)
        for name in ("PYTHONHOME", "PYTHONPATH", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
            environment.pop(name, None)
        environment.update(QT_QPA_PLATFORM="offscreen", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        output = outside / "results"
        result = subprocess.run(
            [str(matches[0]), "--self-test", str(output)], cwd=outside,
            env=environment, capture_output=True, text=True, timeout=240,
        )
        error = output / "error.txt"
        if result.returncode or not (output / "result.json").exists():
            raise RuntimeError(result.stdout + result.stderr + (
                error.read_text(encoding="utf-8") if error.exists() else "\nNo self-test report"
            ))
        report = json.loads((output / "result.json").read_text(encoding="utf-8"))
        assert report["status"] == "ok" and report["frozen"] and report["worker_pids"]
        for name in ["completion", *report["lattices"], *(f"{x}-clean" for x in report["lattices"])]:
            png = output / f"{name}.png"
            assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") and png.stat().st_size > 1000
        print(f"Passed: {archive.name}: bundled Qt resources, FCC/BCC/SC, spawned workers, "
              "PNG exports, sessions and completion preview")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    check_archive(parser.parse_args().archive)
