"""Build and smoke-test an archive for the current OS/CPU (Python 3.12+)."""

import argparse
from importlib.metadata import distribution
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import tomllib

from smoke_executable import check_archive


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    parser.add_argument("--work", type=Path, default=ROOT / "build" / "executable")
    args = parser.parse_args()
    output, work = args.output.resolve(), args.work.resolve()
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    system = {"win32": "windows", "darwin": "macos", "linux": "linux"}[sys.platform]
    machine = platform.machine().lower()
    arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(machine, machine)
    name = f"DichromaticMap-{version}-{system}-{arch}"
    environment = dict(os.environ, PYQTGRAPH_QT_LIB="PySide6")
    subprocess.run([
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--distpath", str(work / "frozen"), "--workpath", str(work / "pyinstaller"),
        str(ROOT / "packaging" / "DichromaticMap.spec"),
    ], cwd=ROOT, env=environment, check=True)
    with tempfile.TemporaryDirectory(prefix="staging-", dir=work) as directory:
        staging = Path(directory)
        bundle = staging / name
        if sys.platform == "darwin":
            bundle.mkdir()
            shutil.copytree(work / "frozen" / "DichromaticMap.app", bundle / "DichromaticMap.app", symlinks=True)
        else:
            shutil.copytree(work / "frozen" / "DichromaticMap", bundle, symlinks=True)
        shutil.copy2(ROOT / "LICENSE", bundle)
        shutil.copy2(ROOT / "docs" / "releases" / f"v{version}.md", bundle / "RELEASE_NOTES.md")
        shutil.copy2(ROOT / "packaging" / "README.txt", bundle)
        shutil.copy2(ROOT / "packaging" / "THIRD_PARTY_NOTICES.txt", bundle)
        shutil.copytree(ROOT / "packaging" / "licenses", bundle / "licenses")
        python_license = next(path for path in (
            Path(sysconfig.get_path("stdlib")) / "LICENSE.txt",
            Path(sys.base_prefix) / "LICENSE.txt",
        ) if path.is_file())
        shutil.copy2(python_license, bundle / "licenses" / "Python.txt")
        pyinstaller = distribution("pyinstaller")
        bootloader_license = next(path for path in pyinstaller.files if path.name == "COPYING.txt")
        shutil.copy2(pyinstaller.locate_file(bootloader_license), bundle / "licenses" / "PyInstaller.txt")
        if sys.platform == "darwin":
            archive = output / f"{name}.zip"
            subprocess.run([
                "ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(bundle), str(archive),
            ], check=True)
        else:
            archive = Path(shutil.make_archive(
                str(output / name), "zip" if sys.platform == "win32" else "gztar",
                root_dir=staging, base_dir=name,
            ))
    print(f"Checking extracted archive: {archive}", flush=True)
    check_archive(archive)
    print(f"Built and verified: {archive}")


if __name__ == "__main__":
    main()
