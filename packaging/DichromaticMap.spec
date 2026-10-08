"""Native, relocatable Qt application; build with scripts/build_executable.py."""

from importlib.metadata import PackageNotFoundError
from pathlib import Path
import sys
import tomllib

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

root = Path(SPECPATH).parent
version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
datas = collect_data_files("dichromatic_map", includes=["ui/resources/**/*"])
# Keep dependency metadata and its license texts with the redistributed libraries.
# Conda combines the PySide6 distributions that pip ships separately.
for package in ("numpy", "pyqtgraph", "threadpoolctl", "PySide6", "shiboken6",
                "PySide6_Essentials", "PySide6_Addons"):
    try:
        datas += copy_metadata(package)
    except PackageNotFoundError:
        if package not in ("PySide6_Essentials", "PySide6_Addons"):
            raise

analysis = Analysis(
    [str(root / "scripts" / "frozen_entry.py")],
    pathex=[str(root / "src")],
    datas=datas,
    hiddenimports=["PySide6.QtSvg", "threadpoolctl"],
    excludes=["PyQt5", "PyQt6", "PySide2", "matplotlib", "scipy", "IPython", "tkinter"],
)
pyz = PYZ(analysis.pure)
executable = EXE(
    pyz, analysis.scripts, [],
    exclude_binaries=True,
    name="DichromaticMap",
    console=sys.platform not in ("win32", "darwin"),
    strip=False,
    upx=False,
)
collection = COLLECT(
    executable, analysis.binaries, analysis.datas,
    name="DichromaticMap", strip=False, upx=False,
)
if sys.platform == "darwin":
    app = BUNDLE(
        collection,
        name="DichromaticMap.app",
        bundle_identifier="io.github.Yazhuo-Liu.DichromaticMap",
        version=version,
        info_plist={"CFBundleShortVersionString": version, "NSHighResolutionCapable": True},
    )
