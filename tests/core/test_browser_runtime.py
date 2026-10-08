"""The browser smoke cache remains pinned and validates runtime package bytes."""

import hashlib
import json
from pathlib import Path
import runpy

import pytest

smoke = runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts/smoke_browser.py"))


def runtime(tmp_path):
    site, cache = tmp_path / "site", tmp_path / "cache"
    site.mkdir()
    (site / "use_worker.js").write_text('import {loadPyodide} from "https://cdn.example/pyodide/v1/full/pyodide.mjs";')
    directory = cache / "v1"
    directory.mkdir(parents=True)
    for name in smoke["CORE_ASSETS"]:
        (directory / name).write_bytes(b"runtime")
    wheel = b"validated numpy bytes"
    lock = {"packages": {"numpy": {"depends": [], "file_name": "numpy.whl", "sha256": hashlib.sha256(wheel).hexdigest()}}}
    (directory / "pyodide-lock.json").write_text(json.dumps(lock))
    (directory / "numpy.whl").write_bytes(wheel)
    return site, cache, directory


def test_browser_runtime_offline_cache_uses_the_worker_pin(tmp_path):
    site, cache, directory = runtime(tmp_path)
    base, actual, assets = smoke["prepare_runtime"](site, cache, offline=True)
    assert base == "https://cdn.example/pyodide/v1/full/"
    assert actual == directory
    assert "numpy.whl" in assets and set(smoke["CORE_ASSETS"]) <= assets


@pytest.mark.parametrize("asset", ["numpy.whl", "pyodide.asm.wasm"])
def test_browser_runtime_offline_cache_rejects_missing_or_corrupt_assets(tmp_path, asset):
    site, cache, directory = runtime(tmp_path)
    if asset == "numpy.whl":
        (directory / asset).write_bytes(b"wrong package")
    else:
        (directory / asset).unlink()
    with pytest.raises(RuntimeError, match="Missing or corrupt cached"):
        smoke["prepare_runtime"](site, cache, offline=True)
