"""Exercise the built online app with Chromium and its real Pyodide runtime.

Download the version pinned by use_worker.js once, then serve verified cached
assets to the browser. --offline rejects missing assets instead of downloading.
"""

from contextlib import contextmanager
from functools import partial
import argparse
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import runpy
import sys
import tempfile
from threading import Thread
from urllib.request import urlopen
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
CORE_ASSETS = ("pyodide.mjs", "pyodide.asm.mjs", "pyodide.asm.wasm", "python_stdlib.zip")


def runtime_source(site):
    source = (site / "use_worker.js").read_text(encoding="utf-8")
    match = re.search(r'from "(https://[^\"]+/pyodide/(v[^/]+)/full/)pyodide\.mjs"', source)
    if match is None:
        raise ValueError("Cannot find the pinned Pyodide runtime in use_worker.js")
    return match.group(1), match.group(2)


def prepare_runtime(site, cache, *, offline=False):
    base, version = runtime_source(site)
    directory = cache / version
    directory.mkdir(parents=True, exist_ok=True)

    def asset(name, digest=None):
        if Path(name).name != name:
            raise ValueError(f"Unexpected runtime asset path: {name}")
        path = directory / name
        if path.is_file() and (digest is None or hashlib.sha256(path.read_bytes()).hexdigest() == digest):
            return path
        if offline:
            raise RuntimeError(f"Missing or corrupt cached Pyodide asset: {path}")
        with urlopen(base + name, timeout=60) as response:
            data = response.read()
        if digest is not None and hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Pyodide package checksum mismatch: {name}")
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        temporary.replace(path)
        return path

    lock = json.loads(asset("pyodide-lock.json").read_text(encoding="utf-8"))
    packages = lock["packages"]
    names = {"pyodide-lock.json", *CORE_ASSETS}
    for name in CORE_ASSETS:
        asset(name)
    visited = set()

    def package(name):
        if name in visited:
            return
        visited.add(name)
        item = packages[name]
        for dependency in item["depends"]:
            package(dependency)
        asset(item["file_name"], item["sha256"])
        names.add(item["file_name"])

    package("numpy")
    return base, directory, names


@contextmanager
def serve(site):
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(site)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def render_fixture(site):
    # Use exactly the source archive sent to WASM, independently of editable
    # packages on the host. This is a correctness comparison, not a benchmark.
    sys.path.insert(0, str(site / "vendor" / "dichromatic_map.zip"))
    dispatch = runpy.run_path(str(site / "web_bridge.py"))["web_dispatch"]
    request = {"action": "render", "lattice": "FCC", "axis": "110", "angle": 22,
               "width": 12, "height": 9, "center": [.125, -.0625],
               "local_matching": True, "local_distance": .1}
    return {"request": request, "expected": json.loads(dispatch(json.dumps(request)))}


def check_app(page, output, fixture):
    page.wait_for_function("typeof pattern !== 'undefined' && pattern && !document.getElementById('tutorial-start').disabled")
    page.evaluate("async () => { if (typeof waitForCurrentRender === 'function') await waitForCurrentRender(); draw(); }")
    initial = page.evaluate("() => ({angle: state.angle, colors: [...state.colors], axis: state.axis, lattice: state.lattice})")
    buffers = page.evaluate("""async ({request: parameters, expected}) => {
      const raw = await request('render', parameters);
      if (!raw.grains.every(array => array instanceof Float64Array)) throw new Error('Render did not transfer Float64 buffers');
      const actual = window.DichromaticRenderData.decodeRenderResult(raw);
      const compare = (rows, reference, integersFrom) => {
        if (rows.length !== reference.length) throw new Error('Render row count mismatch');
        let index = 0;
        for (const row of rows) {
          for (let column = 0; column < row.length; column++) {
            const target = reference[index][column];
            if (!Number.isFinite(row[column]) || !Number.isFinite(target) ||
                (column >= integersFrom ? row[column] !== target : Math.abs(row[column] - target) > 2e-11))
              throw new Error('Render coordinate or physical index mismatch');
          }
          index++;
        }
      };
      actual.grains.forEach((grain, index) => compare(grain, expected.grains[index], 2));
      compare(actual.coincidences, expected.coincidences, 2);
      compare(actual.local, expected.local, 4);
      return {points: actual.grains.map(grain => grain.length), localPairs: actual.local.length};
    }""", fixture)
    cancellation = page.evaluate("""async () => {
      // The urgent metadata result is an event-loop barrier: the real Worker
      // has accepted the search and services another action between blocks.
      const obsolete = request('near_search', {lattice:'FCC', axis:'1 1 15', angle:22, percent:10, index:40})
        .then(() => 'completed', error => error.name);
      const metadata = await request('metadata', {lattice:'FCC', axis:'110'});
      worker.postMessage({type:'cancel', action:'near_search'});
      const cancelled = await obsolete;
      if (cancelled !== 'AbortError' || metadata.axis !== '110') throw new Error('Worker cancellation or interleaving failed');
      const fresh = await request('near_search', {lattice:'FCC', axis:'110', angle:22, percent:2, index:8});
      if (!fresh.length) throw new Error('Fresh search did not recover after cancellation');
      return {cancelled, freshSolutions:fresh.length};
    }""")
    with page.expect_download() as saved:
        page.locator("#save-session").click()
    session = output / "roundtrip.dmap"
    saved.value.save_as(session)
    with ZipFile(session) as archive:
        assert "session.json" in archive.namelist()
    page.locator("#angle-number").fill("22")
    page.locator("#angle-number").dispatch_event("change")
    page.wait_for_function("state.angle === 22 && pattern.angle === 22 && (typeof currentPattern !== 'function' || currentPattern())")
    page.locator("#session-file").set_input_files(session)
    page.wait_for_function("expected => state.angle === expected.angle && pattern.angle === expected.angle && (typeof currentPattern !== 'function' || currentPattern())", arg=initial)
    restored = page.evaluate("() => ({angle: state.angle, colors: [...state.colors], axis: state.axis, lattice: state.lattice})")
    assert restored == initial, (restored, initial)
    page.locator("#field-slider").evaluate("input => { input.value='5'; input.dispatchEvent(new Event('input', {bubbles:true})); }")
    page.wait_for_function("state.scale === 5 && pattern.grains[0].length > 10000 && awaiting.size === 0")
    page.evaluate("async () => { if (typeof waitForCurrentRender === 'function') await waitForCurrentRender(); draw(); }")
    for clean in (False, True):
        page.locator("#clean-png").set_checked(clean)
        with page.expect_download() as exported:
            page.locator("#export-png").click()
        png = output / ("atoms.png" if clean else "pattern.png")
        exported.value.save_as(png)
        data = png.read_bytes()
        assert data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) > 1000
    page.locator("#axis").select_option("112")
    page.wait_for_function("state.axis === '112' && metadata.layers === 6 && pattern.layers === 6 && (typeof currentPattern !== 'function' || currentPattern())")
    gpu = page.evaluate("""async () => {
      if (typeof waitForCurrentRender === 'function') await waitForCurrentRender();
      const symbols = [['o','d','t','s','p','h'], ['star','+','x','t1','t2','t3']];
      const tested = [];
      for (const group of symbols) {
        const controls = [...document.querySelectorAll('#appearance-layers select')];
        controls.forEach((control, index) => {
          control.value = group[index]; control.dispatchEvent(new Event('change', {bubbles:true}));
        });
        draw();
        const style = JSON.stringify([state.colors, state.symbols, state.sizes, state.scale]);
        if (!gpuRenderer || gpuRenderer.disabled || gpuRenderer.count < 12000 || gpuRenderer.signature !== style)
          throw new Error('Large multilayer plot did not render the common symbols with its real GPU shader');
        tested.push(...group);
      }
      return {symbols: tested, points: gpuRenderer.count};
    }""")
    return {"buffers": buffers, "cancellation": cancellation, "session": "restored", "png": ["pattern.png", "atoms.png"], "gpu": gpu}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, default=ROOT / "site" / "_build")
    parser.add_argument("--runtime-cache", type=Path, default=ROOT / ".cache" / "pyodide")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--chromium", help="Use an installed Chromium executable instead of Playwright's browser")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache" / "browser-smoke")
    args = parser.parse_args()
    site, output = args.site.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base, runtime, assets = prepare_runtime(site, args.runtime_cache.resolve(), offline=args.offline)
    fixture = render_fixture(site)
    from playwright.sync_api import sync_playwright

    errors, console = [], []
    with serve(site) as origin, sync_playwright() as playwright:
        options = {"headless": True, "args": ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]}
        if args.chromium:
            options["executable_path"] = args.chromium
        browser = playwright.chromium.launch(**options)
        context = browser.new_context(viewport={"width": 1440, "height": 900}, service_workers="block")
        context.set_default_timeout(60000)

        def cached_asset(route):
            name = route.request.url.removeprefix(base)
            if name not in assets:
                errors.append(f"Unexpected runtime request: {name}")
                route.abort()
                return
            content_type = "text/javascript" if name.endswith((".js", ".mjs")) else mimetypes.guess_type(name)[0] or "application/octet-stream"
            route.fulfill(body=(runtime / name).read_bytes(), content_type=content_type,
                          headers={"access-control-allow-origin": "*"})

        context.route(base + "**", cached_asset)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: console.append(f"{message.type}: {message.text}"))
        try:
            page.goto(origin + "/use.html")
            report = check_app(page, output, fixture)
            # Keep regression helpers in the checkout, outside the deployed
            # artifact, while testing the actual deployed renderer and marker.
            page.add_script_tag(path=str(site.parent / "gpu_renderer.browser.test.js"))
            report["gpuMaskChecks"] = page.evaluate("""async () =>
              runGPUChecks({marker, createRenderer: window.DichromaticGPU.createRenderer})
            """)
            assert not errors, errors
            report["runtime"] = base
            (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("Real Chromium/Pyodide startup, Float64 parity, cancellation, session import, both PNG modes and twelve GPU markers passed.")
        except Exception:
            page.screenshot(path=str(output / "failure.png"), full_page=True)
            (output / "failure.txt").write_text(page.locator("#status").inner_text() + "\n" + "\n".join(errors + console), encoding="utf-8")
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
