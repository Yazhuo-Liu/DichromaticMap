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


def check_angle_drag(page, output):
    """Drive the native range control and inspect real scheduled drawing."""
    page.locator("#angle-slider").scroll_into_view_if_needed()
    page.evaluate("""() => {
      const probe = window.__sliderSmoke = {marker, image: ctx.drawImage,
        toBlob: HTMLCanvasElement.prototype.toBlob, renderer: gpuRenderer,
        exports: [], frame: {markers: [[], []], gpu: false}};
      marker = function(...args) {
        if (args[0] === ctx) {
          if (args[5] === '#2e6799') probe.frame.markers[0].push([args[1], args[2]]);
          else if (args[5] === state.colors[1]) probe.frame.markers[1].push([args[1], args[2]]);
        }
        return probe.marker(...args);
      };
      ctx.drawImage = function(...args) {
        if (args[0] === probe.renderer.canvas) probe.frame.gpu = true;
        return probe.image.apply(this, args);
      };
      HTMLCanvasElement.prototype.toBlob = function(...args) {
        const exact = currentPattern() && state.angle === pattern.angle;
        probe.exports.push({exact, angle: state.angle});
        if (!exact) throw new Error('PNG copied an angle preview instead of the exact result');
        return probe.toBlob.apply(this, args);
      };
    }""")

    def thumb_position(value):
        return page.locator("#angle-slider").evaluate("""(input, value) => {
          const rect = input.getBoundingClientRect();
          const fraction = (value - Number(input.min)) / (Number(input.max) - Number(input.min));
          return [rect.left + 8 + (rect.width - 16) * fraction, rect.top + rect.height / 2];
        }""", value)

    def start_drag():
        value = page.locator("#angle-slider").input_value()
        page.mouse.move(*thumb_position(float(value)))
        page.mouse.down()

    def move_and_check(value):
        page.evaluate("() => { __sliderSmoke.frame = {markers: [[], []], gpu: false}; }")
        page.mouse.move(*thumb_position(value))
        return page.evaluate("""async () => {
          // Never draw manually: exercise the input handler's frame scheduling.
          const checkPaint = () => {
            const probe = __sliderSmoke;
            if (probe.gpu ? !probe.frame.gpu : probe.frame.markers.some(points => !points.length))
              throw new Error('Scheduled angle preview did not paint both grains');
            // Check each sampled frame's real pixels as well as draw calls.
            // The central region excludes the reference-axes inset and labels.
            const r = plotRect(), ratio = canvas.width / canvas.clientWidth;
            const size = Math.floor(Math.min(192, r.width / 2, r.height / 2) * ratio);
            const pixels = ctx.getImageData(Math.floor((r.left + r.width / 2) * ratio - size / 2),
              Math.floor((r.top + r.height / 2) * ratio - size / 2), size, size).data;
            let blue = 0, orange = 0;
            for (let at = 0; at < pixels.length; at += 4) {
              if (pixels[at + 2] > pixels[at] + 35 && pixels[at + 2] > pixels[at + 1] + 15) blue++;
              if (pixels[at] > pixels[at + 1] + 45 && pixels[at + 1] > pixels[at + 2] + 10) orange++;
            }
            if (blue < 10 || orange < 10) throw new Error('An angle-drag frame lost visible grain pixels');
          };
          await new Promise(requestAnimationFrame); checkPaint();
          await new Promise(requestAnimationFrame); checkPaint();
          if (!angleSliderDragging || currentPattern()) throw new Error('Native drag did not remain a preview');
          const data = previewVisibleData(), probe = __sliderSmoke;
          const positions = data.atoms.map((record, grain) => {
            if (!record.count) throw new Error('A grain disappeared during angle dragging');
            const sample = probe.samples[grain];
            let projected = -1;
            for (let index = 0; index < record.count; index++)
              if (record.indices[index] === sample.index) { projected = index; break; }
            if (projected < 0) throw new Error('An interior atom disappeared from the preview');
            const key = Array.from(record.rows.flat.slice(sample.index * 6 + 2, sample.index * 6 + 6));
            if (JSON.stringify(key) !== JSON.stringify(sample.key)) throw new Error('Preview changed physical atom identity');
            const expected = screen(rotate(sample.position, (grain ? -1 : 1) * (state.angle - probe.angle) / 2));
            const point = [record.x[projected], record.y[projected]];
            if (!point.every(Number.isFinite) || Math.hypot(point[0] - expected[0], point[1] - expected[1]) > 2e-8)
              throw new Error('Preview coordinates do not follow the physical grain rotation');
            if (probe.previous && state.angle !== probe.previous.angle &&
                Math.hypot(point[0] - probe.previous.positions[grain][0], point[1] - probe.previous.positions[grain][1]) < 1e-5)
              throw new Error('A grain stayed frozen while the angle slider moved');
            if (probe.gpu) {
              // Query the actual vertex layout and uploaded GL buffer. This
              // does not depend on the renderer's private CPU staging cache.
              const gl = probe.renderer.canvas.getContext('webgl2');
              const location = gl.getAttribLocation(gl.getParameter(gl.CURRENT_PROGRAM), 'a_position');
              const buffer = gl.getVertexAttrib(location, gl.VERTEX_ATTRIB_ARRAY_BUFFER_BINDING);
              const stride = gl.getVertexAttrib(location, gl.VERTEX_ATTRIB_ARRAY_STRIDE) || 8;
              const offset = gl.getVertexAttribOffset(location, gl.VERTEX_ATTRIB_ARRAY_POINTER);
              const previousBuffer = gl.getParameter(gl.ARRAY_BUFFER_BINDING), uploaded = new Float32Array(2);
              const ordinal = projected + (grain ? data.atoms[0].count : 0);
              try {
                gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
                gl.getBufferSubData(gl.ARRAY_BUFFER, offset + ordinal * stride, uploaded);
              } finally { gl.bindBuffer(gl.ARRAY_BUFFER, previousBuffer); }
              if (uploaded[0] !== Math.fround(point[0]) || uploaded[1] !== Math.fround(point[1]))
                throw new Error('GPU displayed stale angle-preview coordinates');
            } else if (!probe.frame.markers[grain].some(drawn => Math.hypot(drawn[0] - point[0], drawn[1] - point[1]) < 2e-8)) {
              throw new Error('Canvas displayed stale angle-preview coordinates');
            }
            return point;
          });
          const origin = screen([0, 0]);
          if (nearestAtom(...positions[0]) !== null || nearestCommon(...origin) !== null)
            throw new Error('An approximate angle preview was selectable');
          const oldMode = state.mode, count = state.atoms.length;
          state.mode = 'vector';
          try { await selectAt(...positions[0]); } finally { state.mode = oldMode; }
          if (state.atoms.length !== count) throw new Error('Picking committed a preview atom');
          probe.previous = {angle: state.angle, positions};
          return {angle: state.angle, points: data.atoms.map(record => record.count)};
        }""")

    def check_exact():
        return page.evaluate("""async () => {
          await waitForCurrentRender();
          if (angleSliderDragging || !currentPattern() || state.angle !== pattern.angle || !coveredView(renderCoverage))
            throw new Error('Released angle slider did not settle to the exact current result');
          const data = visibleData();
          if (data.atoms.some(record => !record.count) || nearestAtom(data.atoms[0].x[0], data.atoms[0].y[0]) === null)
            throw new Error('Exact drawing or picking did not recover after slider release');
          return state.angle;
        }""")

    checks = []
    try:
        for scale, gpu in ((1, False), (5, True)):
            page.locator("#field-slider").evaluate("""(input, scale) => {
              input.value = String(scale); input.dispatchEvent(new Event('input', {bubbles:true}));
            }""", scale)
            page.evaluate("async () => { await waitForCurrentRender(); }")
            maximum = page.locator("#angle-slider").evaluate("input => Number(input.max)")
            page.evaluate("""gpu => {
              const probe = __sliderSmoke, data = visibleData();
              probe.gpu = gpu; probe.angle = state.angle; probe.previous = null;
              probe.source = pattern; probe.originals = pattern.grains.map(rows => rows.flat.slice());
              const radius = Math.min(state.width, state.height) / 5;
              probe.samples = data.atoms.map(record => {
                for (let cursor = 0; cursor < record.count; cursor++) {
                  const index = record.indices[cursor], row = record.rows.flat.slice(index * 6, index * 6 + 6);
                  const distance = Math.hypot(row[0], row[1]);
                  if (distance > .25 && distance < radius)
                    return {index, position: Array.from(row.slice(0, 2)), key: Array.from(row.slice(2))};
                }
                throw new Error('No interior atom is available for angle-drag regression');
              });
            }""", gpu)
            start_drag()
            frames = [move_and_check(maximum * fraction) for fraction in (.24, .265, .29, .315, .34)]
            page.evaluate("""() => {
              const probe = __sliderSmoke;
              if (probe.source.grains.some((rows, grain) => !rows.flat.every((value, index) => Object.is(value, probe.originals[grain][index]))))
                throw new Error('Angle preview modified the scientific Float64 source');
            }""")
            page.mouse.up()
            released_angle = check_exact()

            # A separate held drag verifies export waits for the release and
            # exact Worker result, instead of silently exporting preview rows.
            page.evaluate("""() => {
              const probe = __sliderSmoke, data = visibleData();
              probe.angle = state.angle; probe.previous = null;
              probe.samples = probe.samples.map((sample, grain) => {
                const rows = data.atoms[grain].rows;
                for (let index = 0; index < rows.length; index++) {
                  const row = rows.row(index);
                  if (JSON.stringify(row.slice(2)) === JSON.stringify(sample.key))
                    return {...sample, index, position: row.slice(0, 2)};
                }
                throw new Error('Exact result lost the test atom');
              });
              probe.exports = [];
            }""")
            start_drag()
            move_and_check(maximum * .365)
            with page.expect_download() as exported:
                page.locator("#export-png").evaluate("button => button.click()")
                page.evaluate("""async () => {
                  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                  if (__sliderSmoke.exports.length) throw new Error('PNG exported before slider release');
                }""")
                page.mouse.up()
                angle = check_exact()
            png = output / ("slider-gpu.png" if gpu else "slider-canvas.png")
            exported.value.save_as(png)
            assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            snapshots = page.evaluate("() => __sliderSmoke.exports")
            assert len(snapshots) == 1 and snapshots[0]["exact"] and snapshots[0]["angle"] == angle
            checks.append({"renderer": "gpu" if gpu else "canvas", "frames": frames,
                           "releaseAngle": released_angle, "exactAngle": angle, "png": png.name})
    finally:
        page.mouse.up()
        page.evaluate("""() => {
          const probe = __sliderSmoke;
          marker = probe.marker; ctx.drawImage = probe.image;
          HTMLCanvasElement.prototype.toBlob = probe.toBlob;
          delete window.__sliderSmoke;
        }""")
    return checks


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
    drag = check_angle_drag(page, output)
    return {"buffers": buffers, "cancellation": cancellation, "session": "restored", "png": ["pattern.png", "atoms.png"], "gpu": gpu, "angleDrag": drag}


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
            print("Real Chromium/Pyodide startup, Float64 parity, cancellation, session import, PNG, native angle dragging and twelve GPU markers passed.")
        except Exception:
            page.screenshot(path=str(output / "failure.png"), full_page=True)
            (output / "failure.txt").write_text(page.locator("#status").inner_text() + "\n" + "\n".join(errors + console), encoding="utf-8")
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
