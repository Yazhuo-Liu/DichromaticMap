"""Exercise the built online app with Chromium and its real Pyodide runtime.

Download the version pinned by use_worker.js once, then serve verified cached
assets to the browser. --offline rejects missing assets instead of downloading.
"""

from contextlib import contextmanager
from functools import partial
import argparse
import base64
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


def check_plot_interactions(page, output):
    """Exercise the actual input owners, resized preview and figure legend."""
    report = {}
    viewport = page.viewport_size
    page.evaluate("""() => {
      document.activeElement?.blur();
      state.center = [4, -3]; state.regions = [true, false, false, true];
    }""")
    page.keyboard.press("Control+c")
    page.keyboard.press("Control+f")
    page.keyboard.press("Escape")
    report["shortcuts"] = page.evaluate("""() => {
      if (state.center[0] !== 4 || state.center[1] !== -3 || state.regions.join() !== 'true,false,false,true')
        throw Error('Browser modifier keys executed application shortcuts');
      document.dispatchEvent(new KeyboardEvent('keydown', {key:'c',isComposing:true,bubbles:true}));
      if (state.center[0] !== 4) throw Error('IME composition executed an application shortcut');
      state.center = [0, 0]; state.regions = [true, true, true, true];
      return 'modifier and composition events preserved the view';
    }""")

    point = page.locator("#map").bounding_box()
    page.mouse.move(point["x"] + 80, point["y"] + 80)
    before = page.evaluate("state.scale")
    page.mouse.wheel(100, 0)
    page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
    assert page.evaluate("state.scale") == before, "Horizontal trackpad motion zoomed the plot"
    report["wheel"] = page.evaluate("""async () => {
      const wheel = delta => canvas.dispatchEvent(new WheelEvent('wheel', {deltaY:delta,cancelable:true}));
      state.scale = 1; wheel(120); const whole = state.scale;
      state.scale = 1; for (let index=0; index<12; index++) wheel(10);
      if (Math.abs(whole-state.scale)>1e-12) throw Error('Wheel zoom depends on the number of events');
      state.scale = 1; wheel(1); const small = state.scale;
      state.scale = 1; wheel(100); const large = state.scale;
      if (!(large>small && small>1)) throw Error('Wheel zoom ignores scroll distance');
      state.scale = 1; viewSize(); renderRequest(); await waitForCurrentRender();
      return {whole,small,large};
    }""")

    slider = page.locator("#angle-slider")
    slider.scroll_into_view_if_needed()
    thumb = slider.evaluate("""input => {
      const rect=input.getBoundingClientRect(), fraction=(Number(input.value)-Number(input.min))/(Number(input.max)-Number(input.min));
      return [rect.left+8+(rect.width-16)*fraction, rect.top+rect.height/2];
    }""")
    page.mouse.move(*thumb)
    page.mouse.down()
    try:
        page.wait_for_function("angleSliderDragging")
        point = page.locator("#map").bounding_box()
        # Two different pointer types are each primary. A touchscreen release
        # must not release the mouse pointer which still owns the range input.
        page.touchscreen.tap(point["x"] + 40, point["y"] + 70)
        assert page.evaluate("angleSliderDragging"), "An unrelated touch ended the mouse angle drag"
        page.set_viewport_size({"width":900,"height":650})
        page.wait_for_function("canvas.getBoundingClientRect().width < 600")
        report["heldResize"] = page.evaluate("""async () => {
          await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
          const transform=viewTransform(), x=transform.rect.width/state.width, y=transform.rect.height/state.height;
          if (!angleSliderDragging || Math.abs(x-y)>1e-10 || currentPattern())
            throw Error('A resized held preview lost its owner, equal axis scales or exact-result guard');
          return {x,y};
        }""")
    finally:
        page.mouse.up()
        page.set_viewport_size(viewport)
    page.evaluate("async () => { await waitForCurrentRender(); }")

    page.locator("#axis").select_option("custom")
    page.locator("#custom-axis").fill("1 1 3")
    page.locator("#apply-axis").click()
    page.wait_for_function("metadata.layers===11 && currentPattern()")
    page.evaluate("async () => { await waitForCurrentRender(); }")
    page.locator('[data-main-tab="layers"]').click()
    page.locator("#no-layers").click()
    page.locator("#layer-checks input").nth(12).check()
    page.locator("#layer-checks input").nth(13).check()
    report["highLayer"] = page.evaluate("""() => {
      const entries=plotLegendEntries(), data=visibleData();
      if (!data.atoms.every(record=>record.count) || !entries.some(entry=>entry.label==='G1 · G') ||
          !entries.some(entry=>entry.label==='G2 · G')) throw Error('Visible high layers have no legend');
      return data.atoms.map(record=>record.count);
    }""")
    # Trace real glyph painting in both screen icon contexts and the PNG
    # context; additionally inspect pixels so a glyph call alone is insufficient.
    page.evaluate("""() => {
      window.__legendSmoke={original:CanvasRenderingContext2D.prototype.fillText,screen:[],png:[]};
      CanvasRenderingContext2D.prototype.fillText=function(text,...args) {
        if (String(text)==='12') {
          if (this.canvas.closest('#plot-legend')) __legendSmoke.screen.push(String(text));
          else if (this.canvas!==canvas) __legendSmoke.png.push(String(text));
        }
        return __legendSmoke.original.call(this,text,...args);
      };
    }""")
    try:
        page.get_by_text("VIEW / PERFORMANCE", exact=True).click()
        page.locator('[data-view-tab="appearance"]').click()
        page.locator("#appearance-layers select").nth(6).select_option("number:12")
        report["numberedLegend"] = page.evaluate("""() => {
          const entries=plotLegendEntries().filter(entry=>entry.label.endsWith(' · G'));
          const icons=[...document.querySelectorAll('#plot-legend canvas[data-symbol="number:12"]')];
          if (entries.length!==2 || entries.some(entry=>entry.symbol!=='number:12') || icons.length!==2 || __legendSmoke.screen.length!==2)
            throw Error('Screen legend did not paint the visible numbered symbols');
          const painted=icons.map((icon,index)=>{
            const data=icon.getContext('2d').getImageData(0,0,icon.width,icon.height).data, ratio=icon.width/18;
            let count=0;
            for(let y=Math.ceil(6*ratio);y<12*ratio;y++) for(let x=Math.ceil(6*ratio);x<12*ratio;x++) {
              const at=(y*icon.width+x)*4;
              if(index===0 ? data[at]>220&&data[at+1]>220&&data[at+2]>220 : data[at+3]>128) count++;
            }
            if(!count) throw Error('Numbered legend contains an empty circle instead of real numeral pixels');
            return count;
          });
          return {labels:entries.map(entry=>entry.label),painted};
        }""")
        page.locator("#clean-png").set_checked(False)
        with page.expect_download() as exported:
            page.locator("#export-png").click()
        exported.value.save_as(output / "numbered-legend.png")
        assert page.evaluate("__legendSmoke.png.length") == 2, "Ordinary PNG did not paint both real numbered legend glyphs"
    finally:
        page.evaluate("""() => {
          CanvasRenderingContext2D.prototype.fillText=__legendSmoke.original;
          delete window.__legendSmoke;
        }""")

    page.locator('[data-main-tab="layers"]').click()
    page.locator("#all-layers").click()
    try:
        page.set_viewport_size({"width":390,"height":844})
        page.locator("#plot-legend-panel").evaluate("element => { element.open=true; }")
        assert page.get_by_text("+5 layers", exact=True).is_visible(), "Expanded mobile legend hides its omission notice"
        report["mobileLegend"] = "+5 layers is visible"
    finally:
        page.set_viewport_size(viewport)
    page.evaluate("async () => { await waitForCurrentRender(); }")
    return report


def check_clean_exports(page, output):
    """Decode actual downloads and compare vector exports to the PNG pixels."""
    page.locator('[data-main-tab="orientation"]').click()
    page.locator("#axis").select_option("112")
    page.wait_for_function("state.axis === '112' && metadata.layers === 6 && currentPattern()")
    page.locator("#field-slider").evaluate("""input => {
      input.value='0.5'; input.dispatchEvent(new Event('input', {bubbles:true}));
    }""")
    page.evaluate("async () => { await waitForCurrentRender(); }")
    page.locator("#clean-png").set_checked(True)
    cases = [
        {"name": "hollow-transparent", "background": False, "hollow": True,
         "symbols": ["o", "d", "t", "s", "p", "h"], "layers": [[], [0]]},
        {"name": "hollow-white", "background": True, "hollow": True,
         "symbols": ["o", "d", "t", "s", "p", "h"], "layers": [[], [0]]},
        {"name": "shapes-transparent", "background": False,
         "symbols": ["o", "d", "t", "s", "p", "h"], "layers": [list(range(6)), list(range(6))]},
        {"name": "shapes-white", "background": True,
         "symbols": ["star", "+", "x", "t1", "t2", "t3"], "layers": [list(range(6)), list(range(6))]},
        {"name": "numbered-filtered", "background": False, "filtered": True,
         "symbols": ["number:12", "d", "t", "s", "p", "h"], "layers": [[0, 2, 4], [1, 3, 5]]},
    ]
    reports = []
    for case in cases:
        expected = page.evaluate("""async options => {
          state.displayRotation = 23; state.center = [.125, -.25];
          state.colors = ['#54a84f', '#c52c79']; state.symbols = options.symbols;
          state.sizes = [1.25, .75, 1.6, .6, 1, 1.4];
          state.visibleLayers = options.layers.map(layers => new Set(layers));
          state.boundary = [[0, -3], [0, 3]];
          state.regions = options.filtered ? [true, false, false, true] : [true, true, true, true];
          state.referenceAxes = true;
          renderRequest(); await waitForCurrentRender(); draw();
          const data = visibleData(), rect = plotRect();
          let hollow = null;
          if (options.hollow) {
            const atoms = data.atoms[1];
            for (let index = 0; index < atoms.count; index++) {
              const x = atoms.x[index], y = atoms.y[index];
              if (x < rect.left + 15 || x > rect.left + rect.width - 15 ||
                  y < rect.top + 15 || y > rect.top + rect.height - 15) continue;
              let isolated = true;
              for (let other = 0; other < atoms.count; other++)
                if (other !== index && Math.hypot(x - atoms.x[other], y - atoms.y[other]) < 22) isolated = false;
              if (isolated) { hollow = [x, y]; break; }
            }
            if (!hollow) throw Error('No isolated hollow atom available for export validation');
          }
          const numbers = [];
          for (const atoms of data.atoms) for (let index = 0; index < atoms.count; index++) {
            const layer = atoms.rows.flat[atoms.indices[index] * 6 + 2], symbol = state.symbols[layer];
            if (symbol.startsWith('number:')) numbers.push(symbol.slice(7));
          }
          return {counts:data.atoms.map(atoms => atoms.count), numbers, hollow,
            cssWidth:canvas.clientWidth, cssHeight:canvas.clientHeight, rect};
        }""", case)
        page.locator("#clean-background").set_checked(case["background"])
        with page.expect_download() as exported:
            page.locator("#export-png").click()
        png = output / f"{case['name']}.png"
        exported.value.save_as(png)
        with page.expect_download() as exported:
            page.locator("#export-svg").click()
        svg = output / f"{case['name']}.svg"
        exported.value.save_as(svg)
        report = page.evaluate("""async ({png, svg, background, expected}) => {
          const svgDocument = new DOMParser().parseFromString(svg, 'image/svg+xml');
          if (svgDocument.querySelector('parsererror,image,foreignObject,script'))
            throw Error('SVG is invalid or contains a raster image/non-vector content');
          const root = svgDocument.documentElement;
          if (root.localName !== 'svg' || !root.hasAttribute('viewBox')) throw Error('Missing scalable SVG viewBox');
          const allowed = new Set(['svg','defs','clipPath','rect','g','circle','path','text']);
          for (const node of root.querySelectorAll('*'))
            if (!allowed.has(node.localName)) throw Error('Unexpected SVG content: ' + node.localName);
          const backgroundRects = [...root.children].filter(node => node.localName === 'rect');
          if (backgroundRects.length !== Number(background) || backgroundRects.some(node => node.getAttribute('fill') !== '#ffffff'))
            throw Error('SVG background does not follow the pure-white/transparent choice');
          const counts = [1,2].map(grain => {
            const group = root.querySelector(`g[data-grain="${grain}"]`);
            if (!group) throw Error('SVG is missing an atom grain');
            return [...group.children].filter(node => ['circle','path'].includes(node.localName)).length;
          });
          if (JSON.stringify(counts) !== JSON.stringify(expected.counts)) throw Error('SVG changed atom visibility');
          const numbers = [...root.querySelectorAll('text')].map(node => node.textContent).sort();
          if (JSON.stringify(numbers) !== JSON.stringify(expected.numbers.sort())) throw Error('SVG contains annotations or lost numbered atom glyphs');

          const load = async url => { const img = new Image(); img.src = url; await img.decode(); return img; };
          const pngImage = await load(png), width = pngImage.naturalWidth, height = pngImage.naturalHeight;
          const raster = document.createElement('canvas');
          raster.width = width; raster.height = height;
          const context = raster.getContext('2d', {willReadFrequently:true});
          context.drawImage(pngImage, 0, 0);
          const pixels = context.getImageData(0, 0, width, height).data;
          const url = URL.createObjectURL(new Blob([svg], {type:'image/svg+xml'}));
          let svgPixels;
          try {
            const svgImage = await load(url);
            context.clearRect(0, 0, width, height); context.drawImage(svgImage, 0, 0, width, height);
            svgPixels = context.getImageData(0, 0, width, height).data;
          } finally { URL.revokeObjectURL(url); }
          const sx = width / expected.cssWidth, sy = height / expected.cssHeight;
          const isBackground = (data, index) => background
            ? data[index] === 255 && data[index+1] === 255 && data[index+2] === 255 && data[index+3] === 255
            : data[index+3] === 0;
          let active = 0, differing = 0, absolute = 0, smoothedDiffering = 0, smoothedAbsolute = 0;
          for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
            const at = (y * width + x) * 4, r = expected.rect;
            // The whole outer margin must contain no grid, axes, legend,
            // boundary handles, reference inset or other annotations.
            if (x < (r.left-2)*sx || x > (r.left+r.width+2)*sx ||
                y < (r.top-2)*sy || y > (r.top+r.height+2)*sy)
              if (!isBackground(pixels, at) || !isBackground(svgPixels, at))
                throw Error(`Clean export painted its margin at ${x},${y}`);
            if (background && (pixels[at+3] !== 255 || svgPixels[at+3] !== 255))
              throw Error('White export contains transparent pixels');
            if (isBackground(pixels, at) && isBackground(svgPixels, at)) continue;
            active++;
            // Compare premultiplied channels: RGB under alpha zero is not
            // visible and can legally differ between the image encoders.
            let maximum = Math.abs(pixels[at+3] - svgPixels[at+3]);
            absolute += maximum;
            for (let channel = 0; channel < 3; channel++) {
              const difference = Math.abs(pixels[at+channel] * pixels[at+3] / 255 - svgPixels[at+channel] * svgPixels[at+3] / 255);
              maximum = Math.max(maximum, difference); absolute += difference;
            }
            if (maximum > 40) differing++;
            // Chromium antialiases SVG circles differently from Canvas arcs.
            // Average a 3x3 neighborhood to compare the geometry and color,
            // without treating subpixel stroke coverage as a displaced atom.
            // A real one-pixel displacement still fails this comparison.
            let smoothedMaximum = 0;
            for (let channel=0; channel<4; channel++) {
              let difference = 0;
              for (let dy=-1; dy<=1; dy++) for (let dx=-1; dx<=1; dx++) {
                const near = ((y+dy)*width+x+dx)*4;
                difference += channel === 3 ? pixels[near+3] - svgPixels[near+3]
                  : (pixels[near+channel]*pixels[near+3] - svgPixels[near+channel]*svgPixels[near+3])/255;
              }
              difference = Math.abs(difference / 9);
              smoothedMaximum = Math.max(smoothedMaximum, difference); smoothedAbsolute += difference;
            }
            if (smoothedMaximum > 24) smoothedDiffering++;
          }
          if (active < 50) throw Error('Clean export contains no visible atom pixels');
          if (expected.hollow) {
            const x = Math.round(expected.hollow[0]*sx), y = Math.round(expected.hollow[1]*sy);
            for (let dy=-1; dy<=1; dy++) for (let dx=-1; dx<=1; dx++) {
              const at = ((y+dy)*width+x+dx)*4;
              if (!isBackground(pixels, at) || !isBackground(svgPixels, at))
                throw Error('Hollow atom center does not preserve the selected background');
            }
          }
          const fraction = differing / active, mean = absolute / (4 * active);
          const smoothedFraction = smoothedDiffering / active, smoothedMean = smoothedAbsolute / (4 * active);
          if (smoothedFraction > .1 || smoothedMean > 7.5)
            throw Error(`PNG/SVG rendering differs after antialias averaging: ${(smoothedFraction*100).toFixed(2)}% of atom pixels, mean error ${smoothedMean.toFixed(2)}`);
          return {width,height,atoms:counts,background:background?'white':'transparent',
            numberedGlyphs:numbers.length,activePixels:active,differingFraction:fraction,meanChannelError:mean,
            smoothedDifferingFraction:smoothedFraction,smoothedMeanChannelError:smoothedMean};
        }""", {"png": "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode("ascii"),
                "svg": svg.read_text(encoding="utf-8"), "background": case["background"], "expected": expected})
        reports.append({"name": case["name"], **report})

    # The atoms-only background setting must not remove the ordinary plot's
    # grid, annotations or existing opaque background.
    page.locator("#clean-png").set_checked(False)
    page.locator("#clean-background").set_checked(False)
    with page.expect_download() as exported:
        page.locator("#export-png").click()
    png = output / "ordinary-after-transparent.png"
    exported.value.save_as(png)
    corner = page.evaluate("""async url => {
      const img = new Image(); img.src = url; await img.decode();
      const canvas = document.createElement('canvas'); canvas.width=img.width; canvas.height=img.height;
      const context = canvas.getContext('2d'); context.drawImage(img,0,0);
      return Array.from(context.getImageData(0,0,1,1).data);
    }""", "data:image/png;base64," + base64.b64encode(png.read_bytes()).decode("ascii"))
    assert corner == [255, 255, 255, 255], ("Ordinary PNG background is not pure white", corner)
    page.evaluate("""() => {
      const target=document.createElement('canvas');
      target.width=canvas.clientWidth; target.height=canvas.clientHeight;
      const context=target.getContext('2d'), stroke=context.stroke.bind(context), colors=[];
      context.stroke=function(...args) { colors.push(this.strokeStyle); return stroke(...args); };
      drawGrid(context,plotRect());
      if (!colors.includes('#d9e1ea') || !colors.includes('#ebeff4'))
        throw Error('The white background changed the existing major or minor grid color');
      if ([...context.getImageData(0,0,1,1).data].join()!=='255,255,255,255')
        throw Error('The screen plot background is not pure white');
    }""")
    viewport = page.viewport_size
    page.locator("#clean-png").set_checked(True)
    page.locator("#clean-background").set_checked(True)
    page.locator("#export-controls").screenshot(path=str(output / "export-controls-desktop.png"))
    try:
        page.set_viewport_size({"width":390, "height":844})
        page.locator("#export-controls").screenshot(path=str(output / "export-controls-mobile.png"))
    finally:
        page.set_viewport_size(viewport)
    page.evaluate("async () => { await waitForCurrentRender(); }")
    return reports


def check_translations(page, output):
    """Drive whole-grain and rectangle moves through actual browser controls."""
    page.evaluate("""async () => {
      state.lattice='FCC'; state.axis='111'; resetSelections(); await loadMetadata(true);
      state.scale=.6; state.center=[0,0]; state.displayRotation=23;
      state.visibleLayers=[new Set([0]),new Set([0,1,2])];
      state.boundary=[[-1,0],[1,0]]; state.regions=[true,false,false,true];
      rebuildLayers(); updateSummary(); renderRequest(); await waitForCurrentRender(); draw();
      globalThis.translationSmoke={before:pattern.grains.map(rows=>Array.from(rows)),
        boundary:JSON.stringify(state.boundary),regions:JSON.stringify(state.regions)};
    }""")
    page.locator("#translation-panel > summary").click()
    page.locator("#translation-grain").select_option("0")
    page.locator("#translation-dx").fill("0.3")
    page.locator("#translation-dy").fill("0.15")
    page.locator("#translate-whole").click()
    whole = page.evaluate("""async () => {
      await waitForCurrentRender();
      const probe=translationSmoke, rows=pattern.grains.map(data=>Array.from(data));
      const key=row=>row.slice(2).join(','), moved=new Map(rows[0].map(row=>[key(row),row]));
      const layers=new Set(); let checked=0;
      for (const before of probe.before[0]) {
        if (Math.abs(before[0])>2 || Math.abs(before[1])>2) continue;
        const after=moved.get(key(before));
        if (!after || Math.abs(after[0]-before[0]-.3)>1e-10 || Math.abs(after[1]-before[1]-.15)>1e-10)
          throw Error('Whole-grain move failed to update physical atom coordinates');
        layers.add(before[2]); checked++;
      }
      if (layers.size!==metadata.layers) throw Error('Whole move omitted hidden axial layers');
      if (JSON.stringify(rows[1])!==JSON.stringify(probe.before[1])) throw Error('G1 move changed G2');
      if (JSON.stringify(state.boundary)!==probe.boundary || JSON.stringify(state.regions)!==probe.regions)
        throw Error('Moving atoms changed the fixed GB reference or clipping');
      probe.whole=rows; return {checked,layers:layers.size};
    }""")

    def select_rectangle():
        page.locator("#pick-translation").click()
        points = page.evaluate("""() => [[-1.1,.3],[1.1,2.3]].map(point => {
          const [x,y]=screen(point), r=canvas.getBoundingClientRect(); return [r.left+x,r.top+y];
        })""")
        for x, y in points:
            page.mouse.click(x, y)
        page.wait_for_function("state.translationCorners.length===2 && state.mode==='idle'")

    select_rectangle()
    page.locator("#translation-dx").fill("0.4")
    page.locator("#translation-dy").fill("-0.2")
    page.screenshot(path=str(output / "translation-selection.png"))
    page.locator("#translate-region").click()
    local = page.evaluate("""async () => {
      await waitForCurrentRender(); draw();
      const probe=translationSmoke, rows=pattern.grains.map(data=>Array.from(data));
      const op=state.grainDisplacements[0][1], [xmin,xmax,ymin,ymax]=op.bounds;
      const moved=new Map(rows[0].map(row=>[row.slice(2).join(','),row]));
      let inside=0,outside=0; const layers=new Set();
      for (const before of probe.whole[0]) {
        if (Math.abs(before[0])>3 || Math.abs(before[1])>3) continue;
        const selected=before[0]>=xmin && before[0]<=xmax && before[1]>=ymin && before[1]<=ymax;
        const after=moved.get(before.slice(2).join(','));
        if (!after || Math.abs(after[0]-before[0]-(selected ? .4 : 0))>1e-10 ||
            Math.abs(after[1]-before[1]-(selected ? -.2 : 0))>1e-10)
          throw Error('Rectangle move changed the wrong physical atom or used display axes');
        if (selected) { inside++; layers.add(before[2]); } else outside++;
      }
      if (!inside || !outside || layers.size!==metadata.layers) throw Error('Rectangle did not exercise selected, unselected and hidden-layer atoms');
      if (JSON.stringify(rows[1])!==JSON.stringify(probe.whole[1])) throw Error('Local G1 move changed G2');
      if (JSON.stringify(state.boundary)!==probe.boundary || JSON.stringify(state.regions)!==probe.regions)
        throw Error('Rectangle move changed the GB reference');
      if (!uniformStrainBlocked() || !document.getElementById('apply-strain').disabled)
        throw Error('Manual displacement still allows uniform-strain fitting');
      probe.local=rows; probe.ops=JSON.stringify(state.grainDisplacements);
      return {inside,outside,layers:layers.size,bounds:op.bounds};
    }""")
    with page.expect_download() as saved:
        page.locator("#save-session").click()
    session = output / "translated.dmap"
    saved.value.save_as(session)
    with ZipFile(session) as archive:
        snapshot = json.loads(archive.read("session.json"))
        assert len(snapshot["state"]["grain_displacements"][0]) == 2
    page.locator("#undo-translation").click()
    page.evaluate("""async () => {
      await waitForCurrentRender();
      if (JSON.stringify(pattern.grains.map(rows=>Array.from(rows)))!==JSON.stringify(translationSmoke.whole))
        throw Error('Undo failed to restore the preceding physical pattern');
    }""")
    page.locator("#reset-translations").click()
    page.evaluate("""async () => {
      await waitForCurrentRender();
      if (hasGrainDisplacements()) throw Error('Reset left manual moves behind');
      if (JSON.stringify(state.boundary)!==translationSmoke.boundary) throw Error('Reset removed the GB reference');
    }""")
    page.locator("#session-file").set_input_files(session)
    page.wait_for_function("currentPattern() && JSON.stringify(state.grainDisplacements)===translationSmoke.ops")
    page.evaluate("""() => {
      if (JSON.stringify(pattern.grains.map(rows=>Array.from(rows)))!==JSON.stringify(translationSmoke.local))
        throw Error('Saved manual moves did not reproduce the physical pattern');
      draw();
    }""")
    select_rectangle()
    page.screenshot(path=str(output / "translation.png"))
    return {"whole":whole, "local":local, "undo":"restored", "session":"restored"}


def check_absolute_translations(page, output):
    """Keep a selected material region fixed while its absolute offset changes."""
    page.evaluate("""async () => {
      state.lattice='SC'; state.axis='100'; resetSelections(); await loadMetadata(true);
      state.angle=0; state.scale=.6; state.center=[0,0]; state.displayRotation=31;
      state.regions=[true,true,true,true]; state.nearEnabled=false;
      rebuildLayers(); updateSummary(); renderRequest(); await waitForCurrentRender(); draw();
      const rows=pattern.grains.map(grain=>Array.from(grain));
      const at=(x,y)=>rows[0].find(row=>Math.abs(row[0]-x)<1e-12 && Math.abs(row[1]-y)<1e-12);
      globalThis.absoluteShiftSmoke={before:rows,zero:at(0,0).slice(2).join(','),
        neighbor:at(1,0).slice(2).join(','),center:JSON.stringify(state.center)};
      document.getElementById('translation-panel').open=true;
    }""")
    page.locator("#translation-grain").select_option("0")

    def apply_offset(x, y, *, local=False):
        page.locator("#translation-dx").fill(str(x))
        page.locator("#translation-dy").fill(str(y))
        page.locator("#translate-region" if local else "#translate-whole").click()
        page.evaluate("async () => { await waitForCurrentRender(); draw(); }")

    def assert_whole(x, y):
        return page.evaluate("""([x,y]) => {
          const rows=pattern.grains.map(grain=>Array.from(grain));
          const moved=new Map(rows[0].map(row=>[row.slice(2).join(','),row]));
          let checked=0;
          for (const before of absoluteShiftSmoke.before[0]) {
            if (Math.abs(before[0])>1 || Math.abs(before[1])>1) continue;
            const after=moved.get(before.slice(2).join(','));
            if (!after || Math.abs(after[0]-before[0]-x)>1e-10 || Math.abs(after[1]-before[1]-y)>1e-10)
              throw Error('Whole-grain offset accumulated instead of using the original position');
            checked++;
          }
          if (!checked || JSON.stringify(rows[1])!==JSON.stringify(absoluteShiftSmoke.before[1]))
            throw Error('Absolute whole-grain move changed G2 or checked no atoms');
          return checked;
        }""", [x, y])

    apply_offset(.5, .25)
    whole = assert_whole(.5, .25)
    history = page.evaluate("JSON.stringify(state.grainDisplacements)")
    apply_offset(.5, .25)
    assert_whole(.5, .25)
    assert page.evaluate("JSON.stringify(state.grainDisplacements)") == history, "An unchanged whole offset added a move"
    apply_offset(1, -.5)
    assert_whole(1, -.5)
    apply_offset(0, 0)
    assert_whole(0, 0)
    page.locator("#reset-translations").click()
    page.evaluate("async () => { await waitForCurrentRender(); }")

    def points_on_canvas(points):
        return page.evaluate("""points => points.map(point => {
          const [x,y]=screen(point), rect=canvas.getBoundingClientRect();
          return [rect.left+x,rect.top+y];
        })""", points)

    def drag_rectangle():
        page.locator("#pick-translation").click()
        # Reverse corners exercise normalized model-axis bounds under rotation.
        first, second = points_on_canvas([[.2, .2], [-.2, -.2]])
        page.mouse.move(*first)
        page.mouse.down()
        page.mouse.move(*second, steps=8)
        page.mouse.up()
        page.wait_for_function("state.translationCorners.length===2 && state.mode==='idle'")
        assert page.evaluate("JSON.stringify(state.center)") == page.evaluate("absoluteShiftSmoke.center"), "Rectangle drag panned the plot"
        page.evaluate("""() => {
          const expected=[-.2,.2,-.2,.2], bounds=translationBounds();
          if (bounds.some((value,index)=>Math.abs(value-expected[index])>.015))
            throw Error('Dragged rectangle used display coordinates instead of physical model axes');
        }""")

    def assert_region(x, y):
        return page.evaluate("""([x,y]) => {
          const rows=pattern.grains.map(grain=>Array.from(grain));
          const moved=new Map(rows[0].map(row=>[row.slice(2).join(','),row]));
          const selected=moved.get(absoluteShiftSmoke.zero), neighbor=moved.get(absoluteShiftSmoke.neighbor);
          if (!selected || Math.abs(selected[0]-x)>1e-10 || Math.abs(selected[1]-y)>1e-10)
            throw Error('The original selected atom did not receive the absolute region offset');
          if (!neighbor || Math.abs(neighbor[0]-1)>1e-10 || Math.abs(neighbor[1])>1e-10)
            throw Error('Editing a moved rectangle captured an overlapping unselected atom');
          if (JSON.stringify(rows[1])!==JSON.stringify(absoluteShiftSmoke.before[1]))
            throw Error('Editing the G1 rectangle changed G2');
          return {selected:selected.slice(0,2),neighbor:neighbor.slice(0,2)};
        }""", [x, y])

    drag_rectangle()
    apply_offset(1, 0, local=True)
    overlap = assert_region(1, 0)
    history = page.evaluate("JSON.stringify(state.grainDisplacements)")
    apply_offset(1, 0, local=True)
    assert_region(1, 0)
    assert page.evaluate("JSON.stringify(state.grainDisplacements)") == history, "An unchanged region offset added a move"
    apply_offset(2, 0, local=True)
    assert_region(2, 0)
    apply_offset(0, 0, local=True)
    assert_region(0, 0)
    page.locator("#undo-translation").click()
    page.evaluate("async () => { await waitForCurrentRender(); }")
    assert_region(2, 0)
    apply_offset(.25, .125)
    page.locator("#undo-translation").click()
    page.evaluate("async () => { await waitForCurrentRender(); }")
    assert_region(2, 0)
    assert [float(page.locator(selector).input_value()) for selector in ("#translation-dx", "#translation-dy")] == [2, 0], "Undo restored the rectangle but displayed its whole-grain offset"
    page.locator("#translate-region").click()
    page.evaluate("async () => { await waitForCurrentRender(); }")
    assert_region(2, 0)
    page.locator("#reset-translations").click()
    page.evaluate("async () => { await waitForCurrentRender(); }")
    page.locator("#pick-translation").click()
    first, second = points_on_canvas([[-.6, -.4], [.6, .4]])
    page.mouse.click(*first)
    page.mouse.move(*second, steps=5)
    page.wait_for_function("state.translationCorners.length===1 && translationPreview!==null")
    assert page.locator("#translate-region").is_disabled(), "Hover alone committed a rectangle"
    assert page.evaluate("JSON.stringify(state.center)") == page.evaluate("absoluteShiftSmoke.center")
    page.screenshot(path=str(output / "translation-hover.png"))
    page.keyboard.press("Escape")
    page.wait_for_function("state.mode==='idle' && !state.translationCorners.length && translationPreview===null")

    page.locator("#pick-vector").click()
    for point in points_on_canvas([[0, 0], [1, 1]]):
        page.mouse.click(*point)
    page.wait_for_function("state.atoms.length===2 && state.mode==='idle'")
    page.locator("#axial-repeat").fill("2")
    page.locator("#axial-repeat").dispatch_event("change")
    drag_rectangle()
    page.locator("#translation-use-vector").click()
    vector = page.evaluate("""() => {
      const x=Number(document.getElementById('translation-dx').value);
      const y=Number(document.getElementById('translation-dy').value);
      const length=Number(document.getElementById('translation-vector-length').value);
      if (Math.abs(x-1)>1e-10 || Math.abs(y-1)>1e-10 || Math.abs(length-Math.SQRT2)>1e-10)
        throw Error('Vector shift used display axes or included the axial periodic-image height');
      return {x,y,length,displayRotation:state.displayRotation,axialRepeat:state.axialRepeat};
    }""")
    page.locator("#translation-vector-length").fill(str(-2 ** .5 / 2))
    page.locator("#translation-vector-length").dispatch_event("change")
    page.locator("#translate-region").click()
    page.evaluate("async () => { await waitForCurrentRender(); draw(); }")
    assert_region(-.5, -.5)
    assert page.locator("#translation-vector-length").is_enabled(), "Applying the shift lost the captured vector direction"
    page.locator("#translation-vector-length").fill(str(2 ** .5 / 2))
    page.locator("#translation-vector-length").dispatch_event("change")
    page.locator("#translate-region").click()
    page.evaluate("async () => { await waitForCurrentRender(); draw(); }")
    assert_region(.5, .5)
    page.screenshot(path=str(output / "absolute-translation.png"))
    return {"wholeAtomsChecked": whole, "overlap": overlap,
            "sameOffset": "idempotent", "zeroOffset": "restored", "undo": "restored",
            "hover": "previewed and cancelled", "vector": vector, "signedLength": "reversed and reapplied"}


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
    interactions = check_plot_interactions(page, output)
    exports = check_clean_exports(page, output)
    translations = check_translations(page, output)
    absolute_translations = check_absolute_translations(page, output)
    return {"buffers": buffers, "cancellation": cancellation, "session": "restored", "png": ["pattern.png", "atoms.png"], "gpu": gpu, "angleDrag": drag,
            "plotInteractions": interactions, "cleanExports": exports, "translations": translations,
            "absoluteTranslations": absolute_translations}


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
    from smoke_core_contracts import check_core_contracts

    errors, console = [], []
    with serve(site) as origin, sync_playwright() as playwright:
        options = {"headless": True, "args": ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]}
        if args.chromium:
            options["executable_path"] = args.chromium
        browser = playwright.chromium.launch(**options)
        context = browser.new_context(viewport={"width": 1440, "height": 900}, has_touch=True, service_workers="block")
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
            report["coreContracts"] = check_core_contracts(
                page, base, origin + "/vendor/dichromatic_map.zip",
                (site / "web_bridge.py").read_text(encoding="utf-8"),
            )
            # Keep regression helpers in the checkout, outside the deployed
            # artifact, while testing the actual deployed renderer and marker.
            page.add_script_tag(path=str(site.parent / "gpu_renderer.browser.test.js"))
            report["gpuMaskChecks"] = page.evaluate("""async () =>
              runGPUChecks({marker, createRenderer: window.DichromaticGPU.createRenderer})
            """)
            assert not errors, errors
            report["runtime"] = base
            (output / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            print("Real Chromium/Pyodide startup, Native/WASM scientific edge cases, cancellation, session import, PNG/SVG background and raster comparisons, pointer interactions and twelve GPU markers passed.")
        except Exception:
            page.screenshot(path=str(output / "failure.png"), full_page=True)
            (output / "failure.txt").write_text(page.locator("#status").inner_text() + "\n" + "\n".join(errors + console), encoding="utf-8")
            raise
        finally:
            browser.close()


if __name__ == "__main__":
    main()
