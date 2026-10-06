import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";
import { LatestRequestQueue } from "./worker_queue.mjs";

let enginePromise;
async function engine() {
  if (!enginePromise) {
    enginePromise = (async () => {
      const read = async (url, binary) => {
        const response = await fetch(url);
        if (!response.ok) throw new Error("The app's numerical files could not be loaded.");
        return binary ? response.arrayBuffer() : response.text();
      };
      const [pyodide, [source, archive]] = await Promise.all([
        loadPyodide().then(async runtime => { await runtime.loadPackage("numpy"); return runtime; }),
        Promise.all([read("./web_bridge.py", false), read("./vendor/dichromatic_map.zip", true)]),
      ]);
      pyodide.FS.writeFile("/dichromatic_map.zip", new Uint8Array(archive));
      pyodide.runPython("import sys; sys.path.insert(0, '/dichromatic_map.zip')");
      pyodide.runPython(source);
      self.postMessage({ type: "ready" });
      return pyodide;
    })();
  }
  return enginePromise;
}

function copyArray(proxy) {
  let buffer;
  try {
    buffer = proxy.getBuffer("f64");
    // The Pyodide view belongs to WASM memory and cannot be transferred.
    // Copy it once into an owned buffer, then release the Python references.
    return new Float64Array(buffer.data);
  } finally {
    buffer?.release();
    proxy.destroy();
  }
}

function renderResult(pyodide) {
  const payload = pyodide.runPython("web_render(web_request)");
  let grains;
  try {
    grains = payload.get("grains");
    const result = {
      format: "dichromatic-map-render-v1",
      metadata: JSON.parse(payload.get("metadata")),
      grains: [copyArray(grains.get(0)), copyArray(grains.get(1))],
      coincidences: copyArray(payload.get("coincidences")),
      local: null,
    };
    const local = payload.get("local");
    if (local != null) result.local = copyArray(local);
    return {result, transfer: [...result.grains, result.coincidences, ...(result.local ? [result.local] : [])]
      .map(array => array.buffer)};
  } finally {
    grains?.destroy();
    payload.destroy();
  }
}

const queue = new LatestRequestQueue({
  post: (message, transfer = []) => self.postMessage(message, transfer),
  execute: async (request, cancelled) => {
    const pyodide = await engine();
    if (cancelled()) return;
    pyodide.globals.set("web_request", JSON.stringify(request));
    if (request.action === "render") return renderResult(pyodide);
    return {result: JSON.parse(pyodide.runPython("web_dispatch(web_request)"))};
  },
  startSearch: async (request, cancelled) => {
    const pyodide = await engine();
    if (cancelled()) return null;
    pyodide.globals.set("web_request", JSON.stringify(request));
    const job = pyodide.runPython("web_near_start(web_request)");
    pyodide.globals.set("web_search", job);
    let disposed = false;
    return {
      // Keep the historical eight-row partition: changing partitions can
      // change which equivalent cell wins a roundoff tie. Yield after each
      // fixed block for cancellation and higher-priority requests.
      step: () => pyodide.runPython("web_near_step(web_search)"),
      finish: () => ({result: JSON.parse(pyodide.runPython("web_near_finish(web_search)"))}),
      dispose: () => {
        if (disposed) return;
        disposed = true;
        pyodide.globals.delete("web_search");
        job.destroy();
      },
    };
  },
});

self.onmessage = ({data}) => queue.enqueue(data);
