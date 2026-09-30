import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

let enginePromise;
async function engine() {
  if (!enginePromise) {
    enginePromise = (async () => {
      const pyodide = await loadPyodide();
      await pyodide.loadPackage("numpy");
      const [source, archive] = await Promise.all([
        fetch("./web_bridge.py"), fetch("./vendor/dichromatic_map.zip"),
      ]);
      if (!source.ok || !archive.ok) throw new Error("The app's numerical files could not be loaded.");
      pyodide.FS.writeFile("/dichromatic_map.zip", new Uint8Array(await archive.arrayBuffer()));
      pyodide.runPython("import sys; sys.path.insert(0, '/dichromatic_map.zip')");
      pyodide.runPython(await source.text());
      self.postMessage({ type: "ready" });
      return pyodide;
    })();
  }
  return enginePromise;
}

self.onmessage = async ({ data }) => {
  try {
    const pyodide = await engine();
    pyodide.globals.set("web_request", JSON.stringify(data.request));
    const result = pyodide.runPython("web_dispatch(web_request)");
    self.postMessage({ type: "result", id: data.id, result: JSON.parse(result) });
  } catch (error) {
    self.postMessage({ type: "error", id: data.id, message: error.message || String(error) });
  }
};
