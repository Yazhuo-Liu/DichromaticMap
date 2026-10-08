"use strict";

// Run with a real WebGL 2 browser and the application's marker function.
// CPU unit tests cannot catch shader geometry or drawing-buffer recovery bugs.
function gpuContextEvent(canvas, name) {
  return new Promise((resolve, reject) => {
    const done = () => {clearTimeout(timeout); resolve();};
    const timeout = setTimeout(() => {
      canvas.removeEventListener(name, done);
      reject(Error(`Timed out waiting for ${name}`));
    }, 5000);
    canvas.addEventListener(name, done, {once: true});
  });
}

async function runGPURecoveryChecks({createRenderer}) {
  const checks = [];
  for (const dpr of [1, 2]) {
    let restored = 0;
    const renderer = createRenderer({minimumAtoms: 0, onRestored: () => restored++});
    if (!renderer) throw Error("A WebGL 2 browser is required for recovery tests");
    const extension = renderer.gl.getExtension("WEBGL_lose_context");
    if (!extension) throw Error("WEBGL_lose_context is required for recovery tests");
    const flat = new Float64Array([1.234567890123, 2.345678901234, 0, 4, 5, 6]);
    const original = flat.slice();
    const selection = {rows: {flat, stride: 6}, indices: new Uint32Array([0]),
      x: new Float64Array([24]), y: new Float64Array([32]), count: 1, layers: new Set([0])};
    const options = {atoms: [selection, []], revision: 1, width: 96, height: 64, dpr,
      plot: {left: 0, top: 0, width: 96, height: 64}, symbols: ["o"],
      colors: ["#1677d2", "#e35d35"], sizes: [2], scale: 1};
    if (!renderer.render(options)) throw Error("Initial recovery-test rendering failed");
    const oldProgram = renderer.program, oldBuffer = renderer.buffer, storage = renderer.storage;
    const loss = gpuContextEvent(renderer.canvas, "webglcontextlost");
    extension.loseContext();
    await loss;
    if (renderer.render(options) !== null) throw Error("Lost context did not select Canvas fallback");
    // The view and style can change while Canvas handles the unavailable GPU.
    selection.x[0] = 72;
    const restoration = gpuContextEvent(renderer.canvas, "webglcontextrestored");
    // Chromium completes the loss dispatch before allowing restoreContext.
    // A Promise continuation alone still runs inside that event's task.
    await new Promise(resolve => setTimeout(resolve, 0));
    extension.restoreContext();
    await restoration;
    const gpu = renderer.render({...options, revision: 2, colors: ["#00ff00", "#e35d35"]});
    if (!gpu || restored !== 1) throw Error("Restored context did not resume the idle plot");
    if (renderer.program === oldProgram || renderer.buffer === oldBuffer || renderer.storage !== storage)
      throw Error("Context restoration did not rebuild GL handles and retain CPU staging storage");
    if (renderer.revision !== 2 || renderer.storage[0] !== 72 ||
        !flat.every((value, index) => Object.is(value, original[index])))
      throw Error("Restored upload used stale coordinates or changed scientific rows");
    const copy = document.createElement("canvas"); copy.width = gpu.width; copy.height = gpu.height;
    const context = copy.getContext("2d"); context.drawImage(gpu, 0, 0);
    const newPixel = context.getImageData(72 * dpr, 32 * dpr, 1, 1).data;
    const oldPixel = context.getImageData(24 * dpr, 32 * dpr, 1, 1).data;
    if (newPixel[0] !== 0 || newPixel[1] !== 255 || newPixel[2] !== 0 || newPixel[3] !== 255 || oldPixel[3] !== 0)
      throw Error(`Restored DPR ${dpr} drawing buffer retained stale pixels or styles`);
    checks.push({dpr, restored});
    extension.loseContext();
  }
  return checks;
}

async function runGPUChecks({marker, createRenderer}) {
  const symbols = ["o", "d", "t", "s", "p", "h", "star", "+", "x", "t1", "t2", "t3"];
  const checks = [];
  for (const dpr of [1, 2]) {
    const atoms = [[], []];
    for (let grain = 0; grain < 2; grain++) {
      for (let index = 0; index < symbols.length; index++) atoms[grain].push({point: [0, 0, index],
        x: 16 + index % 6 * 32, y: 16 + (Math.floor(index / 6) + grain * 2) * 32});
    }
    const options = {atoms, width: 192, height: 128, dpr,
      plot: {left: 0, top: 0, width: 192, height: 128}, symbols,
      colors: ["#1677d2", "#e35d35"], sizes: symbols.map(() => 2), scale: 1};
    const renderer = createRenderer({minimumAtoms: 0});
    if (!renderer) throw Error("A WebGL 2 browser is required for GPU regression tests");
    const gpu = renderer.render(options);
    if (!gpu) throw Error("Common markers unexpectedly used Canvas fallback");
    const reference = document.createElement("canvas"); reference.width = gpu.width; reference.height = gpu.height;
    const cpu = reference.getContext("2d"); cpu.setTransform(dpr, 0, 0, dpr, 0, 0);
    for (let grain = 0; grain < 2; grain++) for (const atom of atoms[grain])
      marker(cpu, atom.x, atom.y, 8, symbols[atom.point[2]], grain ? options.colors[1] : "#2e6799",
        grain ? null : options.colors[0]);
    const copy = document.createElement("canvas"); copy.width = gpu.width; copy.height = gpu.height;
    const context = copy.getContext("2d"); context.drawImage(gpu, 0, 0);
    const actual = context.getImageData(0, 0, gpu.width, gpu.height).data;
    const expected = cpu.getImageData(0, 0, gpu.width, gpu.height).data;
    for (let grain = 0; grain < 2; grain++) for (const atom of atoms[grain]) {
      let intersection = 0, union = 0;
      for (let y = (atom.y - 15) * dpr; y < (atom.y + 15) * dpr; y++)
        for (let x = (atom.x - 15) * dpr; x < (atom.x + 15) * dpr; x++) {
          const offset = (y * gpu.width + x) * 4;
          const a = actual[offset + 3] >= 128, b = expected[offset + 3] >= 128;
          if (a && b) intersection++;
          if (a || b) union++;
        }
      const overlap = intersection / union;
      if (overlap < 0.80) throw Error(`${symbols[atom.point[2]]} G${grain + 1} DPR ${dpr} differs from Canvas: ${overlap}`);
      checks.push({symbol: symbols[atom.point[2]], grain: grain + 1, dpr, overlap});
    }
    const extension = renderer.gl.getExtension("WEBGL_lose_context");
    extension?.loseContext();
  }
  await runGPURecoveryChecks({createRenderer});
  return checks;
}
if (typeof module !== "undefined") module.exports = {runGPUChecks, runGPURecoveryChecks};
if (typeof window !== "undefined") Object.assign(window, {runGPUChecks, runGPURecoveryChecks});
