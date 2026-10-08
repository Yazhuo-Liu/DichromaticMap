"use strict";

// Run with a real WebGL 2 browser and the application's marker function.
// CPU unit tests cannot catch shader geometry or drawing-buffer recovery bugs.
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
  return checks;
}
if (typeof module !== "undefined") module.exports = {runGPUChecks};
if (typeof window !== "undefined") window.runGPUChecks = runGPUChecks;
