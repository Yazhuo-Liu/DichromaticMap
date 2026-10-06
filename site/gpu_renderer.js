"use strict";

// The numerical rows and viewport filtering stay in Float64 on the CPU. This
// renderer receives only final pixel positions, so GPU Float32 arithmetic
// cannot change atom identities, counts, picking, or scientific calculations.
const GPU_MIN_ATOMS = 12_000;
const GPU_FLOATS_PER_ATOM = 12;
const GPU_SUPPORTED_SYMBOLS = new Set(["o", "d"]);

function atomCount(atoms) {
  return atoms?.reduce((sum, grain) => sum + (grain.count ?? grain.length), 0) || 0;
}

function gpuColor(value) {
  if (typeof value !== "string" || !value.startsWith("#")) return null;
  let hex = value.slice(1);
  if (/^[0-9a-f]{3}$/i.test(hex)) hex = [...hex].map(c => c + c).join("");
  if (!/^[0-9a-f]{6}$/i.test(hex)) return null;
  return [0, 2, 4].map(start => parseInt(hex.slice(start, start + 2), 16) / 255).concat(1);
}

function gpuStyles({colors, symbols, sizes, scale}) {
  const fill = gpuColor(colors?.[0]), stroke = gpuColor(colors?.[1]);
  if (!fill || !stroke || !(scale > 0) || !Number.isFinite(scale)) return null;
  const blue = gpuColor("#2e6799"), empty = [0, 0, 0, 0];
  return [0, 1].map(grain => symbols.map((symbol, layer) => {
    if (!GPU_SUPPORTED_SYMBOLS.has(symbol)) return null;
    const radius = Math.max(2, Math.min(8, 4.5 * (sizes[layer] || 1) / Math.sqrt(scale)));
    if (!Number.isFinite(radius)) return null;
    return [radius, symbol === "o" ? 0 : 1, ...(grain ? empty : fill), ...(grain ? stroke : blue)];
  }));
}

function packAtoms(options, storage) {
  const styles = gpuStyles(options);
  if (!styles) return null;
  const count = atomCount(options.atoms);
  const floats = count * GPU_FLOATS_PER_ATOM;
  const data = storage?.length >= floats ? storage : new Float32Array(floats);
  let offset = 0;
  for (let grain = 0; grain < options.atoms.length; grain++) {
    const atoms = options.atoms[grain];
    const length = atoms.count ?? atoms.length;
    for (let index = 0; index < length; index++) {
      let layer, x, y;
      if (atoms.rows) {
        layer = atoms.rows.flat[atoms.indices[index] * atoms.rows.stride + 2];
        x = atoms.x[index]; y = atoms.y[index];
      } else {
        const atom = atoms[index];
        layer = atom.point[2]; x = atom.x; y = atom.y;
      }
      const style = styles[grain]?.[layer];
      if (!style || !Number.isFinite(x) || !Number.isFinite(y)) return null;
      data[offset++] = x; data[offset++] = y;
      data.set(style, offset); offset += style.length;
    }
  }
  return {data, count, view: data.subarray(0, floats)};
}

const GPU_VERTEX_SHADER = `#version 300 es
in vec2 a_position;
in float a_radius;
in float a_shape;
in vec4 a_fill;
in vec4 a_stroke;
uniform vec2 u_size;
uniform float u_dpr;
out float v_radius;
out float v_extent;
flat out float v_shape;
flat out vec4 v_fill;
flat out vec4 v_stroke;
void main() {
  gl_Position = vec4(a_position.x / u_size.x * 2.0 - 1.0,
                     1.0 - a_position.y / u_size.y * 2.0, 0.0, 1.0);
  v_radius = a_radius;
  v_extent = 2.0 * (a_radius + 3.0);
  v_shape = a_shape;
  v_fill = a_fill;
  v_stroke = a_stroke;
  gl_PointSize = v_extent * u_dpr;
}`;

const GPU_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in float v_radius;
in float v_extent;
flat in float v_shape;
flat in vec4 v_fill;
flat in vec4 v_stroke;
uniform float u_dpr;
out vec4 color;
void main() {
  vec2 p = (gl_PointCoord - 0.5) * v_extent;
  float distance = v_shape < 0.5 ? length(p) - v_radius
    : (abs(p.x) + abs(p.y) - v_radius) * 0.70710678118;
  float aa = 0.7 / u_dpr;
  float fill = (1.0 - smoothstep(-aa, aa, distance)) * v_fill.a;
  float stroke = (1.0 - smoothstep(0.625 - aa, 0.625 + aa, abs(distance))) * v_stroke.a;
  float alpha = stroke + fill * (1.0 - stroke);
  color = vec4(v_stroke.rgb * stroke + v_fill.rgb * fill * (1.0 - stroke), alpha);
}`;

class AtomRenderer {
  constructor(documentObject, minimumAtoms) {
    this.canvas = documentObject.createElement("canvas");
    this.gl = this.canvas.getContext("webgl2", {
      alpha: true, antialias: false, premultipliedAlpha: true,
      // The canvas is composited into the existing Canvas 2D plot and its PNG.
      preserveDrawingBuffer: true,
    });
    if (!this.gl) throw new Error("WebGL 2 unavailable");
    this.minimumAtoms = minimumAtoms;
    this.disabled = false;
    this.canvas.addEventListener("webglcontextlost", event => {
      event.preventDefault();
      // Canvas 2D remains available during loss and after restoration.
      this.disabled = true;
    });
    const gl = this.gl;
    const shaders = [this.compile(gl.VERTEX_SHADER, GPU_VERTEX_SHADER),
                     this.compile(gl.FRAGMENT_SHADER, GPU_FRAGMENT_SHADER)];
    this.program = gl.createProgram();
    for (const shader of shaders) gl.attachShader(this.program, shader);
    gl.linkProgram(this.program);
    for (const shader of shaders) gl.deleteShader(shader);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error("GPU atom shader could not link");
    }
    this.buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.useProgram(this.program);
    for (const [name, size, offset] of [["a_position", 2, 0], ["a_radius", 1, 2],
      ["a_shape", 1, 3], ["a_fill", 4, 4], ["a_stroke", 4, 8]]) {
      const location = gl.getAttribLocation(this.program, name);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false,
                            GPU_FLOATS_PER_ATOM * 4, offset * 4);
    }
    this.sizeUniform = gl.getUniformLocation(this.program, "u_size");
    this.dprUniform = gl.getUniformLocation(this.program, "u_dpr");
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    if (gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1] < 44 || gl.getError() !== gl.NO_ERROR) {
      throw new Error("GPU atom rendering unavailable");
    }
  }

  compile(kind, source) {
    const gl = this.gl, shader = gl.createShader(kind);
    gl.shaderSource(shader, source); gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      gl.deleteShader(shader);
      throw new Error("GPU atom shader could not compile");
    }
    return shader;
  }

  canRender({atoms, width, height, dpr}) {
    return !this.disabled && !this.gl.isContextLost() && atomCount(atoms) >= this.minimumAtoms &&
      width > 0 && height > 0 && dpr > 0 && dpr <= 2 &&
      Number.isFinite(width) && Number.isFinite(height) && Number.isFinite(dpr);
  }

  render(options) {
    if (!this.canRender(options)) return null;
    try {
      const signature = JSON.stringify([options.colors, options.symbols, options.sizes, options.scale]);
      // Projected typed arrays can be reused and changed in place during pan.
      // Without an explicit revision, upload every frame rather than relying
      // on object identity and accidentally drawing stale pixel positions.
      if (options.revision === undefined || this.revision !== options.revision ||
          this.atoms !== options.atoms || this.signature !== signature) {
        const packed = packAtoms(options, this.storage);
        if (!packed) return null;
        this.storage = packed.data;
        this.count = packed.count;
        this.atoms = options.atoms;
        this.revision = options.revision;
        this.signature = signature;
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffer);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, packed.view, this.gl.DYNAMIC_DRAW);
      }
      const {width, height, dpr, plot} = options;
      const w = Math.round(width * dpr), h = Math.round(height * dpr);
      if (this.canvas.width !== w) this.canvas.width = w;
      if (this.canvas.height !== h) this.canvas.height = h;
      const gl = this.gl;
      gl.viewport(0, 0, w, h);
      gl.disable(gl.SCISSOR_TEST);
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      gl.enable(gl.SCISSOR_TEST);
      // Canvas 2D applies the final fractional plot clip during composition.
      const left = Math.floor(plot.left * dpr);
      const bottom = Math.floor((height - plot.top - plot.height) * dpr);
      gl.scissor(left, bottom, Math.ceil((plot.left + plot.width) * dpr) - left,
                 Math.ceil((height - plot.top) * dpr) - bottom);
      gl.useProgram(this.program);
      gl.uniform2f(this.sizeUniform, width, height);
      gl.uniform1f(this.dprUniform, dpr);
      // One ordered draw retains grain/layer/point paint order, including
      // overlapping filled atoms and the hollow markers of the second grain.
      gl.drawArrays(gl.POINTS, 0, this.count);
      if (gl.isContextLost() || gl.getError() !== gl.NO_ERROR) {
        this.disabled = true;
        return null;
      }
      return this.canvas;
    } catch (_) {
      this.disabled = true;
      return null;
    }
  }
}

function createRenderer({document: documentObject = globalThis.document, minimumAtoms = GPU_MIN_ATOMS} = {}) {
  try { return new AtomRenderer(documentObject, minimumAtoms); }
  catch (_) { return null; }
}

const gpuExports = {createRenderer, packAtoms, gpuColor, GPU_MIN_ATOMS};
if (typeof module !== "undefined") module.exports = gpuExports;
if (typeof window !== "undefined") window.DichromaticGPU = gpuExports;
