"use strict";

// Scientific rows and exact viewport tests remain Float64 on the CPU. GPU
// coordinates are final screen pixels; a style ID preserves ordered painting.
const GPU_MIN_ATOMS = 12_000;
const GPU_FLOATS_PER_ATOM = 3;
const GPU_MAX_STYLES = 64;
const GPU_SYMBOL_IDS = {o: 0, d: 1, t: 2, s: 3, p: 4, h: 5, star: 6,
  "+": 7, x: 8, t1: 9, t2: 10, t3: 11};

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
  const data = new Float32Array(GPU_MAX_STYLES * 12), lookup = [[], []];
  let count = 0;
  for (let grain = 0; grain < 2; grain++) {
    for (let layer = 0; layer < symbols.length; layer++) {
      const shape = GPU_SYMBOL_IDS[symbols[layer]];
      const radius = Math.max(2, Math.min(8, 4.5 * (sizes[layer] || 1) / Math.sqrt(scale)));
      if (shape === undefined || !Number.isFinite(radius)) { lookup[grain][layer] = -1; continue; }
      if (count >= GPU_MAX_STYLES) return null;
      lookup[grain][layer] = count;
      data.set([radius, shape, 0, 0, ...(grain ? empty : fill), ...(grain ? stroke : blue)], count++ * 12);
    }
  }
  return {data, lookup, count, key: JSON.stringify(lookup),
    fastOnly: symbols.every(symbol => symbol === "o" || symbol === "d")};
}
function packAtoms(options, storage, styles = gpuStyles(options)) {
  if (!styles) return null;
  // visibleData supplies these sets once per exact CPU viewport filter. Avoid
  // even a partial coordinate upload when a visible font glyph needs Canvas.
  for (let grain = 0; grain < options.atoms.length; grain++) {
    for (const layer of options.atoms[grain].layers || []) {
      if (!(styles.lookup[grain][layer] >= 0)) return null;
    }
  }
  const count = atomCount(options.atoms), floats = count * GPU_FLOATS_PER_ATOM;
  const capacity = Math.max(64, 2 ** Math.ceil(Math.log2(Math.max(1, floats))));
  const data = storage?.length >= floats ? storage : new Float32Array(capacity);
  let offset = 0;
  for (let grain = 0; grain < options.atoms.length; grain++) {
    const atoms = options.atoms[grain], length = atoms.count ?? atoms.length;
    for (let index = 0; index < length; index++) {
      let layer, x, y;
      if (atoms.rows) {
        layer = atoms.rows.flat[atoms.indices[index] * atoms.rows.stride + 2];
        x = atoms.x[index]; y = atoms.y[index];
      } else { const atom = atoms[index]; layer = atom.point[2]; x = atom.x; y = atom.y; }
      const style = styles.lookup[grain]?.[layer];
      if (!(style >= 0) || !Number.isFinite(x) || !Number.isFinite(y)) return null;
      data[offset++] = x; data[offset++] = y; data[offset++] = style;
    }
  }
  return {data, count, view: data.subarray(0, floats), styles};
}

const GPU_VERTEX_SHADER = `#version 300 es
in vec2 a_position;
in float a_style;
uniform vec2 u_size;
uniform float u_dpr;
uniform vec4 u_styles[192];
out float v_radius;
out float v_extent;
flat out float v_shape;
flat out vec4 v_fill;
flat out vec4 v_stroke;
void main() {
  int style = int(a_style) * 3;
  vec4 geometry = u_styles[style];
  gl_Position = vec4(a_position.x / u_size.x * 2.0 - 1.0,
                     1.0 - a_position.y / u_size.y * 2.0, 0.0, 1.0);
  v_radius = geometry.x;
  float padding = 3.0;
  if (geometry.y == 6.0) padding = max(padding, (0.625 + 0.7 / u_dpr) / 0.3726480017);
  else if (geometry.y == 2.0 || geometry.y >= 9.0) padding = max(padding, 2.0 * (0.625 + 0.7 / u_dpr));
  v_extent = 2.0 * (geometry.x + padding);
  v_shape = geometry.y;
  v_fill = u_styles[style + 1];
  v_stroke = u_styles[style + 2];
  gl_PointSize = v_extent * u_dpr;
}`;
const GPU_SIMPLE_VERTEX_SHADER = GPU_VERTEX_SHADER.replace("u_styles[192]", "u_styles[12]")
  .replace(/  float padding = 3.0;[\s\S]+?  v_extent = 2.0 \* \(geometry.x \+ padding\);/,
    "  v_extent = 2.0 * (geometry.x + 3.0);");
const GPU_FRAGMENT_SHADER = `#version 300 es
precision highp float;
in float v_radius;
in float v_extent;
flat in float v_shape;
flat in vec4 v_fill;
flat in vec4 v_stroke;
uniform float u_dpr;
out vec4 color;
float boxDistance(vec2 p, vec2 halfSize) {
  vec2 q = abs(p) - halfSize;
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
}
float starDistance(vec2 p, float radius) {
  // Fold into the nearest of the five tips without fragment atan/sin/cos.
  p.x = abs(p.x);
  float up = -p.y;
  float right = 0.9510565163 * p.x - 0.3090169944 * p.y;
  float bottom = 0.5877852523 * p.x + 0.8090169944 * p.y;
  if (right > up && right >= bottom)
    p = vec2(0.3090169944 * p.x + 0.9510565163 * p.y, -right);
  else if (bottom > up)
    p = vec2(-0.8090169944 * p.x + 0.5877852523 * p.y, -bottom);
  // Within this angular sector the boundary is one of the two tip-to-inner
  // edges. Offset planes match Canvas miter joins at outer and inner vertices.
  // The inner vertex radius is 0.44, matching the Canvas ten-vertex path.
  return 0.9279727727 * abs(p.x) - 0.3726480017 * p.y - 0.3726480017 * radius;
}
void main() {
  vec2 p = (gl_PointCoord - 0.5) * v_extent;
  float aa = 0.7 / u_dpr;
  int shape = int(v_shape);
  if (shape == 7 || shape == 8) {
    if (shape == 8) p = vec2(p.x + p.y, p.y - p.x) * 0.70710678118;
    // Canvas coverage of diagonal thin strokes extends farther than an axis
    // aligned stroke. Match that coverage at both supported pixel densities.
    float halfWidth = shape == 8 ? 0.75 : 0.625;
    float distance = min(boxDistance(p, vec2(v_radius, halfWidth)),
                         boxDistance(p, vec2(halfWidth, v_radius)));
    float alpha = (1.0 - smoothstep(-aa, aa, distance)) * v_stroke.a;
    color = vec4(v_stroke.rgb * alpha, alpha);
    return;
  }
  float distance;
  if (shape == 0) distance = length(p) - v_radius;
  else if (shape == 1) distance = (abs(p.x) + abs(p.y) - v_radius) * 0.70710678118;
  else if (shape == 3) distance = max(abs(p.x), abs(p.y)) - v_radius * 0.70710678118;
  else if (shape == 4) distance = max(max(0.5877852523 * abs(p.x) - 0.8090169944 * p.y,
    0.9510565163 * abs(p.x) + 0.3090169944 * p.y), p.y) - 0.8090169944 * v_radius;
  else if (shape == 5) distance = max(abs(p.x), 0.5 * abs(p.x) + 0.8660254038 * abs(p.y))
    - 0.8660254038 * v_radius;
  else if (shape == 6) distance = starDistance(p, v_radius);
  else {
    if (shape == 9) p.y = -p.y;
    else if (shape == 10) p = vec2(p.y, -p.x);
    else if (shape == 11) p = vec2(p.y, p.x);
    distance = max(0.8660254038 * abs(p.x) - 0.5 * p.y, p.y) - 0.5 * v_radius;
  }
  float fill = (1.0 - smoothstep(-aa, aa, distance)) * v_fill.a;
  float halfStroke = (v_shape == 1.0 || v_shape == 6.0) ? 0.75 : 0.625;
  float stroke = (1.0 - smoothstep(halfStroke - aa, halfStroke + aa, abs(distance))) * v_stroke.a;
  float alpha = stroke + fill * (1.0 - stroke);
  color = vec4(v_stroke.rgb * stroke + v_fill.rgb * fill * (1.0 - stroke), alpha);
}`;

// Keep the small default circle/diamond fragment program independent: some
// software/mobile drivers execute a large multi-shape shader less efficiently.
const GPU_SIMPLE_FRAGMENT_SHADER = `#version 300 es
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
  float halfStroke = v_shape == 1.0 ? 0.75 : 0.625;
  float stroke = (1.0 - smoothstep(halfStroke - aa, halfStroke + aa, abs(distance))) * v_stroke.a;
  float alpha = stroke + fill * (1.0 - stroke);
  color = vec4(v_stroke.rgb * stroke + v_fill.rgb * fill * (1.0 - stroke), alpha);
}`;

class AtomRenderer {
  constructor(documentObject, minimumAtoms, {now = () => Date.now(), onRestored = null,
    retryDelay = 1000, maximumRetries = 2} = {}) {
    this.canvas = documentObject.createElement("canvas");
    this.gl = this.canvas.getContext("webgl2", {
      alpha: true, antialias: false, premultipliedAlpha: true, preserveDrawingBuffer: true,
    });
    if (!this.gl) throw new Error("WebGL 2 unavailable");
    this.minimumAtoms = minimumAtoms;
    this.now = now; this.onRestored = onRestored;
    this.retryDelay = retryDelay; this.maximumRetries = maximumRetries;
    this.retries = 0; this.retryAt = 0; this.contextLost = false;
    this.disabled = false;
    this.canvas.addEventListener("webglcontextlost", event => {
      event.preventDefault();
      this.contextLost = true; this.disabled = true; this.retryAt = Infinity;
      // Old GL handles are invalid after loss. Keep only the reusable CPU
      // staging buffer; scientific Float64 rows remain owned by the app.
      this.discardResources(false);
    });
    this.canvas.addEventListener("webglcontextrestored", () => {
      this.contextLost = false; this.disabled = true;
      this.retries = 0; this.retryAt = this.now();
      // The next render rebuilds resources and uploads its latest revision.
      // Notify the app so an idle plot can also resume GPU rendering.
      this.onRestored?.();
    });
    this.initializeResources();
  }
  initializeResources() {
    const gl = this.gl;
    this.programs = {simple: this.createProgram(GPU_SIMPLE_FRAGMENT_SHADER, true), general: null};
    this.program = this.programs.simple.program;
    this.buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer); gl.useProgram(this.program);
    for (const [name, size, offset] of [["a_position", 2, 0], ["a_style", 1, 2]]) {
      const location = gl.getAttribLocation(this.program, name);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, GPU_FLOATS_PER_ATOM * 4, offset * 4);
    }
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    if (gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1] < 44 || gl.getError() !== gl.NO_ERROR)
      throw new Error("GPU atom rendering unavailable");
    this.gpuCapacity = 0;
    this.signature = undefined; this.styles = undefined;
    this.uploadedStyleKey = undefined; this.revision = undefined; this.atoms = undefined;
    this.count = 0;
  }
  discardResources(removeGLObjects = true) {
    if (removeGLObjects && !this.gl.isContextLost()) {
      for (const entry of Object.values(this.programs || {}))
        if (entry) this.gl.deleteProgram(entry.program);
      if (this.buffer) this.gl.deleteBuffer(this.buffer);
    }
    this.programs = null; this.program = null; this.buffer = null; this.gpuCapacity = 0;
  }
  scheduleRetry() {
    this.disabled = true;
    this.retryAt = this.retries >= this.maximumRetries ? Infinity
      : this.now() + this.retryDelay * 2 ** this.retries;
  }
  recover() {
    if (this.contextLost || this.gl.isContextLost() || this.now() < this.retryAt ||
        this.retries >= this.maximumRetries) return false;
    this.retries++;
    try {
      this.discardResources();
      // Consume prior error flags before checking a newly rebuilt pipeline.
      for (let count = 0; count < 16 && this.gl.getError() !== this.gl.NO_ERROR; count++) {}
      this.initializeResources(); this.disabled = false;
      return true;
    } catch (_) { this.scheduleRetry(); return false; }
  }
  createProgram(fragmentSource, simple = false) {
    const gl = this.gl;
    const shaders = [this.compile(gl.VERTEX_SHADER, simple ? GPU_SIMPLE_VERTEX_SHADER : GPU_VERTEX_SHADER), this.compile(gl.FRAGMENT_SHADER, fragmentSource)];
    const program = gl.createProgram();
    for (const shader of shaders) gl.attachShader(program, shader);
    gl.bindAttribLocation(program, 0, "a_position"); gl.bindAttribLocation(program, 1, "a_style");
    gl.linkProgram(program);
    for (const shader of shaders) gl.deleteShader(shader);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error("GPU atom shader could not link");
    return {program, styleSlots: simple ? 4 : GPU_MAX_STYLES, size: gl.getUniformLocation(program, "u_size"),
      dpr: gl.getUniformLocation(program, "u_dpr"), styles: gl.getUniformLocation(program, "u_styles[0]")};
  }
  compile(kind, source) {
    const gl = this.gl, shader = gl.createShader(kind);
    gl.shaderSource(shader, source); gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      gl.deleteShader(shader); throw new Error("GPU atom shader could not compile");
    }
    return shader;
  }
  canRender({atoms, width, height, dpr}) {
    return !this.disabled && !this.gl.isContextLost() && atomCount(atoms) >= this.minimumAtoms &&
      width > 0 && height > 0 && dpr > 0 && dpr <= 2 &&
      Number.isFinite(width) && Number.isFinite(height) && Number.isFinite(dpr);
  }
  render(options) {
    if (this.disabled && !this.recover()) return null;
    if (!this.canRender(options)) return null;
    try {
      const gl = this.gl;
      const signature = JSON.stringify([options.colors, options.symbols, options.sizes, options.scale]);
      if (this.signature !== signature) {
        const styles = gpuStyles(options);
        if (!styles) return null;
        this.styles = styles; this.signature = signature;
      }
      const name = this.styles.fastOnly && this.styles.count <= 4 ? "simple" : "general";
      const program = this.programs[name] || (this.programs[name] = this.createProgram(GPU_FRAGMENT_SHADER));
      this.program = program.program;
      gl.useProgram(this.program);
      if (program.signature !== signature) {
        gl.uniform4fv(program.styles, this.styles.data.subarray(0, program.styleSlots * 12)); program.signature = signature;
      }
      if (options.revision === undefined || this.revision !== options.revision ||
          this.atoms !== options.atoms || this.uploadedStyleKey !== this.styles.key) {
        const packed = packAtoms(options, this.storage, this.styles);
        if (!packed) return null;
        this.storage = packed.data; this.count = packed.count; this.atoms = options.atoms;
        this.revision = options.revision; this.uploadedStyleKey = this.styles.key;
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
        if (this.gpuCapacity < this.storage.byteLength) {
          this.gpuCapacity = this.storage.byteLength;
          gl.bufferData(gl.ARRAY_BUFFER, this.gpuCapacity, gl.DYNAMIC_DRAW);
        }
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, packed.view);
      }
      const {width, height, dpr, plot} = options;
      const w = Math.round(width * dpr), h = Math.round(height * dpr);
      if (this.canvas.width !== w) this.canvas.width = w;
      if (this.canvas.height !== h) this.canvas.height = h;
      gl.viewport(0, 0, w, h); gl.disable(gl.SCISSOR_TEST);
      gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT); gl.enable(gl.SCISSOR_TEST);
      const left = Math.floor(plot.left * dpr), bottom = Math.floor((height - plot.top - plot.height) * dpr);
      gl.scissor(left, bottom, Math.ceil((plot.left + plot.width) * dpr) - left,
        Math.ceil((height - plot.top) * dpr) - bottom);
      gl.useProgram(this.program); gl.uniform2f(program.size, width, height); gl.uniform1f(program.dpr, dpr);
      gl.drawArrays(gl.POINTS, 0, this.count);
      if (gl.isContextLost()) { this.disabled = true; return null; }
      if (gl.getError() !== gl.NO_ERROR) { this.scheduleRetry(); return null; }
      this.retries = 0;
      return this.canvas;
    } catch (_) { this.scheduleRetry(); return null; }
  }
}
function createRenderer({document: documentObject = globalThis.document, minimumAtoms = GPU_MIN_ATOMS, ...options} = {}) {
  try { return new AtomRenderer(documentObject, minimumAtoms, options); }
  catch (_) { return null; }
}
const gpuExports = {createRenderer, packAtoms, gpuColor, gpuStyles, GPU_MIN_ATOMS, GPU_SYMBOL_IDS};
if (typeof module !== "undefined") module.exports = gpuExports;
if (typeof window !== "undefined") window.DichromaticGPU = gpuExports;
