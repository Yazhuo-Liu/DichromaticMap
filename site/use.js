"use strict";

const $ = (id) => document.getElementById(id);
const canvas = $("map");
const ctx = canvas.getContext("2d");
const identity = () => [[[1, 0], [0, 1]], [[1, 0], [0, 1]]];
const clone = (x) => JSON.parse(JSON.stringify(x));
const state = {
  lattice: "FCC", axis: "110", angle: 38.94244126898139,
  displayRotation: 0, a0: 3.52,
  center: [0, 0], scale: 1, width: 12, height: 9,
  visibleLayers: [new Set([0, 1]), new Set([0, 1])],
  colors: ["#1677d2", "#e35d35"], symbols: ["o", "d"], sizes: [1, 1],
  mode: "idle", boundary: [], atoms: [], axialRepeat: 0,
  regions: [true, true, true, true], referenceAxes: true,
  nearEnabled: false, nearMethod: "local", nearSolutions: [], nearCell: null,
  deformations: identity(), translations: [[0, 0], [0, 0]],
  manual: [], manualLocalCutoff: null, manualFit: null, manualOriginal: null,
  showCell: false, cleanPNG: false,
};
let metadata;
let pattern;
let worker = new Worker("./use_worker.js", { type: "module" });
let callNumber = 0;
const awaiting = new Map();
let renderGeneration = 0;
let nearGeneration = 0;
let analysisGeneration = 0;
let geometryGeneration = 0;
const analysisRequests = new Map();
let renderTimer;
let nearTimer;
let drawFrame;
let canvasMetrics;
let transformCache;
let visibilityCache;
let visibleRevision = 0;
let gpuRenderer;
let renderCoverage;
let renderPending;
let renderError;
let pointerStart;
let dragDistance = 0;
const activePointers = new Map();
let pinchStart;
let gestureWasPinch = false;
let gestureNeedsRender = false;

if (window.matchMedia("(max-width: 820px)").matches) {
  state.referenceAxes = false;
  $("reference-axes").checked = false;
  $("plot-legend-panel").open = false;
}

function setStatus(message) {
  const g1 = pattern ? visiblePoints(0).length : 0;
  const g2 = pattern ? visiblePoints(1).length : 0;
  const cslPoints = pattern ? visibleCSL() : [];
  const layerCounts = currentPattern() ? visibleData().layerCounts : new Map();
  const byLayer = Array.from(layerCounts, ([layer, count]) => `${layer < 26 ? String.fromCharCode(65 + layer) : `L${layer + 1}`}: ${count}`).join(", ");
  setHTML($("status"), `<strong>${escapeHtml(message)}</strong>Drag to pan · pinch or wheel to zoom<br>Visible G1 / G2: ${g1} / ${g2}<br>Same-layer CSL: ${cslPoints.length}${byLayer ? ` · ${escapeHtml(byLayer)}` : ""}<br>Compute: on this device`);
}
function setHTML(element, html) {
  if (element.innerHTML !== html) element.innerHTML = html;
}
function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"})[c]);
}
function request(action, data = {}) {
  return new Promise((resolve, reject) => {
    const id = ++callNumber;
    awaiting.set(id, {resolve, reject});
    worker.postMessage({id, request: {action, ...data}});
  });
}
worker.onmessage = ({data}) => {
  if (data.type === "ready") { setStatus("Ready · choose an interaction tool"); return; }
  const item = awaiting.get(data.id);
  if (!item) return;
  awaiting.delete(data.id);
  if (data.type === "error") {
    const error = new Error(data.message);
    error.name = data.name || "Error";
    item.reject(error);
  }
  else item.resolve(data.result);
};
worker.onerror = () => fail(new Error("The local engine could not start. Check your connection and reload."));
function fail(error) {
  if (error.name === "AbortError") return;
  const message = error.message || String(error);
  setStatus(message);
  if (!pattern) {
    $("engine-overlay").querySelector("strong").textContent = "Unable to start the online app";
    $("engine-overlay").querySelector("span").textContent = message;
    $("engine-overlay").querySelector(".engine-spinner").classList.add("hidden");
  }
}

function viewSize() {
  const {width: drawW, height: drawH} = refreshCanvasMetrics().plot;
  state.width = 12 * state.scale;
  state.height = state.width * drawH / drawW;
}
function renderDimensions() {
  // The lattice engine crops in unrotated coordinates. Cover every corner of
  // the displayed rectangle after rotating it back into those coordinates.
  const radians = state.displayRotation * Math.PI / 180;
  const cosine = Math.abs(Math.cos(radians));
  const sine = Math.abs(Math.sin(radians));
  return {
    width: Math.max(state.width * 1.43, cosine * state.width + sine * state.height + 2),
    height: Math.max(state.height * 1.43, sine * state.width + cosine * state.height + 2),
  };
}
function rotate(point, degrees) {
  const a = degrees * Math.PI / 180, c = Math.cos(a), s = Math.sin(a);
  return [c * point[0] - s * point[1], s * point[0] + c * point[1]];
}
function plotRect() {
  return (canvasMetrics || refreshCanvasMetrics()).plot;
}
function refreshCanvasMetrics() {
  const rect = canvas.getBoundingClientRect();
  if (!canvasMetrics || canvasMetrics.width !== rect.width || canvasMetrics.height !== rect.height) {
    canvasMetrics = {width: rect.width, height: rect.height,
      plot: {left: 35, top: 30, width: Math.max(100, rect.width - 63),
             height: Math.max(100, rect.height - 66)}};
  }
  return canvasMetrics;
}
function viewTransform() {
  const r = plotRect();
  if (!transformCache || transformCache.rect !== r ||
      transformCache.cx !== state.center[0] || transformCache.cy !== state.center[1] ||
      transformCache.width !== state.width || transformCache.height !== state.height ||
      transformCache.rotation !== state.displayRotation) {
    const radians = state.displayRotation * Math.PI / 180;
    transformCache = {rect: r, cx: state.center[0], cy: state.center[1],
      width: state.width, height: state.height, rotation: state.displayRotation,
      cosine: Math.cos(radians), sine: Math.sin(radians),
      ox: r.left + r.width / 2, oy: r.top + r.height / 2};
  }
  return transformCache;
}
function screen(point) {
  const t = viewTransform();
  return [t.ox + (t.cosine * point[0] - t.sine * point[1] - t.cx) * t.rect.width / t.width,
          t.oy - (t.sine * point[0] + t.cosine * point[1] - t.cy) * t.rect.height / t.height];
}
function modelFromScreen(x, y) {
  const r = plotRect();
  const displayed = [state.center[0] + (x - r.left - r.width / 2) * state.width / r.width,
                     state.center[1] - (y - r.top - r.height / 2) * state.height / r.height];
  return rotate(displayed, -state.displayRotation);
}
function inView(point) {
  const [x, y] = screen(point), r = plotRect();
  return x >= r.left && x <= r.left + r.width && y >= r.top && y <= r.top + r.height;
}
function sideVisible(point, grain) {
  return sideVisibleXY(point[0], point[1], grain);
}
function sideVisibleXY(x, y, grain) {
  if (state.boundary.length !== 2) return true;
  const [a, b] = state.boundary;
  const cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
  return (state.regions[2 * grain] && cross >= -1e-9) ||
         (state.regions[2 * grain + 1] && cross <= 1e-9);
}
function visiblePoints(grain) {
  return currentPattern() ? visibleData().points[grain] : [];
}
function visibleCSL() {
  return currentPattern() ? visibleData().csl : [];
}
function visibleLocal() {
  return currentPattern() ? visibleData().local : [];
}
function currentRenderParameters() {
  return {
    lattice: state.lattice, axis: state.axis, angle: state.angle,
    deformations: state.deformations, translations: state.translations,
    local_matching: state.nearEnabled && state.nearMethod === "local",
    local_distance: Number($("local-distance").value),
  };
}
function currentPattern() {
  return !!pattern && renderCoverage?.signature === JSON.stringify(currentRenderParameters());
}
async function waitForCurrentRender() {
  for (;;) {
    viewSize();
    if (currentPattern() && coveredView(renderCoverage)) return;
    if (!renderPending || renderPending.settled) renderRequest();
    const pending = renderPending;
    await pending.promise;
    if (pending !== renderPending) continue;
    if (renderError) throw renderError;
  }
}
function visibleData() {
  const transform = viewTransform();
  const selection = `${[...state.visibleLayers[0]]}|${[...state.visibleLayers[1]]}|${state.boundary.flat()}|${state.regions}|${state.nearEnabled}|${state.nearMethod}`;
  if (visibilityCache?.pattern === pattern && visibilityCache.transform === transform &&
      visibilityCache.selection === selection) return visibilityCache;
  const {DenseRows, SelectedRows} = window.DichromaticRenderData;
  const makeSelection = (source, stride, local = false) => {
    const rows = DenseRows.from(source || [], stride);
    const result = {rows, indices: new Uint32Array(rows.length), x: new Float64Array(rows.length),
      y: new Float64Array(rows.length), count: 0, layers: new Set(),
      spatial: window.DichromaticSpatialIndex
        ? new window.DichromaticSpatialIndex.RowSpatialIndex(rows, {midpoints: local}) : null};
    if (local) { result.x2 = new Float64Array(rows.length); result.y2 = new Float64Array(rows.length); }
    result.points = new SelectedRows(rows, result);
    return result;
  };
  // Reuse projected buffers and index selections across pans; no row arrays
  // or per-atom records are allocated during a redraw.
  const data = visibilityCache?.pattern === pattern ? visibilityCache : {
    pattern, atoms: [makeSelection(pattern?.grains[0], 6), makeSelection(pattern?.grains[1], 6)],
    cslMarkers: makeSelection(pattern?.coincidences, 3),
    localMarkers: makeSelection(pattern?.local, 5, true), layerCounts: new Map(), maxLocalSeparation: 0,
  };
  data.transform = transform; data.selection = selection; data.revision = ++visibleRevision;
  data.points = [data.atoms[0].points, data.atoms[1].points];
  data.csl = data.cslMarkers.points; data.local = data.localMarkers.points;
  data.layerCounts.clear(); data.maxLocalSeparation = 0;
  const r = transform.rect;
  const {cosine, sine, cx, cy, ox, oy, width, height} = transform;
  for (let grain = 0; grain < 2; grain++) {
    const record = data.atoms[grain], flat = record.rows.flat;
    record.count = 0; record.layers.clear();
    const candidates = record.spatial?.queryViewport(transform);
    const length = candidates?.length ?? record.rows.length;
    for (let cursor = 0; cursor < length; cursor++) {
      const index = candidates ? candidates[cursor] : cursor;
      const offset = index * 6, px = flat[offset], py = flat[offset + 1], layer = flat[offset + 2];
      if (!state.visibleLayers[grain].has(layer) || !sideVisibleXY(px, py, grain)) continue;
      const x = ox + (cosine * px - sine * py - cx) * r.width / width;
      const y = oy - (sine * px + cosine * py - cy) * r.height / height;
      if (x >= r.left && x <= r.left + r.width && y >= r.top && y <= r.top + r.height) {
        const at = record.count++;
        record.indices[at] = index; record.x[at] = x; record.y[at] = y;
        record.layers.add(layer);
      }
    }
  }
  const csl = data.cslMarkers, common = csl.rows.flat;
  csl.count = 0;
  const commonCandidates = csl.spatial?.queryViewport(transform);
  const commonLength = commonCandidates?.length ?? csl.rows.length;
  for (let cursor = 0; cursor < commonLength; cursor++) {
    const index = commonCandidates ? commonCandidates[cursor] : cursor;
    const offset = index * 3, px = common[offset], py = common[offset + 1], layer = common[offset + 2];
    if (!state.visibleLayers[0].has(layer) || !state.visibleLayers[1].has(layer) ||
        !sideVisibleXY(px, py, 0) || !sideVisibleXY(px, py, 1)) continue;
    const x = ox + (cosine * px - sine * py - cx) * r.width / width;
    const y = oy - (sine * px + cosine * py - cy) * r.height / height;
    if (x >= r.left && x <= r.left + r.width && y >= r.top && y <= r.top + r.height) {
      const at = csl.count++;
      csl.indices[at] = index; csl.x[at] = x; csl.y[at] = y;
      data.layerCounts.set(layer, (data.layerCounts.get(layer) || 0) + 1);
    }
  }
  const local = data.localMarkers, pairs = local.rows.flat;
  local.count = 0;
  if (state.nearEnabled && state.nearMethod === "local") {
    const localCandidates = local.spatial?.queryViewport(transform);
    const localLength = localCandidates?.length ?? local.rows.length;
    for (let cursor = 0; cursor < localLength; cursor++) {
      const index = localCandidates ? localCandidates[cursor] : cursor;
      const offset = index * 5, ax = pairs[offset], ay = pairs[offset + 1],
        bx = pairs[offset + 2], by = pairs[offset + 3], layer = pairs[offset + 4];
      if (!state.visibleLayers[0].has(layer) || !state.visibleLayers[1].has(layer) ||
          !sideVisibleXY(ax, ay, 0) || !sideVisibleXY(bx, by, 1)) continue;
      const mx = (ax + bx) / 2, my = (ay + by) / 2;
      const x = ox + (cosine * mx - sine * my - cx) * r.width / width;
      const y = oy - (sine * mx + cosine * my - cy) * r.height / height;
      if (x >= r.left && x <= r.left + r.width && y >= r.top && y <= r.top + r.height) {
        const at = local.count++;
        local.indices[at] = index;
        local.x[at] = ox + (cosine * ax - sine * ay - cx) * r.width / width;
        local.y[at] = oy - (sine * ax + cosine * ay - cy) * r.height / height;
        local.x2[at] = ox + (cosine * bx - sine * by - cx) * r.width / width;
        local.y2[at] = oy - (sine * bx + cosine * by - cy) * r.height / height;
        data.maxLocalSeparation = Math.max(data.maxLocalSeparation, Math.hypot(ax - bx, ay - by));
      }
    }
  }
  visibilityCache = data;
  return data;
}
function renderRequest() {
  clearTimeout(renderTimer);
  renderPending?.finish();
  renderError = undefined;
  const pending = {settled: false};
  pending.promise = new Promise(resolve => {
    pending.finish = () => { pending.settled = true; resolve(); };
  });
  renderPending = pending;
  worker.postMessage({type: "cancel", action: "render"});
  const generation = ++renderGeneration;
  scheduleDraw();
  renderTimer = setTimeout(async () => {
    try {
      viewSize();
      const modelCenter = rotate(state.center, -state.displayRotation);
      const dimensions = renderDimensions();
      const parameters = clone(currentRenderParameters());
      const signature = JSON.stringify(parameters);
      if (renderCoverage?.signature === signature && coveredView(renderCoverage)) {
        draw(); updateSummary(); setStatus("Ready · choose an interaction tool"); return;
      }
      setStatus("Calculating lattice…");
      const raw = await request("render", {
        ...parameters, width: dimensions.width, height: dimensions.height, center: modelCenter,
      });
      if (generation !== renderGeneration || signature !== JSON.stringify(currentRenderParameters())) return;
      const result = window.DichromaticRenderData.decodeRenderResult(raw);
      state.angle = result.angle;
      pattern = result;
      renderCoverage = {signature: JSON.stringify({...parameters, angle: result.angle}),
        center: modelCenter, ...dimensions};
      $("engine-overlay").classList.add("hidden");
      document.querySelector(".control-pane").classList.remove("loading");
      $("tutorial-start").disabled = false;
      $("tutorial-start").title = "Explore the controls with a guided tutorial";
      draw();
      updateSummary();
      setStatus("Ready · choose an interaction tool");
    } catch (error) {
      if (generation === renderGeneration) { renderError = error; fail(error); }
    } finally { pending.finish(); }
  }, 45);
  return pending.promise;
}
function coveredView(coverage) {
  for (const x of [-state.width / 2, state.width / 2]) {
    for (const y of [-state.height / 2, state.height / 2]) {
      const point = rotate([state.center[0] + x, state.center[1] + y], -state.displayRotation);
      if (Math.abs(point[0] - coverage.center[0]) > coverage.width / 2 ||
          Math.abs(point[1] - coverage.center[1]) > coverage.height / 2) return false;
    }
  }
  return true;
}

function resizeCanvas() {
  const rect = canvasMetrics || refreshCanvasMetrics();
  const dpr = Math.min(devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
function marker(context, x, y, radius, symbol, stroke, fill, lineWidth = 1.25) {
  context.beginPath();
  if (symbol === "o" || symbol?.startsWith("number:")) {
    context.arc(x, y, radius, 0, Math.PI * 2);
  } else if (symbol === "+" || symbol === "x") {
    const a = symbol === "x" ? Math.PI / 4 : 0;
    for (let i = 0; i < 2; i++) {
      const t = a + i * Math.PI / 2;
      context.moveTo(x - radius * Math.cos(t), y - radius * Math.sin(t));
      context.lineTo(x + radius * Math.cos(t), y + radius * Math.sin(t));
    }
  } else {
    const sides = {d:4,t:3,s:4,p:5,h:6,star:10,t1:3,t2:3,t3:3}[symbol] || 6;
    const start = {d: 0, s: -Math.PI / 4, t1: Math.PI / 2, t2: 0, t3: Math.PI}[symbol] ?? -Math.PI / 2;
    for (let i = 0; i < sides; i++) {
      const rad = start + i * Math.PI * 2 / sides;
      const rr = symbol === "star" && i % 2 ? radius * .44 : radius;
      const px = x + rr * Math.cos(rad), py = y + rr * Math.sin(rad);
      if (!i) context.moveTo(px, py); else context.lineTo(px, py);
    }
    context.closePath();
  }
  if (fill) { context.fillStyle = fill; context.fill(); }
  context.strokeStyle = stroke;
  context.lineWidth = lineWidth;
  context.stroke();
  if (symbol?.startsWith("number:")) {
    context.save();
    context.textAlign = "center"; context.textBaseline = "middle";
    context.font = `bold ${Math.max(6, Math.min(10, radius * 1.3))}px Arial`;
    context.fillStyle = fill ? "#ffffff" : stroke;
    context.fillText(symbol.slice(7), x, y + .5);
    context.restore();
  }
}
function drawGrid(context, r) {
  context.fillStyle = "#f7f9fb";
  context.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
  context.save();
  context.beginPath(); context.rect(r.left, r.top, r.width, r.height); context.clip();
  const xMin = state.center[0] - state.width / 2, xMax = state.center[0] + state.width / 2;
  const yMin = state.center[1] - state.height / 2, yMax = state.center[1] + state.height / 2;
  for (let step of [0.25, 1]) {
    context.beginPath();
    context.strokeStyle = step === 1 ? "#d9e1ea" : "#ebeff4";
    context.lineWidth = 1;
    for (let x = Math.ceil(xMin / step) * step; x <= xMax; x += step) {
      const px = r.left + (x - xMin) / state.width * r.width;
      context.moveTo(px, r.top); context.lineTo(px, r.top + r.height);
    }
    for (let y = Math.ceil(yMin / step) * step; y <= yMax; y += step) {
      const py = r.top + (yMax - y) / state.height * r.height;
      context.moveTo(r.left, py); context.lineTo(r.left + r.width, py);
    }
    context.stroke();
  }
  context.restore();
  context.strokeStyle = "#8498c0";
  context.strokeRect(r.left + .5, r.top + .5, r.width, r.height);
  context.fillStyle = "#52627b"; context.font = "11px Arial";
  for (let x = Math.ceil(xMin); x <= xMax; x++) {
    const px = r.left + (x - xMin) / state.width * r.width;
    context.textAlign = "center"; context.fillText(String(x), px, r.top + r.height + 14);
  }
  for (let y = Math.ceil(yMin); y <= yMax; y++) {
    const py = r.top + (yMax - y) / state.height * r.height;
    context.textAlign = "right"; context.fillText(String(y), r.left - 5, py + 4);
  }
  context.textAlign = "start";
}
function cellCorners(cell) {
  const c = cell.cell;
  const origin = state.manualFit ? state.manualFit.origin : [0, 0];
  return [origin, [origin[0] + c[0][0], origin[1] + c[1][0]],
    [origin[0] + c[0][0] + c[0][1], origin[1] + c[1][0] + c[1][1]],
    [origin[0] + c[0][1], origin[1] + c[1][1]]];
}
function drawPolygon(context, points, color, dash = []) {
  if (points.length < 2) return;
  context.save(); context.strokeStyle = color; context.lineWidth = 2;
  context.setLineDash(dash); context.beginPath();
  points.forEach((point, i) => { const [x, y] = screen(point); if (!i) context.moveTo(x, y); else context.lineTo(x, y); });
  if (points.length === 4) context.closePath();
  context.stroke(); context.restore();
}
function drawVector(context) {
  const points = state.atoms.map(atom => atom.position);
  drawPolygon(context, points, "#9347aa");
  if (points.length !== 2) return;
  const [x1, y1] = screen(points[0]);
  const [x2, y2] = screen(points[1]);
  const dx = x2 - x1, dy = y2 - y1;
  const length = Math.hypot(dx, dy);
  if (length <= 4) return;
  const ux = dx / length, uy = dy / length;
  const tipOffset = Math.min(8, length * .25);
  const headLength = Math.min(18, (length - tipOffset) * .6);
  const halfWidth = Math.min(7, headLength * .4);
  const tipX = x2 - ux * tipOffset, tipY = y2 - uy * tipOffset;
  const baseX = tipX - ux * headLength, baseY = tipY - uy * headLength;
  context.save();
  context.fillStyle = "#9347aa"; context.strokeStyle = "#9347aa"; context.lineWidth = 1.2;
  context.beginPath();
  context.moveTo(tipX, tipY);
  context.lineTo(baseX - uy * halfWidth, baseY + ux * halfWidth);
  context.lineTo(baseX + uy * halfWidth, baseY - ux * halfWidth);
  context.closePath(); context.fill(); context.stroke();
  context.restore();
}
function clippedBoundary() {
  if (state.boundary.length !== 2) return null;
  const first = screen(state.boundary[0]), second = screen(state.boundary[1]);
  const direction = [second[0] - first[0], second[1] - first[1]];
  const r = plotRect();
  const limits = [[r.left, r.left + r.width], [r.top, r.top + r.height]];
  let start = -Infinity, end = Infinity;
  for (let axis = 0; axis < 2; axis++) {
    if (Math.abs(direction[axis]) < 1e-10) {
      if (first[axis] < limits[axis][0] || first[axis] > limits[axis][1]) return null;
      continue;
    }
    const bounds = limits[axis].map(value => (value - first[axis]) / direction[axis]);
    start = Math.max(start, Math.min(...bounds));
    end = Math.min(end, Math.max(...bounds));
  }
  if (start >= end) return null;
  return [start, end].map(t => [first[0] + t * direction[0], first[1] + t * direction[1]]);
}
function drawBoundary(context) {
  const ends = clippedBoundary();
  if (ends) {
    context.save(); context.strokeStyle = "#17212b"; context.lineWidth = 2.4;
    context.beginPath(); context.moveTo(...ends[0]); context.lineTo(...ends[1]);
    context.stroke(); context.restore();
  }
  state.boundary.forEach((point, i) => {
    const [x, y] = screen(point);
    if (!inView(point)) return;
    marker(context, x, y, 9, "o", "#111827", "#ffffff");
    context.fillStyle = "#23334b"; context.fillText(`B${i + 1}`, x + 10, y - 9);
  });
}
function drawAtoms(context, clean = false) {
  if (!currentPattern()) return;
  const r = plotRect();
  const visible = visibleData();
  context.save(); context.beginPath(); context.rect(r.left, r.top, r.width, r.height); context.clip();
  let gpuCanvas = null;
  if (context === ctx && !clean && window.DichromaticGPU &&
      visible.atoms[0].count + visible.atoms[1].count >= 12000) {
    if (gpuRenderer === undefined) gpuRenderer = window.DichromaticGPU.createRenderer();
    if (gpuRenderer) {
      const metrics = canvasMetrics || refreshCanvasMetrics();
      gpuCanvas = gpuRenderer.render({width: metrics.width, height: metrics.height,
        dpr: Math.min(devicePixelRatio || 1, 2), plot: r, atoms: visible.atoms,
        revision: visible.revision, colors: state.colors, symbols: state.symbols, sizes: state.sizes, scale: state.scale});
    }
  }
  if (gpuCanvas) {
    context.drawImage(gpuCanvas, 0, 0, canvasMetrics.width, canvasMetrics.height);
  } else {
    for (let grain = 0; grain < 2; grain++) {
      const atoms = visible.atoms[grain];
      for (let index = 0; index < atoms.count; index++) {
        const layer = atoms.rows.flat[atoms.indices[index] * 6 + 2], x = atoms.x[index], y = atoms.y[index];
        const radius = Math.max(2, Math.min(8, 4.5 * (state.sizes[layer] || 1) / Math.sqrt(state.scale)));
        marker(context, x, y, radius, state.symbols[layer],
          grain === 0 ? "#2e6799" : state.colors[1], grain === 0 ? state.colors[0] : null);
      }
    }
  }
  if (!clean) {
    const csl = visible.cslMarkers;
    for (let index = 0; index < csl.count; index++) {
      const layer = csl.rows.flat[csl.indices[index] * 3 + 2];
      marker(context, csl.x[index], csl.y[index], 10, state.symbols[layer], "#e5a50a", null, 2.2);
    }
    const local = visible.localMarkers;
    for (let index = 0; index < local.count; index++) {
      const x1 = local.x[index], y1 = local.y[index], x2 = local.x2[index], y2 = local.y2[index];
      context.strokeStyle = "#aa45bb"; context.lineWidth = 1.4; context.setLineDash([2, 2]);
      context.beginPath(); context.moveTo(x1, y1); context.lineTo(x2, y2); context.stroke(); context.setLineDash([]);
      context.beginPath(); context.arc((x1 + x2) / 2, (y1 + y2) / 2, 10, 0, Math.PI * 2);
      context.strokeStyle = "#aa45bb"; context.lineWidth = 2; context.stroke();
    }
  }
  context.restore();
}
function draw() {
  if (drawFrame !== undefined) { cancelAnimationFrame(drawFrame); drawFrame = undefined; }
  refreshCanvasMetrics();
  resizeCanvas();
  const r = plotRect();
  drawGrid(ctx, r);
  // Keep plot overlays, including the supercell, inside the plotted axes.
  ctx.save();
  ctx.beginPath();
  ctx.rect(r.left, r.top, r.width, r.height);
  ctx.clip();
  drawAtoms(ctx);
  if (state.showCell) {
    const cell = state.nearCell || (currentPattern() ? pattern?.exact_cell : null);
    if (cell) drawPolygon(ctx, cellCorners(cell), "#0d9fa1", [7, 4]);
  }
  if (state.manual.length) {
    for (let grain = 0; grain < 2; grain++) {
      drawPolygon(ctx, state.manual.map(v => v.endpoints[grain]), state.colors[grain]);
    }
    state.manual.forEach((v, i) => {
      const [x, y] = screen(v.position);
      ctx.fillStyle = "#27405f"; ctx.font = "bold 11px Arial";
      ctx.fillText(`C${i + 1}`, x + 8, y - 7);
    });
  }
  if (state.boundary.length) drawBoundary(ctx);
  if (state.atoms.length) {
    drawVector(ctx);
    state.atoms.forEach((v, i) => { const [x, y] = screen(v.position); ctx.fillStyle = "#83419a"; ctx.fillText(`P${i + 1}`, x + 6, y - 7); });
  }
  ctx.restore();
  ctx.strokeStyle = "#8498c0";
  ctx.strokeRect(r.left + .5, r.top + .5, r.width, r.height);
  if (state.referenceAxes) drawReferenceAxes(ctx, r);
  updateReferenceAxesToggle();
  positionVectorAnnotation();
  drawLegend();
}
function scheduleDraw() {
  if (drawFrame !== undefined) return;
  drawFrame = requestAnimationFrame(() => { drawFrame = undefined; draw(); });
}
function updateReferenceAxesToggle() {
  const button = $("reference-axes-toggle");
  const box = referenceAxesBounds(plotRect());
  button.style.left = `${state.referenceAxes ? box.left + box.width - 57 : box.left}px`;
  button.style.top = state.referenceAxes ? `${box.top + 4}px` : "";
  button.style.bottom = state.referenceAxes ? "" : "8px";
  button.textContent = state.referenceAxes ? "Axes −" : "Axes +";
  button.setAttribute("aria-label", state.referenceAxes ? "Hide grain reference axes" : "Show grain reference axes");
  button.setAttribute("aria-pressed", String(state.referenceAxes));
}
function positionVectorAnnotation() {
  const annotation = $("vector-annotation");
  if (state.referenceAxes && currentPattern() && pattern?.reference_axes && metadata?.reference_labels) {
    const annotationBottom = referenceAxesBounds(plotRect()).top - 10;
    annotation.style.bottom = `${canvas.clientHeight - annotationBottom}px`;
    annotation.style.maxHeight = `${Math.max(1, annotationBottom - 45)}px`;
  } else {
    annotation.style.bottom = "42px";
    annotation.style.maxHeight = "";
  }
}
function referenceAxesBounds(r) {
  return {left: r.left + 14, top: r.top + r.height - 138, width: 178, height: 123};
}
function drawReferenceAxes(context, r) {
  if (!currentPattern() || !pattern?.reference_axes || !metadata?.reference_labels) return;
  const box = referenceAxesBounds(r);
  const x = box.left + 68, y = box.top + 67;
  context.save();
  context.fillStyle = "#fffffff0"; context.strokeStyle = "#cbd5e1";
  context.fillRect(box.left, box.top, box.width, box.height);
  context.strokeRect(box.left + .5, box.top + .5, box.width - 1, box.height - 1);
  for (let grain = 0; grain < 2; grain++) {
    for (let axis = 0; axis < 2; axis++) {
      const direction = rotate(pattern.reference_axes[grain][axis], state.displayRotation);
      context.strokeStyle = state.colors[grain]; context.fillStyle = state.colors[grain];
      context.lineWidth = 2; context.beginPath(); context.moveTo(x, y);
      const ex = x + direction[0] * 31, ey = y - direction[1] * 31;
      context.lineTo(ex, ey); context.stroke();
      context.beginPath(); context.arc(ex, ey, 2.5, 0, Math.PI * 2); context.fill();
      context.font = "10px Arial";
      context.fillText(metadata.reference_labels[axis], ex + direction[0] * 8 - 13,
                       ey - direction[1] * 8 + 3);
    }
  }
  context.restore();
}
function drawLegend() {
  if (!metadata) return;
  const parts = [];
  const filled = {o:"●",d:"◆",t:"▲",s:"■",p:"⬟",h:"⬢",star:"★","+":"+",x:"×",t1:"▼",t2:"▶",t3:"◀"};
  const outline = {o:"○",d:"◇",t:"△",s:"□",p:"⬠",h:"⬡",star:"☆","+":"+",x:"×",t1:"▽",t2:"▷",t3:"◁"};
  for (let layer = 0; layer < Math.min(metadata.layers, 6); layer++) {
    for (let grain = 0; grain < 2; grain++) {
      if (!state.visibleLayers[grain].has(layer)) continue;
      const name = layer < 26 ? String.fromCharCode(65 + layer) : `L${layer + 1}`;
      const color = state.colors[grain];
      const symbol = state.symbols[layer];
      const glyph = grain ? (outline[symbol] || "○") : (filled[symbol] || "●");
      parts.push(`<span><b class="legend-marker" style="color:${color}">${glyph}</b>G${grain + 1} · ${name}</span>`);
    }
  }
  if (visibleCSL().length) parts.push(`<span><b class="legend-marker" style="color:#e5a50a">◎</b>CSL</span>`);
  if (visibleLocal().length) parts.push(`<span><b class="legend-marker" style="color:#aa45bb">◎</b>Near pair</span>`);
  setHTML($("plot-legend"), parts.join(""));
}

function updateSummary() {
  if (!metadata) return;
  for (const [id, mode] of [["pick-gb","boundary"],["pick-vector","vector"],["pick-cell","cell"]]) {
    $(id).classList.toggle("selected", state.mode === mode);
  }
  const name = `${state.lattice} ⟨${state.axis}⟩`;
  const preset = currentPattern() && pattern?.preset ? ` · ${pattern.preset}` : "";
  $("control-title").textContent = `${name} GB`;
  $("plot-title").textContent = `${name} dichromatic pattern · θ = ${state.angle.toFixed(2)}°${preset}`;
  $("angle-hint").textContent = `Exact θ = ${state.angle.toFixed(8)}° · allowed 0–${metadata.max_angle}°`;
  $("layer-info").textContent = `${metadata.layers} axial layers per grain · spacing = ${metadata.layer_spacing.toPrecision(5)} a₀ · axial repeat = ${metadata.axial_period.toPrecision(5)} a₀. CSL and near pairs require visibility in both grains.`;
  $("gb-options").classList.toggle("hidden", state.boundary.length !== 2);
  $("near-toggle").textContent = state.nearEnabled ? "Disable Near-CSL" : "Enable Near-CSL";
  $("near-toggle").classList.toggle("selected", state.nearEnabled);
  $("near-toggle").disabled = !!state.manualFit;
  $("near-method").disabled = !!state.manualFit;
  $("pick-cell").disabled = !!state.manualFit;
  $("undo-cell").disabled = !!state.manualFit || !state.manual.length;
  $("clear-cell").disabled = !!state.manualFit || !state.manual.length;
  $("run-search").classList.toggle("hidden", !state.nearEnabled || state.nearMethod !== "strain");
  $("local-distance").disabled = !state.nearEnabled || state.nearMethod !== "local";
  $("strain-percent").disabled = !state.nearEnabled || state.nearMethod !== "strain";
  $("search-index").disabled = !state.nearEnabled || state.nearMethod !== "strain";
  $("local-distance").closest("label").classList.toggle("hidden", state.nearMethod !== "local");
  $("strain-percent").closest("label").classList.toggle("hidden", state.nearMethod !== "strain");
  $("search-index").closest("label").classList.toggle("hidden", state.nearMethod !== "strain");
  $("near-results").classList.toggle("hidden", !state.nearSolutions.length);
  if (state.nearEnabled && state.nearMethod === "local" && currentPattern() && pattern?.local) {
    const visible = visibleData();
    $("near-info").textContent = `Local same-layer mutual nearest pairs\nVisible pairs: ${visible.localMarkers.count}\nMaximum visible separation: ${visible.localMarkers.count ? visible.maxLocalSeparation.toFixed(5) : "—"} a₀\nOriginal atom positions retained.`;
  }
  $("fit-cell").disabled = !(state.nearCell || (currentPattern() ? pattern?.exact_cell : null));
  $("complete-cell").disabled = !(state.manual.length === 2 || state.manual.length === 3);
  $("fit-manual").disabled = !state.manual.length;
  $("apply-strain").disabled = state.manual.length !== 4 ||
    (!state.nearEnabled || state.nearMethod !== "local") && !state.manualFit ||
    !state.manual.some(v => v.source !== "CSL") && !state.manualFit;
  $("apply-strain").textContent = state.manualFit ? "Restore original local structure" : "Apply bulk strain to selected cell";
  $("manual-info").textContent = state.manual.length
    ? `Selected ${state.manual.length}/4 vertices in layer ${state.manual[0].layer + 1}.\n${$("manual-info").dataset.count || "Continue around the perimeter in the same layer."}`
    : "No manual cell. Pick one layer; each grain is counted inside its own four actual atom vertices.";
  $("field-label").textContent = `${state.scale.toFixed(1)}×`;
  $("angle-number").value = state.angle.toFixed(2);
  $("angle-slider").value = String(state.angle);
  $("rotation-number").value = String(state.displayRotation);
  $("rotation-slider").value = String(state.displayRotation);
}
function rebuildPreset() {
  $("preset").replaceChildren(new Option("Custom angle", ""));
  for (const preset of metadata.presets) $("preset").add(new Option(preset.label, String(preset.angle)));
  const found = metadata.presets.find(p => Math.abs(p.angle - state.angle) < 1e-8);
  $("preset").value = found ? String(found.angle) : "";
}
function rebuildLayers() {
  const checks = $("layer-checks"); checks.replaceChildren();
  const appearance = $("appearance-layers"); appearance.replaceChildren();
  for (let layer = 0; layer < metadata.layers; layer++) {
    for (let grain = 0; grain < 2; grain++) {
      const label = document.createElement("label");
      const check = document.createElement("input"); check.type = "checkbox";
      check.checked = state.visibleLayers[grain].has(layer);
      check.addEventListener("change", () => {
        if (check.checked) state.visibleLayers[grain].add(layer);
        else state.visibleLayers[grain].delete(layer);
        draw(); updateSummary(); setStatus("Layer visibility updated");
      });
      label.append(check, ` G${grain + 1} ${layer < 26 ? String.fromCharCode(65 + layer) : `L${layer + 1}`}`);
      checks.append(label);
    }
    const row = document.createElement("div"); row.className = "appearance-row";
    const title = document.createElement("span"); title.textContent = layer < 26 ? `Layer ${String.fromCharCode(65 + layer)}` : `L${layer + 1}`;
    const symbol = document.createElement("select");
    const symbols = ["o", "d", "t", "s", "p", "h", "star", "+", "x", "t1", "t2", "t3"];
    const names = ["Circle", "Diamond", "Triangle", "Square", "Pentagon", "Hexagon", "Star", "Plus", "Cross", "Down triangle", "Right triangle", "Left triangle"];
    symbols.forEach((value, i) => symbol.add(new Option(names[i], value)));
    for (let number = 1; number <= Math.max(12, metadata.layers); number++) {
      symbol.add(new Option(`Numbered circle ${number}`, `number:${number}`));
    }
    symbol.value = state.symbols[layer];
    symbol.addEventListener("change", () => {
      if (state.symbols.some((existing, index) => index !== layer && existing === symbol.value)) {
        symbol.value = state.symbols[layer];
        setStatus("Each layer needs a unique marker symbol");
        return;
      }
      state.symbols[layer] = symbol.value; draw();
    });
    const size = document.createElement("input"); size.type = "number"; size.min = "25"; size.max = "400"; size.step = "25";
    size.value = String(Math.round((state.sizes[layer] || 1) * 100)); size.title = "Marker size %";
    size.addEventListener("change", () => { state.sizes[layer] = Math.min(4, Math.max(.25, Number(size.value) / 100)); draw(); });
    row.append(title, symbol, size); appearance.append(row);
  }
}
async function loadMetadata(reset = true, supplied = null) {
  const result = supplied || await request("metadata", {lattice: state.lattice, axis: state.axis});
  applyMetadata(reset, result);
}
function applyMetadata(reset, result) {
  metadata = result;
  state.lattice = result.lattice; state.axis = result.axis;
  if (reset) {
    state.angle = result.default_angle;
    state.visibleLayers = [new Set(Array.from({length: result.layers}, (_, i) => i)),
                           new Set(Array.from({length: result.layers}, (_, i) => i))];
    state.symbols = result.symbols;
    state.sizes = Array(result.layers).fill(1);
  }
  $("angle-slider").max = $("angle-number").max = String(result.max_angle);
  rebuildPreset(); rebuildLayers(); updateSummary(); renderRequest();
}
function resetSelections() {
  invalidateAnalyses();
  cancelNearSearch();
  state.boundary = []; state.atoms = []; state.manual = [];
  state.manualLocalCutoff = null; state.manualFit = null; state.manualOriginal = null;
  state.nearCell = null; state.nearSolutions = [];
  state.deformations = identity(); state.translations = [[0, 0], [0, 0]];
  state.mode = "idle"; $("manual-info").dataset.count = "";
  $("vector-annotation").hidden = true; $("strain-info").classList.add("hidden");
}
async function changeGeometry() {
  const generation = ++geometryGeneration;
  invalidateAnalyses(); cancelNearSearch();
  try {
    const lattice = $("structure").value;
    const axis = $("axis").value === "custom" ? $("custom-axis").value : $("axis").value;
    const next = await request("metadata", {lattice, axis});
    if (generation !== geometryGeneration) return;
    state.lattice = next.lattice; state.axis = next.axis;
    resetSelections();
    await loadMetadata(true, next);
    if (state.nearEnabled && state.nearMethod === "strain") searchNear();
  } catch (error) { if (generation === geometryGeneration) fail(error); }
}
function changeAngle(value) {
  const angle = Number(value);
  if (!Number.isFinite(angle) || angle < 0 || angle > metadata.max_angle) return fail(new Error(`Angle must be 0–${metadata.max_angle}°`));
  geometryGeneration++;
  state.angle = angle; resetSelections(); rebuildPreset(); updateSummary(); renderRequest();
  if (state.nearEnabled && state.nearMethod === "strain") searchNear();
}
function setMode(mode) {
  invalidateAnalyses();
  state.mode = state.mode === mode ? "idle" : mode;
  if (state.mode === "boundary") state.boundary = [];
  if (state.mode === "vector") { state.atoms = []; $("vector-annotation").hidden = true; }
  if (state.mode === "cell") { state.manual = []; $("manual-info").dataset.count = ""; }
  for (const [id, value] of [["pick-gb","boundary"],["pick-vector","vector"],["pick-cell","cell"]]) {
    $(id).classList.toggle("selected", state.mode === value);
  }
  draw(); updateSummary();
  setStatus(({idle:"Ready · choose an interaction tool",boundary:"Pick B1, then B2 on visible atoms",vector:"Pick P1, then P2 on visible atoms",cell:"Pick four same-layer CSL or near-pair markers"})[state.mode]);
}
function nearestAtom(x, y) {
  if (!currentPattern()) return null;
  let bestGrain = -1, bestIndex = -1, distance = Infinity;
  const visible = visibleData();
  for (let grain = 0; grain < 2; grain++) {
    const atoms = visible.atoms[grain];
    for (let index = 0; index < atoms.count; index++) {
      const d = Math.hypot(atoms.x[index] - x, atoms.y[index] - y);
      if (d < distance) { distance = d; bestGrain = grain; bestIndex = atoms.indices[index]; }
    }
  }
  if (distance > 14 || bestGrain < 0) return null;
  const flat = visible.atoms[bestGrain].rows.flat, offset = bestIndex * 6;
  return {position: [flat[offset], flat[offset + 1]], grain: bestGrain, layer: flat[offset + 2],
    half_indices: [flat[offset + 3], flat[offset + 4], flat[offset + 5]]};
}
function nearestCommon(x, y) {
  if (!currentPattern()) return null;
  let bestSource = null, bestIndex = -1, distance = Infinity;
  const tiedLayers = new Set(), visible = visibleData(), csl = visible.cslMarkers;
  for (let index = 0; index < csl.count; index++) {
    const row = csl.indices[index], layer = csl.rows.flat[row * 3 + 2];
    const d = Math.hypot(csl.x[index] - x, csl.y[index] - y);
    if (d < distance - 1e-6) {
      distance = d; tiedLayers.clear(); tiedLayers.add(layer);
      bestSource = "CSL"; bestIndex = row;
    } else if (Math.abs(d - distance) < 1e-6) tiedLayers.add(layer);
  }
  const local = visible.localMarkers, flat = local.rows.flat, t = visible.transform, r = t.rect;
  for (let index = 0; index < local.count; index++) {
    const row = local.indices[index], offset = row * 5, layer = flat[offset + 4];
    const mx = (flat[offset] + flat[offset + 2]) / 2, my = (flat[offset + 1] + flat[offset + 3]) / 2;
    const sx = t.ox + (t.cosine * mx - t.sine * my - t.cx) * r.width / t.width;
    const sy = t.oy - (t.sine * mx + t.cosine * my - t.cy) * r.height / t.height;
    const d = Math.hypot(sx - x, sy - y);
    if (d < distance - 1e-6) {
      distance = d; tiedLayers.clear(); tiedLayers.add(layer);
      bestSource = "local"; bestIndex = row;
    } else if (Math.abs(d - distance) < 1e-6) tiedLayers.add(layer);
  }
  if (distance > 15 || bestIndex < 0) return null;
  if (tiedLayers.size > 1) return {ambiguous: true};
  if (bestSource === "CSL") {
    const offset = bestIndex * 3, common = csl.rows.flat;
    return {position: [common[offset], common[offset + 1]], layer: common[offset + 2], source: "CSL"};
  }
  const offset = bestIndex * 5;
  return {position: [(flat[offset] + flat[offset + 2]) / 2, (flat[offset + 1] + flat[offset + 3]) / 2],
    endpoints: [[flat[offset], flat[offset + 1]], [flat[offset + 2], flat[offset + 3]]],
    layer: flat[offset + 4], source: "local"};
}
function resolveCommon(candidate) {
  if (candidate.endpoints) return candidate;
  const endpoints = [];
  for (let grain = 0; grain < 2; grain++) {
    let best = -1, distance = Infinity;
    const rows = visibleData().atoms[grain].rows, flat = rows.flat;
    for (let index = 0; index < rows.length; index++) {
      const offset = index * 6;
      if (flat[offset + 2] !== candidate.layer) continue;
      const d = Math.hypot(flat[offset] - candidate.position[0], flat[offset + 1] - candidate.position[1]);
      if (d < distance) { distance = d; best = offset; }
    }
    if (distance > 1e-5) throw new Error("The CSL marker's atoms are unavailable. Recalculate the view.");
    endpoints.push([flat[best], flat[best + 1]]);
  }
  return {...candidate, endpoints};
}
function manualPolygons() {
  return [0, 1].map(g => state.manual.map(v => v.endpoints[g]));
}
function invalidateAnalyses() {
  analysisGeneration++;
  if ($("completion-dialog").open) $("completion-dialog").close();
}
function analysisSignature(kind) {
  const snapshot = sessionState();
  // An exact-preset render may normalize the angle during search debounce.
  // Explicit angle edits invalidate the analysis and near generations.
  if (kind === "near") delete snapshot.angle;
  return JSON.stringify([snapshot, sessionSettings()]);
}
function beginAnalysis(kind) {
  const number = (analysisRequests.get(kind) || 0) + 1;
  analysisRequests.set(kind, number);
  return {kind, number, generation:analysisGeneration, signature:analysisSignature(kind)};
}
function analysisCurrent(token) {
  return token.generation === analysisGeneration && analysisRequests.get(token.kind) === token.number &&
    token.signature === analysisSignature(token.kind);
}
async function countManual() {
  if (state.manual.length !== 4) return;
  const token = beginAnalysis("count");
  try {
    const result = await request("count", {
      polygons: manualPolygons(), angle: state.angle,
      deformations: state.deformations, translations: state.translations,
      lattice: state.lattice, axis: state.axis, layer: state.manual[0].layer,
      boundary: state.boundary, use_boundary: $("count-visible").checked,
      region_states: state.regions,
    });
    if (!analysisCurrent(token)) return;
    const layer = state.manual[0].layer;
    const text = `G1: ${result.interior[0][layer]} interior + ${result.boundary[0][layer]} boundary\nG2: ${result.interior[1][layer]} interior + ${result.boundary[1][layer]} boundary\nAreas: ${result.areas.map(a => a.toFixed(4)).join(" / ")} a₀²`;
    $("manual-info").dataset.count = text;
    updateSummary(); setStatus("Manual cell counted");
  } catch (error) {
    if (!analysisCurrent(token)) return;
    state.manual.pop();
    state.mode = "cell";
    draw(); updateSummary(); fail(error);
  }
}
async function selectAt(x, y) {
  if (state.mode === "cell" && state.manual.length >= 4) {
    state.mode = "idle"; updateSummary(); return;
  }
  if (state.mode === "boundary") {
    const atom = nearestAtom(x, y); if (!atom) return setStatus("Click a visible atom for the boundary point");
    state.boundary.push(atom.position);
    if (state.boundary.length === 2) state.mode = "idle";
    draw(); updateSummary(); setStatus(state.boundary.length === 2 ? "Boundary selected" : "Pick B2");
  } else if (state.mode === "vector") {
    const atom = nearestAtom(x, y); if (!atom) return setStatus("Click a visible atom for the vector point");
    state.atoms.push(atom);
    if (state.atoms.length === 2) { state.mode = "idle"; await updateVector(); }
    draw(); updateSummary();
  } else if (state.mode === "cell") {
    let candidate = nearestCommon(x, y);
    if (!candidate) return setStatus("Click a gold CSL or purple near-pair marker");
    if (candidate.ambiguous) return setStatus("Overlapping layers: isolate one layer before picking this marker");
    if (state.manual.length && candidate.layer !== state.manual[0].layer) return setStatus("Choose the same axial layer for all vertices");
    if (state.manual.some(v => Math.hypot(v.position[0] - candidate.position[0], v.position[1] - candidate.position[1]) < 1e-8)) return setStatus("Choose a different vertex");
    try { candidate = resolveCommon(candidate); } catch (error) { return fail(error); }
    state.manual.push(candidate);
    if (candidate.source === "local") state.manualLocalCutoff = Number($("local-distance").value);
    if (state.manual.length === 4) { state.mode = "idle"; await countManual(); }
    draw(); updateSummary();
  }
}
async function updateVector() {
  if (state.atoms.length !== 2) return;
  const token = beginAnalysis("vector");
  try {
    const result = await request("vector", {
      atoms: state.atoms, axial_repeat: state.axialRepeat, angle: state.angle,
      lattice: state.lattice, axis: state.axis, deformations: state.deformations,
    });
    if (!analysisCurrent(token)) return;
    const text = [`P1→P2 · axial ${state.axialRepeat >= 0 ? "+" : ""}${state.axialRepeat}`];
    const vectors = Object.entries(result.vectors);
    for (const [grain, v] of vectors) {
      text.push(`${grain} current (polar): ${v.current_formatted}`);
    }
    if (vectors.some(([, v]) => v.strained)) {
      for (const [grain, v] of vectors) {
        text.push(`${grain} lattice [uvw]: ${v.lattice_formatted}`);
      }
    }
    text.push(`Current |Δr|/a₀ = ${result.length.toFixed(4)}`);
    $("vector-readout").textContent = text.join("\n");
    $("vector-annotation").hidden = false;
    positionVectorAnnotation();
    setStatus("Vector measured");
  } catch (error) { if (analysisCurrent(token)) fail(error); }
}

function fitPoints(points) {
  if (!points.length) return;
  const xs = points.map(p => rotate(p, state.displayRotation)[0]);
  const ys = points.map(p => rotate(p, state.displayRotation)[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  state.center = [(minX + maxX) / 2, (minY + maxY) / 2];
  const r = plotRect();
  const width = Math.max(1, (maxX - minX) * 1.35, (maxY - minY) * 1.35 * r.width / r.height);
  state.scale = Math.max(.1, Math.min(5, width / 12));
  $("field-slider").value = String(state.scale);
  viewSize(); renderRequest(); updateSummary();
}
async function completeManual() {
  if (![2, 3].includes(state.manual.length)) return;
  const token = beginAnalysis("complete");
  try {
    const candidates = await request("complete", {
      polygons: manualPolygons(), angle: state.angle, lattice: state.lattice, axis: state.axis,
      layer: state.manual[0].layer, percent: Number($("manual-strain").value),
      rotation: Number($("manual-rotation").value),
    });
    if (!analysisCurrent(token)) return;
    if (!candidates.length) return setStatus("No compatible completion was found");
    const holder = $("completion-options"); holder.replaceChildren();
    candidates.forEach((c, index) => {
      const label = document.createElement("label"); label.className = "candidate";
      const radio = document.createElement("input"); radio.type = "radio"; radio.name = "completion";
      radio.value = String(index); radio.checked = index === 0 && c.strain !== null;
      radio.disabled = c.strain === null;
      const caption = document.createElement("span");
      caption.textContent = ` ${c.description} · ${c.atoms.join("/")} layer cells`;
      const detail = document.createElement("small");
      detail.textContent = c.error || `Maximum principal strain ${(c.strain * 100).toFixed(5)}%`;
      label.append(radio, caption, detail); holder.append(label);
    });
    const dialog = $("completion-dialog"); dialog.showModal();
    $("completion-accept").onclick = async () => {
      if (!analysisCurrent(token)) { dialog.close(); return; }
      const selected = holder.querySelector('input[name="completion"]:checked');
      if (!selected) return;
      const c = candidates[Number(selected.value)], oldCount = state.manual.length;
      for (let i = oldCount; i < 4; i++) {
        const endpoints = [c.vertices[0][i], c.vertices[1][i]];
        const midpoint = endpoints[0].map((v, j) => (v + endpoints[1][j]) / 2);
        const source = Math.hypot(endpoints[0][0] - endpoints[1][0], endpoints[0][1] - endpoints[1][1]) <= 1e-6 ? "CSL" : c.source;
        state.manual.push({position: midpoint, endpoints, layer: state.manual[0].layer, source});
      }
      state.mode = "idle";
      dialog.close(); await countManual(); draw(); updateSummary();
    };
  } catch (error) { if (analysisCurrent(token)) fail(error); }
}
async function toggleSelectedStrain() {
  if (state.manualFit) {
    invalidateAnalyses();
    state.manual = state.manualOriginal;
    state.manualOriginal = null; state.manualFit = null;
    state.boundary = []; state.atoms = []; $("vector-annotation").hidden = true;
    state.deformations = identity(); state.translations = [[0, 0], [0, 0]];
    state.nearCell = null; $("strain-info").classList.add("hidden");
    renderRequest(); await countManual(); updateSummary();
    setStatus("Original local structure restored");
    return;
  }
  if (state.manual.length !== 4 || !state.manual.some(v => v.source !== "CSL")) return;
  const token = beginAnalysis("fit");
  try {
    const fit = await request("fit_selected", {
      polygons: manualPolygons(), angle: state.angle, lattice: state.lattice,
      axis: state.axis, layer: state.manual[0].layer,
      percent: Number($("manual-strain").value), rotation: Number($("manual-rotation").value),
    });
    if (!analysisCurrent(token)) return;
    invalidateAnalyses(); cancelNearSearch();
    state.manualOriginal = clone(state.manual);
    state.manualFit = fit; state.nearCell = fit.cell;
    state.boundary = []; state.atoms = []; $("vector-annotation").hidden = true;
    state.deformations = [fit.cell.f1, fit.cell.f2];
    state.translations = fit.translations;
    state.manual = state.manual.map((v, i) => {
      const endpoints = [fit.vertices[0][i], fit.vertices[1][i]];
      return {position: endpoints[0].map((x, j) => (x + endpoints[1][j]) / 2),
              endpoints, layer: v.layer, source: "CSL"};
    });
    $("strain-info").classList.remove("hidden");
    $("strain-info").textContent = fit.readout || `Applied selected-cell strain\nMax principal strain: ${(fit.cell.max_strain * 100).toFixed(6)}%`;
    renderRequest(); await countManual(); updateSummary();
  } catch (error) { if (analysisCurrent(token)) fail(error); }
}
function applyNearCell(index) {
  const cell = state.nearSolutions[index];
  if (!cell) return;
  invalidateAnalyses();
  state.boundary = []; state.atoms = []; state.manual = [];
  state.nearCell = cell;
  state.deformations = [cell.f1, cell.f2]; state.translations = [[0, 0], [0, 0]];
  $("near-info").textContent = cell.readout || `${cell.label}\nG1 F = ${JSON.stringify(cell.f1)}\nG2 F = ${JSON.stringify(cell.f2)}`;
  renderRequest(); updateSummary();
}
function cancelNearSearch() {
  nearGeneration++;
  clearTimeout(nearTimer);
  worker.postMessage({type: "cancel", action: "near_search"});
}
function searchNear() {
  cancelNearSearch();
  if (!state.nearEnabled || state.nearMethod !== "strain") return;
  invalidateAnalyses();
  const generation = nearGeneration;
  state.nearCell = null; state.nearSolutions = [];
  state.deformations = identity(); state.translations = [[0, 0], [0, 0]];
  state.boundary = []; state.atoms = []; state.manual = [];
  $("vector-annotation").hidden = true;
  renderRequest(); updateSummary();
  $("near-info").textContent = "Waiting for the current angle…";
  const token = beginAnalysis("near");
  nearTimer = setTimeout(async () => {
    try {
      if (generation !== nearGeneration || !analysisCurrent(token)) return;
      setStatus("Searching strained periodic cells locally…");
      $("near-info").textContent = "Searching compatible periodic cells in the browser worker…";
      const result = await request("near_search", {angle: state.angle,
        percent: Number($("strain-percent").value), index: Number($("search-index").value),
        lattice: state.lattice, axis: state.axis});
      if (generation !== nearGeneration || !analysisCurrent(token) || !state.nearEnabled || state.nearMethod !== "strain") return;
      state.nearSolutions = result;
      $("near-results").replaceChildren();
      result.forEach((cell, i) => $("near-results").add(new Option(cell.label, String(i))));
      if (result.length) applyNearCell(0);
      else $("near-info").textContent = "No compatible cell found within this bounded search. Increase the index or strain limit.";
      updateSummary(); setStatus(result.length ? "Strained common cells ready" : "No strained common cell found");
    } catch (error) { if (generation === nearGeneration && analysisCurrent(token)) fail(error); }
  }, 150);
}
function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = name;
  document.body.append(anchor); anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 60_000);
}
async function exportPNG() {
  try {
  await waitForCurrentRender();
  // A download can precede the animation frame queued by the latest input.
  draw();
  const clean = $("clean-png").checked;
  const copy = document.createElement("canvas"); copy.width = canvas.width; copy.height = canvas.height;
  const context = copy.getContext("2d");
  context.setTransform(canvas.width / canvas.clientWidth, 0, 0, canvas.height / canvas.clientHeight, 0, 0);
  if (clean) {
    context.fillStyle = "white"; context.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    drawAtoms(context, true);
  } else {
    context.drawImage(canvas, 0, 0, canvas.clientWidth, canvas.clientHeight);
    context.fillStyle = "#17212b"; context.textAlign = "center"; context.font = "13px Arial";
    context.fillText($("plot-title").textContent, canvas.clientWidth / 2, 17);
    context.textAlign = "left";
    const legend = Array.from($("plot-legend").children).map(item => item.textContent.trim());
    if (legend.length) {
      const columns = Math.min(3, legend.length), rows = Math.ceil(legend.length / columns);
      context.fillStyle = "#ffffffeb";
      context.fillRect(48, 48, columns * 96 + 8, rows * 17 + 10);
      context.strokeStyle = "#cdd8e4";
      context.strokeRect(48.5, 48.5, columns * 96 + 7, rows * 17 + 9);
      context.font = "11px Arial"; context.fillStyle = "#4e5f7c";
      legend.forEach((label, i) => context.fillText(label, 55 + (i % columns) * 96,
                                                    65 + Math.floor(i / columns) * 17));
    }
    if (!$("vector-annotation").hidden) {
      const lines = $("vector-readout").textContent.split("\n");
      const lowerEdge = state.referenceAxes && pattern?.reference_axes && metadata?.reference_labels
        ? referenceAxesBounds(plotRect()).top - 10 : canvas.clientHeight - 42;
      const x = 48, y = lowerEdge - lines.length * 15 - 5;
      context.fillStyle = "#fffffff0"; context.fillRect(x - 6, y - 15, 410, lines.length * 15 + 20);
      context.fillStyle = "#293b59"; context.font = "11px monospace";
      lines.forEach((line, i) => context.fillText(line, x, y + i * 15));
    }
  }
  copy.toBlob(blob => { if (blob) downloadBlob(blob, clean ? "dichromatic-atoms.png" : "dichromatic-pattern.png"); });
  } catch (error) { fail(error); }
}
function sessionState() {
  return {
    lattice: state.lattice, axis: state.axis, angle: state.angle, a0: state.a0,
    mode: state.mode, display_rotation: state.displayRotation,
    reference_axes: state.referenceAxes, colors: state.colors, symbols: state.symbols,
    sizes: state.sizes, selected_layer: -1,
    visible_layers: state.visibleLayers.map(set => Array.from(set).sort((a, b) => a - b)),
    boundary: state.boundary, atoms: state.atoms, manual: state.manual,
    manual_local_cutoff: state.manualLocalCutoff, axial_repeat: state.axialRepeat,
    near_enabled: state.nearEnabled, near_method: state.nearMethod,
    near_cell: state.nearCell, deformations: state.deformations,
    translations: state.translations, manual_fit: state.manualFit,
    manual_original: state.manualOriginal,
  };
}
function sessionSettings() {
  return {
    region_states: state.regions, manual_visible: $("count-visible").checked,
    show_common_cell: state.showCell,
    local_distance: Number($("local-distance").value),
    strain_percent: Number($("strain-percent").value),
    search_index: Number($("search-index").value),
    manual_strain_percent: Number($("manual-strain").value),
    manual_rotation_deg: Number($("manual-rotation").value),
  };
}
async function saveSession() {
  try {
    viewSize();
    const result = await request("save_session", {
      state: sessionState(), settings: sessionSettings(),
      view_range: [state.center[0] - state.width / 2, state.center[0] + state.width / 2,
                   state.center[1] - state.height / 2, state.center[1] + state.height / 2],
    });
    const binary = atob(result), bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    downloadBlob(new Blob([bytes], {type:"application/zip"}), "dichromatic-map.dmap");
    setStatus("Session saved with numerical CSV tables");
  } catch (error) { fail(error); }
}
async function importSession(file) {
  if (!file) return;
  invalidateAnalyses(); cancelNearSearch(); geometryGeneration++;
  const token = beginAnalysis("import");
  try {
    const dataUrl = await new Promise((resolve, reject) => {
      const reader = new FileReader(); reader.onload = () => resolve(reader.result);
      reader.onerror = () => reject(reader.error); reader.readAsDataURL(file);
    });
    const loaded = await request("load_session", {bytes: dataUrl.split(",")[1]});
    if (!analysisCurrent(token)) return;
    const raw = loaded.state, p = raw.parameters;
    const nextMetadata = await request("metadata", {lattice:p.lattice, axis:p.axis});
    if (!analysisCurrent(token)) return;
    state.lattice = p.lattice; state.axis = p.axis; state.angle = p.angle_deg;
    state.a0 = p.lattice_constant; state.mode = raw.interaction_mode;
    state.displayRotation = raw.display_rotation_deg; state.referenceAxes = raw.show_reference_axes;
    state.colors = raw.grain_colors; state.symbols = raw.layer_symbols;
    state.sizes = raw.layer_size_scales;
    state.visibleLayers = raw.visible_grain_layers.map(layers => new Set(layers));
    state.boundary = raw.selected_points;
    state.atoms = raw.selected_atoms.map(a => ({position:a.position, grain:a.grain_index, layer:a.layer, half_indices:a.half_indices}));
    const vertices = list => (list || []).map(v => ({position:v.position, layer:v.layer, source:v.source, endpoints:v.grain_positions}));
    state.manual = vertices(raw.manual_vertices);
    state.manualOriginal = raw.manual_unstrained_vertices === null ? null : vertices(raw.manual_unstrained_vertices);
    state.manualFit = raw.manual_strain_fit;
    state.manualLocalCutoff = raw.manual_local_cutoff;
    state.axialRepeat = raw.axial_repeat;
    state.nearEnabled = raw.near_enabled; state.nearMethod = raw.near_method;
    state.nearCell = raw.near_cell; state.nearSolutions = raw.near_cell ? [raw.near_cell] : [];
    state.deformations = raw.deformations; state.translations = raw.translations;
    state.regions = loaded.settings.region_states; state.showCell = loaded.settings.show_common_cell;
    state.center = [(loaded.view_range[0] + loaded.view_range[1]) / 2,
                    (loaded.view_range[2] + loaded.view_range[3]) / 2];
    state.scale = Math.max(.1, Math.min(5, (loaded.view_range[1] - loaded.view_range[0]) / 12));
    $("vector-annotation").hidden = true;
    $("vector-readout").textContent = "";
    $("manual-info").dataset.count = "";
    $("near-results").replaceChildren();
    if (state.nearCell) $("near-results").add(new Option(state.nearCell.label || "Imported cell", "0"));
    $("near-info").textContent = state.nearCell ? state.nearCell.readout || "Imported common cell" : "";
    $("strain-info").classList.toggle("hidden", !state.manualFit);
    $("strain-info").textContent = state.manualFit ? state.manualFit.readout || "Imported selected-cell strain" : "";
    applyMetadata(false, nextMetadata);
    $("structure").value = state.lattice;
    const known = ["100","110","111","112"].includes(state.axis);
    $("axis").value = known ? state.axis : "custom";
    $("custom-axis-row").classList.toggle("hidden", known);
    if (!known) $("custom-axis").value = state.axis;
    $("g1-color").value = state.colors[0]; $("g2-color").value = state.colors[1];
    $("lattice-constant").value = state.a0;
    $("near-method").value = state.nearMethod;
    $("reference-axes").checked = state.referenceAxes;
    $("show-cell").checked = state.showCell;
    $("count-visible").checked = loaded.settings.manual_visible;
    $("local-distance").value = loaded.settings.local_distance;
    $("strain-percent").value = loaded.settings.strain_percent;
    $("search-index").value = loaded.settings.search_index;
    $("manual-strain").value = loaded.settings.manual_strain_percent;
    $("manual-rotation").value = loaded.settings.manual_rotation_deg;
    $("axial-repeat").value = state.axialRepeat;
    $("field-slider").value = state.scale;
    document.querySelectorAll("[data-region]").forEach(input => input.checked = state.regions[Number(input.dataset.region)]);
    rebuildPreset(); rebuildLayers(); updateSummary(); renderRequest();
    if (state.atoms.length === 2) updateVector();
    if (state.manual.length === 4) countManual();
    setStatus("Session imported");
  } catch (error) { if (analysisCurrent(token)) fail(error); }
  finally { $("session-file").value = ""; }
}

$("structure").addEventListener("change", changeGeometry);
$("axis").addEventListener("change", () => {
  $("custom-axis-row").classList.toggle("hidden", $("axis").value !== "custom");
  if ($("axis").value !== "custom") changeGeometry();
});
$("apply-axis").addEventListener("click", changeGeometry);
$("custom-axis").addEventListener("keydown", e => { if (e.key === "Enter") changeGeometry(); });
$("preset").addEventListener("change", () => { if ($("preset").value) changeAngle($("preset").value); });
$("angle-number").addEventListener("change", () => changeAngle($("angle-number").value));
$("angle-slider").addEventListener("input", () => changeAngle($("angle-slider").value));
function rotationChanged(value) {
  state.displayRotation = Number(value); updateSummary(); scheduleDraw(); renderRequest();
}
$("rotation-number").addEventListener("change", () => rotationChanged($("rotation-number").value));
$("rotation-slider").addEventListener("input", () => rotationChanged($("rotation-slider").value));
$("rotation-reset").addEventListener("click", () => rotationChanged(0));
document.querySelectorAll("[data-main-tab]").forEach(button => button.addEventListener("click", () => {
  document.querySelectorAll("[data-main-tab]").forEach(b => b.classList.toggle("active", b === button));
  $("orientation-tab").classList.toggle("hidden", button.dataset.mainTab !== "orientation");
  $("layers-tab").classList.toggle("hidden", button.dataset.mainTab !== "layers");
}));
document.querySelectorAll("[data-view-tab]").forEach(button => button.addEventListener("click", () => {
  document.querySelectorAll("[data-view-tab]").forEach(b => b.classList.toggle("active", b === button));
  for (const name of ["view","performance","appearance"]) $(name + "-tab").classList.toggle("hidden", name !== button.dataset.viewTab);
}));
$("all-layers").addEventListener("click", () => {
  state.visibleLayers = [0,1].map(() => new Set(Array.from({length:metadata.layers},(_,i)=>i)));
  rebuildLayers(); draw(); updateSummary(); setStatus("All layers visible");
});
$("no-layers").addEventListener("click", () => {
  state.visibleLayers = [new Set(),new Set()]; rebuildLayers(); draw(); updateSummary(); setStatus("Layers hidden");
});
$("show-cell").addEventListener("change", () => { state.showCell = $("show-cell").checked; draw(); });
$("fit-cell").addEventListener("click", () => {
  const cell = state.nearCell || (currentPattern() ? pattern?.exact_cell : null);
  if (cell) fitPoints(cellCorners(cell));
});
$("pick-gb").addEventListener("click", () => setMode("boundary"));
$("pick-vector").addEventListener("click", () => setMode("vector"));
$("pick-cell").addEventListener("click", () => setMode("cell"));
$("axial-repeat").addEventListener("change", () => { state.axialRepeat = Number($("axial-repeat").value); updateVector(); });
document.querySelectorAll("[data-region]").forEach(input => input.addEventListener("change", () => {
  state.regions[Number(input.dataset.region)] = input.checked;
  draw(); updateSummary(); setStatus("GB side visibility updated"); if (state.manual.length === 4) countManual();
}));
function regionPreset(values) {
  state.regions = values;
  document.querySelectorAll("[data-region]").forEach(input => input.checked = values[Number(input.dataset.region)]);
  draw(); updateSummary(); setStatus("GB side visibility updated"); if (state.manual.length === 4) countManual();
}
$("show-all-sides").addEventListener("click", () => regionPreset([true,true,true,true]));
$("sides-one").addEventListener("click", () => regionPreset([true,false,false,true]));
$("sides-two").addEventListener("click", () => regionPreset([false,true,true,false]));
$("field-slider").addEventListener("input", () => {
  state.scale = Number($("field-slider").value); viewSize(); updateSummary(); scheduleDraw(); renderRequest();
});
document.querySelectorAll("[data-scale]").forEach(button => button.addEventListener("click", () => {
  state.scale = Number(button.dataset.scale); $("field-slider").value = state.scale;
  updateSummary(); renderRequest();
}));
function centerView() { state.center = [0,0]; renderRequest(); draw(); }
$("center-view").addEventListener("click", centerView);
$("fit-view").addEventListener("click", centerView);
$("zoom-in").addEventListener("click", () => { state.scale = Math.max(.1,state.scale / 1.25); $("field-slider").value = state.scale; updateSummary(); renderRequest(); });
$("zoom-out").addEventListener("click", () => { state.scale = Math.min(5,state.scale * 1.25); $("field-slider").value = state.scale; updateSummary(); renderRequest(); });
function setReferenceAxes(visible) {
  state.referenceAxes = visible;
  $("reference-axes").checked = visible;
  if (visible && window.matchMedia("(max-width: 820px)").matches) {
    $("vector-annotation").open = false;
  }
  draw();
}
$("reference-axes").addEventListener("change", () => setReferenceAxes($("reference-axes").checked));
$("reference-axes-toggle").addEventListener("click", () => setReferenceAxes(!state.referenceAxes));
for (const id of ["plot-legend-panel", "vector-annotation"]) {
  const panel = $(id);
  panel.addEventListener("click", event => {
    if (panel.open && !event.target.closest("summary")) panel.open = false;
  });
}
$("g1-color").addEventListener("input", () => { state.colors[0] = $("g1-color").value; scheduleDraw(); });
$("g2-color").addEventListener("input", () => { state.colors[1] = $("g2-color").value; scheduleDraw(); });
$("lattice-constant").addEventListener("change", () => {
  const value = Number($("lattice-constant").value);
  if (Number.isFinite(value) && value > 0) state.a0 = value;
  else { $("lattice-constant").value = state.a0; setStatus("Lattice constant must be positive"); }
});
$("reset-appearance").addEventListener("click", () => {
  state.colors = ["#1677d2","#e35d35"]; state.symbols = metadata.symbols;
  state.sizes = Array(metadata.layers).fill(1);
  $("g1-color").value = state.colors[0]; $("g2-color").value = state.colors[1];
  rebuildLayers(); draw();
});
$("near-method").addEventListener("change", () => {
  cancelNearSearch();
  state.boundary = []; state.atoms = []; state.manual = [];
  $("vector-annotation").hidden = true;
  state.nearMethod = $("near-method").value;
  state.nearCell = null; state.nearSolutions = [];
  state.deformations = identity(); state.translations = [[0,0],[0,0]];
  updateSummary(); renderRequest();
  if (state.nearEnabled && state.nearMethod === "strain") searchNear();
});
$("near-toggle").addEventListener("click", () => {
  cancelNearSearch();
  state.nearEnabled = !state.nearEnabled;
  if (!state.nearEnabled) {
    state.boundary = []; state.atoms = []; state.manual = [];
    $("vector-annotation").hidden = true;
    state.nearCell = null; state.nearSolutions = [];
    state.deformations = identity(); state.translations = [[0,0],[0,0]];
    $("near-info").textContent = "Off. Original unstrained lattices restored.";
  } else if (state.nearMethod === "local") {
    $("near-info").textContent = "Local same-layer mutual nearest pairs. Original atom positions retained.";
  }
  updateSummary(); renderRequest();
  if (state.nearEnabled && state.nearMethod === "strain") searchNear();
});
$("local-distance").addEventListener("change", renderRequest);
$("strain-percent").addEventListener("change", searchNear);
$("search-index").addEventListener("change", searchNear);
$("run-search").addEventListener("click", searchNear);
$("near-results").addEventListener("change", () => applyNearCell(Number($("near-results").value)));
$("undo-cell").addEventListener("click", () => {
  state.manual.pop(); state.manualFit = null; state.manualOriginal = null;
  $("manual-info").dataset.count = ""; draw(); updateSummary();
});
$("clear-cell").addEventListener("click", () => {
  state.manual = []; state.manualFit = null; state.manualOriginal = null;
  $("manual-info").dataset.count = ""; draw(); updateSummary();
});
$("fit-manual").addEventListener("click", () => fitPoints(state.manual.map(v => v.position)));
$("complete-cell").addEventListener("click", completeManual);
$("completion-cancel").addEventListener("click", () => $("completion-dialog").close());
$("count-visible").addEventListener("change", countManual);
$("apply-strain").addEventListener("click", toggleSelectedStrain);
$("export-png").addEventListener("click", exportPNG);
$("save-session").addEventListener("click", saveSession);
$("import-session").addEventListener("click", () => $("session-file").click());
$("session-file").addEventListener("change", () => importSession($("session-file").files[0]));

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {x:event.clientX - rect.left, y:event.clientY - rect.top};
}
function startPan(pointerId, point) {
  pointerStart = {pointerId, x:point.x, y:point.y, center:[...state.center]};
  dragDistance = 0;
}
function startPinch() {
  const [a, b] = Array.from(activePointers.values());
  const midpoint = {x:(a.x + b.x) / 2, y:(a.y + b.y) / 2};
  const r = plotRect();
  viewSize();
  pinchStart = {
    distance:Math.max(1, Math.hypot(a.x - b.x, a.y - b.y)),
    scale:state.scale,
    anchor:[state.center[0] + (midpoint.x - r.left - r.width / 2) * state.width / r.width,
            state.center[1] - (midpoint.y - r.top - r.height / 2) * state.height / r.height],
  };
  pointerStart = undefined;
  gestureWasPinch = true;
}
function updatePinch() {
  const [a, b] = Array.from(activePointers.values());
  const midpoint = {x:(a.x + b.x) / 2, y:(a.y + b.y) / 2};
  const distance = Math.max(1, Math.hypot(a.x - b.x, a.y - b.y));
  state.scale = Math.max(.1, Math.min(5, pinchStart.scale * pinchStart.distance / distance));
  viewSize();
  const r = plotRect();
  state.center = [pinchStart.anchor[0] - (midpoint.x - r.left - r.width / 2) * state.width / r.width,
                  pinchStart.anchor[1] + (midpoint.y - r.top - r.height / 2) * state.height / r.height];
  gestureNeedsRender = true;
  $("field-slider").value = String(state.scale);
  $("field-label").textContent = `${state.scale.toFixed(1)}×`;
  scheduleDraw();
}
function finishPointer(event, cancelled = false) {
  if (!activePointers.has(event.pointerId)) return;
  const point = canvasPoint(event);
  const wasPan = pointerStart?.pointerId === event.pointerId;
  activePointers.delete(event.pointerId);
  pinchStart = undefined;
  if (activePointers.size >= 2) {
    startPinch();
  } else if (activePointers.size === 1) {
    const [pointerId, remaining] = activePointers.entries().next().value;
    startPan(pointerId, remaining);
  } else {
    if (!cancelled && !gestureWasPinch && wasPan && !gestureNeedsRender) selectAt(point.x, point.y);
    else if (gestureNeedsRender) renderRequest();
    pointerStart = undefined;
    dragDistance = 0;
    gestureWasPinch = false;
    gestureNeedsRender = false;
  }
}
canvas.addEventListener("pointerdown", event => {
  if (event.pointerType === "mouse" && event.button !== 0) return;
  if (event.pointerType === "touch") event.preventDefault();
  canvas.setPointerCapture(event.pointerId);
  const point = canvasPoint(event);
  activePointers.set(event.pointerId, point);
  if (activePointers.size === 1) {
    gestureWasPinch = false;
    gestureNeedsRender = false;
    startPan(event.pointerId, point);
  } else {
    startPinch();
  }
});
canvas.addEventListener("pointermove", event => {
  if (!activePointers.has(event.pointerId)) return;
  const point = canvasPoint(event);
  activePointers.set(event.pointerId, point);
  if (pinchStart && activePointers.size >= 2) {
    updatePinch();
  } else if (pointerStart?.pointerId === event.pointerId) {
    const dx = point.x - pointerStart.x, dy = point.y - pointerStart.y;
    dragDistance = Math.max(dragDistance, Math.hypot(dx, dy));
    if (dragDistance > 3) {
      gestureNeedsRender = true;
      const r = plotRect();
      state.center = [pointerStart.center[0] - dx * state.width / r.width,
                      pointerStart.center[1] + dy * state.height / r.height];
      scheduleDraw();
    }
  }
});
canvas.addEventListener("pointerup", event => finishPointer(event));
canvas.addEventListener("pointercancel", event => finishPointer(event, true));
canvas.addEventListener("lostpointercapture", event => finishPointer(event, true));
canvas.addEventListener("wheel", event => {
  event.preventDefault();
  state.scale = Math.max(.1, Math.min(5, state.scale * (event.deltaY > 0 ? 1.13 : 1 / 1.13)));
  $("field-slider").value = state.scale;
  viewSize(); updateSummary(); scheduleDraw(); renderRequest();
}, {passive:false});
document.addEventListener("keydown", event => {
  if (["INPUT","SELECT","TEXTAREA"].includes(document.activeElement?.tagName) || $("completion-dialog").open) return;
  const key = event.key.toLowerCase();
  if (key === "r") setMode("boundary");
  else if (key === "v") setMode("vector");
  else if (key === "m") setMode("cell");
  else if (key === "c") centerView();
  else if (key === "f") regionPreset([true,true,true,true]);
  else if (key === "1") regionPreset([true,false,false,true]);
  else if (key === "2") regionPreset([false,true,true,false]);
  else if (key === "escape") { state.mode = "idle"; updateSummary(); }
});
window.addEventListener("resize", () => { if (metadata) renderRequest(); });

loadMetadata().catch(fail);
