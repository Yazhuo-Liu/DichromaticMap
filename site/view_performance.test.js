"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const renderData = require("./render_data.js");
const spatialIndex = require("./spatial_index.js");

async function main() {
const nodes = new Map(), frames = new Map(), timers = new Map(), messages = [];
let sequence = 0, boundsReads = 0, backingWrites = 0;
const context = new Proxy({}, {get(target, key) { return target[key] ?? (() => {}); }});
function node(id) {
  if (!nodes.has(id)) {
    const element = {
      id, handlers: {}, style: {}, dataset: {}, value: "1", checked: true,
      innerHTML: "", children: [], clientWidth: 1024, clientHeight: 852,
      classList: {add() {}, remove() {}, toggle() {}},
      addEventListener(type, callback) { this.handlers[type] = callback; },
      getBoundingClientRect() { boundsReads++; return this.rect || {left: 0, top: 0, width: 1024, height: 852}; },
      getContext: () => context, replaceChildren() {}, add() {}, setAttribute() {},
      querySelector: () => node("dummy"), closest: () => node("dummy"),
    };
    for (const key of ["width", "height"]) {
      let value = 0;
      Object.defineProperty(element, key, {get: () => value, set(next) { value = next; backingWrites++; }});
    }
    nodes.set(id, element);
  }
  return nodes.get(id);
}
const sandbox = {
  Worker: class {postMessage(message) {messages.push(message);}},
  Option: class {}, devicePixelRatio: 1,
  window: {matchMedia: () => ({matches: false}), addEventListener() {},
    DichromaticRenderData: renderData, DichromaticSpatialIndex: spatialIndex},
  document: {getElementById: node, querySelectorAll: () => [],
    querySelector: () => node("dummy"), addEventListener() {}, createElement: () => ({getContext: () => context, toBlob() {}})},
  requestAnimationFrame(callback) { const id = ++sequence; frames.set(id, callback); return id; },
  cancelAnimationFrame: id => frames.delete(id),
  setTimeout(callback, delay) {const id = ++sequence; timers.set(id, {callback, delay}); return id;},
  clearTimeout: id => timers.delete(id),
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(__dirname, "use.js"), "utf8") + `
  metadata = {layers:2, max_angle:90, presets:[], reference_labels:[], layer_spacing:1, axial_period:1};
  globalThis.app = {state, draw, scheduleDraw, exportPNG, visiblePoints, visibleCSL, visibleLocal, visibleData, screen,
    viewSize, coveredView, searchNear, cancelNearSearch, nearestAtom, nearestCommon, resolveCommon, setStatus, updateSummary, drawAtoms, sessionState,
    setPattern(value) {pattern = value; renderCoverage = {signature:JSON.stringify(currentRenderParameters()),
      center: [0,0], width:1e6, height:1e6};}};
`, sandbox);
const app = sandbox.app;
const pattern = {grains:[[], []], coincidences:[], local:[]};
for (let x = -60; x <= 60; x++) for (let y = -24; y <= 24; y++) for (let layer=0; layer<2; layer++) {
  const point = [x + layer*.2, y + layer*.1, layer, x, y, layer];
  pattern.grains[0].push(point);
  pattern.grains[1].push([...point]);
  if ((x+y)%3===0) pattern.coincidences.push(point.slice(0,3));
  if ((x+y)%4===0) pattern.local.push([point[0],point[1],point[0]+.04,point[1]-.03,layer]);
}
app.setPattern(pattern);

const typed = renderData.decodeRenderResult({format:"dichromatic-map-render-v1", metadata:{},
  grains:pattern.grains.map(rows=>new Float64Array(rows.flat())),
  coincidences:new Float64Array(pattern.coincidences.flat()), local:new Float64Array(pattern.local.flat())});
app.setPattern(typed); verify();
const projected = app.visibleData().atoms[0], projectedX = projected.x, projectedIndices = projected.indices;
const candidates = projected.spatial.queryViewport(app.visibleData().transform);
assert(candidates && candidates.length < projected.rows.length / 4,
  "the app queries a conservative spatial subset before its exact Float64 viewport tests");
app.state.center = [1,-2]; verify();
assert.equal(app.visibleData().atoms[0],projected,"panning reuses the selection object");
assert.equal(app.visibleData().atoms[0].x,projectedX,"panning reuses projected Float64 coordinates");
assert.equal(app.visibleData().atoms[0].indices,projectedIndices,"panning reuses indices without per-atom records");
const row = renderData.DenseRows.prototype.row;
renderData.DenseRows.prototype.row = () => {throw new Error("hot render/picking path must not materialize rows");};
try {
  app.draw(); app.setStatus("Ready"); app.updateSummary();
  app.nearestAtom(512,426); app.nearestCommon(512,426);
} finally {renderData.DenseRows.prototype.row = row;}
assert.deepEqual(JSON.parse(JSON.stringify(typed)),pattern,"explicit JSON output preserves the legacy fixture API");

const tiePattern = renderData.decodeRenderResult({format:"dichromatic-map-render-v1", metadata:{},
  grains:[new Float64Array([0,0,0,9007199254740991,-7,3,0,0,1,4,5,6]),
          new Float64Array([0,0,0,9007199254740991,-7,3,0,0,1,4,5,6])],
  coincidences:new Float64Array([0,0,0,0,0,1]), local:new Float64Array()});
app.state.center=[0,0]; app.state.displayRotation=0; app.state.scale=1;
app.state.boundary=[]; app.state.regions=[true,true,true,true];
app.state.visibleLayers=[new Set([0,1]),new Set([0,1])]; app.viewSize(); app.setPattern(tiePattern);
const [pickX,pickY] = app.screen([0,0]);
assert.equal(app.nearestCommon(pickX,pickY).ambiguous,true,"overlapping layers remain ambiguous");
assert.equal(app.nearestAtom(pickX,pickY).grain,0,"equal-distance atoms retain original grain order");
assert.deepEqual(Array.from(app.nearestAtom(pickX,pickY).half_indices),[9007199254740991,-7,3]);
app.state.atoms = [app.nearestAtom(pickX,pickY)];
assert.deepEqual(JSON.parse(JSON.stringify(app.sessionState().atoms[0].half_indices)),[9007199254740991,-7,3]);
app.state.atoms = [];
app.state.visibleLayers=[new Set([0]),new Set([0])];
const common = app.nearestCommon(pickX,pickY);
assert.equal(common.layer,0); assert.equal(common.source,"CSL");
assert.deepEqual(JSON.parse(JSON.stringify(app.resolveCommon(common).endpoints)),[[0,0],[0,0]]);
assert.equal(app.nearestCommon(pickX+16,pickY),null);
assert.equal(app.nearestAtom(pickX+15,pickY),null);
const localOnly = renderData.decodeRenderResult({format:"dichromatic-map-render-v1",metadata:{},
  grains:[new Float64Array(),new Float64Array()],coincidences:new Float64Array(),
  local:new Float64Array([-.02,0,.02,0,0])});
app.state.nearEnabled=true; app.state.nearMethod="local"; app.setPattern(localOnly);
const localPick = app.nearestCommon(pickX,pickY);
assert.equal(localPick.source,"local");
assert.deepEqual(JSON.parse(JSON.stringify(localPick.endpoints)),[[-.02,0],[.02,0]]);
app.state.visibleLayers=[new Set([0,1]),new Set([0,1])];
app.setPattern(pattern);
app.state.nearEnabled = true;
function referenceVisible(point, grain) {
  const a = app.state.displayRotation * Math.PI/180, c = Math.cos(a), s = Math.sin(a);
  const x = c*point[0]-s*point[1], y = s*point[0]+c*point[1];
  const px = 35+961/2+(x-app.state.center[0])*961/app.state.width;
  const py = 30+786/2-(y-app.state.center[1])*786/app.state.height;
  if (px<35 || px>996 || py<30 || py>816) return false;
  if (app.state.boundary.length !== 2) return true;
  const [p,q] = app.state.boundary;
  const cross = (q[0]-p[0])*(point[1]-p[1])-(q[1]-p[1])*(point[0]-p[0]);
  return app.state.regions[2*grain] && cross>=-1e-9 || app.state.regions[2*grain+1] && cross<=1e-9;
}
function referenceSideVisible(point,grain) {
  if (app.state.boundary.length!==2) return true;
  const [a,b]=app.state.boundary;
  const cross=(b[0]-a[0])*(point[1]-a[1])-(b[1]-a[1])*(point[0]-a[0]);
  return app.state.regions[2*grain] && cross>=-1e-9 || app.state.regions[2*grain+1] && cross<=1e-9;
}
function verify() {
  app.viewSize(); app.setPattern(pattern); app.draw();
  for (let grain=0; grain<2; grain++) {
    assert.deepEqual(Array.from(app.visiblePoints(grain)), pattern.grains[grain].filter(p =>
      app.state.visibleLayers[grain].has(p[2]) && referenceVisible(p,grain)));
  }
  assert.deepEqual(Array.from(app.visibleCSL()), pattern.coincidences.filter(p =>
    app.state.visibleLayers.every(layers=>layers.has(p[2])) && referenceVisible(p,0) && referenceVisible(p,1)));
  const local=app.state.nearEnabled && app.state.nearMethod==="local" ? pattern.local.filter(pair=>{
    const [x,y]=app.screen([(pair[0]+pair[2])/2,(pair[1]+pair[3])/2]);
    return app.state.visibleLayers.every(layers=>layers.has(pair[4])) &&
      referenceSideVisible(pair,0) && referenceSideVisible(pair.slice(2),1) &&
      x>=35 && x<=996 && y>=30 && y<=816;
  }) : [];
  assert.deepEqual(Array.from(app.visibleLocal()),local,"local filtering clips midpoints and uses each grain's own endpoint");
  assert.equal(app.visibleData().maxLocalSeparation,local.reduce((longest,pair)=>
    Math.max(longest,Math.hypot(pair[0]-pair[2],pair[1]-pair[3])),0));
}
verify();
app.state.scale = 3; verify();  // Include atoms exactly on viewport edges.
app.state.scale = 1;
app.state.center = [2,-3]; app.state.displayRotation = 37; verify();
app.state.visibleLayers[0].delete(1); verify();
app.state.boundary = [[-2,-1],[3,4]]; app.state.regions = [true,false,false,true]; verify();
app.state.boundary[1][0] = 0; verify();
app.state.scale = 2; verify();
app.state.nearEnabled = false; assert.equal(app.visibleLocal().length,0);
app.setPattern({...pattern, grains:[[], []], coincidences:[], local:[]});
assert.equal(app.visiblePoints(0).length,0);
app.setPattern(pattern);

backingWrites = 0; boundsReads = 0;
for (let i=0;i<5;i++) app.scheduleDraw();
assert.equal(frames.size,1,"rapid events should schedule one canvas frame");
const callback = frames.values().next().value; frames.clear(); callback();
assert.equal(boundsReads,1,"a draw measures the canvas once regardless of point count");
assert.equal(backingWrites,0,"unchanged canvas size must retain its backing store");
app.scheduleDraw(); app.draw(); assert.equal(frames.size,0,"a synchronous export/selection draw cancels a pending duplicate");
node("clean-png").checked = false; node("vector-annotation").hidden = true;
app.scheduleDraw(); await app.exportPNG();
assert.equal(frames.size,0,"PNG export must flush the latest scheduled view before copying pixels");

const large = new Float64Array(6500*6);
for(let index=0;index<6500;index++) {
  large[index*6] = (index%100)/20-2.5;
  large[index*6+1] = Math.floor(index/100)/20-1.5;
  large[index*6+2] = index%2;
  large[index*6+3] = index;
}
app.state.center=[0,0]; app.state.displayRotation=0; app.state.scale=1;
app.state.boundary=[]; app.state.nearEnabled=false; app.state.visibleLayers=[new Set([0,1]),new Set([0,1])];
app.setPattern(renderData.decodeRenderResult({format:"dichromatic-map-render-v1",metadata:{},
  grains:[large,large.slice()],coincidences:new Float64Array(),local:null}));
let factories=0, gpuCalls=0, blits=0, arcs=0, fallback=false, previousRevision=0;
const gpuCanvas = {};
sandbox.window.DichromaticGPU = {createRenderer() {factories++; return {render(options) {
  gpuCalls++;
  assert(options.atoms.every(atoms=>atoms.rows.flat instanceof Float64Array));
  assert(options.atoms[0].count+options.atoms[1].count>=12000);
  assert(options.revision>=previousRevision); previousRevision=options.revision;
  return fallback ? null : gpuCanvas;
}};}};
context.drawImage = source => {assert.equal(source,gpuCanvas);blits++;};
context.arc = () => {arcs++;};
app.viewSize(); app.draw(); assert.equal(factories,1); assert.equal(blits,1); assert.equal(arcs,0);
const revision = previousRevision;
app.state.center=[.1,.1]; app.draw(); assert(previousRevision>revision,"GPU upload revision tracks in-place pan projections");
assert.equal(factories,1);
const cleanContext = new Proxy({arc(){arcs++;}},{get(target,key){return target[key]??(()=>{});}});
const beforeClean = gpuCalls;
app.drawAtoms(cleanContext,true); assert.equal(gpuCalls,beforeClean,"atoms-only exports use full Canvas markers");
assert(arcs>0);
arcs=0; fallback=true; app.draw(); assert(arcs>0,"unavailable GPU must fall back to Canvas without dropping atoms");
assert.equal(large[3],0); assert.equal(large[large.length-3],6499,"GPU integration leaves crystal indices untouched");
delete context.drawImage; delete context.arc; delete sandbox.window.DichromaticGPU;
app.setPattern(pattern);

app.state.center = [0,0]; app.state.displayRotation = 45; app.state.width=12; app.state.height=9;
assert.equal(app.coveredView({center:[0,0],width:16,height:16}),true);
assert.equal(app.coveredView({center:[0,0],width:12,height:9}),false,"rotation corners must be covered");
app.state.nearEnabled=true; app.state.nearMethod="strain";
messages.length=0;
for (let i=0;i<10;i++) {node("angle-slider").value=String(39+i*.1); node("angle-slider").handlers.input();}
assert.equal(messages.filter(m=>m.request?.action==="near_search").length,0,"slider input waits for quiet time");
assert.equal([...timers.values()].filter(t=>t.delay===150).length,1);
for (const [id,timer] of [...timers]) if (timer.delay===150) {timers.delete(id); timer.callback();}
const searches = messages.filter(m=>m.request?.action==="near_search");
assert.equal(searches.length,1); assert.equal(searches[0].request.angle,39.9);
app.searchNear(); app.cancelNearSearch();
assert.equal([...timers.values()].filter(t=>t.delay===150).length,0,"disabling search clears delayed work");
assert(messages.some(m=>m.type==="cancel" && m.action==="near_search"));
console.log("Viewport filtering, frame coalescing, backing-store reuse, and latest-angle search passed.");

}
main().catch(error => { console.error(error); process.exitCode = 1; });
