"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

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
    DichromaticRenderData: {decodeRenderResult: result => result}},
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
  globalThis.app = {state, draw, scheduleDraw, exportPNG, visiblePoints, visibleCSL, visibleLocal, screen,
    viewSize, coveredView, searchNear, cancelNearSearch,
    setPattern(value) {pattern = value;}};
`, sandbox);
const app = sandbox.app;
const pattern = {grains:[[], []], coincidences:[], local:[]};
for (let x = -20; x <= 20; x++) for (let y = -8; y <= 8; y++) for (let layer=0; layer<2; layer++) {
  const point = [x + layer*.2, y + layer*.1, layer, x, y, layer];
  pattern.grains[0].push(point);
  pattern.grains[1].push([...point]);
  if ((x+y)%3===0) pattern.coincidences.push(point.slice(0,3));
  if ((x+y)%4===0) pattern.local.push([point[0],point[1],point[0]+.04,point[1]-.03,layer]);
}
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
function verify() {
  app.viewSize(); app.draw();
  for (let grain=0; grain<2; grain++) {
    assert.deepEqual(Array.from(app.visiblePoints(grain)), pattern.grains[grain].filter(p =>
      app.state.visibleLayers[grain].has(p[2]) && referenceVisible(p,grain)));
  }
  assert.deepEqual(Array.from(app.visibleCSL()), pattern.coincidences.filter(p =>
    app.state.visibleLayers.every(layers=>layers.has(p[2])) && referenceVisible(p,0) && referenceVisible(p,1)));
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
app.scheduleDraw(); app.exportPNG();
assert.equal(frames.size,0,"PNG export must flush the latest scheduled view before copying pixels");

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
