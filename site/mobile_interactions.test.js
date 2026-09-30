"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const nodes = new Map();
function node(id) {
  if (!nodes.has(id)) nodes.set(id, {
    id, handlers: {}, style: {}, value: "1", checked: true, open: true,
    classList: {add() {}, remove() {}, toggle() {}},
    addEventListener(type, fn) { this.handlers[type] = fn; },
    setPointerCapture() {},
    getBoundingClientRect() { return this.rect || {left: 0, top: 0, width: 400, height: 300}; },
    getContext() { return {}; },
  });
  return nodes.get(id);
}

const sandbox = {
  Worker: class { postMessage() {} },
  window: {matchMedia: () => ({matches: true}), addEventListener() {}},
  document: {
    getElementById: node,
    querySelectorAll: () => [],
    querySelector: () => node("dummy"),
    addEventListener() {},
  },
  setTimeout, clearTimeout,
};
vm.createContext(sandbox);
const source = fs.readFileSync(path.join(__dirname, "use.js"), "utf8");
vm.runInContext(source + `
  let renderCount = 0, selectCount = 0;
  draw = () => {};
  renderRequest = () => { renderCount++; };
  selectAt = () => { selectCount++; };
  globalThis.testApi = {
    state,
    get renderCount() { return renderCount; },
    get selectCount() { return selectCount; },
    get pinch() { return pinchStart; },
    viewSize, renderDimensions, drawVector, screen,
  };
`, sandbox);

function pointer(type, id, x, y) {
  node("map").handlers[type]({
    pointerId:id, clientX:x, clientY:y, pointerType:"touch", button:0,
    preventDefault() {},
  });
}

const app = sandbox.testApi;
assert.equal(app.state.referenceAxes, false);
assert.equal(node("plot-legend-panel").open, false);

pointer("pointerdown", 1, 100, 100);
pointer("pointerdown", 2, 300, 100);
pointer("pointermove", 2, 250, 100);
assert(app.state.scale > 1, "bringing fingers together must zoom out");
const rect = {left:35, top:30, width:337, height:234};
const anchoredX = app.state.center[0] + (175 - rect.left - rect.width / 2) * app.state.width / rect.width;
const anchoredY = app.state.center[1] - (100 - rect.top - rect.height / 2) * app.state.height / rect.height;
assert(Math.abs(anchoredX - app.pinch.anchor[0]) < 1e-10);
assert(Math.abs(anchoredY - app.pinch.anchor[1]) < 1e-10);
pointer("pointerup", 2, 250, 100);
assert.equal(app.renderCount, 0, "recalculation waits until the gesture ends");
pointer("pointerup", 1, 100, 100);
assert.equal(app.selectCount, 0, "pinch release must not pick an atom");
assert.equal(app.renderCount, 1);

const previousScale = app.state.scale;
pointer("pointerdown", 3, 100, 100);
pointer("pointerdown", 4, 200, 100);
pointer("pointermove", 4, 260, 100);
assert(app.state.scale < previousScale, "moving fingers apart must zoom in");
pointer("pointerup", 4, 260, 100);
pointer("pointerup", 3, 100, 100);
assert.equal(app.selectCount, 0);

pointer("pointerdown", 5, 100, 100);
pointer("pointerup", 5, 100, 100);
assert.equal(app.selectCount, 1, "a single tap must still pick an atom");
pointer("pointerdown", 6, 100, 100);
pointer("pointermove", 6, 130, 120);
pointer("pointerup", 6, 130, 120);
assert.equal(app.selectCount, 1, "a drag must not pick an atom");
assert.equal(app.renderCount, 3);

node("plot-legend-panel").open = true;
node("plot-legend-panel").handlers.click({target:{closest:() => null}});
assert.equal(node("plot-legend-panel").open, false);
node("vector-annotation").open = true;
node("vector-annotation").handlers.click({target:{closest:() => null}});
assert.equal(node("vector-annotation").open, false);
node("vector-annotation").open = true;
node("reference-axes-toggle").handlers.click();
assert.equal(app.state.referenceAxes, true);
assert.equal(node("reference-axes").checked, true);
assert.equal(node("vector-annotation").open, false);

for (const [width, height, rotation] of [
  [1600, 900, 32.5],
  [390, 780, 45],
  [1200, 800, 90],
  [1200, 800, 0],
]) {
  node("map").rect = {left:0, top:0, width, height};
  app.state.scale = 2;
  app.state.center = [4, -3];
  app.state.displayRotation = rotation;
  app.viewSize();
  const request = app.renderDimensions();
  const angle = rotation * Math.PI / 180;
  const cosine = Math.cos(angle), sine = Math.sin(angle);
  const sourceCenter = [cosine * 4 + sine * -3, -sine * 4 + cosine * -3];
  for (const x of [-app.state.width / 2, app.state.width / 2]) {
    for (const y of [-app.state.height / 2, app.state.height / 2]) {
      const sourceX = cosine * (4 + x) + sine * (-3 + y);
      const sourceY = -sine * (4 + x) + cosine * (-3 + y);
      assert(Math.abs(sourceX - sourceCenter[0]) < request.width / 2,
        `render must cover horizontal corners at ${rotation}° in ${width}×${height} view`);
      assert(Math.abs(sourceY - sourceCenter[1]) < request.height / 2,
        `render must cover vertical corners at ${rotation}° in ${width}×${height} view`);
    }
  }
}

function arrowTriangle(rotation) {
  app.state.center = [0, 0];
  app.state.scale = 1;
  app.state.displayRotation = rotation;
  app.state.atoms = [{position:[0, 0]}, {position:[2, 0]}];
  app.viewSize();
  let path = [];
  const fills = [];
  const context = {
    save() {}, restore() {}, setLineDash() {}, beginPath() { path = []; },
    moveTo(x, y) { path.push([x, y]); },
    lineTo(x, y) { path.push([x, y]); },
    closePath() {}, stroke() {}, fill() { fills.push([...path]); },
  };
  app.drawVector(context);
  assert.equal(fills.length, 1, "a selected vector must have a filled arrowhead");
  assert.equal(fills[0].length, 3, "the arrowhead must be triangular");
  return {triangle:fills[0], endpoint:app.screen([2, 0])};
}

const horizontal = arrowTriangle(0);
assert(horizontal.triangle[0][0] < horizontal.endpoint[0], "tip sits before P2");
assert(horizontal.triangle[0][0] > horizontal.triangle[1][0], "tip points toward P2");
const vertical = arrowTriangle(90);
assert(vertical.triangle[0][1] > vertical.endpoint[1], "rotated tip sits before P2");
assert(vertical.triangle[0][1] < vertical.triangle[1][1], "rotated tip points toward P2");

console.log("Mobile interactions, rotated viewport coverage, and vector arrow passed.");
