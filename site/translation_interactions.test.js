"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const {harness, fixture} = require("./state_consistency.test.js");

const plain = value => JSON.parse(JSON.stringify(value));
function close(actual, expected) {
  assert.equal(actual.length, expected.length);
  actual.forEach((value, index) => assert(Math.abs(value - expected[index]) < 1e-10, `${actual} != ${expected}`));
}
function setup() {
  const h = harness(), app = h.app;
  app.setPattern(fixture());
  h.node("translation-grain").value = "0";
  h.node("map").setPointerCapture = () => {};
  vm.runInContext(`globalThis.moves = {
    get selection() { return translationSelection; },
    get preview() { return translationPreview; },
    get vector() { return translationVector; },
    get pointer() { return translationPointer; },
    get undo() { return translationUndo; },
    wholeGrainOffset, translationBounds,
  };`, h.sandbox);
  // Register the application's real keyboard listener; the general harness
  // deliberately ignores document-level events used by most other tests.
  let keydown;
  h.sandbox.document.addEventListener = (kind, callback) => { if (kind === "keydown") keydown = callback; };
  const source = fs.readFileSync(path.join(__dirname, "use.js"), "utf8");
  const first = source.indexOf('document.addEventListener("keydown"');
  const last = source.indexOf('window.addEventListener("resize"', first);
  vm.runInContext(source.slice(first, last), h.sandbox);
  const target = (x, y) => {
    for (const [id, value] of [["translation-dx", x], ["translation-dy", y]]) {
      h.node(id).value = String(value); h.node(id).handlers.input();
    }
  };
  target(0, 0);
  const pointer = (type, id, point, pointerType = "mouse") => {
    const [clientX, clientY] = point;
    h.node("map").handlers[type]({pointerId:id, clientX, clientY, pointerType, button:0, preventDefault() {}});
  };
  const click = (point, id = 1) => { pointer("pointerdown", id, point); pointer("pointerup", id, point); };
  const select = async (a = [-.1, -.1], b = [.1, .1]) => {
    h.node("pick-translation").handlers.click();
    await app.selectAt(...app.screen(a)); await app.selectAt(...app.screen(b));
  };
  return {h, app, api:h.sandbox.moves, target, pointer, click, select,
    key(key) { keydown({key, preventDefault() {}}); },
    apply(local = false) { h.node(local ? "translate-region" : "translate-whole").handlers.click(); },
  };
}

// Independent point-by-point oracle for the portable, incremental operations.
function moved(points, history) {
  return points.map(point => history.reduce((current, {bounds:b, delta:d}) => {
    if (!b || (current[0] >= b[0] && current[0] <= b[1] && current[1] >= b[2] && current[1] <= b[3]))
      return [current[0] + d[0], current[1] + d[1]];
    return current;
  }, [...point]));
}

test("absolute whole-grain targets are idempotent, editable and reversible to zero", () => {
  const {h, app, api, target, apply} = setup();
  target(1, -.5); apply();
  const first = plain(app.state.grainDisplacements[0]);
  apply();
  assert.deepEqual(plain(app.state.grainDisplacements[0]), first);
  assert.equal(api.undo[0].length, 1, "a repeated target is not an undo operation");
  target(2, .5); apply();
  close(api.wholeGrainOffset(app.state.grainDisplacements[0]), [2, .5]);
  assert.deepEqual(moved([[0, 0], [1, 1]], app.state.grainDisplacements[0]), [[2, .5], [3, 1.5]]);
  target(0, 0); apply();
  assert.deepEqual(moved([[0, 0], [1, 1]], app.state.grainDisplacements[0]), [[0, 0], [1, 1]]);
  h.node("undo-translation").handlers.click();
  close(api.wholeGrainOffset(app.state.grainDisplacements[0]), [2, .5]);
  assert.equal(h.node("translation-dx").value, "2");
});

test("editing a region retains its original atoms even at an overlapping destination", async () => {
  const {h, app, api, target, apply, select} = setup();
  await select();
  target(1, 0); apply(true);
  const originalBounds = plain(api.selection.bounds);
  assert.deepEqual(moved([[0, 0], [1, 0], [2, 0]], app.state.grainDisplacements[0]), [[1, 0], [1, 0], [2, 0]]);
  target(2, 0); apply(true);
  assert.equal(app.state.grainDisplacements[0].length, 1);
  assert.deepEqual(plain(app.state.grainDisplacements[0][0].bounds), originalBounds);
  assert.deepEqual(moved([[0, 0], [1, 0], [2, 0]], app.state.grainDisplacements[0]), [[2, 0], [1, 0], [2, 0]]);
  close(api.translationBounds(), [1.9, 2.1, -.1, .1]);
  apply(true);
  assert.equal(api.undo[0].length, 2);
  h.node("undo-translation").handlers.click();
  assert.deepEqual(plain(api.selection.target), [1, 0]);
  close(api.translationBounds(), [.9, 1.1, -.1, .1]);
  assert.deepEqual(moved([[0, 0], [1, 0]], app.state.grainDisplacements[0]), [[1, 0], [1, 0]]);
});

test("zero region target remains selected and Undo restores both offset and selection", async () => {
  const {h, app, api, target, apply, select} = setup();
  await select(); target(.5, .25); apply(true);
  target(0, 0); apply(true);
  assert.equal(app.state.grainDisplacements[0].length, 0);
  assert.equal(h.node("undo-translation").disabled, false);
  assert.equal(h.node("translate-region").disabled, false);
  close(api.translationBounds(), [-.1, .1, -.1, .1]);
  h.node("undo-translation").handlers.click();
  assert.deepEqual(plain(api.selection.target), [.5, .25]);
  assert.deepEqual(plain(app.state.grainDisplacements[0][0].delta), [.5, .25]);
  h.node("undo-translation").handlers.click();
  assert.equal(app.state.grainDisplacements[0].length, 0);
  assert.deepEqual(plain(api.selection.target), [0, 0]);
  assert.equal(h.node("translate-region").disabled, false);
});

test("an active region can be edited at the 64-operation limit", async () => {
  const {h, app, api, target, apply, select} = setup();
  app.state.grainDisplacements[0] = Array.from({length:63}, () => ({delta:[.125, 0], bounds:null}));
  app.setPattern(fixture());
  await select(); target(.5, 0); apply(true);
  assert.equal(app.state.grainDisplacements[0].length, 64);
  assert.equal(h.node("translate-region").disabled, false);
  target(1, 0); apply(true);
  assert.equal(app.state.grainDisplacements[0].length, 64);
  assert.deepEqual(plain(app.state.grainDisplacements[0][63].delta), [1, 0]);
  assert.equal(api.selection.baseHistory.length, 63);
  target(0, 0); apply(true);
  assert.equal(app.state.grainDisplacements[0].length, 63);
});

test("whole-grain edits carry existing regions and end the active region editor", async () => {
  const {app, api, target, apply, select} = setup();
  target(1, 0); apply(); app.setPattern(fixture());
  await select([.9, -.1], [1.1, .1]); target(0, .5); apply(true);
  target(2, 0); apply();
  assert.equal(api.selection, null);
  assert.equal(app.state.translationCorners.length, 0);
  assert.deepEqual(moved([[0, 0], [1, 0]], app.state.grainDisplacements[0]), [[2, .5], [3, 0]]);
});

test("changing grains clears the region and shows the chosen grain's absolute whole offset", async () => {
  const {h, app, api, target, apply, select} = setup();
  target(2, -.5); apply(); app.setPattern(fixture());
  await select();
  h.node("translation-grain").value = "1"; h.node("translation-grain").handlers.change();
  assert.equal(api.selection, null);
  assert.equal(h.node("translate-region").disabled, true);
  assert.equal(h.node("translation-dx").value, "0");
  target(-1, .25); apply();
  h.node("translation-grain").value = "0"; h.node("translation-grain").handlers.change();
  assert.equal(h.node("translation-dx").value, "2");
  assert.equal(h.node("translation-dy").value, "-0.5");
  assert.deepEqual(plain(app.state.grainDisplacements.map(history => api.wholeGrainOffset(history))), [[2, -.5], [-1, .25]]);
});

test("measured vector uses physical xy, ignores axial repeat and supports signed length", () => {
  const {h, app, api, apply} = setup();
  app.state.displayRotation = 67; app.state.a0 = 9; app.state.axialRepeat = 4;
  app.setPattern(fixture());
  app.state.atoms = [{position:[1, -2], grain:0, layer:0}, {position:[4, 2], grain:1, layer:7}];
  h.node("translation-use-vector").handlers.click();
  close(api.vector, [.6, .8]);
  assert.equal(h.node("translation-dx").value, "3");
  assert.equal(h.node("translation-dy").value, "4");
  assert.equal(h.node("translation-vector-length").value, "5");
  h.node("translation-vector-length").value = "-2.5";
  h.node("translation-vector-length").handlers.input();
  close([Number(h.node("translation-dx").value), Number(h.node("translation-dy").value)], [-1.5, -2]);
  apply();
  assert.equal(app.state.atoms.length, 0);
  close(api.vector, [.6, .8]);
  assert.equal(h.node("translation-vector-length").disabled, false, "the copied direction survives geometry invalidation");
  h.node("translation-vector-length").value = "10"; h.node("translation-vector-length").handlers.input(); apply();
  close(api.wholeGrainOffset(app.state.grainDisplacements[0]), [6, 8]);
});

test("invalid vector lengths cannot apply a stale shift and manual xy editing drops the copied direction", () => {
  const {h, app, api, apply, target} = setup();
  app.state.atoms = [{position:[0, 0]}, {position:[3, 4]}];
  h.node("translation-use-vector").handlers.click();
  for (const value of ["", "NaN", "Infinity", "100000001"]) {
    h.node("translation-vector-length").value = value; h.node("translation-vector-length").handlers.input(); apply();
    assert.equal(app.state.grainDisplacements[0].length, 0);
    assert.equal(h.node("translation-dx").value, "");
  }
  target(.3, .4);
  assert.equal(api.vector, null);
  assert.equal(h.node("translation-vector-length").disabled, true);
  apply(); assert.deepEqual(plain(app.state.grainDisplacements[0][0].delta), [.3, .4]);
});

test("a measured direction edits the retained region without selecting destination atoms", async () => {
  const {h, app, api, apply, select} = setup();
  await select();
  app.state.atoms = [{position:[-1, 0]}, {position:[1, 0]}];
  h.node("translation-use-vector").handlers.click();
  h.node("translation-vector-length").value = "1"; h.node("translation-vector-length").handlers.input(); apply(true);
  assert.deepEqual(moved([[0, 0], [1, 0]], app.state.grainDisplacements[0]), [[1, 0], [1, 0]]);
  assert.equal(app.state.atoms.length, 0);
  h.node("translation-vector-length").value = "2"; h.node("translation-vector-length").handlers.input(); apply(true);
  assert.deepEqual(moved([[0, 0], [1, 0]], app.state.grainDisplacements[0]), [[2, 0], [1, 0]]);
  assert.deepEqual(plain(api.selection.target), [2, 0]);
});

test("zero in-plane vectors and stale geometry cannot replace a shift", () => {
  const {h, app, api, target} = setup();
  target(.25, .5);
  app.state.atoms = [{position:[1, 2], layer:0}, {position:[1, 2], layer:1}];
  h.node("translation-use-vector").handlers.click();
  assert.equal(api.vector, null); assert.equal(h.node("translation-dx").value, "0.25");
  app.state.atoms[1].position = [3, 4]; app.changeAngle(22);
  h.node("translation-use-vector").handlers.click();
  assert.equal(api.vector, null);
});

test("drag rectangle previews and selects physical axes without panning the plot", () => {
  const {h, app, api, pointer} = setup();
  app.state.displayRotation = 31; app.state.center = [.3, -.4];
  const before = plain(app.state.center), a = [-1, -.5], b = [1, .75];
  h.node("pick-translation").handlers.click();
  pointer("pointerdown", 1, app.screen(a)); pointer("pointermove", 1, app.screen(b));
  close(api.preview, b);
  assert.deepEqual(plain(app.state.center), before);
  assert.equal(h.node("translate-region").disabled, true);
  pointer("pointerup", 1, app.screen(b));
  close(api.selection.bounds, [-1, 1, -.5, .75]);
  assert.equal(app.state.mode, "idle");
  assert.equal(h.node("translate-region").disabled, false);
  assert.deepEqual(plain(app.state.center), before);
});

test("two-click rectangle has a live hover preview and Esc cancels it", () => {
  const {h, app, api, click, pointer, key} = setup();
  h.node("pick-translation").handlers.click();
  click(app.screen([-1, -.5]));
  assert.equal(app.state.translationCorners.length, 1);
  pointer("pointermove", 99, app.screen([1, .75]));
  close(api.preview, [1, .75]);
  click(app.screen([1, .75]));
  close(api.selection.bounds, [-1, 1, -.5, .75]);
  key("Escape");
  assert.equal(api.selection, null); assert.equal(api.preview, null);
  assert.equal(app.state.translationCorners.length, 0);
  assert.equal(h.node("translate-region").disabled, true);
});

test("pointer cancellation restores the first corner and never pans or commits a partial region", () => {
  const {h, app, api, click, pointer} = setup();
  const center = plain(app.state.center);
  h.node("pick-translation").handlers.click(); click(app.screen([-1, -.5]));
  pointer("pointerdown", 1, app.screen([0, .5])); pointer("pointermove", 1, app.screen([1, .75]));
  pointer("pointercancel", 1, app.screen([1, .75]));
  assert.equal(api.selection, null); assert.equal(api.preview, null);
  assert.equal(app.state.translationCorners.length, 1);
  close(app.state.translationCorners[0], [-1, -.5]);
  assert.deepEqual(plain(app.state.center), center);
  assert.equal(h.node("translate-region").disabled, true);
});

test("Esc during a held rectangle drag cannot commit or pan when the pointer is released", () => {
  const {h, app, api, pointer, key} = setup();
  const center = plain(app.state.center);
  h.node("pick-translation").handlers.click();
  pointer("pointerdown", 1, app.screen([-1, -.5])); pointer("pointermove", 1, app.screen([1, .75]));
  key("Escape");
  pointer("pointermove", 1, app.screen([2, 1])); pointer("pointerup", 1, app.screen([2, 1]));
  assert.equal(api.selection, null); assert.equal(api.preview, null);
  assert.equal(app.state.mode, "idle"); assert.equal(app.state.translationCorners.length, 0);
  assert.deepEqual(plain(app.state.center), center);
  assert.equal(h.node("translate-region").disabled, true);
});

test("a second touch changes rectangle selection into pinch without committing a box", () => {
  const {h, app, api, pointer} = setup();
  h.node("pick-translation").handlers.click();
  pointer("pointerdown", 1, [300, 300], "touch");
  pointer("pointermove", 1, [400, 400], "touch");
  assert(api.preview);
  pointer("pointerdown", 2, [600, 400], "touch");
  assert.equal(api.pointer, undefined); assert.equal(api.preview, null);
  const scale = app.state.scale;
  pointer("pointermove", 2, [700, 400], "touch");
  assert(app.state.scale < scale);
  pointer("pointerup", 2, [700, 400], "touch");
  pointer("pointerup", 1, [400, 400], "touch");
  assert.equal(api.selection, null);
  assert.equal(app.state.translationCorners.length, 0);
  assert.equal(h.node("translate-region").disabled, true);
  assert.deepEqual(plain(app.state.grainDisplacements), [[], []]);
});

test("ordinary drags still pan and a rectangle drag ignores unrelated pointer releases", () => {
  const {h, app, api, pointer} = setup();
  pointer("pointerdown", 1, [300, 300]); pointer("pointermove", 1, [350, 325]); pointer("pointerup", 1, [350, 325]);
  assert.notDeepEqual(plain(app.state.center), [0, 0]);
  app.setPattern(fixture());
  h.node("pick-translation").handlers.click();
  pointer("pointerdown", 1, app.screen([-1, -.5])); pointer("pointermove", 1, app.screen([1, .75]));
  pointer("pointerup", 99, [100, 100]);
  assert.equal(api.pointer.pointerId, 1);
  pointer("pointerup", 1, app.screen([1, .75]));
  close(api.selection.bounds, [-1, 1, -.5, .75]);
});
