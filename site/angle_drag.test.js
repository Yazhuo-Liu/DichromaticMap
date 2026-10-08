"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const vm = require("node:vm");
const {harness, fixture} = require("./state_consistency.test.js");

function rotate(point, degrees) {
  const angle = degrees * Math.PI / 180;
  return [Math.cos(angle) * point[0] - Math.sin(angle) * point[1],
    Math.sin(angle) * point[0] + Math.cos(angle) * point[1]];
}
const reference = [[1.1, .7], [-1.2, .8]];
function patternAt(angle, deformations, translations) {
  return {...fixture(angle), grains: reference.map((point, grain) => {
    const [x, y] = rotate(point, (1 - 2 * grain) * angle / 2);
    const f = deformations[grain], t = translations[grain];
    return [[f[0][0] * x + f[0][1] * y + t[0], f[1][0] * x + f[1][1] * y + t[1],
      0, 9007199254740991, grain, -7]];
  }), coincidences: [[0, 0, 0]], local: [[0, 0, .01, .02, 0]]};
}
const identity = () => [[[1, 0], [0, 1]], [[1, 0], [0, 1]]];
const zero = () => [[0, 0], [0, 0]];
function setup() {
  const h = harness();
  h.app.state.angle = 30;
  h.app.state.deformations = [[[1.2, .3], [.1, .9]], [[.95, -.2], [.15, 1.1]]];
  h.app.state.translations = [[1, -.5], [-.2, .3]];
  const source = patternAt(30, h.app.state.deformations, h.app.state.translations);
  h.app.setPattern(source);
  h.sandbox.drawn = [];
  vm.runInContext("marker = (context, x, y, radius, symbol, stroke, fill) => drawn.push({x,y,stroke,fill});", h.sandbox);
  return {h, app: h.app, source, saved: JSON.stringify(source)};
}
function input(h, angle) {
  h.node("angle-slider").value = String(angle);
  h.node("angle-slider").handlers.input();
}
function assertPose(app, angle) {
  const preview = app.previewVisibleData();
  assert(preview);
  for (let grain = 0; grain < 2; grain++) {
    const record = preview.atoms[grain];
    assert.equal(record.count, 1);
    const target = app.screen(rotate(reference[grain], (1 - 2 * grain) * angle / 2));
    assert(Math.abs(record.x[0] - target[0]) < 1e-10);
    assert(Math.abs(record.y[0] - target[1]) < 1e-10);
    assert.equal(record.rows.flat[record.indices[0] * 6 + 3], 9007199254740991);
  }
}

test("held angle drag draws both grains continuously, undoing old strain and preserving scientific rows", () => {
  const {h, app, source, saved} = setup();
  h.node("angle-slider").handlers.pointerdown({button: 0});
  for (const angle of [44, 45, 46]) {
    input(h, angle); assertPose(app, angle);
    h.sandbox.drawn.length = 0;
    h.flushFrame();
    assert.equal(h.sandbox.drawn.length, 2, "actual Canvas drawing cannot omit either grain");
    assert(!app.currentPattern());
    assert.equal(app.nearestAtom(...app.screen([0, 0])), null);
    assert.equal(app.nearestCommon(...app.screen([0, 0])), null);
    assert.equal(app.visiblePoints(0).length, 0, "preview rows cannot be used as numerical selections");
    assert.equal(app.previewVisibleData().cslMarkers.count, 0);
    assert.equal(app.previewVisibleData().localMarkers.count, 0);
    h.flush(45);
  }
  assert.equal(h.messages.filter(message => message.request?.action === "render").length, 0,
    "holding the slider does not enqueue numerical jobs per mouse move");
  app.renderRequest(); h.flush(45); h.flushFrame();
  assert.equal(h.messages.filter(message => message.request?.action === "render").length, 0,
    "resize or other view requests during a held drag also wait for release");
  assert.equal(JSON.stringify(source), saved);
  assert(h.node("status").innerHTML.includes("Visible G1 / G2: 1 / 1"));
  const before = app.previewVisibleData().revision;
  for (const angle of [47, 48, 49]) input(h, angle);
  assert.equal(h.frameCount, 1, "rapid input is painted once per animation frame");
  assert(app.previewVisibleData().revision > before, "GPU uploads must follow the new angle");
  assertPose(app, 49);
});

test("PNG waits through held drag and final computation while the preview stays drawn", async () => {
  const {h, app} = setup();
  h.node("angle-slider").handlers.pointerdown({button: 0});
  input(h, 54);
  const exporting = app.exportPNG();
  await Promise.resolve(); h.flush(45);
  assert.equal(h.exports, 0);
  h.windowEvents.get("pointerup")();
  assert(!app.angleSliderDragging);
  await Promise.resolve();
  assertPose(app, 54);
  h.sandbox.drawn.length = 0; h.flushFrame();
  assert.equal(h.sandbox.drawn.length, 2, "releasing cannot clear the grains before the worker responds");
  h.flush(45);
  h.reply(h.last("render"), patternAt(54, identity(), zero()));
  await exporting;
  assert(app.currentPattern());
  assert.equal(app.previewVisibleData(), null);
  assert.equal(h.exports, 1);
  const [x, y] = app.screen(rotate(reference[0], 27));
  assert.equal(app.nearestAtom(x, y).grain, 0);
});

test("obsolete renders and unrelated geometry cannot replace or reuse an angle preview", async () => {
  const {h, app} = setup();
  app.changeAngle(34); h.flush(45); const old = h.last("render");
  h.node("angle-slider").handlers.pointerdown({button: 0});
  input(h, 52);
  h.reply(old, patternAt(34, identity(), zero())); await Promise.resolve();
  assertPose(app, 52);
  app.state.axis = "100";
  assert.equal(app.previewVisibleData(), null);
  input(h, 54);
  assert.equal(app.previewVisibleData(), null, "different-axis source rows cannot become an angle preview");
  h.windowEvents.get("pointercancel")();
  assert(!app.angleSliderDragging);
});

test("engine failure releases a held export without publishing preview data", async () => {
  const {h, app} = setup();
  h.node("angle-slider").handlers.pointerdown({button: 0}); input(h, 54);
  const exporting = app.exportPNG();
  app.worker.onerror();
  await exporting;
  assert(!app.angleSliderDragging);
  assert.equal(h.exports, 0);
  assert(!app.currentPattern());
});

test("limited preview coverage is explicit and never advertised as exact", () => {
  const {h, app, source} = setup();
  app.setPattern(source, {width: 15, height: 15});
  h.node("angle-slider").handlers.pointerdown({button: 0}); input(h, 90);
  assert.equal(app.previewVisibleData().coverageComplete, false);
  assert(h.node("status").innerHTML.includes("Release to update edge atoms."));
  assert(!app.currentPattern());
  h.windowEvents.get("blur")();
  assert(!app.angleSliderDragging);
});

test("export follows a new drag started immediately after the previous release", async () => {
  const {h, app} = setup();
  h.node("angle-slider").handlers.pointerdown({button: 0}); input(h, 44);
  const exporting = app.exportPNG();
  h.windowEvents.get("pointerup")();
  h.node("angle-slider").handlers.pointerdown({button: 0}); input(h, 54);
  await Promise.resolve(); h.flush(45);
  assert.equal(h.exports, 0);
  assert.equal(h.messages.filter(message => message.request?.action === "render").length, 0);
  h.windowEvents.get("pointerup")(); await Promise.resolve();
  h.flush(45); h.reply(h.last("render"), patternAt(54, identity(), zero()));
  await exporting;
  assert.equal(h.exports, 1);
  assert(app.currentPattern());
});
