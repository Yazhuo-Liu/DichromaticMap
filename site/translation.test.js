"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const vm = require("node:vm");
const {harness, fixture} = require("./state_consistency.test.js");

const plain = value => JSON.parse(JSON.stringify(value));
function setup() {
  const h = harness();
  h.app.setPattern(fixture());
  h.node("translation-grain").value = "0";
  h.node("translation-dx").value = "0.25";
  h.node("translation-dy").value = "-0.1";
  vm.runInContext("globalThis.moveAPI = {translateGrain, undoGrainTranslation, resetGrainTranslations, translationBounds, uniformStrainBlocked, analysisInputs, currentRenderParameters};", h.sandbox);
  return {h, app:h.app, api:h.sandbox.moveAPI};
}

test("whole-grain moves regenerate physical geometry across all layers while retaining GB clipping", () => {
  const {h, app, api} = setup();
  app.state.visibleLayers = [new Set(), new Set([0])];
  app.state.boundary = [[-1, 0], [1, 0]];
  app.state.regions = [true, false, false, true];
  app.state.atoms = [{position:[0, 0]}];
  app.state.manual = [{position:[0, 0]}];
  h.node("translate-whole").handlers.click();
  assert.deepEqual(plain(app.state.grainDisplacements), [[{delta:[.25, -.1], bounds:null}], []]);
  assert.deepEqual(plain(app.state.boundary), [[-1, 0], [1, 0]]);
  assert.deepEqual(plain(app.state.regions), [true, false, false, true]);
  assert.equal(app.state.atoms.length, 0);
  assert.equal(app.state.manual.length, 0);
  assert.equal(app.currentPattern(), false, "old positions cannot be picked or exported");
  h.flush(45);
  assert.deepEqual(plain(h.last("render").request.grain_displacements), plain(app.state.grainDisplacements));
  assert.equal("visible_layers" in h.last("render").request, false, "physical moves must include hidden layers");
  assert.deepEqual(plain(api.analysisInputs("count").grain_displacements), plain(app.state.grainDisplacements));
});

test("two rectangle clicks use physical axes after display rotation and panning", async () => {
  const {h, app, api} = setup();
  app.state.displayRotation = 31;
  app.state.center = [.2, -.3];
  h.node("translation-grain").value = "1";
  h.node("translation-grain").handlers.change();
  h.node("pick-translation").handlers.click();
  await app.selectAt(...app.screen([-.8, -.6]));
  await app.selectAt(...app.screen([1.2, .9]));
  assert.equal(app.state.mode, "idle");
  assert.equal(h.node("translate-region").disabled, false);
  plain(api.translationBounds()).forEach((value, index) => assert(Math.abs(value - [-.8, 1.2, -.6, .9][index]) < 1e-12));
  h.node("translation-dx").value = ".25";
  h.node("translation-dy").value = "-.1";
  h.node("translate-region").handlers.click();
  const operation = app.state.grainDisplacements[1][0];
  assert.deepEqual(plain(operation.delta), [.25, -.1]);
  operation.bounds.forEach((value, index) => assert(Math.abs(value - [-.8, 1.2, -.6, .9][index]) < 1e-12));
  assert.equal(app.state.translationCorners.length, 2, "the active selection stays editable");
  plain(api.translationBounds()).forEach((value, index) => assert(Math.abs(value - [-.55, 1.45, -.7, .8][index]) < 1e-12));
});

test("moves invalidate superseded worker replies and retain exact final coordinates", async () => {
  const {h, app, api} = setup();
  app.changeAngle(22); h.flush(45); const old = h.last("render");
  h.node("translation-dx").value = ".25";
  h.node("translation-dy").value = "-.1";
  api.translateGrain(); h.flush(45); const fresh = h.last("render");
  h.reply(old, fixture(22)); await Promise.resolve();
  assert.equal(app.currentPattern(), false);
  h.reply(fresh, {...fixture(22), grains:[[[.25, -.1, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0]]]});
  await Promise.resolve();
  assert.equal(app.currentPattern(), true);
  const atom = app.nearestAtom(...app.screen([.25, -.1]));
  assert.equal(atom.grain, 0);
  assert.deepEqual(plain(atom.position), [.25, -.1]);
});

test("undo/reset preserve the other grain and the existing uniform F/t base", () => {
  const {h, app, api} = setup();
  app.state.deformations = [[[1.02, 0], [0, 1]], [[1, 0], [0, 1]]];
  app.state.translations = [[.1, .2], [0, 0]];
  app.state.nearCell = {cell:[[1, 0], [0, 1]]};
  const base = plain([app.state.deformations, app.state.translations]);
  api.translateGrain();
  h.node("translation-grain").value = "1"; api.translateGrain();
  h.node("undo-translation").handlers.click();
  assert.deepEqual(plain(app.state.grainDisplacements.map(ops => ops.length)), [1, 0]);
  h.node("reset-translations").handlers.click();
  assert.deepEqual(plain(app.state.grainDisplacements), [[], []]);
  assert.deepEqual(plain([app.state.deformations, app.state.translations]), base);
  assert.equal(api.uniformStrainBlocked(), true, "a retained custom base cannot masquerade as unstrained geometry");
  app.state.nearEnabled = true; app.state.nearMethod = "local";
  h.node("near-toggle").handlers.click();
  assert.deepEqual(plain([app.state.deformations, app.state.translations]), base);
  app.changeAngle(22);
  assert.equal(api.uniformStrainBlocked(), false);
});

test("manual operations block uniform-strain workflows without resetting physical state", async () => {
  const {h, app, api} = setup();
  api.translateGrain();
  const before = plain(app.state.grainDisplacements);
  app.state.manual = Array.from({length:3}, () => ({position:[0, 0], endpoints:[[0, 0], [0, 0]], layer:0, source:"local"}));
  await app.completeManual();
  app.state.manual.push(app.state.manual[0]);
  await app.toggleSelectedStrain();
  app.state.nearEnabled = true; app.state.nearMethod = "local";
  h.node("near-method").value = "strain"; h.node("near-method").handlers.change();
  assert.equal(app.state.nearMethod, "local");
  app.searchNear(); h.flush(150);
  assert.equal(h.last("complete"), undefined);
  assert.equal(h.last("fit_selected"), undefined);
  assert.equal(h.last("near_search"), undefined);
  assert.deepEqual(plain(app.state.grainDisplacements), before);
  assert.deepEqual(plain(app.sessionState().grain_displacements), before);
});

test("angle resets discard moves and never reinterpret displaced atoms as a rotation preview", () => {
  const {h, app, api} = setup();
  api.translateGrain();
  app.setPattern({...fixture(), grains:[[[.25, -.1, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0]]]});
  h.node("angle-slider").handlers.pointerdown({button:0});
  h.node("angle-slider").value = "54"; h.node("angle-slider").handlers.input();
  assert.deepEqual(plain(app.state.grainDisplacements), [[], []]);
  assert.equal(app.previewVisibleData(), null);
  h.windowEvents.get("pointerup")();
});

test("invalid or excessive moves never change physical state", () => {
  const {h, app, api} = setup();
  for (const value of ["", "NaN", "Infinity", "100000001"]) {
    h.node("translation-dx").value = value; api.translateGrain();
    assert.deepEqual(plain(app.state.grainDisplacements), [[], []]);
  }
  h.node("translation-dx").value = "0"; h.node("translation-dy").value = "0"; api.translateGrain();
  assert.deepEqual(plain(app.state.grainDisplacements), [[], []]);
  h.node("translation-dx").value = "1";
  app.state.translationCorners = [[0, 0], [0, 1]]; api.translateGrain(true);
  assert.deepEqual(plain(app.state.grainDisplacements), [[], []]);
  app.state.grainDisplacements[0] = Array.from({length:64}, () => ({delta:[.1, 0], bounds:null}));
  api.translateGrain(); assert.equal(app.state.grainDisplacements[0].length, 64);
});
