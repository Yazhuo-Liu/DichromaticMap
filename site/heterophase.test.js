"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const vm = require("node:vm");
const {harness} = require("./state_consistency.test.js");

const plain = value => JSON.parse(JSON.stringify(value));
// FCC [110] (layers A,B) and BCC [111] (layers A,B,C), as returned by metadata.
const METADATA = {
  lattice: "FCC", axis: "110", lattice2: "BCC", axis2: "111", homophase: false, scale: 2.87 / 3.59,
  label: "FCC [110] / BCC [111]", layers: 3, grain_layers: [2, 3], symbols: ["o", "d", "t"],
  layer_spacing: .35, axial_period: .7, reference_labels: ["[-1 1 0]", "[0 0 1]"],
  grains: [
    {lattice: "FCC", axis: "110", axis_label: "[1 1 0]", layers: 2, layer_spacing: .35, axial_period: .71,
     reference_labels: ["[-1 1 0]", "[0 0 1]"]},
    {lattice: "BCC", axis: "111", axis_label: "[1 1 1]", layers: 3, layer_spacing: .23, axial_period: .69,
     reference_labels: ["[-1 1 0]", "[-1 -1 2]"]},
  ],
  max_angle: 30, default_angle: 24.735610317245346, preset_kind: "orientation relationship",
  presets: [{angle: 24.735610317245346, name: "KS", label: "KS · 24.74°"}],
};

function setup() {
  const h = harness(), app = h.app;
  vm.runInContext("globalThis.heteroAPI = {analysisInputs, visibleData, applyMetadata, currentRenderParameters};",
    h.sandbox);
  const api = h.sandbox.heteroAPI;
  Object.assign(app.state, {lattice2: "BCC", axis2: "111", ratio: METADATA.scale,
    angle: METADATA.default_angle, nearEnabled: true, nearMethod: "local"});
  api.applyMetadata(false, METADATA);
  app.state.visibleLayers = [new Set([0, 1]), new Set([0, 1, 2])];
  // Rows keep the G1 layer; parallel arrays give each match's G2 layer.
  app.setPattern({angle: METADATA.default_angle, preset: "KS", exact_cell: null,
    grains: [[[0, 0, 0, 0, 0, 0], [1, 0, 1, 1, 1, 0]], [[0, 0, 0, 0, 0, 0], [1, .02, 2, 1, 1, 1]]],
    coincidences: [[0, 0, 0]], coincidence_partners: new Float64Array([0]),
    local: [[1, 0, 1, .02, 1]], local_partners: new Float64Array([2])});
  return {h, app, api};
}

test("different grains send both crystals and the a0 ratio with every calculation", () => {
  const {h, app, api} = setup();
  const crystal = {lattice: "FCC", axis: "110", lattice2: "BCC", axis2: "111", ratio: 2.87 / 3.59};
  for (const [key, value] of Object.entries(crystal)) assert.equal(api.currentRenderParameters()[key], value, key);
  app.changeAngle(20); h.flush(45);
  const render = h.last("render").request;
  for (const [key, value] of Object.entries(crystal)) assert.equal(render[key], value, key);
  for (const kind of ["near", "count", "vector"]) {
    const inputs = api.analysisInputs(kind);
    for (const [key, value] of Object.entries(crystal)) assert.equal(inputs[key], value, `${kind} ${key}`);
  }
  // A homophase bicrystal keeps its original request shape.
  Object.assign(app.state, {lattice2: null, axis2: null, ratio: 1});
  for (const key of ["lattice2", "axis2", "ratio"]) assert.equal(key in api.analysisInputs("near"), false, key);
});

test("coincident columns and near pairs need each grain's own layer to be visible", () => {
  const {app, api} = setup();
  let visible = api.visibleData();
  assert.equal(visible.cslMarkers.count, 1);
  assert.equal(visible.localMarkers.count, 1);
  app.state.visibleLayers = [new Set([0, 1]), new Set([0, 1])];
  visible = api.visibleData();
  assert.equal(visible.cslMarkers.count, 1, "the coincidence uses G2 layer A");
  assert.equal(visible.localMarkers.count, 0, "the pair's G2 endpoint is in hidden layer C");
  app.state.visibleLayers = [new Set([1]), new Set([0, 1, 2])];
  visible = api.visibleData();
  assert.equal(visible.cslMarkers.count, 0, "the coincidence uses G1 layer A");
  assert.equal(visible.localMarkers.count, 1);
});

test("manual cells keep one layer per grain through picking, counting and sessions", async () => {
  const {h, app, api} = setup();
  const [x, y] = app.screen([1, .01]);
  const picked = app.nearestCommon(x, y);
  assert.equal(picked.layer, 1);
  assert.equal(picked.layer2, 2);
  app.setMode("cell");
  await app.selectAt(x, y);
  assert.equal(app.state.manual.length, 1);
  assert.equal(app.state.manual[0].layer2, 2);
  // Same G1 layer but another G2 layer is a different projected-column family.
  const [cx, cy] = app.screen([0, 0]);
  app.state.manual[0].layer = 0;
  await app.selectAt(cx, cy);
  assert.equal(app.state.manual.length, 1, "G2 layer mismatch is rejected");
  app.state.manual = [0, 1, 2, 3].map(i => ({position: [i, 0], layer: 1, layer2: 2, source: "local",
    endpoints: [[i, 0], [i, .01]]}));
  assert.deepEqual(plain(api.analysisInputs("count").layer), [1, 2]);
  assert.deepEqual(plain(api.analysisInputs("fit").layer), [1, 2]);
  const counting = app.countManual();
  h.reply(h.last("count"), {interior: [[0, 5, 0], [0, 0, 7]], boundary: [[0, 1, 0], [0, 0, 2]],
    areas: [1, 1]});
  await counting;
  assert.match(h.node("manual-info").dataset.count, /G1: 6 atoms[\s\S]*G2: 9 atoms/);
  const saved = app.sessionState();
  assert.deepEqual([saved.lattice2, saved.axis2, saved.ratio], ["BCC", "111", 2.87 / 3.59]);
  assert.deepEqual(saved.manual.map(v => v.layer2), [2, 2, 2, 2]);
});

test("interface labels describe both grains", () => {
  const {h, app} = setup();
  app.updateSummary();
  assert.equal(h.node("control-title").textContent, "FCC [110] / BCC [111] interface");
  assert.match(h.node("layer-info").textContent, /G1 FCC \[1 1 0\]: 2 layers[\s\S]*G2 BCC \[1 1 1\]: 3 layers/);
  assert.equal(h.node("preset-label").textContent, "Orientation relationship");
});

test("checking the G2 option only reveals G2 controls copied from G1", () => {
  const h = harness(), app = h.app;
  const box = h.node("heterophase"), sent = () => h.messages.filter(m => m.request?.action === "metadata").length;
  const before = sent();
  box.checked = true; box.handlers.change();
  assert.equal(sent(), before, "checking alone requests no new structure");
  assert.deepEqual([h.node("structure2").value, h.node("axis2").value, h.node("lattice-ratio").value],
    [app.state.lattice, app.state.axis, 1]);
  box.checked = false; box.handlers.change();
  assert.equal(sent(), before, "unchecking an unchanged copy requests nothing");
  box.checked = true; box.handlers.change();
  h.node("structure2").value = "BCC"; h.node("lattice-ratio").value = "0.7994";
  h.node("lattice-ratio").handlers.change();
  const request = h.last("metadata").request;
  assert.deepEqual([request.lattice2, request.axis2, request.ratio], ["BCC", "110", .7994]);
});
