"use strict";
const assert = require("node:assert/strict");
const test = require("node:test");
const vm = require("node:vm");
const {harness, fixture} = require("./state_consistency.test.js");

function setup() {
  const h = harness();
  vm.runInContext(`
    metadata.layers = 27;
    state.visibleLayers = [new Set([6, 12]), new Set([12])];
    state.symbols = ["o","d","t","s","p","h","star","+","x","t1","t2","t3",
      ...Array.from({length:15}, (_, index) => "number:" + (index + 13))];
    globalThis.legendAPI = {draw:drawLegend, entries:() => plotLegendEntries(), drawPNG:(...args) => drawPlotLegend(...args)};
  `, h.sandbox);
  h.app.setPattern({...fixture(), coincidences: []});
  return {h, api: h.sandbox.legendAPI};
}

function recordingContext() {
  const text = [], circles = [], fills = [];
  return {text, circles, fills, save() {}, restore() {}, beginPath() {}, closePath() {},
    moveTo() {}, lineTo() {}, stroke() {}, fillRect() {}, strokeRect() {}, clearRect() {},
    setTransform() {}, arc(x, y, radius) { circles.push({x,y,radius}); },
    fill() { fills.push(this.fillStyle); }, fillText(value) { text.push(String(value)); }};
}

test("legend follows visible high layers and retains real numbered-circle symbols", () => {
  const {h, api} = setup();
  api.draw();
  const html = h.node("plot-legend").innerHTML;
  assert(html.includes("G1 · G") && html.includes("G1 · M") && html.includes("G2 · M"));
  assert(!html.includes("G1 · A"));
  const entries = Array.from(api.entries());
  assert.deepEqual(entries.map(entry => [entry.label, entry.symbol]),
    [["G1 · G", "star"], ["G1 · M", "number:13"], ["G2 · M", "number:13"]]);
  assert(html.includes("canvas"), "screen legend draws actual marker paths instead of fallback text glyphs");
});

test("screen and PNG render the same numbered glyphs, labels and grain fill conventions", () => {
  const {h, api} = setup();
  const icons = [recordingContext(), recordingContext(), recordingContext()];
  h.sandbox.document.querySelectorAll = selector => selector === "#plot-legend canvas"
    ? icons.map((context, index) => ({dataset: {legendIndex: String(index)}, style: {}, getContext: () => context})) : [];
  api.draw();
  assert.deepEqual(icons.map(context => context.text), [[], ["13"], ["13"]]);
  assert.deepEqual(icons[1].fills, [h.app.state.colors[0]]);
  assert.deepEqual(icons[2].fills, []);
  const png = recordingContext();
  api.drawPNG(png);
  assert.deepEqual(png.text, ["G1 · G", "13", "G1 · M", "13", "G2 · M"]);
  assert.equal(png.circles.length, 2);
});

test("compact legends disclose omitted visible layers without inventing a marker", () => {
  const {h, api} = setup();
  vm.runInContext("state.visibleLayers = [new Set(Array.from({length:27}, (_, i) => i)), new Set()];", h.sandbox);
  api.draw();
  const entries = Array.from(api.entries());
  assert.equal(entries.length, 7);
  assert.equal(entries.at(-1).label, "+21 layers");
  assert.equal(entries.at(-1).symbol, undefined);
  const html = h.node("plot-legend").innerHTML;
  assert(html.includes("+21 layers"));
  assert.equal((html.match(/<canvas/g) || []).length, 6);
  const png = recordingContext(); api.drawPNG(png);
  assert.equal(png.text.at(-1), "+21 layers");
  assert.equal(png.circles.length, 1, "only the actual circle layer draws a circular icon");
});
