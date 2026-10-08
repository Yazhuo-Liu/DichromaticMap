"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const {createRenderer, packAtoms, gpuColor, GPU_SYMBOL_IDS, GPU_MIN_ATOMS} = require("./gpu_renderer.js");

const style = {colors: ["#1677d2", "#e35d35"], symbols: ["o", "d"], sizes: [1, 2], scale: 5};
const rows = {flat: new Float64Array([1, 2, 0, 4, 5, 6, 10, 20, 1, 40, 50, 60]), stride: 6};
const selection = {rows, indices: new Uint32Array([1, 0, 99]),
  x: new Float64Array([100.123456789, 200.987654321, NaN]),
  y: new Float64Array([300.123456789, 400.987654321, NaN]), count: 2};

test("GPU uses final pixel coordinates and retains selected row and grain order", () => {
  const before = rows.flat.slice();
  const packed = packAtoms({...style, atoms: [selection, [{point: [0, 0, 0], x: 501, y: 502}]]});
  assert.equal(packed.count, 3);
  assert.equal(packed.view.length, 9);
  assert.equal(packed.view[0], Math.fround(selection.x[0]));
  assert.equal(packed.view[1], Math.fround(selection.y[0]));
  assert.equal(packed.view[2], 1); // Layer 1 has the second style.
  assert.equal(packed.styles.data[13], 1); // Its geometry is a diamond.
  assert.equal(packed.view[3], Math.fround(selection.x[1]));
  assert.equal(packed.view[6], 501); // Second grain remains last.
  assert.equal(packed.styles.data[31], 0); // Hollow second grain has no fill alpha.
  assert.equal(packed.styles.data[35], 1);
  assert.deepEqual(rows.flat, before);
});

test("visible unsupported symbols and invalid coordinates retain Canvas rendering", () => {
  assert.equal(packAtoms({...style, symbols: ["o", "number:2"], atoms: [selection, []]}), null);
  assert.equal(packAtoms({...style, atoms: [[{point: [0, 0, 0], x: NaN, y: 5}], []]}), null);
  // A hidden unsupported layer does not prevent accelerating visible circles.
  assert.notEqual(packAtoms({...style, symbols: ["o", "number:2"],
    atoms: [[{point: [0, 0, 0], x: 10, y: 20}], []]}), null);
});

test("marker radius follows the Canvas size clamps and buffers can be reused", () => {
  const storage = new Float32Array(100);
  const packed = packAtoms({...style, sizes: [100, 0.001],
    atoms: [[{point: [0, 0, 0], x: 1, y: 2}, {point: [0, 0, 1], x: 3, y: 4}], []]}, storage);
  assert.equal(packed.data, storage);
  assert.equal(packed.view.length, 6);
  assert.equal(packed.styles.data[0], 8);
  assert.equal(packed.styles.data[12], 2);
});

test("CSS hex colors stay faithful and unsupported colors use Canvas", () => {
  assert.deepEqual(gpuColor("#f00"), [1, 0, 0, 1]);
  assert.deepEqual(gpuColor("#ABCDEF"), [171 / 255, 205 / 255, 239 / 255, 1]);
  assert.equal(gpuColor("rgba(1,2,3,.5)"), null);
  assert.equal(gpuColor("abcdef"), null);
  assert.equal(packAtoms({...style, colors: ["red", "blue"], atoms: [selection, []]}), null);
});

test("unavailable WebGL or initialization errors return the Canvas fallback", () => {
  assert.equal(GPU_MIN_ATOMS, 12000);
  assert.equal(createRenderer({document: {createElement: () => ({getContext: () => null})}}), null);
  assert.equal(createRenderer({document: {createElement: () => { throw Error("unavailable"); }}}), null);
});

function fakeRenderer() {
  const uploads = [], draws = [], allocations = [], styleUpdates = [];
  let lost = false, error = 0;
  const functions = {NO_ERROR: 0, getParameter: () => [1, 1024],
    getShaderParameter: () => true, getProgramParameter: () => true,
    getAttribLocation: () => 0, isContextLost: () => lost, getError: () => error,
    bufferData: (_, size) => allocations.push(size),
    bufferSubData: (_, __, data) => uploads.push(Float32Array.from(data)),
    uniform4fv: (_, data) => styleUpdates.push(Float32Array.from(data)),
    drawArrays: (_, __, count) => draws.push(count)};
  const gl = new Proxy(functions, {get: (target, name) => name in target ? target[name]
    : /^[A-Z_]+$/.test(name) ? 1 : () => {}});
  const canvas = {width: 0, height: 0, getContext: () => gl, addEventListener: () => {}};
  const renderer = createRenderer({document: {createElement: () => canvas}, minimumAtoms: 0});
  const options = {...style, width: 100, height: 100, dpr: 1,
    plot: {left: 0, top: 0, width: 100, height: 100}, atoms: [selection, []]};
  return {renderer, options, uploads, draws, allocations, styleUpdates, setLost: value => {lost = value;}, setError: value => {error = value;}};
}

test("reused projections upload on viewport revision and style changes", () => {
  const {renderer, options, uploads, styleUpdates} = fakeRenderer();
  assert.ok(renderer.render({...options, revision: 1}));
  assert.equal(uploads.length, 1);
  assert.ok(renderer.render({...options, revision: 1}));
  assert.equal(uploads.length, 1);
  const original = selection.x[0];
  try {
    selection.x[0] = 80;
    assert.ok(renderer.render({...options, revision: 2}));
    assert.equal(uploads.length, 2);
    assert.equal(uploads[1][0], 80);
    assert.ok(renderer.render({...options, revision: 2, colors: ["#ff0000", "#00ff00"]}));
    assert.equal(uploads.length, 2, "color-only changes do not upload coordinates again");
    assert.equal(styleUpdates.at(-1)[4], 1); // New fill reaches the style table.
    // Calls without an explicit revision also remain safe for mutable arrays.
    assert.ok(renderer.render(options));
    selection.x[0] = 75;
    assert.ok(renderer.render(options));
    assert.equal(uploads.at(-1)[0], 75);
  } finally { selection.x[0] = original; }
});

test("context loss and runtime GPU errors immediately select Canvas fallback", () => {
  const first = fakeRenderer();
  first.setLost(true);
  assert.equal(first.renderer.render(first.options), null);
  assert.equal(first.draws.length, 0);
  const second = fakeRenderer();
  second.setError(1285);
  assert.equal(second.renderer.render(second.options), null);
  second.setError(0);
  assert.equal(second.renderer.render(second.options), null);
  assert.equal(second.draws.length, 1);
});

test("all twelve base symbols have distinct GPU styles; font glyphs keep Canvas", () => {
  const symbols = Object.keys(GPU_SYMBOL_IDS);
  const atoms = symbols.map((symbol, index) => ({point: [0, 0, index], x: index, y: 10}));
  const packed = packAtoms({...style, symbols, sizes: symbols.map(() => 1), atoms: [atoms, []]});
  assert.equal(packed.count, symbols.length);
  assert.equal(packed.view.length, symbols.length * 3);
  for (let index = 0; index < symbols.length; index++)
    assert.equal(packed.styles.data[index * 12 + 1], GPU_SYMBOL_IDS[symbols[index]]);
  const unsafeRows = {get flat() { throw Error("unsupported layers should fail before coordinate packing"); }};
  assert.equal(packAtoms({...style, symbols: ["o", "number:2"], atoms: [
    {...selection, rows: unsafeRows, layers: new Set([0, 1])}, []]}), null);
});

test("CPU and GPU capacities grow geometrically and upload only the used prefix", () => {
  const {renderer, options, uploads, allocations} = fakeRenderer();
  assert.ok(renderer.render({...options, revision: 1}));
  const capacity = allocations[0];
  assert.ok(renderer.render({...options, revision: 2}));
  assert.equal(allocations.length, 1, "pan uses bufferSubData without reallocating GPU storage");
  const atoms = Array.from({length: 35}, (_, index) => ({point: [0, 0, 0], x: index, y: 10}));
  assert.ok(renderer.render({...options, atoms: [atoms, []], revision: 3}));
  assert.equal(uploads.at(-1).length, 35 * 3);
  assert.equal(allocations.length, 2);
  assert.ok(allocations[1] > capacity);
  assert.ok(renderer.render({...options, revision: 4}));
  assert.equal(allocations.length, 2, "a smaller selection reuses the larger allocation");
  assert.equal(uploads.at(-1).length, selection.count * 3);
});
