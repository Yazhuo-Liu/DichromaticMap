"use strict";
const assert = require("node:assert/strict");
const test = require("node:test");
const {RowSpatialIndex} = require("./spatial_index.js");

function grid({shift = 0, midpoints = false} = {}) {
  const values = [];
  for (let x = -50; x <= 50; x++) for (let y = -50; y <= 50; y++)
    values.push(...(midpoints ? [x + shift - 100, y - 10, x + shift + 100, y + 10, 0]
      : [x + shift, y, (x + y + 100) % 3, x, y, 0]));
  const stride = midpoints ? 5 : 6;
  return {flat: new Float64Array(values), stride, length: values.length / stride};
}
function verify(rows, options, midpoints = false) {
  const original = rows.flat.slice();
  const index = new RowSpatialIndex(rows, {midpoints});
  const candidates = index.queryViewport(options);
  assert.ok(candidates, "a narrow viewport should query spatial candidates");
  const selected = new Set(candidates);
  let visible = 0;
  for (let row = 0; row < rows.length; row++) {
    const [x, y] = index.coordinates(row);
    const px = options.cosine * x - options.sine * y - options.cx;
    const py = options.sine * x + options.cosine * y - options.cy;
    if (Math.abs(px) <= options.width / 2 && Math.abs(py) <= options.height / 2) {
      assert.ok(selected.has(row), `the exact visible row ${row} must be a candidate`); visible++;
    }
  }
  assert.ok(visible > 0);
  assert.ok(candidates.length < rows.length / 10, "coarse query substantially reduces exact projection work");
  assert.ok(candidates.every((value, at) => !at || value > candidates[at - 1]), "candidates retain original paint/picking order");
  assert.deepEqual(rows.flat, original, "the numerical source remains unchanged");
  return index;
}

test("rotated and translated views conservatively query Float64 rows", () => {
  for (const degrees of [0, 37, 89, 137]) {
    const radians = degrees * Math.PI / 180;
    verify(grid(), {cosine: Math.cos(radians), sine: Math.sin(radians), cx: 7.25, cy: -3.9, width: 12, height: 9});
  }
});
test("large coordinates and exact viewport edges remain candidates", () => {
  const shift = 1e10;
  const rows = grid({shift}), radians = 37 * Math.PI / 180;
  const cosine = Math.cos(radians), sine = Math.sin(radians);
  verify(rows, {cosine, sine, cx: cosine * shift, cy: sine * shift, width: 12, height: 9});
  const index = new RowSpatialIndex(grid());
  const candidates = new Set(index.queryViewport({cosine: 1, sine: 0, cx: 0, cy: 0, width: 12, height: 8}));
  for (let row = 0; row < index.rows.length; row++) {
    const [x, y] = index.coordinates(row);
    if (Math.abs(x) === 6 && Math.abs(y) <= 4 || Math.abs(y) === 4 && Math.abs(x) <= 6)
      assert.ok(candidates.has(row));
  }
});
test("local-pair candidates use midpoints instead of either grain endpoint", () => {
  verify(grid({midpoints: true}), {cosine: 1, sine: 0, cx: 0, cy: 0, width: 12, height: 9}, true);
});
test("broad, small, degenerate or invalid data use the simple exact scan", () => {
  const index = new RowSpatialIndex(grid());
  assert.equal(index.queryBounds({left: -100, right: 100, bottom: -100, top: 100}), null);
  assert.equal(index.queryBounds({left: NaN, right: 0, bottom: 0, top: 0}), null);
  assert.equal(index.queryBounds({left: 100, right: 110, bottom: 100, top: 110}).length, 0);
  assert.equal(new RowSpatialIndex({flat: new Float64Array(12), stride: 6, length: 2}).bins, null);
  const invalid = grid(); invalid.flat[0] = NaN;
  assert.equal(new RowSpatialIndex(invalid).bins, null);
});
