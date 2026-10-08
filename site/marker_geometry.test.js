"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "use.js"), "utf8");
const marker = vm.runInNewContext(`(${source.slice(source.indexOf("function marker("),
  source.indexOf("function drawGrid("))})`);
const close = (a, b) => Math.abs(a - b) < 1e-9;

function vertices(symbol) {
  const points = [];
  const context = {beginPath() {}, closePath() {}, fill() {}, stroke() {},
    moveTo: (x, y) => points.push([x, y]), lineTo: (x, y) => points.push([x, y])};
  marker(context, 0, 0, 10, symbol, "#000000", "#ffffff");
  return points;
}

test("square edges align with the axes and remain distinct from diamonds", () => {
  const square = vertices("s"), diamond = vertices("d");
  assert.equal(square.length, 4);
  for (let index = 0; index < square.length; index++) {
    const [x, y] = square[index], next = square[(index + 1) % square.length];
    assert.ok(close(x, next[0]) !== close(y, next[1]), "each square edge is horizontal or vertical");
    assert.ok(close(Math.abs(x), Math.abs(y)), "square corners have equal x/y distances");
  }
  assert.ok(diamond.every(([x, y]) => close(x, 0) !== close(y, 0)), "diamond tips lie on the axes");
});

for (const [symbol, direction] of [["t", [0, -1]], ["t1", [0, 1]], ["t2", [1, 0]], ["t3", [-1, 0]]]) {
  test(`${symbol} points in the direction advertised by the menu`, () => {
    const points = vertices(symbol);
    assert.equal(points.length, 3);
    assert.ok(points.some(([x, y]) => close(x, direction[0] * 10) && close(y, direction[1] * 10)),
      "a triangle tip points in the requested direction");
    const projections = points.map(([x, y]) => x * direction[0] + y * direction[1]);
    assert.equal(projections.filter(value => value < 0).length, 2, "the base lies opposite the tip");
  });
}

test("different triangle orientations yield distinct point sets", () => {
  const sets = ["t", "t1", "t2", "t3"].map(symbol => vertices(symbol)
    .map(point => point.map(value => value.toFixed(6)).join(",")).sort().join(";"));
  assert.equal(new Set(sets).size, 4);
});
