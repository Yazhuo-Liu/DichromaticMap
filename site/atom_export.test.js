"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const vm = require("node:vm");
const {harness, fixture} = require("./state_consistency.test.js");

function setup() {
  const h = harness();
  h.sandbox.Blob = Blob;
  h.sandbox.downloads = [];
  vm.runInContext(`
    downloadBlob = (blob, name) => downloads.push({blob, name});
    globalThis.exportAPI = {svgMarker, atomsSVG, exportSVG};
  `, h.sandbox);
  h.node("clean-background").checked = true;
  h.node("vector-annotation").hidden = true;
  h.app.setPattern(fixture());
  return {h, api: h.sandbox.exportAPI, downloads: h.sandbox.downloads};
}

function recordPNG(h) {
  const fills = [], paths = [], images = [];
  const context = new Proxy({
    fillRect(...bounds) { fills.push({color: this.fillStyle, bounds}); },
    fill() { paths.push(this.fillStyle); },
    drawImage(...args) { images.push(args); },
  }, {get(target, key) { return target[key] ?? (() => {}); }});
  h.sandbox.document.createElement = tag => {
    assert.equal(tag, "canvas");
    return {getContext: () => context, toBlob(callback) { callback(new Blob(["PNG"])); }};
  };
  return {fills, paths, images};
}

test("clean PNG selects pure white or leaves the canvas and hollow markers transparent", async () => {
  for (const background of [true, false]) {
    const {h, downloads} = setup(), recording = recordPNG(h);
    h.app.state.visibleLayers = [new Set(), new Set([0])];
    h.node("clean-png").checked = true;
    h.node("clean-background").checked = background;
    await h.app.exportPNG();
    assert.deepEqual(recording.fills, background ? [{color: "#ffffff", bounds: [0, 0, 1024, 852]}] : []);
    assert.deepEqual(recording.paths, [], "hollow atom interiors must not be painted white");
    assert.deepEqual(recording.images, [], "clean export must not copy plot decorations");
    assert.equal(downloads[0].name, "dichromatic-atoms.png");
  }
});

test("the atoms-only background option leaves the regular PNG export unchanged", async () => {
  const {h, downloads} = setup(), recording = recordPNG(h);
  h.node("clean-png").checked = false;
  h.node("clean-background").checked = false;
  await h.app.exportPNG();
  assert.equal(recording.images.length, 1);
  assert.equal(recording.images[0][0], h.node("map"));
  assert.equal(downloads[0].name, "dichromatic-pattern.png");
});

test("SVG retains rotated visible atoms, colors and sizes while excluding overlays and hidden atoms", () => {
  const {h, api} = setup(), a = h.app;
  a.state.colors = ["#12ab34", "#ab1234"];
  a.state.scale = .5;
  a.state.sizes = [1.2, 1];
  a.state.displayRotation = 23;
  a.state.center = [.2, -.1];
  a.state.visibleLayers = [new Set([0]), new Set([0])];
  a.state.boundary = [[-2, 0], [2, 0]];
  a.state.regions = [true, true, true, false];
  a.state.showCell = true;
  a.state.nearEnabled = true;
  a.setPattern({...fixture(),
    grains: [[[-1, 0, 0], [0, -1, 0], [1, 0, 1]], [[0, 1, 0], [0, -1, 0], [0, 2, 1]]],
    local: [[0, 0, .01, .02, 0]],
  });
  const svg = api.atomsSVG(false);
  assert.match(svg, /viewBox="0 0 1024 852"/);
  assert.match(svg, /<clipPath id="plot-clip"><rect x="35" y="30" width="961" height="786"\/>/);
  assert.equal((svg.match(/<circle /g) || []).length, 3);
  assert.equal((svg.match(/fill="#12ab34"/g) || []).length, 2);
  assert.equal((svg.match(/stroke="#ab1234"/g) || []).length, 1);
  assert.match(svg, /<circle[^>]+fill="none" stroke="#ab1234"/);
  assert.doesNotMatch(svg, /image|foreignObject|<text|#e5a50a|#aa45bb|#0d9fa1|#ffffff/);
  const number = value => String(Number(value.toFixed(6)));
  const [x, y] = a.screen([0, 1]);
  const radius = 4.5 * 1.2 / Math.sqrt(.5);
  assert(svg.includes(`<circle cx="${number(x)}" cy="${number(y)}" r="${number(radius)}"`));
  const white = api.atomsSVG(true);
  assert.match(white, /<rect width="100%" height="100%" fill="#ffffff"\/>/);
  assert.equal(white.replace('<rect width="100%" height="100%" fill="#ffffff"/>\n', ""), svg);
});

test("every marker exports as vector geometry, including numbered circles", () => {
  const {api} = setup();
  const symbols = ["o", "d", "t", "s", "p", "h", "star", "+", "x", "t1", "t2", "t3", "number:27"];
  for (const symbol of symbols) {
    for (const fill of [null, "#123456"]) {
      const svg = api.svgMarker(12, 13, 8, symbol, "#abcdef", fill);
      assert.equal((svg.match(/<(circle|path) /g) || []).length, 1, symbol);
      assert.match(svg, new RegExp(`fill="${fill || "none"}" stroke="#abcdef" stroke-width="1.25"`));
      assert.doesNotMatch(svg, /image|data:/);
      if (symbol.startsWith("number:")) {
        assert.match(svg, /<text[^>]+>27<\/text>/);
        assert.match(svg, new RegExp(`<text[^>]+fill="${fill ? "#ffffff" : "#abcdef"}"`));
      } else assert.doesNotMatch(svg, /<text/);
    }
  }
  assert.match(api.svgMarker(1, 2, 3, "number:<&", "#000000", null), /&lt;&amp;<\/text>/);
});

test("SVG waits for the current render and discards superseded calculations", async () => {
  const {h, api, downloads} = setup(), a = h.app;
  a.changeAngle(22);
  const exporting = api.exportSVG();
  assert.equal(downloads.length, 0);
  h.flush(45); const old = h.last("render");
  a.changeAngle(24); await Promise.resolve();
  h.reply(old, fixture(22)); await Promise.resolve();
  assert.equal(downloads.length, 0);
  h.flush(45); h.reply(h.last("render"), fixture(24));
  await exporting;
  assert.equal(a.state.angle, 24);
  assert.equal(downloads.length, 1);
  assert.equal(downloads[0].name, "dichromatic-atoms.svg");
  assert.equal(downloads[0].blob.type, "image/svg+xml;charset=utf-8");
  assert.match(await downloads[0].blob.text(), /<svg /);

  a.changeAngle(26);
  const failed = api.exportSVG();
  h.flush(45); h.reject(h.last("render"));
  await failed;
  assert.equal(downloads.length, 1, "a failed calculation must not publish the old SVG");
});

test("SVG waits for a held angle slider and exports only the final committed geometry", async () => {
  const {h, api, downloads} = setup();
  h.node("angle-slider").handlers.pointerdown({button: 0});
  h.node("angle-slider").value = "54";
  h.node("angle-slider").handlers.input();
  const exporting = api.exportSVG();
  await Promise.resolve(); h.flush(45);
  assert.equal(downloads.length, 0);
  h.windowEvents.get("pointerup")(); await Promise.resolve();
  h.flush(45); h.reply(h.last("render"), {...fixture(54), grains: [[], []], coincidences: []});
  await exporting;
  assert.equal(downloads.length, 1);
  assert.doesNotMatch(await downloads[0].blob.text(), /<circle|<path /);
});
