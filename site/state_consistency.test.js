"use strict";
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const renderData = require("./render_data.js");

function harness() {
  const nodes = new Map(), timers = new Map(), messages = [], titles = [];
  let sequence = 0, exports = 0;
  const context = new Proxy({fillText(text, x, y) { if (y === 17) titles.push(text); }},
    {get(target, key) { return target[key] ?? (() => {}); }});
  const node = id => {
    if (!nodes.has(id)) nodes.set(id, {id, value: "1", checked: false, style: {}, dataset: {},
      children: [], innerHTML: "", handlers: {}, clientWidth: 1024, clientHeight: 852,
      classList: {add() {}, remove() {}, toggle() {}},
      getBoundingClientRect: () => ({left:0, top:0, width:1024, height:852}),
      getContext: () => context, querySelector: () => node("dummy"), closest: () => node("dummy"),
      addEventListener(type, fn) {this.handlers[type] = fn;}, setAttribute() {},
      replaceChildren() {}, add() {}, append() {}, close() {}, showModal() {},
    });
    return nodes.get(id);
  };
  const sandbox = {Worker: class {postMessage(message) {messages.push(message);}},
    Option: class {}, devicePixelRatio:1,
    window: {matchMedia: () => ({matches:false}), addEventListener() {}, DichromaticRenderData:renderData},
    document: {getElementById:node, querySelector: () => node("dummy"), querySelectorAll: () => [],
      addEventListener() {}, createElement: () => ({getContext: () => context, toBlob() {exports++;}})},
    setTimeout(fn, delay) {const id = ++sequence; timers.set(id,{fn,delay}); return id;},
    clearTimeout: id => timers.delete(id), requestAnimationFrame: () => ++sequence, cancelAnimationFrame() {},
  };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(__dirname,"use.js"),"utf8") + `
    metadata = {layers:1,max_angle:90,presets:[],reference_labels:[],layer_spacing:1,axial_period:1};
    globalThis.app = {state, renderRequest, nearestAtom, nearestCommon, screen, exportPNG,
      updateSummary, currentPattern, changeAngle, resetSelections,
      updateVector, countManual, completeManual, toggleSelectedStrain, importSession,
      searchNear, cancelNearSearch,
      get worker() {return worker;},
      setPattern(value) {pattern=value; renderCoverage={signature:JSON.stringify(currentRenderParameters()),
        center:[0,0],width:100,height:100}; viewSize();},
    };
  `,sandbox);
  return {app:sandbox.app, node, messages, titles, sandbox,
    get exports() {return exports;},
    flush(delay) {for (const [id,timer] of [...timers]) if (timer.delay===delay) {timers.delete(id); timer.fn();}},
    last(action) {return messages.filter(m => m.request?.action===action).at(-1);},
    reply(message,result) {sandbox.app.worker.onmessage({data:{id:message.id,type:"result",result}});},
    reject(message) {sandbox.app.worker.onmessage({data:{id:message.id,type:"error",message:"Failed calculation"}});},
  };
}
function fixture(angle=38.94244126898139) {
  return {angle, preset:angle===38.94244126898139 ? "Σ9" : null,
    exact_cell:angle===38.94244126898139 ? {cell:[[1,0],[0,1]]} : null,
    grains:[[[0,0,0,0,0,0]],[[0,0,0,0,0,0]]], coincidences:[[0,0,0]], local:[]};
}
async function main() {
  const h=harness(), app=h.app;
  app.setPattern(fixture());
  app.updateSummary();
  assert.equal(h.node("fit-cell").disabled,false);
  const [x,y]=app.screen([0,0]);
  assert(app.nearestAtom(x,y)); assert(app.nearestCommon(x,y));
  app.changeAngle(22);
  assert.equal(app.nearestAtom(x,y),null,"old atom identities must not be selectable");
  assert.equal(app.nearestCommon(x,y),null,"old CSL markers must not be selectable");
  assert(!h.node("plot-title").textContent.includes("Σ9"),"old preset cannot label a new angle");
  assert.equal(h.node("fit-cell").disabled,true,"old exact-cell controls cannot fit a new geometry");
  const exporting=app.exportPNG();
  assert.equal(h.exports,0,"export waits for numerical render");
  h.flush(45); const old=h.last("render");
  app.changeAngle(24); await Promise.resolve();
  h.reply(old,fixture(22)); await Promise.resolve();
  assert.equal(app.currentPattern(),false,"a superseded response never commits");
  h.flush(45); h.reply(h.last("render"),fixture(24));
  await exporting;
  assert.equal(h.exports,1);
  assert(h.titles.at(-1).includes("24.00°"));
  assert(app.currentPattern());
  app.changeAngle(26);
  const failing=app.exportPNG(); h.flush(45); h.reject(h.last("render"));
  await failing;
  assert.equal(h.exports,1,"failed render must not export an old pattern");
  console.log("Committed renders, stale picking, superseded exports, and render failures passed.");
}
main().catch(error => {console.error(error); process.exitCode=1;});
