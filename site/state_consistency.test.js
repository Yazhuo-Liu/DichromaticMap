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
  const sandbox = {Worker: class {
      postMessage(message) {if (sandbox.failPost) throw new Error("Cannot post message"); messages.push(message);}
      terminate() {this.terminated=true;}
    },
    Option: class {}, devicePixelRatio:1,
    window: {matchMedia: () => ({matches:false}), addEventListener() {}, DichromaticRenderData:renderData},
    document: {getElementById:node, querySelector: () => node("dummy"), querySelectorAll: () => [],
      addEventListener() {}, createElement: () => Object.assign(node(`created-${++sequence}`), {toBlob() {exports++;}})},
    setTimeout(fn, delay) {const id = ++sequence; timers.set(id,{fn,delay}); return id;},
    clearTimeout: id => timers.delete(id), requestAnimationFrame: () => ++sequence, cancelAnimationFrame() {},
  };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(path.join(__dirname,"use.js"),"utf8") + `
    metadata = {layers:1,max_angle:90,presets:[],reference_labels:[],layer_spacing:1,axial_period:1};
    globalThis.app = {state, request, retryEngine, sessionState, renderRequest, nearestAtom, nearestCommon, screen, exportPNG,
      updateSummary, currentPattern, changeAngle, resetSelections,
      updateVector, countManual, completeManual, toggleSelectedStrain, importSession,
      searchNear, cancelNearSearch, selectAt,
      get worker() {return worker;},
      get awaitingCount() {return awaiting.size;},
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
  const atoms = [{position:[0,0],grain:0,layer:0,half_indices:[0,0,0]},
    {position:[1,0],grain:0,layer:0,half_indices:[2,0,0]}];
  app.state.atoms=atoms; h.node("vector-annotation").hidden=true;
  const vector=app.updateVector(), vectorRequest=h.last("vector");
  app.resetSelections(); h.reply(vectorRequest,{}); await vector;
  assert(h.node("vector-annotation").hidden,"reset invalidates old vector replies");
  const vertices = [[0,0],[1,0],[1,1],[0,1]].map(position =>
    ({position,layer:0,source:"local",endpoints:[position,position]}));
  app.state.manual=vertices;
  const fitting=app.toggleSelectedStrain(), fitRequest=h.last("fit_selected");
  app.changeAngle(28); h.reply(fitRequest,{}); await fitting;
  assert.equal(app.state.manualFit,null,"old fit cannot apply after changing angle");
  assert.equal(app.state.manualOriginal,null);
  app.state.manual=vertices;
  const counting=app.countManual(), countRequest=h.last("count");
  app.state.regions=[false,true,true,true]; h.reject(countRequest); await counting;
  assert.equal(app.state.manual.length,4,"old count failures cannot delete current vertices");
  app.state.manual=vertices.slice(0,2);
  const completing=app.completeManual(), completionRequest=h.last("complete");
  app.resetSelections(); h.reply(completionRequest,[]); await completing;
  assert.equal(app.state.manual.length,0,"stale completion cannot resurrect vertices");
  app.state.manual=vertices.slice(0,2); app.state.mode="cell";
  const completingNow=app.completeManual();
  h.node("dummy").value="0";
  h.reply(h.last("complete"),[{vertices:[vertices.map(v=>v.position),vertices.map(v=>v.position)],
    source:"local",strain:0,atoms:[1,1],description:"cell"}]);
  await completingNow;
  const accepting=h.node("completion-accept").onclick();
  assert.equal(app.state.mode,"idle","completion stops picking before count awaits");
  h.reply(h.last("count"),{interior:[[0],[0]],boundary:[[0],[0]],areas:[1,1]}); await accepting;
  await app.selectAt(x,y);
  assert.equal(app.state.manual.length,4,"completion cannot append a fifth vertex");
  app.state.nearEnabled=true; app.state.nearMethod="strain";
  app.changeAngle(fixture().angle+1e-8);
  h.flush(45); h.reply(h.last("render"),fixture()); await Promise.resolve();
  h.flush(150);
  h.reply(h.last("near_search"),[{label:"current exact angle",f1:[[1,0],[0,1]],f2:[[1,0],[0,1]]}]);
  await Promise.resolve();
  assert.equal(app.state.nearCell.label,"current exact angle","render angle normalization cannot discard a valid search");
  const beforeImportAngle=app.state.angle;
  app.searchNear(); h.flush(150); const search=h.last("near_search");
  h.node("vector-annotation").hidden=false; h.node("vector-readout").textContent="OLD VECTOR";
  h.node("manual-info").dataset.count="OLD COUNT";
  h.sandbox.FileReader=class {readAsDataURL() {this.result="data:application/zip;base64,AA=="; this.onload();}};
  const importing=app.importSession({}); await Promise.resolve();
  const loaded={state:{parameters:{lattice:"FCC",axis:"110",angle_deg:22,lattice_constant:3.52},
    interaction_mode:"idle",display_rotation_deg:0,show_reference_axes:true,
    grain_colors:["#1677d2","#e35d35"],layer_symbols:["o"],layer_size_scales:[1],
    visible_grain_layers:[[0],[0]],selected_points:[[0,0],[1,0]],selected_atoms:[],manual_vertices:[],
    manual_unstrained_vertices:null,manual_strain_fit:null,manual_local_cutoff:null,axial_repeat:0,
    near_enabled:true,near_method:"strain",near_cell:{label:"imported"},
    deformations:[[[1,0],[0,1]],[[1,0],[0,1]]],translations:[[0,0],[0,0]]},
    settings:{region_states:[true,true,true,true],show_common_cell:false,manual_visible:false,
      local_distance:.1,strain_percent:2,search_index:12,manual_strain_percent:2,manual_rotation_deg:1},
    view_range:[-6,6,-4.5,4.5]};
  h.reply(h.last("load_session"),loaded); await Promise.resolve();
  assert.equal(app.state.angle,beforeImportAngle,"import waits for metadata validation before replacing state");
  h.reply(h.last("metadata"),{lattice:"FCC",axis:"110",layers:1,max_angle:90,presets:[],
    reference_labels:[],layer_spacing:1,axial_period:1});
  await importing;
  assert.equal(h.node("vector-annotation").hidden,true,"import clears an absent vector readout");
  assert.equal(h.node("vector-readout").textContent,"");
  assert.equal(h.node("manual-info").dataset.count,"","import clears old cell counts");
  h.reply(search,[{label:"obsolete"}]); await Promise.resolve();
  assert.equal(app.state.nearCell.label,"imported","old search cannot overwrite imported cell");
  assert.equal(app.state.boundary.length,2,"imported boundary survives the old search");
  h.node("session-file").value="selected.dmap";
  const obsoleteImport=app.importSession({}); await Promise.resolve();
  app.changeAngle(30); h.reply(h.last("load_session"),loaded); await obsoleteImport;
  assert.equal(app.state.angle,30,"cancelled import preserves newer parameters");
  assert.equal(h.node("session-file").value,"","cancelled import allows choosing the same file again");
  console.log("Committed renders, stale picking, superseded exports, and render failures passed.");
  console.log("Stale vector, fit, count errors, completion, and import/search races passed.");
}
module.exports={harness,fixture};
if (require.main===module) main().catch(error => {console.error(error); process.exitCode=1;});
