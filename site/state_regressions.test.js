"use strict";
const assert = require("node:assert/strict");
const test = require("node:test");
const {harness, fixture} = require("./state_consistency.test.js");

const identity = () => [[[1,0],[0,1]],[[1,0],[0,1]]];
const metadata = {lattice:"FCC",axis:"110",layers:1,max_angle:90,default_angle:fixture().angle,
  presets:[{name:"Σ9",angle:fixture().angle,label:"Σ9"}],reference_labels:[],layer_spacing:1,axial_period:1};
const vector = repeat => ({vectors:{G1:{current_formatted:`repeat-${repeat}`,strained:false}},length:repeat+1});
const atoms = () => [{position:[0,0],grain:0,layer:0,half_indices:[0,0,0]},
  {position:[1,0],grain:0,layer:0,half_indices:[2,0,0]}];

test("preset normalization precedes delayed rendering and near search", async () => {
  const h=harness(), a=h.app;
  a.setMetadata(metadata); a.state.angle=22; a.setPattern(fixture(22));
  a.state.nearEnabled=true; a.state.nearMethod="strain";
  a.changeAngle(38.94); h.flush(45); const render=h.last("render");
  h.flush(150); const search=h.last("near_search");
  assert.equal(render.request.angle,fixture().angle);
  assert.equal(search.request.angle,render.request.angle);
  h.reply(render,fixture()); await Promise.resolve();
  h.reply(search,[{label:"same angle",f1:identity()[0],f2:identity()[1]}]); await Promise.resolve();
  assert.equal(a.state.nearCell.label,"same angle");
});

test("cell mode cannot erase an applied fit through its shortcut entry point", () => {
  const h=harness(), a=h.app; a.setPattern(fixture());
  a.state.manual=Array.from({length:4},(_,i)=>({position:[i,i],layer:0,source:"CSL",endpoints:[[i,i],[i,i]]}));
  a.state.manualFit={cell:{f1:identity()[0],f2:identity()[1]}};
  a.state.manualOriginal=a.state.manual.map(v=>({...v,source:"local"}));
  const before=JSON.stringify(a.sessionState()); a.setMode("cell");
  assert.equal(JSON.stringify(a.sessionState()),before);
  a.updateSummary(); assert.equal(h.node("apply-strain").disabled,false);
});

test("appearance changes preserve a pending physical search, actual inputs reject it", async () => {
  for (const beforeRequest of [true,false]) {
    const h=harness(), a=h.app; a.setPattern(fixture());
    a.state.nearEnabled=true; a.state.nearMethod="strain"; a.searchNear();
    if (!beforeRequest) h.flush(150);
    h.node("g1-color").value="#009900"; h.node("g1-color").handlers.input();
    a.state.displayRotation=31; a.state.referenceAxes=false; a.state.visibleLayers=[new Set(),new Set()];
    if (beforeRequest) h.flush(150);
    assert(h.last("near_search"),"unrelated appearance must not suppress a request");
    h.reply(h.last("near_search"),[{label:"still valid",f1:identity()[0],f2:identity()[1]}]); await Promise.resolve();
    assert.equal(a.state.nearCell.label,"still valid");
    a.searchNear(); h.flush(150); const old=h.last("near_search");
    a.state.angle=22; h.reply(old,[{label:"wrong angle"}]); await Promise.resolve();
    assert.equal(a.state.nearCell,null,"the actual computed angle must be protected");
  }
});

test("PNG waits for the latest vector and follows changes during the wait", async () => {
  const h=harness(), a=h.app; a.setPattern(fixture()); a.state.atoms=atoms();
  const first=a.updateVector(); h.reply(h.last("vector"),vector(0)); await first;
  h.node("axial-repeat").value="1"; h.node("axial-repeat").handlers.change(); const old=h.last("vector");
  const exporting=a.exportPNG(); await Promise.resolve(); await Promise.resolve();
  assert.equal(h.exports,0,"an old readout cannot be exported during a newer calculation");
  h.node("axial-repeat").value="2"; h.node("axial-repeat").handlers.change(); const fresh=h.last("vector");
  h.reply(old,vector(1)); await Promise.resolve(); assert.equal(h.exports,0);
  h.reply(fresh,vector(2)); await exporting;
  assert.equal(h.exports,1); assert(h.textDraws.includes("P1→P2 · axial +2"));
  assert(!h.textDraws.includes("P1→P2 · axial +1"));
});

test("a failed current vector cannot export an older readout", async () => {
  const h=harness(), a=h.app; a.setPattern(fixture()); a.state.atoms=atoms();
  const initial=a.updateVector(); h.reply(h.last("vector"),vector(0)); await initial;
  h.node("axial-repeat").value="1"; h.node("axial-repeat").handlers.change();
  const exporting=a.exportPNG(); await Promise.resolve(); h.reject(h.last("vector")); await exporting;
  assert.equal(h.exports,0);
});
