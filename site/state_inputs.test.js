"use strict";
const assert=require("node:assert/strict");
const test=require("node:test");
const {harness,fixture}=require("./state_consistency.test.js");

function loaded(view_range) {
  return {state:{parameters:{lattice:"FCC",axis:"110",angle_deg:22,lattice_constant:3.52},
    interaction_mode:"idle",display_rotation_deg:0,show_reference_axes:true,
    grain_colors:["#1677d2","#e35d35"],layer_symbols:["o","d"],layer_size_scales:[1,1],
    visible_grain_layers:[[0,1],[0,1]],selected_points:[],selected_atoms:[],manual_vertices:[],
    manual_unstrained_vertices:null,manual_strain_fit:null,manual_local_cutoff:null,axial_repeat:0,
    near_enabled:false,near_method:"local",near_cell:null,
    deformations:[[[1,0],[0,1]],[[1,0],[0,1]]],translations:[[0,0],[0,0]]},
    settings:{region_states:[true,true,true,true],show_common_cell:false,manual_visible:false,
      local_distance:.1,strain_percent:2,search_index:12,manual_strain_percent:2,manual_rotation_deg:1},view_range};
}
async function importInto(h, data) {
  h.sandbox.FileReader=class {readAsDataURL(){this.result="data:application/zip;base64,AA==";this.onload();}};
  const pending=h.app.importSession({}); await Promise.resolve();
  h.reply(h.last("load_session"),data); await Promise.resolve();
  h.reply(h.last("metadata"),{lattice:"FCC",axis:"110",layers:2,max_angle:90,presets:[],
    reference_labels:[],layer_spacing:1,axial_period:1});
  await pending;
}

test("import fits both saved view spans and preserves its center", async () => {
  for (const [view,rotation] of [[[-2,2,-20,20],0],[[-20,20,-2,2],31],[[3,7,-23,17],-45]]) {
    const h=harness(),a=h.app; a.setPattern(fixture()); const data=loaded(view);
    data.state.display_rotation_deg=rotation; await importInto(h,data); h.flush(45);
    const restored=[a.state.center[0]-a.state.width/2,a.state.center[0]+a.state.width/2,
      a.state.center[1]-a.state.height/2,a.state.center[1]+a.state.height/2];
    assert(restored[0]<=view[0] && restored[1]>=view[1] && restored[2]<=view[2] && restored[3]>=view[3]);
    assert.deepEqual(Array.from(a.state.center),[(view[0]+view[1])/2,(view[2]+view[3])/2]);
    assert.equal(a.state.displayRotation,rotation);
  }
});

test("an unsupported imported view is reported without replacing the workspace", async () => {
  const h=harness(),a=h.app;a.setPattern(fixture());a.state.boundary=[[0,0],[1,0]];
  const original=JSON.stringify(a.sessionState());
  await importInto(h,loaded([-40,40,-40,40]));
  assert.equal(JSON.stringify(a.sessionState()),original);
  assert.match(h.node("status").innerHTML,/view.*browser|browser.*view/i);
});

for (const axis of [0,1]) test(`large rounded centers retain both original ${axis ? "Y" : "X"} boundaries exactly`, async () => {
  for (const sign of [1,-1]) {
    const endpoints=[67108862.000000015,67108867.20000002];
    if (sign < 0) endpoints.splice(0,2,-endpoints[1],-endpoints[0]);
    const view=axis ? [-.5,.5,...endpoints] : [...endpoints,-.5,.5];
    const h=harness(),a=h.app;a.setPattern(fixture());await importInto(h,loaded(view));h.flush(45);
    const restored=[a.state.center[0]-a.state.width/2,a.state.center[0]+a.state.width/2,
      a.state.center[1]-a.state.height/2,a.state.center[1]+a.state.height/2];
    for (const start of [0,2]) {
      assert(restored[start]<=view[start],`${restored[start]} cropped ${view[start]}`);
      assert(restored[start+1]>=view[start+1],`${restored[start+1]} cropped ${view[start+1]}`);
    }
  }
});

test("a view beyond the exact maximum width keeps the previous workspace", async () => {
  const h=harness(),a=h.app;a.setPattern(fixture());a.state.boundary=[[0,0],[1,0]];
  const original=JSON.stringify(a.sessionState());
  await importInto(h,loaded([-30,30.000000000000014,-.5,.5]));
  assert.equal(JSON.stringify(a.sessionState()),original);
  assert.match(h.node("status").innerHTML,/view.*browser|browser.*view/i);
});

test("axial repeat rejects fractional, empty, nonfinite and out-of-range input", () => {
  const h=harness(),a=h.app;a.setPattern(fixture());
  a.state.atoms=[{position:[0,0],grain:0,layer:0,half_indices:[0,0,0]},
    {position:[1,0],grain:0,layer:0,half_indices:[2,0,0]}];
  for (const value of ["1.5","","NaN","Infinity","5","-5"]) {
    const count=h.messages.filter(m=>m.request?.action==="vector").length;
    h.node("axial-repeat").value=value;h.node("axial-repeat").handlers.change();
    assert.equal(a.state.axialRepeat,0,value);
    assert.equal(h.node("axial-repeat").value,"0");
    assert.equal(h.messages.filter(m=>m.request?.action==="vector").length,count);
  }
  for (const value of [-4,0,4]) {
    h.node("axial-repeat").value=String(value);h.node("axial-repeat").handlers.change();
    assert.equal(h.last("vector").request.axial_repeat,value);
  }
});

test("count readouts distinguish unique representatives, closed copies and unavailable half-open counts", async () => {
  const h=harness(),a=h.app;a.setPattern(fixture());
  a.state.manual=[[0,0],[1,0],[1,1],[0,1]].map(position=>({position,layer:0,source:"CSL",endpoints:[position,position]}));
  const counted=a.countManual();h.reply(h.last("count"),{interior:[[8],[8]],boundary:[[4],[4]],
    half_open:[[9],[9]],half_open_edges:[[0],[0]],half_open_corners:[[1],[1]],half_open_available:[true,true],areas:[6.364,6.364]});
  await counted;const text=h.node("manual-info").dataset.count;
  assert.match(text,/9.*unique|unique.*9/i);assert.match(text,/12.*closed|closed.*12/i);
  assert.match(text,/0.*edge/i);assert.match(text,/1.*corner/i);
  const irregular=a.countManual();h.reply(h.last("count"),{interior:[[1],[2]],boundary:[[3],[4]],
    half_open:[[2],[0]],half_open_edges:[[0],[0]],half_open_corners:[[1],[0]],half_open_available:[true,false],areas:[1,2]});
  await irregular;const mixed=h.node("manual-info").dataset.count;
  assert.match(mixed,/G2[\s\S]*not a parallelogram/i);
  assert.match(mixed,/6.*closed|closed.*6/i);
  assert(!/G2[^\n]*0.*unique/i.test(mixed));
});
