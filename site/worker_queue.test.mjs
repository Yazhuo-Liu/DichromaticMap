import assert from "node:assert/strict";
import {test} from "node:test";
import {LatestRequestQueue} from "./worker_queue.mjs";
import renderData from "./render_data.js";
import {readFile} from "node:fs/promises";
import vm from "node:vm";

const request = (id, action, value = id) => ({id, request: {action, value}});
const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return {promise, resolve};
};
const settledOnce = (messages, ids) => {
  for (const id of ids) assert.equal(messages.filter(message => message.id === id).length, 1);
};

test("Worker engine initialization retries after an initial download failure", async () => {
  const source=(await readFile(new URL("./use_worker.js",import.meta.url),"utf8"))
    .replace(/^import .+\n/gm,"");
  const first=deferred(),second=deferred(),messages=[];
  let attempts=0;
  const runtime={globals:new Map(),FS:{writeFile(){}},loadPackage:async()=>{},
    runPython(code){if(code==="web_dispatch(web_request)") return JSON.stringify({ok:true});}};
  const self={postMessage(message){messages.push(message); if(message.id===1) first.resolve(); if(message.id===2) second.resolve();}};
  vm.runInNewContext(source,{self,LatestRequestQueue,
    loadPyodide:async()=>{if(++attempts===1) throw new Error("download failed");return runtime;},
    fetch:async()=>({ok:true,text:async()=>"bridge",arrayBuffer:async()=>new ArrayBuffer(1)}),
    Uint8Array,JSON,Promise});
  self.onmessage({data:request(1,"metadata")}); await first.promise;
  assert.equal(messages.find(m=>m.id===1).type,"error");
  self.onmessage({data:request(2,"metadata")}); await second.promise;
  assert.equal(attempts,2); assert.deepEqual(messages.find(m=>m.id===2).result,{ok:true});
  settledOnce(messages,[1,2]);
});

test("replace pending renders and searches, settle every obsolete request", async () => {
  const boot = deferred(), calls = [], messages = [];
  const queue = new LatestRequestQueue({
    execute: async (data, cancelled) => {
      if (data.action === "metadata") await boot.promise;
      if (!cancelled()) calls.push(data.value);
      return {result: data.value};
    },
    startSearch: async data => {
      calls.push(data.value);
      return {step: () => true, finish: () => ({result: data.value}), dispose() {}};
    },
    post: message => messages.push(message),
  });
  queue.enqueue(request(0, "metadata"));
  queue.enqueue(request(1, "render"));
  queue.enqueue(request(2, "render"));
  queue.enqueue(request(3, "near_search"));
  queue.enqueue(request(4, "near_search"));
  boot.resolve();
  await queue.whenIdle();
  assert.deepEqual(calls, [0, 2, 4]);
  assert.deepEqual(messages.filter(message => message.name === "AbortError").map(message => message.id), [1, 3]);
  settledOnce(messages, [0, 1, 2, 3, 4]);
});

test("supersede a render waiting for engine startup before expensive execution", async () => {
  const boot = deferred(), computed = [], messages = [];
  const buffer = new Float64Array([Math.PI]);
  const queue = new LatestRequestQueue({
    execute: async (data, cancelled) => {
      await boot.promise;
      if (cancelled()) return;
      computed.push(data.value);
      return {result: buffer, transfer: [buffer.buffer]};
    },
    post: (message, transfer) => messages.push({message, transfer}),
  });
  queue.enqueue(request(1, "render"));
  queue.enqueue(request(2, "render"));
  boot.resolve();
  await queue.whenIdle();
  assert.deepEqual(computed, [2]);
  assert.equal(messages[0].message.name, "AbortError");
  assert.deepEqual(messages[1].transfer, [buffer.buffer]);
  settledOnce(messages.map(item => item.message), [1, 2]);
});

test("yield search blocks, let user actions through, cancel and replace an active search", async () => {
  const firstBlock = deferred(), calls = [], messages = [], disposed = [];
  const queue = new LatestRequestQueue({
    execute: async data => { calls.push(`${data.action}:${data.value}`); return {result: data.value}; },
    startSearch: async data => {
      calls.push(`start:${data.value}`);
      let steps = 0;
      return {
        step: () => {
          calls.push(`step:${data.value}`);
          if (data.value === 1) firstBlock.resolve();
          return ++steps === 3;
        },
        finish: () => { calls.push(`finish:${data.value}`); return {result: data.value}; },
        dispose: () => disposed.push(data.value),
      };
    },
    post: message => messages.push(message),
  });
  queue.enqueue(request(1, "near_search"));
  await firstBlock.promise;
  queue.enqueue(request(2, "render"));
  queue.enqueue(request(3, "vector"));
  queue.enqueue(request(4, "near_search"));
  await queue.whenIdle();
  assert.equal(calls.filter(call => call === "step:1").length, 1);
  assert.ok(!calls.includes("finish:1"));
  assert.ok(calls.indexOf("render:2") < calls.indexOf("start:4"));
  assert.ok(calls.indexOf("vector:3") < calls.indexOf("start:4"));
  assert.deepEqual(disposed, [1, 4]);
  assert.equal(messages.find(message => message.id === 1).name, "AbortError");
  settledOnce(messages, [1, 2, 3, 4]);
});

test("a render resumes between blocks without cancelling the ongoing search", async () => {
  const firstBlock = deferred(), calls = [], messages = [];
  const queue = new LatestRequestQueue({
    execute: async () => { calls.push("render"); return {result: 2}; },
    startSearch: async () => {
      let steps = 0;
      return {
        step: () => { calls.push("step"); firstBlock.resolve(); return ++steps === 2; },
        finish: () => { calls.push("finish"); return {result: 1}; }, dispose() {},
      };
    },
    post: message => messages.push(message),
  });
  queue.enqueue(request(1, "near_search"));
  await firstBlock.promise;
  queue.enqueue(request(2, "render"));
  await queue.whenIdle();
  assert.deepEqual(calls, ["step", "render", "step", "finish"]);
  assert.ok(messages.every(message => message.type === "result"));
});

test("explicit cancellation stops a search and errors do not strand later requests", async () => {
  const firstBlock = deferred(), messages = [];
  let steps = 0, disposed = 0;
  const queue = new LatestRequestQueue({
    execute: async () => { throw new Error("bad request"); },
    startSearch: async () => ({
      step: () => { steps++; firstBlock.resolve(); return false; },
      finish: () => { throw new Error("cancelled search must not finish"); },
      dispose: () => { disposed++; },
    }),
    post: message => messages.push(message),
  });
  queue.enqueue(request(1, "near_search"));
  await firstBlock.promise;
  queue.enqueue({type: "cancel", action: "near_search"});
  queue.enqueue(request(2, "vector"));
  await queue.whenIdle();
  assert.equal(steps, 1);
  assert.equal(disposed, 1);
  assert.equal(messages.find(message => message.id === 1).name, "AbortError");
  assert.equal(messages.find(message => message.id === 2).message, "bad request");
  settledOnce(messages, [1, 2]);
});

test("failed search releases its state and a later request succeeds", async () => {
  const messages = [], firstBlock = deferred();
  let disposed = 0;
  const queue = new LatestRequestQueue({
    execute: async () => ({result: "ready"}),
    startSearch: async () => ({
      step: () => { firstBlock.resolve(); throw new Error("search failed"); },
      dispose: () => { disposed++; },
    }),
    post: message => messages.push(message),
  });
  queue.enqueue(request(1, "near_search"));
  await firstBlock.promise;
  queue.enqueue(request(2, "render"));
  await queue.whenIdle();
  assert.equal(disposed, 1);
  assert.equal(messages.find(message => message.id === 2).result, "ready");
  settledOnce(messages, [1, 2]);
});

test("transfer render coordinates losslessly and preserve Array picking/session semantics", () => {
  const {decodeRenderResult} = renderData;
  const raw = {
    format: "dichromatic-map-render-v1", metadata: {angle: 22, layers: 2},
    grains: [new Float64Array([Math.PI, -Math.E, 1, -7, 9, 3]), new Float64Array()],
    coincidences: new Float64Array([1e-14, -0, 1]),
    local: new Float64Array([Math.PI, 1, Math.PI + .03, 1.000001, 1]),
  };
  const received = structuredClone(raw, {transfer: [...raw.grains, raw.coincidences, raw.local].map(array => array.buffer)});
  assert.equal(raw.grains[0].byteLength, 0);
  const result = decodeRenderResult(received);
  assert.equal(result.grains[0].flat, received.grains[0], "decoding keeps the transferred buffer without row allocation");
  assert.deepEqual(result.grains[0].row(0), [Math.PI, -Math.E, 1, -7, 9, 3]);
  assert.equal(result.grains[1].length, 0);
  assert.deepEqual(result.coincidences.row(0), [1e-14, -0, 1]);
  assert.deepEqual(result.local.row(0), [Math.PI, 1, Math.PI + .03, 1.000001, 1]);
  assert.equal(JSON.stringify(result.grains[0].row(0).slice(3, 6)), "[-7,9,3]");
  assert.ok(Array.isArray(result.grains[0].filter(point => point[2] === 1)[0]));
  assert.equal(result.angle, 22);
  assert.deepEqual(JSON.parse(JSON.stringify(result)).grains, [[ [Math.PI,-Math.E,1,-7,9,3] ],[]],
    "explicit serialization materializes the original row API");
});

test("retain legacy render responses and distinguish no local matching from no matches", () => {
  const {decodeRenderResult} = renderData;
  const legacy = {grains: [[], []], coincidences: [], local: null};
  assert.equal(decodeRenderResult(legacy), legacy);
  const raw = {format: "dichromatic-map-render-v1", metadata: {},
    grains: [new Float64Array(), new Float64Array()], coincidences: new Float64Array(), local: null};
  assert.equal(decodeRenderResult(raw).local, null);
  raw.local = new Float64Array();
  assert.equal(decodeRenderResult(raw).local.length, 0);
  const originalRow = renderData.DenseRows.prototype.row;
  renderData.DenseRows.prototype.row = () => {throw new Error("decoding must not materialize rows");};
  try {
    const decoded = decodeRenderResult(raw);
    assert.equal(decoded.grains[0].flat,raw.grains[0]);
  } finally {renderData.DenseRows.prototype.row = originalRow;}
});

test("Worker downloads resources during NumPy startup and transfers copied Pyodide buffers", async () => {
  const source = (await readFile(new URL("./use_worker.js", import.meta.url), "utf8"))
    .replace(/^import .+\n/gm, "");
  const numpy = deferred(), numpyStarted = deferred(), resultReceived = deferred();
  const downloads = [], dispatches = [], messages = [], released = [], destroyed = [];
  const wasmArrays = [new Float64Array([Math.PI, -Math.E, 1, -7, 9, 3]), new Float64Array(), new Float64Array([.125, .25, 1])];
  const proxy = (data, name) => ({
    getBuffer: type => {
      assert.equal(type, "f64");
      return {data, release: () => released.push(name)};
    },
    destroy: () => destroyed.push(name),
  });
  const globals = new Map();
  const runtime = {
    globals, FS: {writeFile() {}},
    loadPackage: async name => { assert.equal(name, "numpy"); numpyStarted.resolve(); await numpy.promise; },
    runPython: code => {
      if (code === "web_render(web_request)") {
        dispatches.push(JSON.parse(globals.get("web_request")));
        const grains = {get: index => proxy(wasmArrays[index], `grain${index}`), destroy: () => destroyed.push("grains")};
        const items = new Map([["metadata", JSON.stringify({angle: 22, layers: 2})], ["grains", grains],
          ["coincidences", proxy(wasmArrays[2], "csl")], ["local", undefined]]);
        return {get: key => items.get(key), destroy: () => destroyed.push("payload")};
      }
    },
  };
  const self = {postMessage: (message, transfer = []) => {
    messages.push(structuredClone(message, {transfer}));
    if (message.type === "result") resultReceived.resolve();
  }};
  vm.runInNewContext(source, {self, LatestRequestQueue, loadPyodide: async () => runtime,
    fetch: async url => { downloads.push(url); return {ok: true, text: async () => "bridge source", arrayBuffer: async () => new ArrayBuffer(4)}; },
    Float64Array, Uint8Array, JSON, Promise, Map});
  self.onmessage({data: request(1, "render")});
  await numpyStarted.promise;
  assert.deepEqual(downloads, ["./web_bridge.py", "./vendor/dichromatic_map.zip"]);
  self.onmessage({data: request(2, "render")});
  numpy.resolve();
  await resultReceived.promise;
  assert.deepEqual(dispatches.map(data => data.value), [2]);
  assert.equal(messages.find(message => message.id === 1).name, "AbortError");
  const result = renderData.decodeRenderResult(messages.find(message => message.id === 2).result);
  assert.deepEqual(result.grains[0].row(0), [Math.PI, -Math.E, 1, -7, 9, 3]);
  assert.deepEqual(result.coincidences.row(0), [.125, .25, 1]);
  assert.equal(wasmArrays[0].length, 6, "transferring the owned copy must not detach WASM data");
  assert.deepEqual(released, ["grain0", "grain1", "csl"]);
  assert.deepEqual(destroyed, ["grain0", "grain1", "csl", "grains", "payload"]);
  settledOnce(messages, [1, 2]);
});

test("Worker resumes fixed Python blocks and releases the completed search proxy", async () => {
  const source = (await readFile(new URL("./use_worker.js", import.meta.url), "utf8"))
    .replace(/^import .+\n/gm, "");
  const finished = deferred(), batches = [], messages = [], globals = new Map();
  let destroyed = 0;
  const pythonJob = {destroy: () => { destroyed++; }};
  const runtime = {
    globals, FS: {writeFile() {}}, loadPackage: async () => {},
    runPython: code => {
      if (code === "web_near_start(web_request)") return pythonJob;
      if (code === "web_near_step(web_search)") {
        assert.equal(globals.get("web_search"), pythonJob);
        batches.push("fixed");
        return batches.length === 5;
      }
      if (code === "web_near_finish(web_search)") return JSON.stringify([{label: "cell"}]);
    },
  };
  const self = {postMessage: message => {
    messages.push(message);
    if (message.type === "result") finished.resolve();
  }};
  vm.runInNewContext(source, {self, LatestRequestQueue, loadPyodide: async () => runtime,
    fetch: async () => ({ok: true, text: async () => "bridge", arrayBuffer: async () => new ArrayBuffer(1)}),
    Uint8Array, JSON, Promise});
  self.onmessage({data: request(1, "near_search")});
  await finished.promise;
  assert.deepEqual(batches, ["fixed", "fixed", "fixed", "fixed", "fixed"]);
  assert.equal(destroyed, 1);
  assert.equal(globals.has("web_search"), false);
  assert.deepEqual(messages.find(message => message.id === 1).result, [{label: "cell"}]);
});
