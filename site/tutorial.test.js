"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");
const {tutorialPosition, tutorialSteps} = require("./tutorial.js");

const markup = fs.readFileSync(path.join(__dirname, "use.html"), "utf8");
for (const step of tutorialSteps) {
  assert(markup.includes(`id="${step.target}"`), `Missing tutorial target: ${step.target}`);
  for (const language of ["en", "zh"]) {
    assert.equal(step[language].length, 3);
    assert(step[language].every(text => text.trim().length > 0));
  }
}
assert(/id="tutorial-bubble"[^>]*hidden/.test(markup), "Guide must start hidden");
assert(/id="tutorial-highlight"[^>]*hidden/.test(markup), "Highlight must start hidden");
assert(/id="tutorial-start"[^>]*disabled/.test(markup), "Wait for a working pattern before guiding interactions");

const overlap = (a, b) => Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left)) *
  Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
for (const [viewport, target, size, hasSpace] of [
  [{width:1440,height:900}, {left:1060,right:1410,top:240,bottom:410}, {width:340,height:300}, true],
  [{width:1440,height:900}, {left:0,right:1024,top:0,bottom:850}, {width:340,height:300}, true],
  [{width:390,height:844}, {left:18,right:372,top:270,bottom:450}, {width:340,height:300}, true],
  [{width:390,height:844}, {left:18,right:372,top:540,bottom:720}, {width:340,height:300}, true],
  [{width:390,height:420}, {left:18,right:372,top:140,bottom:240}, {width:340,height:300}, false],
  [{width:844,height:390}, {left:500,right:800,top:140,bottom:250}, {width:340,height:340}, true],
  [{left:100,top:200,width:390,height:844}, {left:118,right:472,top:470,bottom:650}, {width:340,height:300}, true],
  [{left:67,top:88,width:195,height:422}, {left:67,right:262,top:88,bottom:510}, {width:171,height:398}, false],
]) {
  const p = tutorialPosition(target, size, viewport);
  const placed = {...p, right:p.left + size.width, bottom:p.top + size.height};
  const left = viewport.left || 0, top = viewport.top || 0;
  assert(p.left >= left + 12 && p.top >= top + 12);
  assert(placed.right <= left + viewport.width - 12 && placed.bottom <= top + viewport.height - 12,
    "Navigation must remain inside the viewport");
  if (hasSpace) assert.equal(overlap(placed, target), 0, "Keep the highlighted control usable when there is room");
}
console.log("Tutorial targets, opt-in defaults, translations, and responsive placement passed.");

function eventTarget() {
  const listeners = new Map();
  return {
    addEventListener(type, callback) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(callback);
    },
    removeEventListener(type, callback) { listeners.get(type)?.delete(callback); },
    dispatch(type, properties = {}) {
      const event = {type, target:this, preventDefault() {}, stopImmediatePropagation() {}, ...properties};
      for (const callback of listeners.get(type) || []) callback(event);
    },
    listenerCount(type) { return listeners.get(type)?.size || 0; },
  };
}

function tutorialHarness({visualViewport = {width:195, height:422, offsetLeft:0, offsetTop:0}} = {}) {
  const nodes = new Map(), frames = new Map();
  let nextFrame = 0, focused = null;
  const content = {scrollTop:0};
  function node(id) {
    if (!nodes.has(id)) {
      const element = {
        ...eventTarget(), id, hidden:true, disabled:false, value:"zh", style:{}, dataset:{},
        classList:{toggle() {}},
        tagName:id === "tutorial-language" ? "SELECT" : "BUTTON", parentElement:null,
        setAttribute(name, value) { this[name] = value; },
        focus() { focused = this; }, click() { this.dispatch("click"); },
        append() {}, replaceChildren() {}, scrollIntoView() {}, closest() { return null; },
        getClientRects() { return [{}]; },
        contains(element) { return element.id?.startsWith("tutorial-"); },
        querySelector() { return content; },
        getBoundingClientRect() {
          if (id !== "tutorial-bubble") return this.rect || {left:0, top:0, right:390, bottom:844};
          const width = Math.min(340, parseFloat(this.style.maxWidth) || 366);
          const height = Math.min(650, parseFloat(this.style.maxHeight) || 820);
          const left = parseFloat(this.style.left) || 0, top = parseFloat(this.style.top) || 0;
          return {left, top, right:left + width, bottom:top + height, width, height};
        },
      };
      nodes.set(id, element);
    }
    return nodes.get(id);
  }
  const document = {
    ...eventTarget(), getElementById:node, querySelectorAll:() => [], createElement:() => node("kbd"),
    querySelector(selector) {
      if (selector === ".control-scroll") return node("control-scroll");
      return null;
    },
  };
  const viewport = visualViewport && {...eventTarget(), ...visualViewport};
  const window = {...eventTarget(), visualViewport:viewport, scrollTo() {}};
  const sandbox = {
    document, window, innerWidth:390, innerHeight:844, scrollX:0, scrollY:0,
    getComputedStyle:() => ({overflowY:"auto"}),
    requestAnimationFrame(callback) { frames.set(++nextFrame, callback); return nextFrame; },
    cancelAnimationFrame(id) { frames.delete(id); },
    ResizeObserver:class { observe() {} disconnect() {} },
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "tutorial.js"), "utf8"), sandbox);
  return {
    node, document, viewport, content, sandbox,
    get focused() { return focused; },
    get pendingFrames() { return frames.size; },
    flush() {
      const callbacks = [...frames.values()]; frames.clear();
      for (const callback of callbacks) callback();
    },
  };
}

function assertInsideVisibleViewport(harness) {
  const viewport = harness.viewport || {width:harness.sandbox.innerWidth, height:harness.sandbox.innerHeight};
  const left = viewport.offsetLeft || 0, top = viewport.offsetTop || 0;
  const rect = harness.node("tutorial-bubble").getBoundingClientRect();
  assert(rect.left >= left + 12 && rect.top >= top + 12, "Guide must respect the visible viewport origin");
  assert(rect.right <= left + viewport.width - 12, "Zoomed guide and navigation must fit horizontally");
  assert(rect.bottom <= top + viewport.height - 12, "Guide must fit above an on-screen keyboard");
}

test("pinch zoom sizes the guide to the visible viewport without changing its language defaults", () => {
  const harness = tutorialHarness();
  assert(harness.node("tutorial-bubble").hidden, "The guide remains opt-in");
  harness.node("tutorial-start").click(); harness.flush();
  assert.equal(harness.node("tutorial-language").value, "en");
  assert.equal(harness.node("tutorial-language-label").textContent, "语言");
  assert.equal(harness.focused, harness.node("tutorial-next"));
  assertInsideVisibleViewport(harness);
  harness.node("tutorial-language").value = "zh";
  harness.node("tutorial-language").dispatch("change"); harness.flush();
  assert.equal(harness.node("tutorial-language-label").textContent, "language");
  assertInsideVisibleViewport(harness);
});

test("zoom panning and the on-screen keyboard update the active guide and clipped highlight", () => {
  const harness = tutorialHarness();
  harness.node("tutorial-start").click(); harness.flush();
  harness.viewport.offsetLeft = 67; harness.viewport.offsetTop = 88;
  harness.viewport.dispatch("scroll");
  assert.equal(harness.pendingFrames, 1, "Visual viewport panning schedules placement");
  harness.flush(); assertInsideVisibleViewport(harness);
  assert(parseFloat(harness.node("tutorial-highlight").style.left) >= 67);
  assert(parseFloat(harness.node("tutorial-highlight").style.top) >= 88);
  harness.viewport.height = 180;
  harness.viewport.dispatch("resize"); harness.flush();
  assertInsideVisibleViewport(harness);
});

test("guide navigation resets scrolled content and closing removes viewport listeners", () => {
  const harness = tutorialHarness();
  harness.node("tutorial-start").click(); harness.flush();
  assert.equal(harness.viewport.listenerCount("resize"), 1);
  assert.equal(harness.viewport.listenerCount("scroll"), 1);
  harness.content.scrollTop = 200;
  harness.document.dispatch("keydown", {key:"ArrowRight", target:harness.node("tutorial-next")});
  harness.flush();
  assert.equal(harness.node("tutorial-bubble").dataset.step, "2");
  assert.equal(harness.content.scrollTop, 0, "New steps begin at the start of their explanation");
  harness.document.dispatch("keydown", {key:"Escape"});
  assert(harness.node("tutorial-bubble").hidden);
  assert.equal(harness.viewport.listenerCount("resize"), 0);
  assert.equal(harness.viewport.listenerCount("scroll"), 0);
  harness.viewport.dispatch("resize"); harness.viewport.dispatch("scroll");
  assert.equal(harness.pendingFrames, 0, "Closing must cancel all placement work");
  harness.node("tutorial-start").click(); harness.flush();
  assert.equal(harness.viewport.listenerCount("scroll"), 1, "Restarting does not duplicate listeners");
  harness.node("tutorial-close").click();
  assert.equal(harness.focused, harness.node("tutorial-start"));
});

test("browsers without visualViewport retain layout viewport placement", () => {
  const harness = tutorialHarness({visualViewport:null});
  harness.node("tutorial-start").click(); harness.flush();
  assertInsideVisibleViewport(harness);
});
