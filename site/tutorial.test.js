"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
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
]) {
  const p = tutorialPosition(target, size, viewport);
  const placed = {...p, right:p.left + size.width, bottom:p.top + size.height};
  assert(p.left >= 12 && p.top >= 12);
  assert(placed.right <= viewport.width - 12 && placed.bottom <= viewport.height - 12,
    "Navigation must remain inside the viewport");
  if (hasSpace) assert.equal(overlap(placed, target), 0, "Keep the highlighted control usable when there is room");
}
console.log("Tutorial targets, opt-in defaults, translations, and responsive placement passed.");
