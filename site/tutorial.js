"use strict";

// Keep the highlighted controls usable, and place the guide beside them.
function tutorialPosition(target, size, viewport) {
  const margin = 12, gap = 12;
  const clamp = (n, min, max) => Math.max(min, Math.min(n, max));
  const maxX = Math.max(margin, viewport.width - size.width - margin);
  const maxY = Math.max(margin, viewport.height - size.height - margin);
  const x = clamp((target.left + target.right - size.width) / 2, margin, maxX);
  const y = clamp((target.top + target.bottom - size.height) / 2, margin, maxY);
  const choices = [
    {left: target.left - size.width - gap, top: y},
    {left: target.right + gap, top: y},
    {left: x, top: target.bottom + gap},
    {left: x, top: target.top - size.height - gap},
  ];
  const fits = p => p.left >= margin && p.top >= margin &&
    p.left + size.width <= viewport.width - margin &&
    p.top + size.height <= viewport.height - margin;
  const fitted = choices.find(fits);
  if (fitted) return fitted;
  const corners = [{left: margin, top: margin}, {left: maxX, top: margin},
                   {left: margin, top: maxY}, {left: maxX, top: maxY}];
  const overlap = p => Math.max(0, Math.min(p.left + size.width, target.right) - Math.max(p.left, target.left)) *
    Math.max(0, Math.min(p.top + size.height, target.bottom) - Math.max(p.top, target.top));
  return corners.reduce((best, p) => overlap(p) < overlap(best) ? p : best);
}

const tutorialSteps = [
  {target: "map", en: ["Explore the pattern", "The plot overlays two grains. Colors identify grains; marker shapes identify axial layers.", "Drag to pan, then use the wheel or pinch to zoom. You can try each feature while this guide is open."],
   zh: ["先感受一下双色图", "图中叠加了两个晶粒：颜色区分晶粒，符号形状区分轴向层。", "拖动平移，再用滚轮或双指缩放。引导期间可以直接操作控件，不必完成操作也可进入下一步。"], keys: [["C", "Center view", "视野居中"]]},
  {target: "crystal-controls", tab: "orientation", en: ["Choose a crystal and axis", "Select FCC, BCC or SC and a viewing axis. Custom [h k l] accepts an integer triple.", "Try a different structure or axis, then return to FCC / ⟨110⟩. Changing geometry clears earlier selections."],
   zh: ["选择晶体与晶轴", "选择 FCC、BCC 或 SC，再选择观察晶轴。Custom [h k l] 可输入整数三元组。", "试着切换结构或晶轴，再回到 FCC / ⟨110⟩。改变几何参数会清除已有选点。"]},
  {target: "angle-controls", tab: "orientation", en: ["Find an exact CSL", "CSL presets give commensurate orientations with gold coincidence markers. You can also enter or slide the misorientation angle.", "Try the Σ9 preset for FCC ⟨110⟩. Later, 22° is useful for exploring local near pairs."],
   zh: ["找到精确重合点", "CSL 预设给出公度取向，精确重合点显示为金色标记。也可输入错取向角或拖动滑块。", "先试 FCC ⟨110⟩ 的 Σ9 预设。后面体验局部近邻配对时，可使用 22°。"]},
  {target: "rotation-controls", tab: "orientation", en: ["Rotate the drawing", "Display rotation turns the view without changing the grains' relative misorientation.", "Move the slider, then use 0° to return. The grain reference axes rotate with the view."],
   zh: ["旋转显示方向", "Display rotation 旋转画面，不改变两个晶粒之间的错取向角。", "拖动滑块，再点击 0° 恢复。晶粒参考轴也会跟着画面旋转。"]},
  {target: "layers-tab", tab: "layers", en: ["Separate the axial layers", "Each grain has independent layer visibility. Coincidence and near-pair markers need that layer visible in both grains.", "Click No layers, then enable only G1 A and G2 A. This makes later cell picking easier."],
   zh: ["分开观察轴向层", "两个晶粒可以分别显示或隐藏各层。重合点和近邻配对要求同一层在两个晶粒中都可见。", "点击 No layers，再只勾选 G1 A 与 G2 A。这也能让后面的晶胞选点更清楚。"]},
  {target: "common-cell-controls", tab: "layers", en: ["Show an automatic common cell", "Automatic common cell displays an available exact CSL cell or a cell from the strain search.", "At the Σ9 preset, enable the cell and click Fit cell to frame it. Fit is unavailable when no cell exists."],
   zh: ["显示自动共同胞", "Automatic common cell 显示可用的精确 CSL 胞，或应变搜索得到的共同胞。", "在 Σ9 预设下勾选该选项，再点击 Fit cell 居中显示。没有可用晶胞时，Fit 不可用。"]},
  {target: "pick-gb", en: ["Define a grain boundary", "Pick two visible atoms, B1 then B2, to define a line. Left and right are relative to B1 → B2.", "Click Pick GB, then two atoms in the plot. The side controls appear after B2; try arrangements 1 and 2, then show all sides."],
   zh: ["定义晶界参考线", "依次选择两个可见原子 B1、B2 定义直线。左右两侧以 B1 → B2 的方向为准。", "点击 Pick GB，再在图中选两个原子。B2 选完后出现侧别控件；可试排列 1、2，再恢复全部两侧。"], keys: [["R", "Pick GB", "选晶界"], ["1 / 2", "Side arrangements", "切换侧别排列"], ["F", "All sides", "显示全部两侧"]]},
  {target: "pick-vector", en: ["Measure a crystal vector", "Choose P1 and P2 to see the vector length and coordinates in the relevant grain frames. P2 axial periodic image adds an axial repeat.", "Click Measure vector and pick two distinct atoms. Compare a same-grain vector with one across the grains."],
   zh: ["测量晶体向量", "选择 P1、P2，查看向量长度及相应晶粒坐标。P2 axial periodic image 可为第二个端点增加轴向周期。", "点击 Measure vector，再选两个不同原子。分别试同一晶粒内的向量与跨晶粒向量。"], keys: [["V", "Measure vector", "测量向量"]]},
  {target: "view-tab", view: "view", en: ["Frame the view and reference axes", "Field size changes the visible region. Grain reference axes show each grain's orientation, with colors matching the grains.", "Try 2× and Center view. Toggle the reference axes; you can also collapse the legend in the plot."],
   zh: ["调整视野与参考轴", "Field size 改变可见范围。Grain reference axes 显示各晶粒的方向，颜色与晶粒一致。", "试试 2× 和 Center view，再开关参考轴。图中的 Legend 也可以折叠。"], keys: [["C", "Center view", "视野居中"]]},
  {target: "appearance-tab", view: "appearance", en: ["Customize the markers", "Set grain colors and each layer's unique symbol and marker size. These settings are included in PNG exports and saved sessions.", "Change a grain color or layer size, then try Reset appearance."],
   zh: ["调整颜色、形状与大小", "设置晶粒颜色、各层的唯一符号和标记大小。这些设置会保留在 PNG 和保存的会话中。", "换一个晶粒颜色或调整层标记大小，再试 Reset appearance。"]},
  {target: "performance-tab", view: "performance", en: ["Understand the length units", "Lattice constant a₀ sets the reference length in Å. Plot coordinates remain normalized: 1 means one lattice constant.", "Inspect this value. Larger fields and strain-search bounds need more computation on your device."],
   zh: ["理解长度单位", "Lattice constant a₀ 给出以 Å 为单位的参考长度。图中坐标仍归一化，1 表示一个晶格常数。", "查看该数值。更大的视野和应变搜索范围会增加设备上的计算量。"]},
  {target: "near-method", en: ["Find local near pairs", "Local matching finds close, same-layer mutual nearest pairs without moving atoms. Purple rings mark the pairs.", "At 22°, choose Local matching, enable Near-CSL and adjust Local pair distance. Keep one layer visible in both grains."],
   zh: ["寻找局部近邻配对", "Local matching 寻找同层、互为最近邻的原子配对，不移动原子。配对用紫色圆环标出。", "在 22° 下选择 Local matching，启用 Near-CSL，再调整 Local pair distance。两个晶粒都保留同一可见层。"]},
  {target: "near-method", en: ["Search strained periodic cells", "Homogeneous strain + periodic cell searches for common cells within strain and index limits. Choosing a result applies its uniform strain.", "Select that method and enable Near-CSL. Start with a small index bound; inspect the strain readout. Disable Near-CSL to restore the unstrained grains."],
   zh: ["搜索应变共同周期胞", "Homogeneous strain + periodic cell 在应变和搜索指数限制内寻找共同胞。选择结果会施加相应的均匀应变。", "选择该方法并启用 Near-CSL。先用较小的搜索范围，查看应变读数。关闭 Near-CSL 可恢复未应变晶粒。"]},
  {target: "pick-cell", en: ["Pick and count a manual cell", "Pick four gold CSL or purple near-pair markers of one layer around a convex cell. Each grain is counted inside its own actual atom vertices.", "Return to an exact CSL preset or Local matching. Click Pick 4 CSL vertices and pick around the perimeter. Read the separate G1/G2 interior and boundary counts."],
   zh: ["手动选胞并计数", "在同一层上沿凸晶胞周边选择四个金色 CSL 或紫色近邻标记。两个晶粒按各自实际原子顶点计数。", "回到精确 CSL 预设或 Local matching，点击 Pick 4 CSL vertices 并依次选点。查看 G1/G2 各自的内部和边界原子数。"], keys: [["M", "Pick a manual cell", "选择手动胞"]]},
  {target: "complete-cell", en: ["Complete a cell with fewer picks", "After two vertices, symmetry can suggest completions. After three, parallelogram completion can supply the fourth vertex.", "With two or three vertices selected, open Complete by symmetry… and inspect a candidate. Undo vertex removes the last pick; Clear removes the cell."],
   zh: ["用更少的选点补全晶胞", "选完两个顶点后可按对称性建议补全；选完三个顶点后，可按平行四边形生成第四个顶点。", "保留两个或三个顶点，打开补全按钮并查看候选。Undo vertex 撤销最后一个点，Clear 清除晶胞。"]},
  {target: "apply-strain", en: ["Fit strain to your selected cell", "A four-vertex local-pair cell can be fitted within the selected strain and rotation limits. This deforms the grains; it does not relax atoms.", "With a local-pair cell selected, apply bulk strain and inspect the readout. The same button restores the original local structure."],
   zh: ["对所选晶胞拟合应变", "四顶点局部配对胞可在应变和转动限制内拟合。这会改变晶粒几何形状，不会进行原子弛豫。", "选好局部配对胞后施加应变，查看读数。再次点击同一按钮可恢复原始局部结构。"]},
  {target: "export-controls", en: ["Export your figure", "Export PNG saves the plot without the controls or this tutorial. Clean PNG keeps only the visible atom symbols.", "Try a normal or atoms-only export. The colors and marker sizes you chose are retained."],
   zh: ["导出图片", "Export PNG 保存图形，不包含控制栏或引导气泡。Clean PNG 只保留可见原子符号。", "试着导出普通图或仅原子图。你设置的颜色、符号大小都会保留。"]},
  {target: "save-session", en: ["Save a restorable session", "Save session stores the structure, selections, strain, display settings and numerical CSV tables in one .dmap file.", "Save a session to your device so you can return to this analysis later."],
   zh: ["保存可恢复的会话", "Save session 把结构、选点、应变、显示设置和数值 CSV 表保存到一个 .dmap 文件中。", "将会话保存到设备，之后可以继续这次分析。"]},
  {target: "import-session", tab: "orientation", en: ["Restore a session and keep exploring", "Import session restores a .dmap file in this window. You can restart this guide whenever you need it.", "Try importing the session you just saved. Click Finish to close the guide; your own changes stay in the app."],
   zh: ["恢复会话，继续探索", "Import session 在当前窗口恢复 .dmap 文件。需要时可以再次点击 Tutorial 重启引导。", "试着导入刚保存的会话。点击完成结束引导；你自己做的修改会保留在应用中。"]},
];

if (typeof module !== "undefined") module.exports = {tutorialPosition, tutorialSteps};

if (typeof document !== "undefined") (() => {
  const get = id => document.getElementById(id);
  const start = get("tutorial-start"), bubble = get("tutorial-bubble"), highlight = get("tutorial-highlight");
  const language = get("tutorial-language");
  language.value = navigator.language?.startsWith("zh") ? "zh" : "en";
  let index = -1, target = null, saved = null, frame = 0;
  const observer = typeof ResizeObserver !== "undefined" ? new ResizeObserver(schedule) : null;

  function schedule() {
    if (index < 0 || frame) return;
    frame = requestAnimationFrame(() => { frame = 0; position(); });
  }
  function position() {
    if (index < 0) return;
    const raw = target.getBoundingClientRect();
    let rect = {left: Math.max(0, raw.left - 5), top: Math.max(0, raw.top - 5),
                right: Math.min(innerWidth, raw.right + 5), bottom: Math.min(innerHeight, raw.bottom + 5)};
    const scroll = target.closest(".control-scroll");
    if (scroll && getComputedStyle(scroll).overflowY !== "visible") {
      const clip = scroll.getBoundingClientRect();
      rect = {left: Math.max(rect.left, clip.left), top: Math.max(rect.top, clip.top),
              right: Math.min(rect.right, clip.right), bottom: Math.min(rect.bottom, clip.bottom)};
    }
    const visible = target.getClientRects().length && rect.right > rect.left && rect.bottom > rect.top;
    highlight.hidden = !visible;
    if (visible) Object.assign(highlight.style, {left: `${rect.left}px`, top: `${rect.top}px`,
      width: `${rect.right - rect.left}px`, height: `${rect.bottom - rect.top}px`});
    else rect = {left: innerWidth, right: innerWidth, top: 0, bottom: 0};
    const p = tutorialPosition(rect, bubble.getBoundingClientRect(), {width: innerWidth, height: innerHeight});
    Object.assign(bubble.style, {left: `${p.left}px`, top: `${p.top}px`});
  }
  function selectTab(attribute, name) {
    if (name) document.querySelector(`[${attribute}="${name}"]`)?.click();
  }
  function show(nextIndex, focus = true) {
    index = nextIndex;
    const step = tutorialSteps[index], zh = language.value === "zh";
    const [title, description, trial] = step[zh ? "zh" : "en"];
    selectTab("data-main-tab", step.tab);
    selectTab("data-view-tab", step.view);
    target = get(step.target);
    for (let parent = target.parentElement; parent; parent = parent.parentElement) {
      if (parent.tagName === "DETAILS") parent.open = true;
    }
    bubble.hidden = false;
    bubble.lang = zh ? "zh-CN" : "en";
    bubble.dataset.step = String(index + 1);
    get("tutorial-title").textContent = title;
    get("tutorial-description").textContent = description;
    get("tutorial-try").textContent = `${zh ? "试一试：" : "Try it: "}${trial}`;
    get("tutorial-progress").textContent = zh ? `${index + 1} / ${tutorialSteps.length} 步` : `Step ${index + 1} of ${tutorialSteps.length}`;
    get("tutorial-close").setAttribute("aria-label", zh ? "关闭引导" : "Close tutorial");
    get("tutorial-back").textContent = zh ? "上一步" : "Back";
    get("tutorial-back").disabled = index === 0;
    get("tutorial-next").textContent = index === tutorialSteps.length - 1 ? (zh ? "完成" : "Finish") : (zh ? "下一步" : "Next");
    get("tutorial-hint").textContent = zh ? "Esc 退出引导 · 气泡内 ← / → 切换步骤 · 功能快捷键在输入框外使用" : "Esc closes guide · ← / → inside guide change steps · App shortcuts work outside inputs";
    const shortcuts = get("tutorial-shortcuts"); shortcuts.replaceChildren();
    shortcuts.hidden = !step.keys?.length;
    for (const [key, en, chinese] of step.keys || []) {
      const kbd = document.createElement("kbd"); kbd.textContent = key;
      shortcuts.append(kbd, ` ${zh ? chinese : en}  `);
    }
    observer?.disconnect(); observer?.observe(target); observer?.observe(bubble);
    target.scrollIntoView({block: "center", inline: "nearest", behavior: "instant"});
    position(); schedule();
    if (focus) get("tutorial-next").focus({preventScroll: true});
  }
  function close() {
    if (index < 0) return;
    index = -1; bubble.hidden = true; highlight.hidden = true;
    start.setAttribute("aria-expanded", "false");
    observer?.disconnect(); cancelAnimationFrame(frame); frame = 0;
    for (const [section, open] of saved.sections) section.open = open;
    selectTab("data-main-tab", saved.main); selectTab("data-view-tab", saved.view);
    document.querySelector(".control-scroll").scrollTop = saved.scroll;
    window.scrollTo({left: saved.x, top: saved.y, behavior: "instant"});
    start.focus({preventScroll: true});
  }
  start.addEventListener("click", () => {
    if (index >= 0 || start.disabled) return;
    saved = {sections: Array.from(document.querySelectorAll("details.section"), section => [section, section.open]),
      main: document.querySelector("[data-main-tab].active")?.dataset.mainTab,
      view: document.querySelector("[data-view-tab].active")?.dataset.viewTab,
      scroll: document.querySelector(".control-scroll").scrollTop, x: scrollX, y: scrollY};
    start.setAttribute("aria-expanded", "true");
    show(0);
  });
  get("tutorial-close").addEventListener("click", close);
  get("tutorial-back").addEventListener("click", () => { if (index > 0) show(index - 1); });
  get("tutorial-next").addEventListener("click", () => { if (index === tutorialSteps.length - 1) close(); else if (index >= 0) show(index + 1); });
  language.addEventListener("change", () => { if (index >= 0) show(index, false); });
  document.addEventListener("keydown", event => {
    if (index < 0 || event.ctrlKey || event.metaKey || event.altKey || document.querySelector("dialog[open]")) return;
    if (event.key === "Escape") { event.preventDefault(); event.stopImmediatePropagation(); close(); }
    else if (bubble.contains(event.target) && !["INPUT", "SELECT", "TEXTAREA"].includes(event.target.tagName)) {
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault(); event.stopImmediatePropagation();
        get(event.key === "ArrowRight" ? "tutorial-next" : "tutorial-back").click();
      }
    }
  }, true);
  document.addEventListener("scroll", schedule, true);
  document.addEventListener("click", schedule);
  window.addEventListener("resize", schedule);
  window.visualViewport?.addEventListener("resize", schedule);
})();
