"use strict";

document.querySelectorAll("[data-copy-target]").forEach(button => {
  const originalLabel = button.textContent;
  let resetTimer;
  button.addEventListener("click", async () => {
    const source = document.getElementById(button.dataset.copyTarget);
    if (!source) return;
    let copied = false;
    try {
      await navigator.clipboard.writeText(source.textContent.trim());
      copied = true;
    } catch {
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(source);
      selection.removeAllRanges();
      selection.addRange(range);
      try { copied = document.execCommand?.("copy") ?? false; } catch { /* Keep text selected. */ }
      if (copied) selection.removeAllRanges();
    }
    button.textContent = copied ? "Copied" : "Selected — press Ctrl+C";
    clearTimeout(resetTimer);
    resetTimer = setTimeout(() => { button.textContent = originalLabel; }, 2500);
  });
});
