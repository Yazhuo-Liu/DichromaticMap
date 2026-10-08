"use strict";

// A coarse, immutable index into the original Float64 rows. Queries return
// conservative candidates in original row order; use.js keeps the final exact
// Float64 layer/boundary/viewport tests and all scientific data unchanged.
class RowSpatialIndex {
  constructor(rows, {midpoints = false, minimumRows = 8192} = {}) {
    this.rows = rows; this.midpoints = midpoints; this.bins = null;
    if (rows.length < minimumRows) return;
    let left = Infinity, right = -Infinity, bottom = Infinity, top = -Infinity;
    for (let index = 0; index < rows.length; index++) {
      const [x, y] = this.coordinates(index);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;
      left = Math.min(left, x); right = Math.max(right, x);
      bottom = Math.min(bottom, y); top = Math.max(top, y);
    }
    const width = right - left, height = top - bottom;
    if (!(width > 0) || !(height > 0) || !Number.isFinite(width + height)) return;
    const cells = Math.ceil(rows.length / 64);
    this.nx = Math.min(256, Math.max(1, Math.round(Math.sqrt(cells * width / height))));
    this.ny = Math.min(256, Math.max(1, Math.ceil(cells / this.nx)));
    this.bounds = {left, right, bottom, top};
    this.dx = width / this.nx; this.dy = height / this.ny;
    if (!(this.dx > 0) || !(this.dy > 0)) return;
    this.bins = new Map(); this.candidates = new Uint32Array(rows.length);
    for (let index = 0; index < rows.length; index++) {
      const [x, y] = this.coordinates(index);
      const bx = Math.min(this.nx - 1, Math.floor((x - left) / this.dx));
      const by = Math.min(this.ny - 1, Math.floor((y - bottom) / this.dy));
      const key = by * this.nx + bx;
      if (!this.bins.has(key)) this.bins.set(key, []);
      this.bins.get(key).push(index);
    }
  }
  coordinates(index) {
    const offset = index * this.rows.stride, flat = this.rows.flat;
    return this.midpoints ? [(flat[offset] + flat[offset + 2]) / 2, (flat[offset + 1] + flat[offset + 3]) / 2]
      : [flat[offset], flat[offset + 1]];
  }
  queryViewport({cosine, sine, cx, cy, width, height}) {
    if (!this.bins) return null;
    let left = Infinity, right = -Infinity, bottom = Infinity, top = -Infinity;
    for (const dx of [-width / 2, width / 2]) for (const dy of [-height / 2, height / 2]) {
      const x = cx + dx, y = cy + dy;
      const px = cosine * x + sine * y, py = -sine * x + cosine * y;
      left = Math.min(left, px); right = Math.max(right, px);
      bottom = Math.min(bottom, py); top = Math.max(top, py);
    }
    return this.queryBounds({left, right, bottom, top});
  }
  queryBounds({left, right, bottom, top}) {
    if (!this.bins || ![left, right, bottom, top].every(Number.isFinite)) return null;
    // Inverting the view rotation can differ by a few ulps from the exact
    // forward projection. Expand the coarse query so edge points cannot drop.
    const padding = 64 * Number.EPSILON * Math.max(1, Math.abs(left), Math.abs(right), Math.abs(bottom), Math.abs(top));
    left -= padding; right += padding; bottom -= padding; top += padding;
    const bounds = this.bounds;
    if (right < bounds.left || left > bounds.right || top < bounds.bottom || bottom > bounds.top)
      return this.candidates.subarray(0, 0);
    const x0 = Math.max(0, Math.floor((left - bounds.left) / this.dx));
    const x1 = Math.min(this.nx - 1, Math.floor((right - bounds.left) / this.dx));
    const y0 = Math.max(0, Math.floor((bottom - bounds.bottom) / this.dy));
    const y1 = Math.min(this.ny - 1, Math.floor((top - bounds.bottom) / this.dy));
    let count = 0;
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++)
      count += this.bins.get(y * this.nx + x)?.length || 0;
    // A broad/rotated view benefits from the simple full scan instead of
    // copying and sorting almost the entire buffer.
    if (count >= this.rows.length * 0.75) return null;
    let at = 0;
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      const indices = this.bins.get(y * this.nx + x);
      if (indices) { this.candidates.set(indices, at); at += indices.length; }
    }
    return this.candidates.subarray(0, count).sort();
  }
}
if (typeof module !== "undefined") module.exports = {RowSpatialIndex};
if (typeof window !== "undefined") window.DichromaticSpatialIndex = {RowSpatialIndex};
