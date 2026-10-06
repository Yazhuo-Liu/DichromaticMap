"use strict";

// Keep transferred Float64 buffers as the numerical source of truth. Only
// picking, external iteration or serialization materializes ordinary rows.
class DenseRows {
  constructor(flat, stride) {
    if (flat.length % stride) throw new Error("Invalid render buffer length");
    this.flat = flat;
    this.stride = stride;
    this.length = flat.length / stride;
  }
  row(index) {
    if (index < 0 || index >= this.length) return undefined;
    const point = new Array(this.stride), start = index * this.stride;
    for (let column = 0; column < this.stride; column++) point[column] = this.flat[start + column];
    return point;
  }
  at(index) { return this.row(index < 0 ? this.length + index : index); }
  *[Symbol.iterator]() { for (let row = 0; row < this.length; row++) yield this.row(row); }
  map(callback) { return Array.from(this, (row, index) => callback(row, index, this)); }
  filter(callback) { return Array.from(this).filter((row, index) => callback(row, index, this)); }
  toJSON() { return Array.from(this); }
  static from(rows, stride) {
    if (rows instanceof DenseRows) return rows;
    const flat = new Float64Array(rows.length * stride);
    for (let row = 0; row < rows.length; row++) {
      for (let column = 0; column < stride; column++) flat[row * stride + column] = rows[row][column];
    }
    return new DenseRows(flat, stride);
  }
}

class SelectedRows {
  constructor(source, selection) { this.source = source; this.selection = selection; }
  get length() { return this.selection.count; }
  row(index) { return index >= 0 && index < this.length ? this.source.row(this.selection.indices[index]) : undefined; }
  at(index) { return this.row(index < 0 ? this.length + index : index); }
  *[Symbol.iterator]() { for (let row = 0; row < this.length; row++) yield this.row(row); }
  map(callback) { return Array.from(this, (row, index) => callback(row, index, this)); }
  filter(callback) { return Array.from(this).filter((row, index) => callback(row, index, this)); }
  toJSON() { return Array.from(this); }
}

function decodeRenderResult(value) {
  if (value?.format !== "dichromatic-map-render-v1") return value;
  return {...value.metadata, grains: value.grains.map(grain => new DenseRows(grain, 6)),
    coincidences: new DenseRows(value.coincidences, 3), local: value.local === null ? null : new DenseRows(value.local, 5)};
}

if (typeof module !== "undefined") module.exports = {decodeRenderResult, DenseRows, SelectedRows};
if (typeof window !== "undefined") window.DichromaticRenderData = {decodeRenderResult, DenseRows, SelectedRows};
