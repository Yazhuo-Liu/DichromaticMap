"use strict";

// The Worker transfers dense Float64 buffers, keeping JSON out of the large
// render response. Restore ordinary rows so picking and session serialization
// keep exactly the same Array/slice semantics as the original UI.
function decodeRenderResult(value) {
  if (value?.format !== "dichromatic-map-render-v1") return value;
  const rows = (flat, columns) => {
    const result = new Array(flat.length / columns);
    for (let start = 0, row = 0; start < flat.length; start += columns, row++) {
      const point = new Array(columns);
      for (let column = 0; column < columns; column++) point[column] = flat[start + column];
      result[row] = point;
    }
    return result;
  };
  return {...value.metadata, grains: value.grains.map(grain => rows(grain, 6)),
    coincidences: rows(value.coincidences, 3), local: value.local === null ? null : rows(value.local, 5)};
}

if (typeof module !== "undefined") module.exports = {decodeRenderResult};
if (typeof window !== "undefined") window.DichromaticRenderData = {decodeRenderResult};
