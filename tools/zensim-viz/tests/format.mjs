import assert from "node:assert/strict";
import { formatNumber } from "../web/format.mjs";
for (const [value, expected] of [
  [1000, "1000"],
  [100, "100"],
  [0.001, "0.001"],
  [1.23, "1.23"],
  [-0.0001, "-0.0001"],
  [0, "0"],
  [12345, "1.235e+4"],
  [null, "unavailable"],
])
  assert.equal(formatNumber(value), expected);
console.log(
  "Number formatting preserves integer zeros and four significant digits.",
);
