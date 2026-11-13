import assert from "node:assert/strict";
import { submitAgentRun } from "../web/src/submitRun.js";
import { loadTestMaterials } from "./lib/test_materials.mjs";
const _TEST_MATERIALS = loadTestMaterials(import.meta.url);

const keys = [];

function fakeApi(failures) {
  return async (_path, options) => {
    const key = options.headers["Idempotency-Key"];
    keys.push(key);
    if (failures > 0) {
      failures -= 1;
      const error = new TypeError("Failed to fetch");
      throw error;
    }
    return { id: "run-1", delivery: "accepted" };
  };
}

const first = await submitAgentRun(fakeApi(1), { goal: _TEST_MATERIALS["text_001"] });
assert.equal(first.id, "run-1");
assert.equal(keys.length, 2);
assert.equal(keys[0], keys[1]);
assert.ok(keys[0].length > 8 && keys[0].length <= 80);
const reused = keys[0];

keys.length = 0;
await submitAgentRun(fakeApi(0), { goal: _TEST_MATERIALS["text_002"] });
assert.equal(keys.length, 1);
assert.notEqual(keys[0], reused);

let statusCalls = 0;
await assert.rejects(
  () =>
    submitAgentRun(
      async () => {
        statusCalls += 1;
        const error = new Error(_TEST_MATERIALS["text_003"]);
        error.status = 409;
        throw error;
      },
      { goal: _TEST_MATERIALS["text_004"] }
    ),
  (error) => error.status === 409
);
assert.equal(statusCalls, 1);

console.log("PASS submit idempotency");
