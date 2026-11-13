/** Run contracts while distinguishing unavailable local text cases from passes. */
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const groups = {
  unit: ["test_source_location.mjs", "test_submit_idempotency.mjs"],
  ui: ["test_chat_history_ui.mjs"],
};
const group = process.argv[2] || "unit";
if (!groups[group]) throw new Error("Expected unit or ui.");
let passed = 0,
  skipped = 0;
for (const name of groups[group]) {
  const result = spawnSync(process.execPath, [path.join(root, "scripts", name)], {
    cwd: root,
    stdio: "inherit",
    env: process.env,
  });
  if (result.error) throw result.error;
  if (result.status === 77) skipped++;
  else if (result.status === 0) passed++;
  else process.exit(result.status || 1);
}
console.log(
  `${group} checks finished: ${passed} passed, ${skipped} skipped${skipped ? " (local materials unavailable)" : ""}.`
);
