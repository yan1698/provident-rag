/** Text cases are supplied locally, independently of the public test code. */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export function loadTestMaterials(scriptUrl) {
  const root = fileURLToPath(new URL("../..", import.meta.url));
  const directory =
    process.env.PROVIDENT_TEST_MATERIALS_DIR || path.join(root, "data/test-materials");
  const name = path.parse(fileURLToPath(scriptUrl)).name;
  const file = path.join(directory, `${name}.json`);
  if (!fs.existsSync(file)) {
    console.log(
      `SKIP ${name}: local text materials are unavailable; configure PROVIDENT_TEST_MATERIALS_DIR.`
    );
    process.exit(77);
  }
  const payload = JSON.parse(fs.readFileSync(file, "utf8"));
  if (
    payload.schema_version !== 1 ||
    !payload.texts ||
    typeof payload.texts !== "object" ||
    Array.isArray(payload.texts) ||
    Object.values(payload.texts).some((value) => typeof value !== "string")
  ) {
    throw new Error(`Invalid local materials for ${name}.`);
  }
  return payload.texts;
}
