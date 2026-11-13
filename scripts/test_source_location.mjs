import assert from "node:assert/strict";
import { orderedChunks, resolveSource, sourceHref } from "../web/src/sourceLocation.js";
import { loadTestMaterials } from "./lib/test_materials.mjs";
const _TEST_MATERIALS = loadTestMaterials(import.meta.url);

const chunks = [
  { chunk_id: "a", chunk_hash: "ha", start: 0, end: 8, page: 1, text: _TEST_MATERIALS["text_001"] },
  {
    chunk_id: "b",
    chunk_hash: "hb",
    start: 3,
    end: 11,
    page: 1,
    text: _TEST_MATERIALS["text_002"],
  },
];
assert.deepEqual(
  orderedChunks([...chunks].reverse()).map((x) => x.text),
  chunks.map((x) => x.text)
);
assert.equal(
  resolveSource(chunks, { chunkId: "b", quote: _TEST_MATERIALS["text_003"], hash: "hb" }).target
    .chunk_id,
  "b"
);
assert.ok(
  resolveSource(chunks, { quote: _TEST_MATERIALS["text_004"] }).error,
  _TEST_MATERIALS["text_005"]
);
assert.equal(
  resolveSource(chunks, { quote: _TEST_MATERIALS["text_006"] }).target.chunk_id,
  "a",
  _TEST_MATERIALS["text_007"]
);
assert.ok(resolveSource(chunks, { chunkId: "b", quote: _TEST_MATERIALS["text_008"] }).error);
assert.ok(resolveSource(chunks, { chunkId: "gone" }).error);
assert.ok(resolveSource(chunks, { chunkId: "b", hash: "old" }).error);
assert.ok(
  resolveSource([{ ...chunks[0], text: _TEST_MATERIALS["text_009"] }], {
    chunkId: "a",
    quote: _TEST_MATERIALS["text_010"],
  }).error
);
assert.equal(resolveSource(chunks, {}).target, null);
assert.equal(resolveSource(chunks, { chunkId: "a" }).end, chunks[0].text.length);
const href = sourceHref({
  doc_id: "doc",
  chunk_id: "b",
  chunk_hash: "hb",
  text: _TEST_MATERIALS["text_011"],
});
const url = new URL(href, "http://test");
assert.equal(url.searchParams.get("chunk"), "b");
assert.equal(url.searchParams.get("hash"), "hb");
assert.equal(url.hash, "#chunk-b");
console.log(_TEST_MATERIALS["text_012"]);
