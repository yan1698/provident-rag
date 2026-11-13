/** Built React UI in Edge; isolated API replay, no real database or model calls. */
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import crypto from "node:crypto";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import { loadTestMaterials } from "./lib/test_materials.mjs";
const _TEST_MATERIALS = loadTestMaterials(import.meta.url);

const require = createRequire(new URL("../web/package.json", import.meta.url));
const { chromium } = require("playwright");
const root = fileURLToPath(new URL("..", import.meta.url));
process.chdir(root);
const dir = path.join(root, "tmp/ui-regression");
await fs.mkdir(dir, { recursive: true });
const protectedFiles = [
  "web/src/pages/Library.jsx",
  "web/src/pages/Graph.jsx",
  "web/src/pages/Document.jsx",
  "web/public/assets/styles.css",
  "web/public/assets/sky.jpg",
  "web/src/components/ProfileChime.jsx",
  "web/src/api.js",
  "web/src/submitRun.js",
  "web/src/sourceLocation.js",
];
const hashes = {};
for (const file of protectedFiles)
  hashes[file] = crypto
    .createHash("sha256")
    .update(
      file.endsWith("/Document.jsx")
        ? (await fs.readFile(file, "utf8")).replaceAll('to="/"', 'to="/agent"')
        : await fs.readFile(file)
    )
    .digest("hex");
// Freeze all backend source/config and the existing user changes, not only Git HEAD.
async function hashTree(folder) {
  for (const e of await fs.readdir(folder, { withFileTypes: true })) {
    const file = path.join(folder, e.name);
    if (e.isDirectory()) {
      if (!["__pycache__", ".pytest_cache"].includes(e.name)) await hashTree(file);
    } else if (/\.(py|txt|toml)$/.test(file))
      hashes[path.relative(root, file)] = crypto
        .createHash("sha256")
        .update(await fs.readFile(file))
        .digest("hex");
  }
}
await hashTree(path.join(root, "backend"));
const scope = { mode: "hard", collection_ids: ["lib-one"], labels: [_TEST_MATERIALS["text_001"]] };
const evidence = [
  {
    index: 1,
    chunk_id: "chunk-one",
    chunk_hash: "hash-one",
    doc_id: "manual-one",
    page: 2,
    text: _TEST_MATERIALS["text_002"],
  },
];
const original = {
  id: "run-one",
  session_id: "session-one",
  status: "finished",
  goal: _TEST_MATERIALS["text_003"],
  task_kind: "synthesis",
  answer_source: "document_only",
  answer: _TEST_MATERIALS["text_004"],
  scope,
  steps: [{ action: "retrieve", observation: _TEST_MATERIALS["text_005"], elapsed_ms: 40 }],
  step_count: 1,
  gaps: [{ topic: _TEST_MATERIALS["text_006"], reason: _TEST_MATERIALS["text_007"] }],
  outline: {
    evidence,
    reading: {
      documents: {
        "manual-one": {
          doc_id: "manual-one",
          filename: _TEST_MATERIALS["text_008"],
          read_chunk_ids: ["chunk-one"],
          required_chunk_ids: ["chunk-one"],
          goals: { steps: { read_chunk_ids: ["chunk-one"] } },
        },
      },
    },
  },
  verification: { claims: [{ text: _TEST_MATERIALS["text_009"], status: "unverified" }] },
};
original.created_at = "2026-11-12T00:00:00Z";
const runs = [original],
  traces = [
    {
      id: "trace-one",
      trace_id: "trace-one",
      query: _TEST_MATERIALS["text_010"],
      answer: original.answer,
      answer_source: "document_only",
      scope,
      diagnostics: { scope },
      evidence,
      citations: evidence,
      verification: original.verification,
      citation_integrity: [{ index: 1, chunk_id: "chunk-one", status: "ok" }],
    },
  ];
const posts = [],
  deletes = [],
  gets = [],
  errors = [],
  external = [],
  checks = [];
let routeFault = false,
  dropNextAgentResponse = false;
const server = http.createServer(async (req, res) => {
  const u = new URL(req.url, "http://local");
  if (req.method === "GET" && !u.pathname.startsWith("/ui")) gets.push(u.pathname + u.search);
  const reply = (data, status = 200) => {
    res.statusCode = status;
    res.setHeader("Content-Type", "application/json");
    res.end(JSON.stringify(data));
  };
  if (u.pathname.startsWith("/ui")) {
    const rel = u.pathname.replace(/^\/ui\/?/, "");
    const file = path.join(root, "web/dist", !rel || !path.extname(rel) ? "index.html" : rel);
    try {
      res.setHeader(
        "Content-Type",
        file.endsWith(".js")
          ? "text/javascript"
          : file.endsWith(".css")
            ? "text/css"
            : file.endsWith(".jpg")
              ? "image/jpeg"
              : "text/html"
      );
      res.end(await fs.readFile(file));
    } catch {
      res.statusCode = 404;
      res.end();
    }
    return;
  }
  if (u.pathname === "/auth/status") return reply({ auth_enabled: false });
  if (u.pathname === "/library")
    return reply({
      collections: [
        { id: "lib-one", name: _TEST_MATERIALS["text_011"] },
        { id: "lib-two", name: _TEST_MATERIALS["text_012"] },
      ],
    });
  if (u.pathname === "/documents")
    return reply({ results: [{ doc_id: "manual-one", filename: _TEST_MATERIALS["text_013"] }] });
  if (u.pathname === "/documents/manual-one/sections") return reply({ sections: [] });
  if (u.pathname === "/documents/manual-one")
    return reply({
      filename: _TEST_MATERIALS["text_014"],
      chunks: evidence,
      chunk_count: 1,
      embedded_count: 1,
    });
  if (u.pathname.startsWith("/graph/")) return reply({ concepts: [] });
  if (u.pathname === "/traces")
    return reply({
      results: traces.slice(
        Number(u.searchParams.get("offset") || 0),
        Number(u.searchParams.get("offset") || 0) + Number(u.searchParams.get("limit") || 20)
      ),
    });
  if (u.pathname === "/agent")
    return reply({
      results: runs.slice(
        Number(u.searchParams.get("offset") || 0),
        Number(u.searchParams.get("offset") || 0) + Number(u.searchParams.get("limit") || 20)
      ),
    });
  if (req.method === "DELETE") {
    deletes.push(u.pathname);
    const key = u.pathname.split("/").at(-1);
    if (u.pathname.startsWith("/traces/")) {
      const i = traces.findIndex((r) => r.id === key);
      if (i >= 0) traces.splice(i, 1);
    } else {
      for (let i = runs.length - 1; i >= 0; i--)
        if (
          u.pathname.startsWith("/agent/session/") ? runs[i].session_id === key : runs[i].id === key
        )
          runs.splice(i, 1);
    }
    return reply({ deleted: true });
  }
  if (u.pathname === "/agent/sessions")
    return reply({
      results: [...new Set(runs.map((r) => r.session_id))]
        .filter(Boolean)
        .map((id) => {
          const rows = runs.filter((r) => r.session_id === id);
          return {
            id,
            title: rows[0].goal,
            latest_goal: rows.at(-1).goal,
            turn_count: rows.length,
            status: rows.at(-1).status,
            task_kind: rows[0].task_kind,
          };
        })
        .slice(
          Number(u.searchParams.get("offset") || 0),
          Number(u.searchParams.get("offset") || 0) + Number(u.searchParams.get("limit") || 30)
        ),
    });
  if (u.pathname.startsWith("/agent/session/"))
    return reply({
      id: u.pathname.split("/").at(-1),
      runs: runs.filter((r) => r.session_id === u.pathname.split("/").at(-1)),
    });
  if (u.pathname.endsWith("/stream")) return reply({ detail: "poll fallback" }, 503);
  if (req.method === "POST") {
    let raw = "";
    for await (const b of req) raw += b;
    const body = raw ? JSON.parse(raw) : {};
    posts.push({ path: u.pathname, body, key: req.headers["idempotency-key"] });
    if (u.pathname === "/query/route") {
      if (routeFault) return reply({ detail: _TEST_MATERIALS["text_015"] }, 503);
      if (
        body.answer_source === "document_only" &&
        ["hard", "soft"].includes(body.scope_mode) &&
        (!body.collection_ids?.length ||
          body.collection_ids.some((id) => !["lib-one", "lib-two"].includes(id)))
      )
        return reply({ detail: _TEST_MATERIALS["text_016"] }, 400);
      const reading =
        body.analysis_mode === "digest" || body.query.includes(_TEST_MATERIALS["text_017"]);
      return reply({
        requested: body.analysis_mode,
        actual:
          body.answer_source === "general_explanation"
            ? "general_explanation"
            : reading && !body.target_document_id
              ? "quick"
              : reading
                ? "digest"
                : body.analysis_mode === "synthesis" ||
                    body.query.includes(_TEST_MATERIALS["text_018"])
                  ? "synthesis"
                  : "quick",
        fallback_reason: reading && !body.target_document_id ? "reading_target_missing" : "",
        reason: body.analysis_mode === "auto" ? "existing_classifier_rules" : "user_override",
      });
    }
    if (u.pathname.endsWith("/cancel")) {
      const row = runs.find((r) => r.id === u.pathname.split("/")[2]);
      row.status = "cancelled";
      return reply(row);
    }
    if (u.pathname === "/agent/run") {
      if (
        body.analysis_mode &&
        ["hard", "soft"].includes(body.scope_mode) &&
        (!body.collection_ids?.length ||
          body.collection_ids.some((id) => !["lib-one", "lib-two"].includes(id)))
      )
        return reply({ detail: _TEST_MATERIALS["text_019"] }, 400);
      const replay = runs.find((r) => r.key === req.headers["idempotency-key"]);
      if (replay) return reply(replay);
      const general = body.answer_source === "general_explanation";
      const row = {
        ...original,
        id: `run-${runs.length + 1}`,
        key: req.headers["idempotency-key"],
        session_id: body.session_id || `session-${runs.length + 1}`,
        created_at: new Date(Date.UTC(2026, 10, 12, 1, runs.length + traces.length)).toISOString(),
        goal: body.goal,
        task_kind: body.task_kind,
        answer_source: body.answer_source || "document_only",
        source_run_id: body.source_run_id,
        target_document_id: body.target_document_id,
        status: body.goal.includes(_TEST_MATERIALS["text_020"]) ? "running" : "finished",
        scope: {
          mode: body.scope_mode || scope.mode,
          collection_ids: body.collection_ids || scope.collection_ids,
        },
        routing: { requested: body.analysis_mode, actual: body.task_kind },
        outline: general ? {} : original.outline,
        gaps: general ? [] : original.gaps,
        steps: general ? [] : original.steps,
        answer: general ? _TEST_MATERIALS["text_021"] : original.answer,
      };
      runs.push(row);
      if (dropNextAgentResponse) {
        dropNextAgentResponse = false;
        return reply({ detail: _TEST_MATERIALS["text_022"] }, 503);
      }
      return reply(row);
    }
    if (u.pathname === "/query") {
      if (body.query.includes(_TEST_MATERIALS["text_023"]))
        await new Promise((resolve) => setTimeout(resolve, 800));
      const general = body.answer_source === "general_explanation";
      const row = {
        ...traces[0],
        id: `trace-${traces.length + 1}`,
        trace_id: `trace-${traces.length + 1}`,
        created_at: new Date(Date.UTC(2026, 10, 12, 1, runs.length + traces.length)).toISOString(),
        query: body.query,
        answer_source: body.answer_source,
        source_trace_id: body.source_trace_id,
        scope: { mode: body.scope_mode, collection_ids: body.collection_ids || [] },
        diagnostics: {
          scope: { mode: body.scope_mode, collection_ids: body.collection_ids || [] },
        },
        answer: general ? _TEST_MATERIALS["text_024"] : original.answer,
        evidence: general ? [] : evidence,
        citations: general ? [] : evidence,
      };
      traces.push(row);
      return reply(row);
    }
    return reply({ detail: "unexpected mutation" }, 400);
  }
  const id = u.pathname.split("/").at(-1);
  if (u.pathname.startsWith("/agent/"))
    return reply(
      runs.find((r) => r.id === id) || { detail: _TEST_MATERIALS["text_025"] },
      runs.some((r) => r.id === id) ? 200 : 404
    );
  if (u.pathname.startsWith("/traces/"))
    return reply(
      traces.find((r) => r.id === id) || { detail: _TEST_MATERIALS["text_026"] },
      traces.some((r) => r.id === id) ? 200 : 404
    );
  return reply({ detail: "unexpected API" }, 404);
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const base = `http://127.0.0.1:${server.address().port}/ui/`,
  browser = await chromium.launch({
    channel: process.env.BROWSER_CHANNEL || undefined,
    headless: true,
  });
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
page.setDefaultTimeout(12000);
await page.route("**/*", (r) => {
  if (r.request().url().startsWith(base.replace("/ui/", "/"))) return r.continue();
  external.push(r.request().url());
  return r.abort();
});
page.on("pageerror", (e) => errors.push(String(e)));
async function settle() {
  await page.evaluate(() => {
    window.scrollTo({ top: 0, behavior: "instant" });
    for (const a of document.getAnimations()) {
      if (a.effect.getTiming().iterations !== Infinity) a.finish();
      else {
        a.pause();
        a.currentTime = 600;
      }
    }
  });
}
async function go(url = "") {
  await page.goto(base + url);
  await page.locator(".app-shell").waitFor();
}
const ready = async () => {
  await page.locator("#q").waitFor();
  await page.waitForFunction(
    (selectors) =>
      !document.querySelector(selectors[0])?.getAttribute("aria-busy") ||
      document.querySelector(selectors[1]).getAttribute("aria-busy") === "false",
    [_TEST_MATERIALS["text_027"], _TEST_MATERIALS["text_028"]]
  );
};
const mode = async (value) => {
  const t = page.getByRole("button", { name: _TEST_MATERIALS["text_029"], exact: true });
  if ((await t.getAttribute("aria-expanded")) !== "true") await t.click();
  await page.getByLabel(_TEST_MATERIALS["text_030"], { exact: true }).selectOption(value);
};
const send = async (text) => {
  await page.locator("#q").fill(text);
  await page.locator("#ask").click();
};
const info = async (name = _TEST_MATERIALS["text_031"]) => {
  if (!(await page.locator("dialog").evaluate((e) => e.open)))
    await page
      .locator(".chat-actions")
      .first()
      .getByRole("button", {
        name:
          name === _TEST_MATERIALS["text_032"]
            ? _TEST_MATERIALS["text_033"]
            : _TEST_MATERIALS["text_034"],
        exact: true,
      })
      .click();
  await page.getByRole("tab", { name, exact: true }).click();
};
const close = async () => {
  await page.getByRole("button", { name: _TEST_MATERIALS["text_035"], exact: true }).click();
  await page.locator("dialog").waitFor({ state: "hidden" });
};
const history = async () => {
  await page
    .locator(".topbar")
    .getByRole("link", { name: _TEST_MATERIALS["text_036"], exact: true })
    .click();
  await page.getByText(new RegExp(_TEST_MATERIALS["regex_001"], "")).waitFor();
};
const count = (endpoint) => posts.filter((row) => row.path === endpoint),
  last = (endpoint) => count(endpoint).at(-1).body;
async function screenshot(name) {
  await settle();
  if (await page.locator(".chat-messages").count())
    await page
      .locator(".chat-messages")
      .evaluate((e) => e.scrollTo({ top: e.scrollHeight, behavior: "instant" }));
  await page.screenshot({ path: path.join(dir, name + ".png"), fullPage: true });
}
try {
  await go("agent");
  await ready();
  assert.equal(await page.locator(_TEST_MATERIALS["text_037"]).count(), 0);
  assert.equal(await page.locator("textarea").count(), 1);
  assert.equal(await page.locator("#q").inputValue(), "");
  await screenshot("after-new");
  await page.locator("#q").fill(_TEST_MATERIALS["text_038"]);
  await mode("quick");
  await page.getByRole("button", { name: _TEST_MATERIALS["text_039"], exact: true }).click();
  assert.equal(await page.locator("#q").inputValue(), _TEST_MATERIALS["text_040"]);
  await history();
  assert.equal(await page.locator(_TEST_MATERIALS["text_041"]).count(), 0);
  await screenshot("after-history");
  await page.getByRole("link", { name: _TEST_MATERIALS["text_042"], exact: true }).click();
  await ready();
  assert.equal(await page.locator("#q").inputValue(), _TEST_MATERIALS["text_043"]);
  await mode("quick");
  assert.equal(
    await page.getByLabel(_TEST_MATERIALS["text_044"], { exact: true }).inputValue(),
    "quick"
  );
  await page.reload();
  await ready();
  assert.equal(await page.locator("#q").inputValue(), _TEST_MATERIALS["text_045"]);
  await page.getByRole("button", { name: _TEST_MATERIALS["text_046"], exact: true }).click();
  await ready();
  assert.equal(await page.locator("#q").inputValue(), "");
  assert.equal(await page.locator(".chat-exchange").count(), 0);
  checks.push(_TEST_MATERIALS["text_047"]);
  await page.locator("#q").fill(_TEST_MATERIALS["text_048"]);
  const ime = posts.length;
  await page.locator("#q").evaluate((e) =>
    e.dispatchEvent(
      new KeyboardEvent("keydown", {
        key: "Enter",
        code: "Enter",
        isComposing: true,
        bubbles: true,
      })
    )
  );
  assert.equal(posts.length, ime);
  await page.locator("#q").press("Shift+Enter");
  assert.ok((await page.locator("#q").inputValue()).includes("\n"));
  await mode("quick");
  const beforeQ = count("/query").length,
    beforeR = count("/agent/run").length,
    beforePlan = count("/query/route").length;
  await page.locator("#q").fill(_TEST_MATERIALS["text_049"]);
  await page.locator("#ask").evaluate((e) => {
    e.click();
    e.click();
  });
  await page.waitForURL(/trace=trace-/);
  await ready();
  assert.equal(count("/query").length, beforeQ + 1);
  assert.equal(count("/agent/run").length, beforeR);
  assert.equal(count("/query/route").length, beforePlan + 1);
  assert.equal(await page.locator(".chat-user").count(), 1);
  await send(_TEST_MATERIALS["text_050"]);
  await page.waitForURL(/trace=trace-3$/);
  await ready();
  assert.equal(await page.locator(".chat-user").count(), 2);
  assert.equal(count("/agent/run").length, beforeR);
  assert.equal(last("/query").session_id, undefined);
  checks.push(_TEST_MATERIALS["text_051"]);
  await page.locator("#q").fill(_TEST_MATERIALS["text_052"]);
  await page.locator(_TEST_MATERIALS["text_053"]).first().click();
  await page.locator(_TEST_MATERIALS["text_054"]).waitFor();
  const href = await page
    .getByRole("link", { name: _TEST_MATERIALS["text_055"], exact: true })
    .getAttribute("href");
  assert.ok(href.includes("chunk=chunk-one"));
  assert.ok(href.includes("hash=hash-one"));
  assert.ok(href.includes("#chunk-chunk-one"));
  await screenshot("after-sources");
  await page.keyboard.press("Escape");
  await page.locator("dialog").waitFor({ state: "hidden" });
  assert.equal(await page.locator("#q").inputValue(), _TEST_MATERIALS["text_056"]);
  await mode("synthesis");
  await send(_TEST_MATERIALS["text_057"]);
  await page.waitForURL(/agent\/run-/);
  await ready();
  const synthesis = last("/agent/run");
  assert.equal(synthesis.task_kind, "synthesis");
  assert.equal(await page.locator(".chat-exchange").count(), 3);
  assert.equal(await page.locator(".gap-card").count(), 0);
  assert.equal(await page.locator("dialog[open]").count(), 0);
  const beforeReload = posts.length;
  await page.reload();
  await ready();
  assert.equal(posts.length, beforeReload);
  assert.equal(await page.locator(".chat-user").count(), 3);
  assert.deepEqual(await page.locator(".chat-user p").allTextContents(), [
    _TEST_MATERIALS["text_058"],
    _TEST_MATERIALS["text_059"],
    _TEST_MATERIALS["text_060"],
  ]);
  await screenshot("after-chat");
  checks.push(_TEST_MATERIALS["text_061"]);
  await page
    .locator(".chat-actions")
    .last()
    .getByRole("button", { name: _TEST_MATERIALS["text_062"], exact: true })
    .click();
  await page.getByRole("tab", { name: _TEST_MATERIALS["text_063"], exact: true }).click();
  assert.equal(await page.locator(".gap-card").count(), 1);
  await screenshot("after-details");
  await page.getByRole("tab", { name: _TEST_MATERIALS["text_064"], exact: true }).click();
  await page.locator(".reading-coverage").waitFor();
  await close();
  const followBefore = count("/agent/run").length;
  await mode("auto");
  await send(_TEST_MATERIALS["text_065"]);
  await page.waitForURL(/agent\/run-3$/);
  await ready();
  assert.equal(count("/agent/run").length, followBefore + 1);
  assert.equal(last("/agent/run").session_id, "session-2");
  assert.equal(last("/agent/run").source_run_id, "run-2");
  checks.push(_TEST_MATERIALS["text_066"]);
  await page.locator("#q").fill(_TEST_MATERIALS["text_067"]);
  await history();
  await page.locator(_TEST_MATERIALS["text_068"]).click();
  await ready();
  assert.equal(await page.locator("#q").inputValue(), _TEST_MATERIALS["text_069"]);
  assert.equal(await page.locator(".chat-exchange").count(), 4);
  checks.push(_TEST_MATERIALS["text_070"]);
  await page
    .locator(".chat-actions")
    .last()
    .getByRole("button", { name: _TEST_MATERIALS["text_071"], exact: true })
    .click();
  assert.equal(
    await page.getByLabel(_TEST_MATERIALS["text_072"], { exact: true }).inputValue(),
    "general_explanation"
  );
  const explainingDraft = await page.locator("#q").inputValue();
  await history();
  await page.goBack();
  await ready();
  assert.equal(await page.locator("#q").inputValue(), explainingDraft);
  assert.equal(
    await page.getByLabel(_TEST_MATERIALS["text_073"], { exact: true }).inputValue(),
    "general_explanation"
  );
  await send(_TEST_MATERIALS["text_074"]);
  await page.waitForURL(/agent\/run-4$/);
  await ready();
  assert.equal(last("/agent/run").answer_source, "general_explanation");
  assert.equal(last("/agent/run").source_run_id, "run-3");
  assert.equal(await page.locator(_TEST_MATERIALS["text_075"]).count(), 0);
  await page.reload();
  await ready();
  await page.getByRole("button", { name: _TEST_MATERIALS["text_076"], exact: true }).click();
  assert.equal(
    await page.getByLabel(_TEST_MATERIALS["text_077"], { exact: true }).inputValue(),
    "document_only"
  );
  checks.push(_TEST_MATERIALS["text_078"]);
  await page.getByRole("button", { name: _TEST_MATERIALS["text_079"], exact: true }).click();
  await ready();
  await mode("digest");
  const missing = posts.length;
  await send(_TEST_MATERIALS["text_080"]);
  await page.getByText(_TEST_MATERIALS["text_081"], { exact: true }).waitFor();
  assert.equal(posts.length, missing);
  await page.getByLabel(_TEST_MATERIALS["text_082"], { exact: true }).selectOption("manual-one");
  const digestCount = count("/agent/run").length;
  await send(_TEST_MATERIALS["text_083"]);
  await page.waitForURL(/agent\/run-5$/);
  await ready();
  assert.equal(count("/agent/run").length, digestCount + 1);
  assert.equal(last("/agent/run").target_document_id, "manual-one");
  await page.getByRole("button", { name: _TEST_MATERIALS["text_084"], exact: true }).waitFor();
  assert.equal(await page.locator("dialog[open]").count(), 0);
  await screenshot("after-progress");
  await page.getByRole("button", { name: _TEST_MATERIALS["text_085"], exact: true }).click();
  await page
    .getByRole("button", { name: _TEST_MATERIALS["text_086"], exact: true })
    .waitFor({ state: "detached" });
  checks.push(_TEST_MATERIALS["text_087"]);
  await go("agent/run-one");
  await ready();
  await mode("quick");
  await send(_TEST_MATERIALS["text_088"]);
  await page.waitForURL(/trace=trace-4$/);
  await ready();
  assert.equal(last("/query").source_run_id, undefined);
  assert.equal(last("/query").session_id, undefined);
  await page
    .locator(".chat-actions")
    .last()
    .getByRole("button", { name: _TEST_MATERIALS["text_089"], exact: true })
    .click();
  await send(_TEST_MATERIALS["text_090"]);
  await page.waitForURL(/trace=trace-5$/);
  await ready();
  assert.equal(last("/query").source_trace_id, "trace-4");
  assert.equal(last("/query").answer_source, "general_explanation");
  checks.push(_TEST_MATERIALS["text_091"]);
  // Include duplicates and records beyond a single API page, and old runs without sessions.
  runs.push({ ...original, id: "legacy-run", session_id: null, goal: _TEST_MATERIALS["text_092"] });
  for (let i = 0; i < 105; i++)
    traces.push({
      ...traces[0],
      id: `archived-${i}`,
      trace_id: `archived-${i}`,
      query: i === 104 ? _TEST_MATERIALS["text_093"] : _TEST_MATERIALS["text_094"],
    });
  for (let i = 0; i < 102; i++)
    runs.push({
      ...original,
      id: `old-${i}`,
      session_id: `old-session-${i}`,
      goal: i === 101 ? _TEST_MATERIALS["text_095"] : _TEST_MATERIALS["text_096"],
    });
  await history();
  await page
    .getByLabel(_TEST_MATERIALS["text_097"], { exact: true })
    .fill(_TEST_MATERIALS["text_098"]);
  assert.equal(await page.locator("[data-history-key]").count(), 2);
  await screenshot("after-history-search");
  assert.ok(gets.some((u) => u.includes("/traces?unique=false") && u.includes("offset=100")));
  assert.ok(gets.some((u) => u === "/agent?limit=100&offset=100"));
  await page
    .getByLabel(_TEST_MATERIALS["text_099"], { exact: true })
    .fill(_TEST_MATERIALS["text_100"]);
  assert.equal(await page.locator("[data-history-key]").count(), 104);
  await page
    .getByLabel(_TEST_MATERIALS["text_101"], { exact: true })
    .fill(_TEST_MATERIALS["text_102"]);
  const oldPosts = posts.length;
  await page.getByRole("link", { name: new RegExp(_TEST_MATERIALS["regex_002"], "") }).click();
  await ready();
  assert.equal(await page.locator(_TEST_MATERIALS["text_103"]).count(), 1);
  assert.equal(posts.length, oldPosts);
  await history();
  await page
    .getByLabel(_TEST_MATERIALS["text_104"], { exact: true })
    .fill(_TEST_MATERIALS["text_105"]);
  await screenshot("after-history-filter");
  checks.push(_TEST_MATERIALS["text_106"]);
  const initialDelete = deletes.length;
  page.once("dialog", (d) => d.dismiss());
  await page.locator(_TEST_MATERIALS["text_107"]).click();
  assert.equal(deletes.length, initialDelete);
  page.once("dialog", (d) => d.accept());
  await page.locator(_TEST_MATERIALS["text_108"]).click();
  await page.locator(_TEST_MATERIALS["text_109"]).waitFor({ state: "detached" });
  assert.equal(deletes.at(-1), "/agent/session/session-one");
  await page
    .getByLabel(_TEST_MATERIALS["text_110"], { exact: true })
    .fill(_TEST_MATERIALS["text_111"]);
  page.once("dialog", (d) => d.accept());
  await page.locator(_TEST_MATERIALS["text_112"]).click();
  await page.locator(_TEST_MATERIALS["text_113"]).waitFor({ state: "detached" });
  assert.equal(deletes.at(-1), "/agent/legacy-run");
  await page
    .getByLabel(_TEST_MATERIALS["text_114"], { exact: true })
    .fill(_TEST_MATERIALS["text_115"]);
  page.once("dialog", (d) => d.accept());
  await page.locator(_TEST_MATERIALS["text_116"]).click();
  await page.locator(_TEST_MATERIALS["text_117"]).waitFor({ state: "detached" });
  assert.equal(deletes.at(-1), "/traces/trace-one");
  await page.reload();
  await page.getByText(new RegExp(_TEST_MATERIALS["regex_003"], "")).waitFor();
  assert.equal(await page.locator(_TEST_MATERIALS["text_118"]).count(), 0);
  checks.push(
    "delete confirmation cancellation issues no mutation; query, session and legacy run delete use exact existing identities and survive refresh"
  );
  await go("agent?history=1");
  await page.waitForURL(/\/traces$/);
  await page.getByText(new RegExp(_TEST_MATERIALS["regex_004"], "")).waitFor();
  await go("?trace=trace-2");
  await page.waitForURL(/agent\?trace=trace-2/);
  await ready();
  const restoredPosts = posts.length;
  await go("?run=run-2");
  await page.waitForURL(/agent\/run-2/);
  await ready();
  await go("agent/session/session-2");
  await ready();
  assert.equal(posts.length, restoredPosts);
  checks.push(_TEST_MATERIALS["text_119"]);
  traces.push({
    ...traces[0],
    id: "missing-scope",
    trace_id: "missing-scope",
    scope: null,
    diagnostics: {},
  });
  await go(_TEST_MATERIALS["text_120"]);
  await ready();
  const legacy = posts.length;
  await send(_TEST_MATERIALS["text_121"]);
  await page.getByText(_TEST_MATERIALS["text_122"], { exact: true }).waitFor();
  assert.equal(posts.length, legacy);
  traces.push({
    ...traces[0],
    id: "bad-hash",
    trace_id: "bad-hash",
    citation_integrity: [{ index: 1, chunk_id: "chunk-one", status: "hash_mismatch" }],
  });
  await go(_TEST_MATERIALS["text_123"]);
  await ready();
  await info();
  await page.locator('[data-source-status="hash_mismatch"]').waitFor();
  assert.equal(
    await page.getByRole("link", { name: _TEST_MATERIALS["text_124"], exact: true }).count(),
    0
  );
  await close();
  await go("agent/missing");
  await page.getByText(new RegExp(_TEST_MATERIALS["regex_005"], "")).waitFor();
  assert.equal(await page.locator("#ask").isDisabled(), true);
  checks.push(_TEST_MATERIALS["text_125"]);
  traces.push({
    ...traces[0],
    id: "orphan",
    trace_id: "orphan",
    answer_source: "general_explanation",
    source_trace_id: "missing-source",
    answer: _TEST_MATERIALS["text_126"],
    evidence: [],
    citations: [],
  });
  await go("agent?trace=orphan");
  await page.getByText(_TEST_MATERIALS["text_127"], { exact: true }).waitFor();
  await page.getByText(new RegExp(_TEST_MATERIALS["regex_006"], "")).waitFor();
  assert.equal(await page.locator("#ask").isDisabled(), true);
  await page.evaluate(
    (storageKey) =>
      sessionStorage.setItem(
        storageKey,
        JSON.stringify([
          { kind: "query", id: "missing-source" },
          { kind: "query", id: "trace-2" },
        ])
      ),
    _TEST_MATERIALS["text_128"]
  );
  await go("agent?trace=trace-2");
  await ready();
  await page.getByText(new RegExp(_TEST_MATERIALS["regex_007"], "")).waitFor();
  await page.locator("#q").fill(_TEST_MATERIALS["text_129"]);
  assert.equal(await page.locator("#ask").isDisabled(), false);
  checks.push(_TEST_MATERIALS["text_130"]);
  await go("agent");
  await ready();
  routeFault = true;
  const executor = count("/query").length + count("/agent/run").length;
  await send(_TEST_MATERIALS["text_131"]);
  await page.getByText(_TEST_MATERIALS["text_132"], { exact: true }).waitFor();
  assert.equal(count("/query").length + count("/agent/run").length, executor);
  routeFault = false;
  await page.getByRole("button", { name: _TEST_MATERIALS["text_133"], exact: true }).click();
  await ready();
  await mode("synthesis");
  dropNextAgentResponse = true;
  const actualRuns = runs.length;
  await send(_TEST_MATERIALS["text_134"]);
  await page.waitForURL(/agent\/run-/);
  await ready();
  assert.equal(runs.length, actualRuns + 1);
  const attempts = count("/agent/run").slice(-2);
  assert.equal(attempts[0].key, attempts[1].key);
  checks.push(_TEST_MATERIALS["text_135"]);
  await page.getByRole("button", { name: _TEST_MATERIALS["text_136"], exact: true }).click();
  await ready();
  const autoQuick = count("/query").length,
    autoAgent = count("/agent/run").length;
  const nextTrace = `trace-${traces.length + 1}`;
  await send(_TEST_MATERIALS["text_137"]);
  await page.waitForURL(new RegExp(`trace=${nextTrace}$`));
  await ready();
  assert.equal(last("/query/route").analysis_mode, "auto");
  assert.equal(count("/query").length, autoQuick + 1);
  assert.equal(count("/agent/run").length, autoAgent);
  await page.getByRole("button", { name: _TEST_MATERIALS["text_138"], exact: true }).click();
  await ready();
  const autoSynth = `run-${runs.length + 1}`;
  await send(_TEST_MATERIALS["text_139"]);
  await page.waitForURL(new RegExp(`agent/${autoSynth}$`));
  await ready();
  assert.equal(last("/agent/run").analysis_mode, "auto");
  assert.equal(last("/agent/run").task_kind, "synthesis");
  await page.getByRole("button", { name: _TEST_MATERIALS["text_140"], exact: true }).click();
  await ready();
  const noTarget = count("/agent/run").length;
  await send(_TEST_MATERIALS["text_141"]);
  await page.getByLabel(_TEST_MATERIALS["text_142"], { exact: true }).waitFor();
  assert.equal(count("/agent/run").length, noTarget);
  await page.getByLabel(_TEST_MATERIALS["text_143"], { exact: true }).selectOption("manual-one");
  const autoDigest = `run-${runs.length + 1}`;
  await page.locator("#ask").click();
  await page.waitForURL(new RegExp(`agent/${autoDigest}$`));
  await ready();
  assert.equal(count("/agent/run").length, noTarget + 1);
  assert.equal(last("/agent/run").task_kind, "digest");
  checks.push(_TEST_MATERIALS["text_144"]);
  await go(_TEST_MATERIALS["text_145"]);
  await ready();
  const scopedRun = `run-${runs.length + 1}`;
  await page.locator("#ask").click();
  await page.waitForURL(new RegExp(`agent/${scopedRun}$`));
  await ready();
  assert.deepEqual(last("/agent/run").collection_ids, ["lib-one", "lib-two"]);
  assert.equal(last("/agent/run").scope_mode, "hard");
  await mode("quick");
  const scopedTrace = `trace-${traces.length + 1}`;
  await send(_TEST_MATERIALS["text_146"]);
  await page.waitForURL(new RegExp(`trace=${scopedTrace}$`));
  await ready();
  assert.deepEqual(last("/query").collection_ids, ["lib-one", "lib-two"]);
  assert.equal(last("/query").scope_mode, "hard");
  await page.reload();
  await ready();
  assert.ok(
    (await page.locator(".ask-scope-summary").textContent()).includes(_TEST_MATERIALS["text_147"])
  );
  checks.push(_TEST_MATERIALS["text_148"]);
  await page.getByRole("button", { name: _TEST_MATERIALS["text_149"], exact: true }).click();
  await ready();
  const slowTrace = "trace-" + (traces.length + 1);
  await send(_TEST_MATERIALS["text_150"]);
  await page.locator(_TEST_MATERIALS["text_151"]).waitFor();
  assert.equal(await page.locator("#q").inputValue(), "");
  await page.locator("#q").fill(_TEST_MATERIALS["text_152"]);
  await page.waitForURL((url) => url.searchParams.get("trace") === slowTrace);
  await ready();
  assert.equal(await page.locator("#q").inputValue(), _TEST_MATERIALS["text_153"]);
  assert.equal(await page.locator(".chat-pending").count(), 0);
  assert.equal(await page.locator(".chat-user").count(), 1);
  checks.push(_TEST_MATERIALS["text_154"]);
  // Use a clean surviving record for responsive and keyboard tests.

  for (const width of [320, 390, 768, 1024, 1440]) {
    await page.setViewportSize({ width, height: 900 });
    await go("agent?trace=trace-2");
    await ready();
    assert.equal(
      await page.evaluate(() => document.documentElement.scrollWidth > innerWidth),
      false,
      `overflow ${width}`
    );
    const dock = await page.locator(".chat-input-dock").boundingBox();
    assert.ok(dock.y + dock.height <= 902, `dock bottom ${width}`);
    const input = await page.locator("#q").boundingBox();
    assert.ok(input.y >= 0 && input.y + input.height <= 900, `visible input ${width}`);
    await page.locator("#q").fill(`${_TEST_MATERIALS["text_155"]}${width}`);
    await page.locator(_TEST_MATERIALS["text_156"]).first().click();
    assert.equal(await page.locator("dialog").evaluate((e) => e.open), true);
    await page.keyboard.press("Escape");
    assert.equal(await page.locator("#q").inputValue(), `${_TEST_MATERIALS["text_157"]}${width}`);
    if (width === 390) await screenshot("after-mobile-chat");
    if (width === 1440) await screenshot("after-desktop-chat");
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await history();
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  await screenshot("after-mobile-history");
  checks.push(_TEST_MATERIALS["text_158"]);
  assert.deepEqual(errors, []);
  assert.ok(
    external.every(
      (url) => url.includes("fonts.googleapis.com") || url.includes("fonts.gstatic.com")
    )
  );
  await fs.writeFile(
    path.join(dir, "results.json"),
    JSON.stringify({ checks, errors, external, posts, deletes, hashes }, null, 2)
  );
  console.log(`PASS chat/history: ${checks.length}${_TEST_MATERIALS["text_159"]}`);
} catch (err) {
  await page.screenshot({ path: path.join(dir, "failure.png"), fullPage: true });
  console.error(checks);
  throw err;
} finally {
  await browser.close();
  await new Promise((resolve) => server.close(resolve));
}
