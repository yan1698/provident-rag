import removeMd from "remove-markdown";

export const CITATION_RE = /\[S(\d+)\]/g;

export const STATUS_LABEL = {
  supported: "有依据",
  unsupported: "依据不足",
  unverified: "未核实",
  uncited: "未标注出处",
  invalid_citation: "引用无效",
  insufficient: "证据不足",
  meta: "说明",
};

export const GATE_LABEL = {
  avg_coverage_rate: "有依据",
  avg_hallucination_rate: "依据不足",
  avg_recall_at_k: "召回",
  mrr: "排序质量",
  refusal_ok_rate: "拒答正确",
  term_hit_rate: "术语命中",
  off_corpus_documents: "语料外文档",
  off_corpus_evidence_rate: "语料外证据",
  forbidden_documents: "测试夹具",
};

export const AGENT_TEMPLATES = [
  "根据知识库整理关于……的要点",
  "对比知识库中……与……",
  "知识库能覆盖……的哪些部分",
];

export const AGENT_STEP_LABEL = {
  general_explanation: "通用知识解释",
  analyze: "理解任务",
  survey: "扫描目录",
  decompose: "规划子问题",
  digest: "按篇阅读",
  retrieve: "检索",
  graph_walk: "图谱维度",
  compose: "起草长回答",
  answer: "撰写",
  verify: "核对",
  finish: "交付",
  refuse: "拒绝",
  error: "异常",
};

export const TASK_KIND_LABEL = {
  factoid: "短问答",
  synthesis: "综合检索",
  digest: "按篇阅读",
};

export const PURPOSE_LABEL = {
  planner: "规划",
  analyze: "理解任务",
  classify: "分类",
  rewrite: "改写",
  digest: "阅读",
  compose: "起草",
  answer: "撰写",
  extract: "抽取",
};

export const INTENT_LABEL = {
  explain: "说明",
  compare: "对比",
  plan: "规划",
  inventory: "盘点",
};

export function fmtTime(iso) {
  try {
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return String(iso || "");
    return date.toLocaleString("zh-CN", { hour12: false });
  } catch {
    return String(iso || "");
  }
}

export function pct(value) {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(Number(value) * 100).toFixed(0)}%`;
}

export function num(value, digits = 3) {
  if (value == null || Number.isNaN(value)) return "—";
  return Number(value).toFixed(digits);
}

export function looksLikeRefusal(answer) {
  return /现有证据不足以回答/.test(answer || "");
}

export function srcTone(index) {
  const n = Number(index);
  return ((Number.isFinite(n) ? n - 1 : 0) % 8) + 1;
}

export function fileKind(item) {
  const name = String(item.filename || "").toLowerCase();
  const type = String(item.content_type || "").toLowerCase();
  if (name.endsWith(".pdf") || type.includes("pdf")) return "PDF";
  if (name.endsWith(".md") || name.endsWith(".markdown") || type.includes("markdown")) return "MD";
  return "TXT";
}

export function fileMeta(item) {
  const ready = item.embedded_count >= item.chunk_count && item.chunk_count > 0;
  return `${item.chunk_count} 片段 · 向量 ${item.embedded_count}/${item.chunk_count}${ready ? " · 可检索" : ""}`;
}

export function corpusLine(stats) {
  if (!stats) return "正在读取知识库…";
  const docs = stats.documents ?? stats.count ?? 0;
  const chunks = stats.chunks ?? 0;
  const embedded = stats.embedded_chunks ?? 0;
  if (!docs) return "知识库还是空的。先上传 PDF / Markdown / TXT，再提问。";
  const ready = chunks && embedded >= chunks;
  return ready
    ? `当前知识库：${docs} 份文档，${chunks} 个片段，已可检索。`
    : `当前知识库：${docs} 份文档，${chunks} 个片段，其中 ${embedded} 个已向量化。未向量化的部分主要靠关键词检索。`;
}

export function citedIndices(data) {
  const indices = new Set();
  for (const c of data.citations || []) {
    if (c && c.index != null) indices.add(Number(c.index));
  }
  const modules = (data.outline && data.outline.modules) || data.modules || [];
  for (const mod of modules) {
    for (const item of mod.paragraphs || mod.bullets || []) {
      for (const n of item.sources || []) indices.add(Number(n));
    }
  }
  if (!indices.size && data.answer) {
    for (const match of String(data.answer).matchAll(CITATION_RE)) {
      indices.add(Number(match[1]));
    }
  }
  return indices;
}

export function citedEvidence(data) {
  const indices = citedIndices(data);
  const all = (data.outline && data.outline.evidence) || data.evidence || [];
  if (!indices.size) return [];
  return all.filter((item) => indices.has(Number(item.index)));
}

export function stripMd(text) {
  const parked = String(text || "").replace(/\[S(\d+)\]/g, (_, n) => `@@CITE${n}@@`);
  const out = removeMd(parked, { stripListLeaders: true, gfm: true, useImgAltText: true });
  return String(out || "")
    .replace(/@@CITE(\d+)@@/g, "[S$1]")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function usageTokens(item) {
  return (Number(item && item.prompt_tokens) || 0) + (Number(item && item.completion_tokens) || 0);
}

function usageByPurpose(items) {
  const totals = new Map();
  for (const item of items || []) {
    const n = usageTokens(item);
    const key = String((item && item.purpose) || "");
    totals.set(key, (totals.get(key) || 0) + n);
  }
  return [...totals.entries()].map(([purpose, tokens]) => ({
    purpose,
    label: PURPOSE_LABEL[purpose] || purpose || "模型",
    tokens,
  }));
}

export function usageLine(items) {
  return usageByPurpose(items)
    .map((item) => `${item.label} ${item.tokens}`)
    .join(" · ");
}

export function usageNote(items, { recorded = true } = {}) {
  if (!recorded || items == null) return "本次无用量记录";
  const line = usageLine(items);
  if (line) return line;
  return "未调用生成模型";
}

export function stepUsageLine(step) {
  return usageByPurpose((step && step.usage) || [])
    .map((item) => `${item.label} · ${item.tokens}`)
    .join(" · ");
}

export function runUsageLine(steps) {
  const items = [];
  for (const step of steps || []) {
    items.push(...((step && step.usage) || []));
  }
  return usageLine(items);
}

export function stepsUsageNote(steps) {
  const recorded = (steps || []).some((step) => Array.isArray(step && step.usage));
  const items = [];
  for (const step of steps || []) {
    items.push(...((step && step.usage) || []));
  }
  return usageNote(items, { recorded });
}

export function liveNodes(steps) {
  let nodes = [];
  for (const step of steps || []) {
    const obs = step.observation || {};
    if (step.action === "decompose" && Array.isArray(obs.nodes)) {
      nodes = obs.nodes.map((node) => ({
        id: node.id,
        title: node.question || node.title || "",
        status: node.status || "pending",
      }));
    }
    const nodeId = step.node_id || obs.node_id;
    if (!nodeId || (step.action !== "retrieve" && step.action !== "digest")) continue;
    const hits = obs.fresh_count ?? obs.count ?? obs.bullet_count;
    const empty =
      obs.status === "empty" ||
      (typeof hits === "number" && hits <= 0 && step.action === "retrieve");
    nodes = nodes.map((node) => {
      if (node.id !== nodeId) return node;
      return {
        ...node,
        status: empty ? "empty" : "retrieved",
        hits: typeof hits === "number" ? hits : node.hits,
      };
    });
  }
  return nodes;
}

export const NODE_STATUS_LABEL = {
  pending: "等待",
  retrieved: "已找到",
  empty: "未命中",
};

export function stepDetail(step) {
  if (step.action === "general_explanation") return "基于模型知识解释，未检索文档，未经文档核验。";
  const obs = step.observation || {};
  if (step.action === "analyze") {
    const qs = (obs.questions || []).join("；");
    return `${INTENT_LABEL[obs.intent] || obs.intent || "任务"}${qs ? ` · ${qs}` : ""}`;
  }
  if (step.action === "survey") {
    const docs = (obs.documents || []).map((d) => d.filename).filter(Boolean);
    return `${docs.join("、") || "当前知识库"} · ${obs.section_count ?? 0} 个章节${obs.truncated ? "（已截断）" : ""}`;
  }
  if (step.action === "decompose") {
    const nodes = (obs.nodes || []).map((n) => n.question || n.title).filter(Boolean);
    return nodes.length ? `子问题：${nodes.join("、")}` : "未规划子问题";
  }
  if (step.action === "digest") {
    const name = obs.filename || "";
    const n = obs.bullet_count ?? 0;
    const mode =
      obs.mode === "deepseek" ? "DeepSeek" : obs.mode === "extractive" ? "抽取" : obs.mode || "";
    return `${name || "文档"} · ${n} 条要点${mode ? ` · ${mode}` : ""}${obs.done ? " · 读完" : ""}`;
  }
  if (step.action === "retrieve") {
    const snippet = ((obs.hits || [])[0] || {}).snippet || "";
    const n = obs.fresh_count ?? obs.count ?? 0;
    const extra = obs.scope_note || obs.note ? ` · ${obs.scope_note || obs.note}` : "";
    return `${obs.query || step.input || ""} · 命中 ${n}${snippet ? ` · ${snippet.slice(0, 80)}` : ""}${extra}`;
  }
  if (step.action === "compose") {
    const mode = obs.compose_mode === "written" ? "长回答" : "证据摘要";
    const extra = obs.note ? ` · ${obs.note}` : "";
    return `起草 ${(obs.modules || []).length} 段（${mode}）${extra}`;
  }
  if (step.action === "verify") {
    const cov = obs.module_coverage_rate ?? obs.coverage_rate;
    const task = obs.task_coverage;
    return `依据覆盖 ${pct(cov)}${task != null && task !== "" ? ` · 任务覆盖 ${pct(task)}` : ""}`;
  }
  if (step.action === "finish" || step.action === "refuse") {
    if (obs.partial) return "部分交付（已覆盖部分 + 缺口）";
    return obs.status === "refused" || step.action === "refuse" ? "整题拒绝" : "交付结果";
  }
  return step.thought || "";
}
