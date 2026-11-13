import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api";
import { IcoFolder } from "../icons";
import AgentMenu from "../components/AgentMenu";
import { sourceHref } from "../sourceLocation";

const KIND_META = [
  { kind: "document", label: "文档", fill: "#8b6cf6" },
  { kind: "doc_section", label: "章节", fill: "#eef0ff" },
  { kind: "concept", label: "概念", fill: "#e5f8f2" },
  { kind: "method", label: "方法", fill: "#fff3dd" },
  { kind: "metric", label: "指标", fill: "#ffe9ef" },
  { kind: "tool", label: "工具", fill: "#f1ebff" },
  { kind: "role", label: "角色", fill: "#ffeaf6" },
];

const REL_META = [
  { relation: "part_of", label: "从属", stroke: "#94a3b8" },
  { relation: "related_to", label: "相关", stroke: "#94a3b8" },
  { relation: "defines", label: "定义", stroke: "#3b82f6" },
  { relation: "requires", label: "前置依赖", stroke: "#f59e0b" },
  { relation: "causes", label: "因果", stroke: "#ec4899" },
  { relation: "contrasts_with", label: "对照", stroke: "#ec4899" },
  { relation: "measured_by", label: "度量", stroke: "#22c55e" },
  { relation: "instance_of", label: "实例", stroke: "#a855f7" },
];

const EXTRACT_KINDS = new Set(["concept", "method", "metric", "tool", "role"]);
const KIND_FILL = Object.fromEntries(KIND_META.map((item) => [item.kind, item.fill]));
const REL_STROKE = Object.fromEntries(REL_META.map((item) => [item.relation, item.stroke]));
const REL_LABEL = Object.fromEntries(REL_META.map((item) => [item.relation, item.label]));
const DEFAULT_TREE_DEPTH = 1;
const NODE_MAX_W = 168;
const NODE_MIN_W = 92;
const NODE_H = 36;
const COL_GUTTER = 88;
const COL_STEP = NODE_MAX_W + COL_GUTTER;
const ROW_STEP = 62;

function nodeFill(kind) {
  return KIND_FILL[kind] || "#64748b";
}

function nodeTextFill(kind) {
  return kind === "document" ? "#ffffff" : "#36415f";
}

function edgeStroke(relation) {
  return REL_STROKE[relation] || "#94a3b8";
}

function isWideChar(ch) {
  const code = ch.codePointAt(0) || 0;
  return code > 0x2e7f;
}

function textWidth(value) {
  let width = 0;
  for (const ch of String(value || "")) width += isWideChar(ch) ? 13 : 7.2;
  return width;
}

function measureNode(name) {
  const title = String(name || "");
  const pad = 28;
  const budget = NODE_MAX_W - pad;
  if (textWidth(title) <= budget) {
    return {
      label: title,
      title,
      width: Math.min(NODE_MAX_W, Math.max(NODE_MIN_W, pad + textWidth(title))),
      height: NODE_H,
    };
  }
  let used = textWidth("…");
  let cut = 0;
  for (const ch of title) {
    const add = isWideChar(ch) ? 13 : 7.2;
    if (used + add > budget) break;
    used += add;
    cut += 1;
  }
  return {
    label: `${title.slice(0, Math.max(1, cut))}…`,
    title,
    width: NODE_MAX_W,
    height: NODE_H,
  };
}

function buildPartOfTree(entities, edges) {
  const ids = new Set(entities.map((item) => item.id));
  const children = {};
  const parent = {};
  for (const edge of edges) {
    if (edge.relation !== "part_of") continue;
    if (!ids.has(edge.src_id) || !ids.has(edge.dst_id)) continue;
    parent[edge.src_id] = edge.dst_id;
    if (!children[edge.dst_id]) children[edge.dst_id] = [];
    if (!children[edge.dst_id].includes(edge.src_id)) children[edge.dst_id].push(edge.src_id);
  }
  return { children, parent };
}

function kindRank(kind) {
  if (kind === "document") return 0;
  if (kind === "doc_section") return 1;
  return 2;
}

function entityDocPos(entity) {
  if (entity.doc_pos) {
    const pos = entity.doc_pos;
    return [pos.page ?? 0, pos.offset ?? 0, pos.start_char ?? 0];
  }
  const mention = (entity.mentions || [])[0];
  return [0, 0, mention?.start_char ?? 9_999_999];
}

function compareEntities(a, b) {
  const kindDelta = kindRank(a.kind) - kindRank(b.kind);
  if (kindDelta) return kindDelta;
  const pa = entityDocPos(a);
  const pb = entityDocPos(b);
  for (let index = 0; index < 3; index += 1) {
    if (pa[index] !== pb[index]) return pa[index] - pb[index];
  }
  return a.name.localeCompare(b.name, "zh");
}

function layoutForest(entities, edges, { expanded = {}, maxDepth = DEFAULT_TREE_DEPTH } = {}) {
  const { children, parent } = buildPartOfTree(entities, edges);
  const byId = Object.fromEntries(entities.map((item) => [item.id, item]));
  const roots = entities.filter((item) => !parent[item.id]).sort(compareEntities);

  const col = {};
  const visit = (id, depth, seen) => {
    if (seen.has(id)) return;
    seen.add(id);
    col[id] = Math.max(col[id] || 0, depth);
    for (const child of children[id] || []) visit(child, depth + 1, seen);
  };
  for (const root of roots) visit(root.id, 0, new Set());
  for (const item of entities) {
    if (col[item.id] == null) col[item.id] = 0;
  }

  for (const id of Object.keys(children)) {
    children[id].sort((left, right) => compareEntities(byId[left], byId[right]));
  }

  const hiddenKids = {};
  const visible = new Set();
  const walkVisible = (id, depth) => {
    visible.add(id);
    const kids = children[id] || [];
    const open = depth < maxDepth || Boolean(expanded[id]);
    if (!open && kids.length) {
      hiddenKids[id] = kids.length;
      return;
    }
    for (const child of kids) walkVisible(child, depth + 1);
  };
  for (const root of roots) walkVisible(root.id, 0);

  const columns = {};
  for (const item of entities) {
    if (!visible.has(item.id)) continue;
    const depth = col[item.id];
    if (!columns[depth]) columns[depth] = [];
    columns[depth].push(item);
  }
  const positioned = {};
  let maxX = 160;
  let maxY = 80;
  const depths = Object.keys(columns)
    .map(Number)
    .sort((a, b) => a - b);
  for (const depth of depths) {
    const rows = columns[depth].sort((a, b) => compareEntities(a, b));
    rows.forEach((item, index) => {
      const box = measureNode(item.name);
      const x = 36 + depth * COL_STEP;
      const y = 28 + index * ROW_STEP;
      positioned[item.id] = { ...item, ...box, x, y, hiddenKids: hiddenKids[item.id] || 0 };
      maxX = Math.max(maxX, x + box.width + 48);
      maxY = Math.max(maxY, y + box.height + 48);
    });
  }
  const leftover = entities.filter((item) => !positioned[item.id] && !parent[item.id]);
  leftover.forEach((item) => {
    const linked = edges.find((edge) => edge.src_id === item.id || edge.dst_id === item.id);
    const anchorId = linked ? (linked.src_id === item.id ? linked.dst_id : linked.src_id) : "";
    const anchor = positioned[anchorId];
    const box = measureNode(item.name);
    const depth = anchor
      ? Math.round((anchor.x - 36) / COL_STEP) + 1
      : (depths[depths.length - 1] || 0) + 1;
    const x = 36 + depth * COL_STEP;
    const columnCount = Object.values(positioned).filter((node) => node.x === x).length;
    const y = 28 + columnCount * ROW_STEP;
    positioned[item.id] = { ...item, ...box, x, y, hiddenKids: 0 };
    maxX = Math.max(maxX, x + box.width + 48);
    maxY = Math.max(maxY, y + box.height + 48);
  });
  return { nodes: positioned, width: maxX, height: maxY, hiddenKids };
}

function layoutRadial(entities, edges, seedId) {
  const ids = new Set(entities.map((item) => item.id));
  const adj = {};
  for (const edge of edges) {
    if (!ids.has(edge.src_id) || !ids.has(edge.dst_id)) continue;
    if (!adj[edge.src_id]) adj[edge.src_id] = [];
    if (!adj[edge.dst_id]) adj[edge.dst_id] = [];
    adj[edge.src_id].push(edge.dst_id);
    adj[edge.dst_id].push(edge.src_id);
  }
  const hop = {};
  const queue = [];
  const origin = ids.has(seedId) ? seedId : entities[0]?.id;
  if (origin) {
    hop[origin] = 0;
    queue.push(origin);
  }
  while (queue.length) {
    const cur = queue.shift();
    for (const next of adj[cur] || []) {
      if (hop[next] != null) continue;
      hop[next] = hop[cur] + 1;
      queue.push(next);
    }
  }
  const rings = {};
  for (const item of entities) {
    const d = hop[item.id] == null ? 3 : hop[item.id];
    if (!rings[d]) rings[d] = [];
    rings[d].push(item);
  }
  const cx = 420;
  const cy = 340;
  const positioned = {};
  let maxX = 840;
  let maxY = 680;
  Object.keys(rings)
    .map(Number)
    .sort((a, b) => a - b)
    .forEach((d) => {
      const rows = rings[d];
      if (d === 0 && rows.length === 1) {
        const item = rows[0];
        const box = measureNode(item.name);
        positioned[item.id] = {
          ...item,
          ...box,
          x: cx - box.width / 2,
          y: cy - box.height / 2,
          hiddenKids: 0,
        };
        return;
      }
      const radius = 90 + d * 190;
      rows.forEach((item, index) => {
        const box = measureNode(item.name);
        const angle = (Math.PI * 2 * index) / Math.max(rows.length, 1) - Math.PI / 2;
        const x = cx + Math.cos(angle) * radius - box.width / 2;
        const y = cy + Math.sin(angle) * radius - box.height / 2;
        positioned[item.id] = { ...item, ...box, x, y, hiddenKids: 0 };
        maxX = Math.max(maxX, x + box.width + 48);
        maxY = Math.max(maxY, y + box.height + 48);
      });
    });
  return {
    nodes: positioned,
    width: Math.max(maxX, 840),
    height: Math.max(maxY, 680),
    hiddenKids: {},
    byId: {},
  };
}

function SchemaCanvas({ entities, edges, selectedId, onSelect, mode, seedId, expanded }) {
  const layout = useMemo(() => {
    if (mode === "radial") return layoutRadial(entities, edges, seedId);
    return layoutForest(entities, edges, { expanded });
  }, [entities, edges, mode, seedId, expanded]);
  const { nodes, width, height } = layout;
  const placed = Object.values(nodes);
  if (!placed.length) {
    return (
      <p className="lede">
        {mode === "radial" ? "这个概念还没有可展开的邻居。" : "这篇还没有整理出章节关系。"}
      </p>
    );
  }
  return (
    <div className="kg-canvas" id="kg-canvas">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width={width}
        height={height}
        role="img"
        aria-label="知识图谱"
      >
        <defs>
          {REL_META.map((item) => (
            <marker
              key={item.relation}
              id={`kg-arrow-${item.relation}`}
              markerWidth="8"
              markerHeight="8"
              refX="7"
              refY="4"
              orient="auto"
            >
              <path d="M0,0.8 L7.2,4 L0,7.2 Z" fill={item.stroke} />
            </marker>
          ))}
        </defs>
        {edges.map((edge) => {
          const from = nodes[edge.src_id];
          const to = nodes[edge.dst_id];
          const parent = nodes[edge.dst_id];
          const child = nodes[edge.src_id];
          const start = edge.relation === "part_of" ? parent : from;
          const end = edge.relation === "part_of" ? child : to;
          if (!start || !end) return null;
          const x1 = start.x + start.width;
          const y1 = start.y + start.height / 2;
          const x2 = end.x;
          const y2 = end.y + end.height / 2;
          const mx = (x1 + x2) / 2;
          const my = (y1 + y2) / 2;
          const color = edgeStroke(edge.relation);
          const rel = String(edge.relation || "related_to");
          const markerId = REL_STROKE[rel] ? `kg-arrow-${rel}` : "kg-arrow-related_to";
          const d = `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
          const label = REL_LABEL[rel] || rel;
          const boxW = Math.max(56, 10 + label.length * 12);
          return (
            <g key={edge.id} className="kg-edge">
              <path
                d={d}
                fill="none"
                stroke={color}
                strokeWidth="1.7"
                markerEnd={`url(#${markerId})`}
              />
              <rect
                x={mx - boxW / 2}
                y={my - 10}
                width={boxW}
                height="18"
                rx="4"
                fill="rgba(255,255,255,0.92)"
                stroke="rgba(125,132,170,0.14)"
              />
              <text x={mx} y={my + 4} textAnchor="middle" className="kg-edge-label">
                {label}
              </text>
            </g>
          );
        })}
        {placed.map((node) => (
          <g
            key={node.id}
            className={node.id === selectedId ? "kg-node on" : "kg-node"}
            transform={`translate(${node.x} ${node.y})`}
            onClick={() => onSelect(node.id, node.kind, node.hiddenKids)}
            style={{ cursor: "pointer" }}
          >
            <title>{node.title || node.name}</title>
            <svg className="kg-node-clip" width={node.width} height={node.height}>
              <defs>
                <clipPath id={`kg-clip-${node.id}`}>
                  <rect x="12" y="0" width={Math.max(8, node.width - 24)} height={node.height} />
                </clipPath>
              </defs>
              <rect
                width={node.width}
                height={node.height}
                rx={node.height / 2}
                ry={node.height / 2}
                fill={nodeFill(node.kind)}
                stroke={node.id === selectedId ? "#7c5ce4" : "rgba(118,126,169,0.22)"}
                strokeWidth={node.id === selectedId ? 2.2 : 1}
              />
              <text
                x={node.width / 2}
                y={node.height / 2 + 4}
                textAnchor="middle"
                className="kg-node-label"
                fill={nodeTextFill(node.kind)}
                clipPath={`url(#kg-clip-${node.id})`}
              >
                {node.label}
              </text>
              {node.hiddenKids ? (
                <text x={node.width - 10} y="12" textAnchor="middle" className="kg-node-badge">
                  +{node.hiddenKids}
                </text>
              ) : null}
            </svg>
          </g>
        ))}
      </svg>
    </div>
  );
}

const WORK_KEY = "provident.graph.workbench";

function readWorkbench() {
  try {
    return localStorage.getItem(WORK_KEY) === "1";
  } catch {
    return false;
  }
}

function fmtTokens(n) {
  return Number(n || 0).toLocaleString("zh-CN");
}

export default function GraphPage() {
  const [params, setParams] = useSearchParams();
  const wanted = params.get("doc") || "";
  const entityParam = params.get("entity") || "";
  const nameParam = params.get("name") || "";
  const [docs, setDocs] = useState([]);
  const [catalog, setCatalog] = useState([]);
  const [status, setStatus] = useState(null);
  const [payload, setPayload] = useState(null);
  const [selected, setSelected] = useState("");
  const [picked, setPicked] = useState({});
  const [workbench, setWorkbench] = useState(readWorkbench);
  const [expanded, setExpanded] = useState({});
  const [error, setError] = useState("");
  const [jobKind, setJobKind] = useState("");
  const [libraries, setLibraries] = useState([]);
  const [collectionId, setCollectionId] = useState("");
  const [staleGraph, setStaleGraph] = useState(false);
  const collectionRef = useRef("");

  const refreshStatus = async () => {
    try {
      const info = await api("/graph");
      setStatus(info);
    } catch (err) {
      setError(err.message);
    }
  };

  const refreshCatalog = async () => {
    try {
      const id = collectionRef.current;
      const query = id ? `?collection_id=${encodeURIComponent(id)}` : "";
      const data = await api(`/graph/catalog${query}`);
      setCatalog(data.documents || []);
      setStaleGraph(Boolean(data.stale));
    } catch {
      /* catalog is optional until the graph tables exist */
    }
  };

  const setWorkbenchOn = (on) => {
    setWorkbench(on);
    try {
      localStorage.setItem(WORK_KEY, on ? "1" : "0");
    } catch {
      /* ignore */
    }
  };

  useEffect(() => {
    document.body.classList.add("kg-light-mode");
    refreshStatus();
    refreshCatalog();
    api("/library")
      .then((data) => setLibraries(data.collections || []))
      .catch(() => {});
    api("/documents")
      .then((listing) => {
        const rows = listing.results || [];
        setDocs(rows);
      })
      .catch((err) => setError(err.message));
    const timer = setInterval(() => {
      refreshStatus();
      refreshCatalog();
    }, 2000);
    return () => {
      clearInterval(timer);
      document.body.classList.remove("kg-light-mode");
    };
  }, []);

  useEffect(() => {
    if (wanted || entityParam || nameParam || !docs.length) return;
    pickDoc(String(docs[0].doc_id));
  }, [docs, wanted, entityParam, nameParam]);

  useEffect(() => {
    if (entityParam || nameParam) {
      const query = entityParam
        ? `entity_id=${encodeURIComponent(entityParam)}`
        : `name=${encodeURIComponent(nameParam)}`;
      let cancelled = false;
      api(`/graph/neighborhood?${query}&hops=2`)
        .then((data) => {
          if (cancelled) return;
          setPayload(data);
          setSelected(data.seed_id || "");
          setExpanded({});
          setError("");
        })
        .catch((err) => {
          if (!cancelled) {
            setPayload(null);
            setError(err.message);
          }
        });
      return () => {
        cancelled = true;
      };
    }
    if (!wanted) {
      setPayload(null);
      return undefined;
    }
    let cancelled = false;
    api(`/graph/documents/${wanted}`)
      .then((data) => {
        if (!cancelled) {
          setPayload(data);
          setSelected("");
          setExpanded({});
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setPayload(null);
          setError(err.message);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [
    wanted,
    entityParam,
    nameParam,
    status && status.pending_docs,
    status && status.entities,
    status && status.concepts,
  ]);

  const catalogById = Object.fromEntries(catalog.map((item) => [String(item.doc_id), item]));
  const pickedIds = Object.keys(picked).filter((id) => picked[id]);
  const selectedSkeletonIds = pickedIds.filter((id) => {
    const row = catalogById[id];
    return row && !row.skeleton;
  });
  const selectedExtractIds = pickedIds.filter((id) => {
    const row = catalogById[id];
    return row && !row.extracted && Number(row.calls || 0) > 0;
  });
  const estimate = selectedExtractIds.reduce(
    (acc, id) => {
      const row = catalogById[id];
      if (!row) return acc;
      acc.calls += Number(row.calls || 0);
      acc.input += Number(row.input_tokens || 0);
      acc.output += Number(row.output_tokens || 0);
      acc.total += Number(row.total_tokens || 0);
      return acc;
    },
    { calls: 0, input: 0, output: 0, total: 0 }
  );

  const postGraphJob = async (extract, docIds) => {
    const info = await api("/graph", {
      method: "POST",
      body: { extract, doc_ids: docIds },
    });
    setStatus(info);
    setPicked({});
    await refreshCatalog();
  };

  const runSkeleton = async () => {
    if (!workbench || !selectedSkeletonIds.length || graphRunning) return;
    const ok = window.confirm(
      `将为所选 ${selectedSkeletonIds.length} 篇生成章节结构（不调模型，0 token）。\nMarkdown 有 # 标题时会得到完整章节树；TXT/PDF 可能只有文档节点。确认继续？`
    );
    if (!ok) return;
    setJobKind("skeleton");
    try {
      await postGraphJob(false, selectedSkeletonIds);
      setError("");
    } catch (err) {
      setStatus((cur) => ({ ...(cur || {}), error: err.message }));
    } finally {
      setJobKind("");
    }
  };

  const runExtract = async () => {
    if (!workbench || !selectedExtractIds.length || graphRunning) return;
    const ok = window.confirm(
      `将只抽取所选 ${selectedExtractIds.length} 篇（已形成的图谱不会重抽）。\n约 ${estimate.calls} 次调用，预计消耗 ${fmtTokens(estimate.total)} token（输入约 ${fmtTokens(estimate.input)} + 输出上限 ${fmtTokens(estimate.output)}）。确认继续？`
    );
    if (!ok) return;
    setJobKind("extract");
    try {
      await postGraphJob(true, selectedExtractIds);
      setError("");
    } catch (err) {
      setStatus((cur) => ({ ...(cur || {}), error: err.message }));
    } finally {
      setJobKind("");
    }
  };

  const pickDoc = (docId) => {
    const next = new URLSearchParams(params);
    if (docId) next.set("doc", docId);
    else next.delete("doc");
    next.delete("entity");
    next.delete("name");
    setParams(next, { replace: true });
  };

  const pickEntity = (entityId) => {
    const next = new URLSearchParams(params);
    if (entityId) next.set("entity", entityId);
    else next.delete("entity");
    next.delete("name");
    setParams(next, { replace: true });
  };

  const neighborhood = Boolean(entityParam || nameParam);
  const entities = payload?.entities || [];
  const edges = payload?.edges || [];
  const active = entities.find((item) => item.id === selected) || null;
  const mention = active?.mentions?.[0] || null;
  const mentionDoc = mention?.doc_id || wanted;

  const graphDocs = Number(status?.documents || 0);
  const graphRunning = Boolean(status?.running) || Boolean(jobKind);
  const graphConcepts = Number(status?.concepts || 0);
  const graphReady = Number(status?.entities || 0) > 0;
  const skeletonCount = catalog.filter((item) => item.skeleton).length;
  const extractedCount = catalog.filter((item) => item.extracted).length;
  const canSkeleton = workbench && selectedSkeletonIds.length > 0;
  const canExtract = Boolean(status?.extract_enabled) && workbench && selectedExtractIds.length > 0;
  const graphPercent = graphDocs ? Math.round((skeletonCount / graphDocs) * 100) : 100;
  const listing = catalog.length
    ? catalog
    : docs.map((item) => ({
        doc_id: item.doc_id,
        filename: item.filename,
        extracted: false,
        skeleton: false,
        atx: false,
        calls: 0,
        total_tokens: 0,
      }));

  let statusLine = "";
  if (status?.error) statusLine = null;
  else if (!workbench) {
    statusLine = "功能已关闭：已有图谱仍可查看。打开开关后可免费建章节树，或再选文档抽取概念。";
  } else if (graphRunning) {
    statusLine =
      jobKind === "skeleton" ? "正在生成章节结构（不调模型）…" : "正在抽取所选文档的概念…";
  } else if (!pickedIds.length) {
    statusLine = `章节树 ${skeletonCount}/${listing.length} 篇 · 概念已抽 ${extractedCount} 篇 · 共 ${graphConcepts} 个概念。勾选文档后可建章节树或抽概念。`;
  } else if (selectedSkeletonIds.length && selectedExtractIds.length) {
    statusLine = `已选 ${pickedIds.length} 篇：${selectedSkeletonIds.length} 篇可建章节树（0 token），${selectedExtractIds.length} 篇可抽概念。`;
  } else if (selectedSkeletonIds.length) {
    statusLine = `已选 ${selectedSkeletonIds.length} 篇可建章节树，不调模型。`;
  } else if (selectedExtractIds.length) {
    statusLine = `已选 ${selectedExtractIds.length} 篇可抽概念，预计 ${fmtTokens(estimate.total)} token。`;
  } else if (!status?.extract_enabled) {
    statusLine = "未配置 DeepSeek，只能生成章节树。已形成的图谱仍可查看。";
  } else {
    statusLine = "所选文档已形成或正文过短，请改选其他文档。";
  }

  const enabled = status?.enabled !== false;
  const expandableIds = entities
    .filter((item) => item.kind === "doc_section" || item.kind === "document")
    .map((item) => item.id);

  const onSelectNode = (id, kind, hiddenKids) => {
    if (EXTRACT_KINDS.has(kind)) {
      pickEntity(id);
      return;
    }
    setSelected(id);
    if (hiddenKids) {
      setExpanded((cur) => ({ ...cur, [id]: true }));
    } else if (expanded[id] && selected === id) {
      setExpanded((cur) => {
        const next = { ...cur };
        delete next[id];
        return next;
      });
    }
  };

  const skeletonLabel =
    graphRunning && jobKind === "skeleton"
      ? "章节树生成中…"
      : !workbench
        ? "功能已关闭"
        : selectedSkeletonIds.length
          ? "生成章节树"
          : "勾选未建树的文档";

  const extractLabel =
    graphRunning && jobKind === "extract"
      ? "概念抽取中…"
      : !workbench
        ? "功能已关闭"
        : !status?.extract_enabled
          ? "未配置模型"
          : selectedExtractIds.length
            ? "抽出所选概念"
            : "勾选可抽取的文档";

  return (
    <div className="kg-dashboard">
      <section className="kg-overview">
        <div className="kg-intro">
          <span className="kg-intro-orbit" aria-hidden="true">
            ✦
          </span>
          <h1>知识图谱</h1>
          {libraries.length ? (
            <label className="tiny">
              当前库{" "}
              <AgentMenu
                label="当前库"
                value={collectionId}
                placeholder="全部文档"
                options={[
                  { value: "", label: "全部文档" },
                  ...libraries.map((item) => ({ value: item.id, label: item.name })),
                ]}
                onChange={(next) => {
                  collectionRef.current = next;
                  setCollectionId(next);
                  refreshCatalog();
                }}
              />
            </label>
          ) : null}
          {staleGraph ? (
            <p className="tiny kg-stale">图谱尚未按库重抽。当前图仍可能混有建库前的合并结果。</p>
          ) : null}
          <p>
            已形成的图谱随时可看。打开功能后先可免费生成章节树（Markdown 的 <code>#</code>{" "}
            标题），再按需勾选抽取概念；无 <code>#</code>{" "}
            时概念抽取会按原文标题行建章（必须对得上原文）。
          </p>
        </div>
        <label className={`kg-switch ${workbench ? "on" : ""}`}>
          <input
            type="checkbox"
            checked={workbench}
            onChange={(event) => setWorkbenchOn(event.target.checked)}
          />
          <span>知识图谱功能</span>
          <strong>{workbench ? "已开启" : "已关闭"}</strong>
        </label>
        <div className="kg-stat">
          <span className="kg-stat-icon violet">{IcoFolder}</span>
          <div>
            <small>文档总数</small>
            <strong>{graphDocs}</strong>
          </div>
        </div>
        <div className="kg-stat">
          <span className="kg-stat-icon mint">⌘</span>
          <div>
            <small>实体总数</small>
            <strong>{Number(status?.entities || 0)}</strong>
          </div>
        </div>
        <div className="kg-stat">
          <span className="kg-stat-icon amber">⌁</span>
          <div>
            <small>关系总数</small>
            <strong>{Number(status?.edges || 0)}</strong>
          </div>
        </div>
      </section>

      {!enabled ? (
        <section className="kg-white-card">
          <p>图谱已关闭（GRAPH_ENABLED=false）。</p>
        </section>
      ) : graphDocs === 0 ? (
        <section className="kg-white-card">
          <p>
            还没有文档。先到 <Link to="/library">知识库</Link> 上传后再回来看图。
          </p>
        </section>
      ) : (
        <section className="kg-workspace" id="kg-view">
          <aside className="kg-left-rail">
            <div
              className="kg-white-card kg-progress-card"
              id="graph-panel"
              aria-label="整理进度：概念图已就绪"
            >
              <h2>图谱构建进度</h2>
              <div className="kg-progress-body">
                <div
                  className="kg-progress-ring"
                  style={{ "--kg-progress": `${graphPercent * 3.6}deg` }}
                >
                  <strong>{graphPercent}%</strong>
                </div>
                <div>
                  <p>
                    {graphRunning
                      ? jobKind === "skeleton"
                        ? "正在生成章节树…"
                        : "正在抽取概念…"
                      : graphReady
                        ? "已有图谱可查看"
                        : "勾选文档后生成章节树或抽取概念"}
                  </p>
                  <small>
                    章节树 {skeletonCount} / 概念 {extractedCount} / 共 {listing.length} 篇
                  </small>
                </div>
              </div>
              <div className="kg-run-actions">
                <button
                  type="button"
                  className="kg-run-button kg-run-secondary"
                  id="graph-run-skeleton"
                  disabled={graphRunning || !canSkeleton}
                  onClick={runSkeleton}
                >
                  {skeletonLabel}
                </button>
                <div className="kg-extract-block">
                  <button
                    type="button"
                    className="kg-run-button"
                    id="graph-run-extract"
                    disabled={graphRunning || !canExtract}
                    onClick={runExtract}
                  >
                    {extractLabel}
                  </button>
                  {workbench && selectedExtractIds.length ? (
                    <p className="kg-estimate">
                      概念抽取预计 {fmtTokens(estimate.total)} token · {estimate.calls} 次调用
                      <small>
                        输入约 {fmtTokens(estimate.input)}，输出上限 {fmtTokens(estimate.output)}
                      </small>
                    </p>
                  ) : null}
                </div>
              </div>
              <p className="kg-status-line" id="graph-status">
                {status?.error ? (
                  <span className="error">整理失败：{status.error}</span>
                ) : (
                  statusLine
                )}
              </p>
            </div>

            <div className="kg-white-card kg-documents">
              <h2>文档源</h2>
              <p className="kg-doc-hint">
                {workbench
                  ? "勾选文档：未建树的可免费生成章节结构；够长且未抽过的可再抽概念。点名称查看图谱。"
                  : "点名称查看已有图谱。开启功能后可勾选整理。"}
              </p>
              <div className="kg-doc-list" id="kg-doc">
                {listing.map((item) => {
                  const id = String(item.doc_id);
                  const done = Boolean(item.extracted);
                  const hasSkeleton = Boolean(item.skeleton);
                  const canPick = !done && (!hasSkeleton || Number(item.calls || 0) > 0);
                  const checked = Boolean(picked[id]);
                  let tokenHint = "过短";
                  if (done) tokenHint = "概念已形成";
                  else if (!hasSkeleton && item.atx) tokenHint = "可建树 · MD";
                  else if (!hasSkeleton) tokenHint = "可建树";
                  else if (Number(item.calls || 0) > 0)
                    tokenHint = `可抽 · ${fmtTokens(item.total_tokens)}`;
                  else tokenHint = "过短 · 仅骨架";
                  return (
                    <div
                      key={id}
                      className={`kg-doc-row ${id === wanted ? "on" : ""} ${done ? "done" : ""}`}
                    >
                      <input
                        type="checkbox"
                        disabled={!workbench || !canPick || graphRunning}
                        checked={checked && canPick}
                        onChange={(event) => {
                          const on = event.target.checked;
                          setPicked((cur) => {
                            const next = { ...cur };
                            if (on) next[id] = true;
                            else delete next[id];
                            return next;
                          });
                        }}
                      />
                      <button type="button" onClick={() => pickDoc(id)}>
                        <span className="kg-file-mark">
                          {done ? "OK" : hasSkeleton ? "树" : "DOC"}
                        </span>
                        <span title={item.filename}>{item.filename}</span>
                        <em>{tokenHint}</em>
                      </button>
                    </div>
                  );
                })}
              </div>
              <Link className="kg-all-docs" to="/library">
                查看全部文档 →
              </Link>
            </div>
          </aside>

          <section className="kg-white-card kg-graph-panel">
            <header className="kg-graph-toolbar">
              <div>
                <h2>图谱视图</h2>
                <span className="kg-mode-pill">{neighborhood ? "⌘ 概念邻域" : "⌘ 层级布局"}</span>
              </div>
              <div className="kg-tree-actions">
                {neighborhood ? (
                  <button
                    type="button"
                    id="kg-back-tree"
                    onClick={() => pickDoc(wanted || String(docs[0]?.doc_id || ""))}
                  >
                    返回章节树
                  </button>
                ) : (
                  <>
                    <button
                      type="button"
                      id="kg-expand"
                      onClick={() =>
                        setExpanded(Object.fromEntries(expandableIds.map((id) => [id, true])))
                      }
                    >
                      展开全部
                    </button>
                    <button type="button" id="kg-collapse" onClick={() => setExpanded({})}>
                      收起下层
                    </button>
                  </>
                )}
              </div>
            </header>
            {error ? <p className="error kg-inline-error">{error}</p> : null}
            <p className="kg-format-note" id="kg-format-note">
              TXT / PDF 也可抽取。无 <code>#</code> 时章节名必须是原文标题行，不能编目录。
            </p>
            <div className="kg-panel-body">
              {(wanted || neighborhood) && payload ? (
                <SchemaCanvas
                  entities={entities}
                  edges={edges}
                  selectedId={selected}
                  onSelect={onSelectNode}
                  mode={neighborhood ? "radial" : "tree"}
                  seedId={payload.seed_id || ""}
                  expanded={expanded}
                />
              ) : (
                <div className="kg-loading">正在读取关系…</div>
              )}

              <aside className="kg-right-rail">
                <div className="kg-side-card kg-legend" id="kg-legend">
                  <h3>图例</h3>
                  <div className="kg-legend-row">
                    {KIND_META.map((item) => (
                      <span key={item.kind} className="kg-swatch">
                        <i style={{ background: item.fill }} />
                        {item.label}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="kg-side-card kg-mention" id="kg-mention" data-testid="GraphMention">
                  <h3>原文追溯</h3>
                  {active ? (
                    <>
                      <p className="kg-active-name">{active.name}</p>
                      <small>{active.kind}</small>
                      {mention ? (
                        <p className="kg-quote">“{mention.quote}”</p>
                      ) : (
                        <p>这条还没有原文摘录。</p>
                      )}
                      {EXTRACT_KINDS.has(active.kind) && !neighborhood ? (
                        <button type="button" onClick={() => pickEntity(active.id)}>
                          查看概念邻域
                        </button>
                      ) : null}
                      {mentionDoc ? (
                        <Link
                          to={sourceHref({ ...mention, doc_id: mentionDoc, text: mention?.quote })}
                        >
                          打开原文 →
                        </Link>
                      ) : null}
                    </>
                  ) : (
                    <p>点一个节点查看它在文档里的原句；点概念节点可展开邻域。</p>
                  )}
                </div>
              </aside>
            </div>
          </section>
        </section>
      )}
    </div>
  );
}
