import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";
import { IcoCloud, IcoFolder, IcoTrash } from "../icons";
import { fileKind, fmtTime } from "../lib";
import AgentMenu from "../components/AgentMenu";

export default function Library() {
  const [status, setStatus] = useState("");
  const [statusErr, setStatusErr] = useState(false);
  const [docs, setDocs] = useState(null);
  const [embed, setEmbed] = useState(null);
  const [library, setLibrary] = useState(null);
  const [error, setError] = useState("");
  const [q, setQ] = useState("");
  const [kind, setKind] = useState("");
  const [sort, setSort] = useState("new");
  const [dropOn, setDropOn] = useState(false);
  const [filter, setFilter] = useState("all");
  const [preview, setPreview] = useState(null);
  const [createOpen, setCreateOpen] = useState(false);
  const [draftName, setDraftName] = useState("");
  const [draftDesc, setDraftDesc] = useState("");
  const [createBusy, setCreateBusy] = useState(false);
  const [createError, setCreateError] = useState("");

  const refresh = async ({ start = false } = {}) => {
    try {
      let info = await api("/embed");
      if (start && info.pending > 0 && !info.running) {
        info = await api("/embed", { method: "POST", body: {} });
      }
      setEmbed(info);
      const listing = await api("/documents");
      setDocs(listing.results || []);
      setLibrary(await api("/library"));
      setError("");
    } catch (err) {
      setError(err.message);
    }
  };

  useEffect(() => {
    refresh({ start: true });
    const timer = setInterval(() => {
      api("/embed")
        .then((info) => setEmbed(info))
        .catch(() => {});
    }, 2000);
    return () => clearInterval(timer);
  }, []);

  const upload = async (file) => {
    if (!file) return;
    setStatus(`正在入库：${file.name}`);
    setStatusErr(false);
    const body = new FormData();
    body.append("file", file);
    try {
      const data = await api("/ingest", { method: "POST", body });
      setStatus(`${data.filename} 已切成 ${data.chunk_count} 个片段，正在生成卡片。`);
      setFilter("inbox");
      await refresh({ start: true });
    } catch (err) {
      setStatus(err.message);
      setStatusErr(true);
    }
  };

  const openCreate = () => {
    setDraftName("");
    setDraftDesc("");
    setCreateError("");
    setCreateOpen(true);
  };

  const createLibrary = async () => {
    const name = draftName.trim();
    if (!name || createBusy) return;
    setCreateBusy(true);
    setCreateError("");
    try {
      const row = await api("/collections", {
        method: "POST",
        body: { name, description: draftDesc.trim() },
      });
      setCreateOpen(false);
      setFilter(row.id);
      setStatus(`已新建「${row.name}」`);
      setStatusErr(false);
      await refresh();
    } catch (err) {
      setCreateError(err.message);
    } finally {
      setCreateBusy(false);
    }
  };

  const moveDoc = async (doc, collectionId) => {
    try {
      await api(`/documents/${doc.doc_id}/collection`, {
        method: "POST",
        body: { collection_id: collectionId || null },
      });
      await refresh();
    } catch (err) {
      setStatus(err.message);
      setStatusErr(true);
    }
  };

  const openOrganize = async () => {
    try {
      setPreview(await api("/library/organize"));
    } catch (err) {
      setStatus(err.message);
      setStatusErr(true);
    }
  };

  const acceptOrganize = async () => {
    const assignments = (preview?.groups || []).flatMap((group) =>
      (group.documents || []).map((doc) => ({
        doc_id: doc.doc_id,
        collection_id: group.collection_id,
      }))
    );
    if (!assignments.length) return;
    try {
      await api("/library/organize", { method: "POST", body: { assignments } });
      setPreview(null);
      setStatus("已按预览归入库，可撤销这一批");
      setStatusErr(false);
      await refresh();
    } catch (err) {
      setStatus(err.message);
      setStatusErr(true);
    }
  };

  const undo = async () => {
    if (!library?.open_batch_id) return;
    try {
      await api(`/library/batches/${library.open_batch_id}/undo`, { method: "POST", body: {} });
      setStatus("已撤销上一批整理");
      setStatusErr(false);
      await refresh();
    } catch (err) {
      setStatus(err.message);
      setStatusErr(true);
    }
  };

  const chunks = Number(embed?.chunks || 0);
  const embedded = Number(embed?.embedded || 0);
  const embedPct = chunks ? Math.round((embedded / chunks) * 100) : 0;
  const list = docs || [];
  const filtered = list
    .filter((item) => {
      if (filter === "inbox") return !item.collection_id;
      if (filter !== "all") return String(item.collection_id || "") === filter;
      return true;
    })
    .filter((item) => !kind || fileKind(item) === kind)
    .filter((item) => {
      const blob = `${item.display_title || ""} ${item.filename || ""}`.toLowerCase();
      return !q || blob.includes(q.trim().toLowerCase());
    })
    .sort((a, b) =>
      sort === "name"
        ? String(a.display_title || a.filename).localeCompare(
            String(b.display_title || b.filename),
            "zh"
          )
        : new Date(b.created_at) - new Date(a.created_at)
    );
  const ratio = Number(library?.inbox_ratio || 0);
  const showBanner =
    library && ratio >= Number(library.watermark || 0.4) && library.inbox_count > 0;
  const collections = library?.collections || [];

  return (
    <>
      <section className="hero">
        <p>上传文档、管理知识库，确保回答可追溯。</p>
      </section>
      <div className="library-grid">
        <aside className="panel library-side">
          <div className="section-head">
            <span className="ico">{IcoFolder}</span>
            <h2>我的知识库</h2>
          </div>
          <div className="side-embed">
            <div className="side-embed-head">
              <span>向量化</span>
              <strong>{embedPct}%</strong>
            </div>
            <div
              className="embed-track"
              role="progressbar"
              aria-valuenow={embedPct}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label="向量化进度"
            >
              <i style={{ width: `${embedPct}%` }} />
            </div>
            {embed?.error ? <p className="tiny error">{embed.error}</p> : null}
          </div>
          <button type="button" className="lib-new" onClick={openCreate}>
            + 新建库
          </button>
          <div className="library-nav">
            <button
              type="button"
              className={filter === "all" ? "on" : ""}
              onClick={() => setFilter("all")}
            >
              <span>全部文档</span>
              <span>{library?.total ?? list.length}</span>
            </button>
            {collections.map((item) => (
              <button
                key={item.id}
                type="button"
                className={filter === item.id ? "on" : ""}
                onClick={() => setFilter(item.id)}
              >
                <span>{item.name}</span>
                <span>{item.doc_count}</span>
              </button>
            ))}
            <button
              type="button"
              className={filter === "inbox" ? "on" : ""}
              onClick={() => setFilter("inbox")}
            >
              <span>待归类</span>
              <span>{library?.inbox_count ?? 0}</span>
            </button>
          </div>
        </aside>
        <div className="library-main">
          <section className="panel">
            <label
              className={`drop${dropOn ? " on" : ""}`}
              onDragOver={(event) => {
                event.preventDefault();
                setDropOn(true);
              }}
              onDragLeave={() => setDropOn(false)}
              onDrop={(event) => {
                event.preventDefault();
                setDropOn(false);
                upload(event.dataTransfer.files[0]);
              }}
            >
              <input
                type="file"
                accept=".pdf,.txt,.md,.markdown,application/pdf,text/plain,text/markdown"
                hidden
                onChange={(event) => upload(event.target.files && event.target.files[0])}
              />
              <span className="drop-ico">{IcoCloud}</span>
              <strong>点击选择文件</strong>
              <span className="lede">支持 PDF、TXT、Markdown</span>
            </label>
            <p className={`tiny${statusErr ? " error" : ""}`}>{status}</p>
          </section>
          {showBanner ? (
            <section className="panel organize-banner">
              <div className="banner-head">
                <div>
                  <strong>有些文档还没放进知识库</strong>
                  <p>
                    目前 {library.inbox_count}{" "}
                    份文档尚未归类。整理后，按知识库查找和管理文档会更方便。
                  </p>
                </div>
                <button type="button" className="btn" onClick={openOrganize}>
                  整理文档
                </button>
              </div>
              <div className="metric-row">
                <div className="metric-card">
                  <small>待归类文档</small>
                  <strong>{library.inbox_count} 份</strong>
                  <span>还没有指定所属知识库</span>
                </div>
                <div className="metric-card">
                  <small>跨范围补充检索</small>
                  <strong>{library.fallback_7d || 0} 次</strong>
                  <span>{library.fallback_7d ? "近 7 天" : "近 7 天未触发"}</span>
                </div>
                <div className="metric-card">
                  <small>手动调整归属</small>
                  <strong>{library.moves_7d || 0} 次</strong>
                  <span>近 7 天</span>
                </div>
              </div>
              <p className="organize-note">
                整理前，待归类文档仍可参与检索。整理不会自动改变文档归属，确认整理结果后才会生效。
              </p>
              {(library.suggestions || []).map((item) => (
                <p className="tiny" key={item.doc_id}>
                  有 {item.count} 次提问在所选库没有命中，最后落在一篇已归入其他库的文档。
                </p>
              ))}
              {library.open_batch_id ? (
                <button type="button" className="btn ghost" onClick={undo}>
                  撤销上一批整理
                </button>
              ) : null}
            </section>
          ) : null}
          {preview ? (
            <section className="panel stack">
              <h2>整理预览</h2>
              {(preview.groups || []).length ? (
                preview.groups.map((group) => (
                  <p key={group.collection_id}>
                    归入「{group.name}」：
                    {(group.documents || []).map((doc) => doc.title).join("、")}
                  </p>
                ))
              ) : (
                <p className="lede">没有足够接近某个库的待归类文档。</p>
              )}
              <div className="row">
                <button
                  type="button"
                  className="btn"
                  onClick={acceptOrganize}
                  disabled={!(preview.groups || []).length}
                >
                  接受
                </button>
                <button type="button" className="btn ghost" onClick={() => setPreview(null)}>
                  取消
                </button>
              </div>
            </section>
          ) : null}
          <section className="panel">
            {error ? <p className="error">{error}</p> : null}
            <div className="docs-head">
              <div className="section-head">
                <span className="ico">{IcoFolder}</span>
                <h2>文档列表</h2>
              </div>
              <div className="docs-tools">
                <input
                  type="text"
                  placeholder="搜索展示名或原始文件名"
                  value={q}
                  onChange={(event) => setQ(event.target.value)}
                />
                <AgentMenu
                  label="文档类型"
                  value={kind}
                  placeholder="类型"
                  options={[
                    { value: "", label: "全部类型" },
                    { value: "MD", label: "Markdown" },
                    { value: "PDF", label: "PDF" },
                    { value: "TXT", label: "TXT" },
                  ]}
                  onChange={setKind}
                />
                <AgentMenu
                  label="排序"
                  value={sort}
                  placeholder="最新上传"
                  options={[
                    { value: "new", label: "最新上传" },
                    { value: "name", label: "按名称" },
                  ]}
                  onChange={setSort}
                />
              </div>
            </div>
            <div className="list">
              {filtered.map((item) => {
                const kindLabel = fileKind(item);
                const title = item.display_title || item.filename;
                return (
                  <div className="doc-row" key={item.doc_id}>
                    <Link className="doc-link" to={`/library/${item.doc_id}`}>
                      <span className={`file-ico ${kindLabel.toLowerCase()}`}>{kindLabel}</span>
                      <div className="doc-main">
                        <strong>{title}</strong>
                        <div className="tiny">
                          {title !== item.filename ? (
                            <span className="origin">原文件：{item.filename} · </span>
                          ) : null}
                          {item.chunk_count} 片段 · {fmtTime(item.created_at)} ·{" "}
                          {item.card_status === "ready" ? "卡片已生成" : "卡片生成中"}
                        </div>
                        <div className="doc-tags">
                          <span className="tag">{item.collection_name || "待归类"}</span>
                          {item.doc_type_label ? (
                            <span className="tag warn">{item.doc_type_label}</span>
                          ) : null}
                          {item.mixed ? <span className="tag muted">混合</span> : null}
                        </div>
                      </div>
                    </Link>
                    <AgentMenu
                      className="doc-move"
                      label="移入库"
                      value={item.collection_id || ""}
                      placeholder="移回待归类"
                      options={[
                        { value: "", label: "移回待归类" },
                        ...collections.map((libraryItem) => ({
                          value: libraryItem.id,
                          label: libraryItem.name,
                        })),
                      ]}
                      onChange={(next) => moveDoc(item, next)}
                    />
                    <button
                      type="button"
                      className="doc-del"
                      title="从知识库删除"
                      onClick={async () => {
                        if (!window.confirm(`确定删除「${title}」？`)) return;
                        try {
                          await api(`/documents/${item.doc_id}`, { method: "DELETE" });
                          await refresh();
                        } catch (err) {
                          setStatus(err.message);
                          setStatusErr(true);
                        }
                      }}
                    >
                      {IcoTrash}
                    </button>
                  </div>
                );
              })}
              {docs && !filtered.length ? <p className="lede">没有符合筛选的文档。</p> : null}
              {docs === null && !error ? <p className="lede">加载中…</p> : null}
            </div>
          </section>
        </div>
      </div>
      {createOpen ? (
        <div className="modal-back" onClick={() => !createBusy && setCreateOpen(false)}>
          <div
            className="lib-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="new-lib-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="lib-modal-head">
              <span className="ico">{IcoFolder}</span>
              <div>
                <h2 id="new-lib-title">新建知识库</h2>
                <p>创建一个新的知识库，方便你更好地管理和组织文档。</p>
              </div>
              <button
                type="button"
                className="modal-x"
                aria-label="关闭"
                onClick={() => !createBusy && setCreateOpen(false)}
              >
                ×
              </button>
            </div>
            <label className="lib-field">
              <span>
                知识库名称 <i>*</i>
              </span>
              <input
                type="text"
                maxLength={30}
                autoFocus
                placeholder="请输入知识库名称"
                value={draftName}
                onChange={(event) => setDraftName(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") createLibrary();
                  if (event.key === "Escape") setCreateOpen(false);
                }}
              />
              <em>{draftName.length}/30</em>
            </label>
            <label className="lib-field">
              <span>知识库描述（选填）</span>
              <textarea
                maxLength={200}
                placeholder="请输入对该知识库的简单描述，例如：产品需求文档、技术方案等"
                value={draftDesc}
                onChange={(event) => setDraftDesc(event.target.value)}
              />
              <em>{draftDesc.length}/200</em>
            </label>
            <div className="lib-after">
              <div>
                <h3>创建后可进行</h3>
                <ul>
                  <li>从待归类中整理文档</li>
                  <li>手动将文档放入该库</li>
                  <li>在提问时选择该库进行检索</li>
                </ul>
              </div>
            </div>
            {createError ? <p className="error">{createError}</p> : null}
            <div className="lib-modal-actions">
              <button
                type="button"
                className="btn ghost"
                disabled={createBusy}
                onClick={() => setCreateOpen(false)}
              >
                取消
              </button>
              <button
                type="button"
                className="btn"
                disabled={createBusy || !draftName.trim()}
                onClick={createLibrary}
              >
                {createBusy ? "创建中…" : "创建"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </>
  );
}
