import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api";
import ConceptChips from "../components/ConceptChips";
import ResultView from "../components/ResultView";
import { agentDraftHref } from "../agentHandoff";

export default function Trace() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [fb, setFb] = useState("");
  const [concepts, setConcepts] = useState([]);
  const [docNames, setDocNames] = useState({});

  useEffect(() => {
    let cancelled = false;
    setConcepts([]);
    api(`/traces/${id}`)
      .then((payload) => {
        if (!cancelled) setData(payload);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, [id]);

  useEffect(() => {
    const ids = new Set((data?.evidence || []).map((item) => item.doc_id).filter(Boolean));
    if (!ids.size) {
      setDocNames({});
      return undefined;
    }
    let cancelled = false;
    api("/documents")
      .then((payload) => {
        if (cancelled) return;
        const map = {};
        for (const item of payload.results || []) {
          const id = String(item.doc_id);
          if (ids.has(id) || ids.has(item.doc_id))
            map[id] = item.display_title || item.filename || "";
        }
        setDocNames(map);
      })
      .catch(() => {
        if (!cancelled) setDocNames({});
      });
    return () => {
      cancelled = true;
    };
  }, [data]);

  useEffect(() => {
    if (!data) return undefined;
    const cited = (data.citations || []).map((item) => item.chunk_id).filter(Boolean);
    if (!cited.length) {
      setConcepts([]);
      return undefined;
    }
    let cancelled = false;
    api(`/graph/related?chunk_ids=${encodeURIComponent(cited.join(","))}`)
      .then((payload) => {
        if (!cancelled) setConcepts(payload.concepts || []);
      })
      .catch(() => {
        if (!cancelled) setConcepts([]);
      });
    return () => {
      cancelled = true;
    };
  }, [data]);

  return (
    <>
      <section className="hero">
        <p>
          <Link to="/traces">← 提问历史</Link>
        </p>
        <h1>历史回放</h1>
        <p>这是已保存的回答、引用和核对结果，打开时不会重新提问。</p>
      </section>
      <section className="panel" id="box">
        {error ? <p className="error">{error}</p> : null}
        {data ? (
          <ResultView
            data={data}
            docNames={docNames}
            mode="replay"
            onExplore={(text) =>
              navigate(
                agentDraftHref({
                  query: text,
                  scope: data.scope || data.diagnostics?.scope,
                  traceId: id,
                })
              )
            }
            extra={
              <>
                <ConceptChips
                  concepts={concepts}
                  onPick={(name) => navigate(`/agent?q=${encodeURIComponent(name)}`)}
                />
                {data.tags && data.tags.length ? (
                  <div className="trace-tag-line">
                    标签：{data.tags.map((t) => t.name).join(" · ")}
                  </div>
                ) : null}
                <div className="feedback-bar">
                  这条回答有帮助吗？
                  <button
                    type="button"
                    data-rate="up"
                    onClick={async () => {
                      try {
                        await api("/feedback", {
                          method: "POST",
                          body: { rating: "up", trace_id: id },
                        });
                        setFb("已记下你的反馈");
                      } catch (err) {
                        setFb(err.message);
                      }
                    }}
                  >
                    有
                  </button>
                  <button
                    type="button"
                    data-rate="down"
                    onClick={async () => {
                      try {
                        await api("/feedback", {
                          method: "POST",
                          body: { rating: "down", trace_id: id },
                        });
                        setFb("已记下你的反馈");
                      } catch (err) {
                        setFb(err.message);
                      }
                    }}
                  >
                    没有
                  </button>
                  <span id="fb">{fb}</span>
                </div>
              </>
            }
          />
        ) : error ? null : (
          "加载中…"
        )}
      </section>
    </>
  );
}
