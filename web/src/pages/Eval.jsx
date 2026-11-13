import { useCallback, useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useAuth } from "../auth";
import { api } from "../api";
import { GATE_LABEL, fmtTime, num, pct } from "../lib";
import AgentMenu from "../components/AgentMenu";

function EvalHero({ evalSet }) {
  const setFlag = evalSet ? ` --set ${evalSet}` : "";
  return (
    <>
      <h1>回归闸门</h1>
      <p>
        维护者专用：对当前<b>评测集</b>的固定题跑批量验收。报告回放本账号已落库的 trace，
        <strong>不重新调用生成模型</strong>。
        与「问答」里的随意提问无关。当前固定题对着已入库的面试笔记{" "}
        <code>golden-agent-notes-v1</code>。换一批文档时另开 set，不要改这一份的哈希标注。
      </p>
      <p className="tiny">
        更新数字：本机执行 <code>python scripts/run_eval.py{setFlag}</code>
        （已默认当前开发者账号），再点「刷新报告」。
      </p>
    </>
  );
}

export default function EvalPage() {
  const { user } = useAuth();
  const [params, setParams] = useSearchParams();
  const evalSet = (params.get("set") || "").trim();
  const [sets, setSets] = useState([]);
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api("/eval/sets")
      .then((payload) => setSets(payload.results || []))
      .catch(() => setSets([]));
  }, []);

  const reload = useCallback(() => {
    setError("");
    setReport(null);
    const q = evalSet ? `?eval_set=${encodeURIComponent(evalSet)}` : "";
    api(`/eval/report${q}`)
      .then(setReport)
      .catch((err) => setError(err.message));
  }, [evalSet]);

  useEffect(() => {
    reload();
  }, [reload]);

  if (error) {
    return (
      <>
        <section className="hero">
          <EvalHero evalSet={evalSet} />
        </section>
        <section className="panel" id="box">
          <p className="error">{error}</p>
        </section>
      </>
    );
  }

  if (!report) {
    return (
      <>
        <section className="hero">
          <EvalHero evalSet={evalSet} />
        </section>
        <section className="panel" id="box">
          加载中…
        </section>
      </>
    );
  }

  const metrics = report.metrics || {};
  const gates = report.gates || [];
  const results = report.results || [];
  const corpus = report.corpus || {};
  const allDocs = corpus.documents || [];
  const expectedDocs = allDocs.filter((item) => item.expected);
  const offCorpusDocs = allDocs.filter((item) => !item.expected);
  const shownDocs = expectedDocs.length ? expectedDocs : allDocs;
  const forbidden = corpus.forbidden_documents || [];
  const missing = results.filter((item) => !item.trace_id);
  const corpusText = shownDocs.length
    ? shownDocs.map((item) => item.filename).join("、")
    : corpus.path || "未声明语料";

  return (
    <>
      <section className="hero">
        <EvalHero evalSet={report.eval_set || evalSet} />
      </section>
      <section className="panel" id="box">
        <div className="row" style={{ alignItems: "center", justifyContent: "space-between" }}>
          <h2>{report.eval_set || "eval"}</h2>
          <div className="row" style={{ gap: 8 }}>
            {sets.length ? (
              <AgentMenu
                label="评测集"
                value={report.eval_set || evalSet}
                placeholder="选择评测集"
                options={sets.map((item) => ({ value: item.name, label: item.name }))}
                onChange={(next) => {
                  if (next) setParams({ set: next });
                  else setParams({});
                }}
              />
            ) : null}
            <button type="button" className="btn ghost" onClick={reload}>
              刷新报告
            </button>
            {report.passed ? (
              <span className="stamp pass">通过</span>
            ) : (
              <span className="stamp fail">未过</span>
            )}
          </div>
        </div>
        <p className="tiny">
          {fmtTime(report.generated_at)} · {corpusText}
        </p>
        {missing.length ? (
          <p className="lede">
            还有 {missing.length} 道固定题没有 trace。在仓库根目录运行{" "}
            <code>
              python scripts/run_eval.py --set{" "}
              {report.eval_set || evalSet || "golden-agent-notes-v1"} --username{" "}
              {user?.username || "当前账号"}
            </code>
            ，完成后点「刷新报告」。
          </p>
        ) : null}
        {forbidden.length ? (
          <p className="lede">
            知识库里有 {forbidden.length} 份验收夹具（{forbidden.join("、")}），它们的填充文本会挤进
            Top-K 让指标失真。跑 <code>python scripts/reset_corpus.py</code> 清掉后再刷新。
          </p>
        ) : null}
        {offCorpusDocs.length ? (
          <p className="tiny">
            另有 {offCorpusDocs.length} 份 golden 语料之外的文档：
            {offCorpusDocs.map((item) => item.filename).join("、")}。
            它们本身不影响闸门，只有挤进固定题的 Top-K 时「语料外证据」才会变红。
          </p>
        ) : null}
        <div className="kpis" style={{ margin: "16px 0" }}>
          <div className="kpi">
            召回<b>{pct(metrics.avg_recall_at_k)}</b>
          </div>
          <div className="kpi">
            排序质量 MRR<b>{num(metrics.mrr)}</b>
          </div>
          <div className="kpi">
            有依据<b>{pct(metrics.avg_coverage_rate)}</b>
          </div>
          <div className="kpi">
            依据不足<b>{pct(metrics.avg_hallucination_rate)}</b>
          </div>
        </div>
        <div className="stack">
          {gates.map((gate) => {
            const actual = gate.actual == null ? 0 : Number(gate.actual);
            const width = Math.max(
              4,
              Math.min(
                100,
                Math.abs(actual) *
                  (gate.name.includes("rate") ||
                  gate.name.includes("recall") ||
                  gate.name.includes("mrr")
                    ? 100
                    : 20)
              )
            );
            const label = GATE_LABEL[gate.name] || gate.name;
            return (
              <div className="gate" key={gate.name}>
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <span>
                    {label} {gate.op} {gate.threshold}
                  </span>
                  <span className={`pill ${gate.passed ? "ok" : "bad"}`}>
                    {gate.passed ? "达标" : "未达标"}
                  </span>
                </div>
                <div className={`bar ${gate.passed ? "" : "fail"}`}>
                  <i style={{ width: `${width}%` }} />
                </div>
              </div>
            );
          })}
        </div>
        <table style={{ marginTop: 18 }}>
          <thead>
            <tr>
              <th>固定题</th>
              <th>召回</th>
              <th>有依据</th>
              <th>依据不足</th>
              <th>语料外证据</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {results.map((item) => (
              <tr key={item.eval_id || item.query}>
                <td>
                  {item.query}
                  <div className="tiny">
                    {item.eval_id}
                    {item.expect === "refusal" ? " · 应拒答" : ""}
                  </div>
                </td>
                <td>{pct(item.recall_at_k)}</td>
                <td>{pct(item.coverage_rate)}</td>
                <td>{pct(item.hallucination_rate)}</td>
                <td>{item.expect === "refusal" ? "—" : pct(item.off_corpus_evidence_rate)}</td>
                <td>{item.trace_id ? <Link to={`/traces/${item.trace_id}`}>查看</Link> : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </>
  );
}
