import { Link } from "react-router-dom";

export default function ConceptChips({ concepts, onPick }) {
  const rows = concepts || [];
  if (!rows.length) return null;
  return (
    <div className="concept-chips" id="concept-chips">
      <p className="tiny">相关概念</p>
      <div className="chips">
        {rows.map((item) => {
          const graphTo = item.id
            ? `/graph?entity=${encodeURIComponent(item.id)}`
            : `/graph?name=${encodeURIComponent(item.name)}`;
          return (
            <span key={`${item.kind}-${item.id || item.name}`} className="chip-wrap">
              <button type="button" className="chip" onClick={() => onPick(item.name)}>
                {item.name}
              </button>
              <Link className="chip-graph" to={graphTo} title="在图谱中查看邻域">
                图谱
              </Link>
            </span>
          );
        })}
      </div>
    </div>
  );
}
