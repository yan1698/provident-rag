import AgentMenu from "./AgentMenu";

const SCOPE_OPTIONS = [
  { value: "auto", label: "自动选择范围", hint: "按问题措辞决定查哪些库" },
  { value: "all", label: "全部文档", hint: "包含待归类文档" },
  { value: "soft", label: "库和待归类", hint: "指定知识库，并包含待归类" },
  { value: "hard", label: "只查此库", hint: "只查指定知识库，不含待归类" },
];

export default function ScopeFields({
  libraries,
  scopeMode,
  onScopeMode,
  collectionId,
  onCollectionId,
  pinScope = false,
  onPinScope = null,
  invalid = false,
}) {
  if (!libraries.length) return null;
  const showCollection = scopeMode === "soft" || scopeMode === "hard";
  return (
    <>
      <AgentMenu
        label="检索范围"
        value={scopeMode}
        placeholder="自动选择范围"
        options={SCOPE_OPTIONS}
        onChange={onScopeMode}
      />
      {showCollection ? (
        <AgentMenu
          label="知识库"
          value={collectionId}
          placeholder="选择知识库"
          options={libraries.map((item) => ({ value: item.id, label: item.name }))}
          onChange={onCollectionId}
          invalid={invalid && !collectionId}
        />
      ) : null}
      {showCollection && scopeMode === "soft" && onPinScope ? (
        <label className="tiny">
          <input
            type="checkbox"
            checked={pinScope}
            onChange={(event) => onPinScope(event.target.checked)}
          />{" "}
          固定在此库
        </label>
      ) : null}
    </>
  );
}
