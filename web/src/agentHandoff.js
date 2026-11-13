// URLs carry draft parameters or an existing record identity, never authority or model memory.
export const agentTaskHref = (row) => `/agent/${encodeURIComponent(row.id)}`;

export function agentDraftHref({
  query,
  scope,
  mode = "synthesis",
  targetDocumentId = "",
  traceId = "",
}) {
  const params = new URLSearchParams({
    goal: query || "",
    mode: mode === "digest" ? "digest" : "synthesis",
  });
  if (scope?.mode) {
    params.set("scope", scope.mode);
    for (const id of scope.collection_ids || []) params.append("collection", id);
  } else params.set("scope_confirmed", "0");
  if (targetDocumentId) params.set("target", targetDocumentId);
  if (traceId) params.set("from_trace", traceId);
  return `/agent?${params}`;
}

export function readAgentDraft(search) {
  const params = new URLSearchParams(search);
  const validScope =
    !params.has("scope") || ["auto", "all", "soft", "hard"].includes(params.get("scope"));
  return {
    goal: (params.get("goal") || params.get("q") || "").slice(0, 4000),
    mode: ["auto", "quick", "synthesis", "digest"].includes(params.get("mode"))
      ? params.get("mode")
      : "auto",
    scope: {
      mode: validScope ? params.get("scope") || "auto" : "auto",
      collection_ids: params.getAll("collection"),
    },
    scopeConfirmed: validScope && params.get("scope_confirmed") !== "0",
    targetDocumentId: params.get("target") || "",
    traceId: params.get("from_trace") || "",
  };
}
