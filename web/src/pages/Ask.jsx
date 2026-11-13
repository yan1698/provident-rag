import { Navigate, useLocation } from "react-router-dom";

// Legacy question bookmarks remain read-only redirects into the Agent workspace.
export default function Ask() {
  const location = useLocation(),
    params = new URLSearchParams(location.search);
  const run = params.get("run");
  if (run) {
    params.delete("run");
    params.delete("trace");
  }
  const search = params.toString();
  return (
    <Navigate
      replace
      to={`${run ? `/agent/${encodeURIComponent(run)}` : "/agent"}${search ? `?${search}` : ""}${location.hash}`}
    />
  );
}
