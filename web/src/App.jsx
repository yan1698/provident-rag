import { Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./auth";
import Layout from "./components/Layout";
import Agent from "./pages/Agent";
import Ask from "./pages/Ask";
import DocumentPage from "./pages/Document";
import EvalPage from "./pages/Eval";
import GraphPage from "./pages/Graph";
import Library from "./pages/Library";
import Login from "./pages/Login";
import Trace from "./pages/Trace";
import Traces from "./pages/Traces";

function RequireAuth({ children }) {
  const { ready, authEnabled, user } = useAuth();
  if (!ready) {
    return (
      <div className="login-page">
        <p className="tiny">正在进入…</p>
      </div>
    );
  }
  if (authEnabled && !user) {
    return <Navigate to="/login" replace />;
  }
  return children;
}

function RequireEvalGate({ children }) {
  const { ready, evalGate } = useAuth();
  if (!ready) {
    return (
      <div className="login-page">
        <p className="tiny">正在进入…</p>
      </div>
    );
  }
  if (!evalGate) {
    return <Navigate to="/agent" replace />;
  }
  return children;
}

function Shell() {
  return (
    <RequireAuth>
      <Layout>
        <Routes>
          <Route path="/" element={<Ask />} />
          <Route path="/library" element={<Library />} />
          <Route path="/library/:docId" element={<DocumentPage />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/traces" element={<Traces />} />
          <Route path="/traces/:id" element={<Trace />} />
          <Route
            path="/eval"
            element={
              <RequireEvalGate>
                <EvalPage />
              </RequireEvalGate>
            }
          />
          <Route path="/agent" element={<Agent />} />
          <Route path="/agent/session/:sessionId" element={<Agent />} />
          <Route path="/agent/:id" element={<Agent />} />
        </Routes>
      </Layout>
    </RequireAuth>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/*" element={<Shell />} />
      </Routes>
    </AuthProvider>
  );
}
