import { useState } from "react";
import { Navigate } from "react-router-dom";
import { useAuth } from "../auth";

export default function Login() {
  const { ready, authEnabled, user, login, register } = useAuth();
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  if (!ready) {
    return (
      <div className="login-page">
        <p className="tiny">正在检查登录状态…</p>
      </div>
    );
  }
  if (!authEnabled || user) {
    return <Navigate to="/agent" replace />;
  }

  const submit = async (event) => {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      if (mode === "register") {
        await register(username, password, displayName);
      } else {
        await login(username, password);
      }
    } catch (err) {
      setError(err.message || "登录失败");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-page">
      <div className="sky" aria-hidden="true" />
      <section className="panel login-card">
        <p className="login-brand">Provident</p>
        <h1>{mode === "register" ? "创建账号" : "登录"}</h1>
        <p className="lede">每个账号有自己的知识库和问答记录，互不可见。</p>
        <form className="stack" onSubmit={submit}>
          <label className="login-field">
            用户名
            <input
              type="text"
              autoComplete="username"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              minLength={2}
              maxLength={32}
              required
            />
          </label>
          {mode === "register" ? (
            <label className="login-field">
              显示名（可选）
              <input
                type="text"
                autoComplete="nickname"
                value={displayName}
                onChange={(event) => setDisplayName(event.target.value)}
                maxLength={64}
                placeholder="默认同用户名"
              />
            </label>
          ) : null}
          <label className="login-field">
            密码
            <input
              type="password"
              autoComplete={mode === "register" ? "new-password" : "current-password"}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              minLength={mode === "register" ? 6 : 1}
              required
            />
          </label>
          {error ? <p className="error">{error}</p> : null}
          <button className="btn" type="submit" disabled={busy}>
            {busy ? "请稍候…" : mode === "register" ? "注册并进入" : "进入"}
          </button>
        </form>
        <p className="tiny login-switch">
          {mode === "register" ? "已有账号？" : "还没有账号？"}
          <button
            type="button"
            className="linkish"
            onClick={() => {
              setMode(mode === "register" ? "login" : "register");
              setError("");
            }}
          >
            {mode === "register" ? "去登录" : "创建账号"}
          </button>
        </p>
      </section>
    </div>
  );
}
