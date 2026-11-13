import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { api, TOKEN_KEY } from "./api";

const USER_KEY = "provident.user";
const AuthContext = createContext(null);

export function getStoredToken() {
  try {
    return localStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

function readStoredUser() {
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export function AuthProvider({ children }) {
  const [ready, setReady] = useState(false);
  const [authEnabled, setAuthEnabled] = useState(true);
  const [evalGate, setEvalGate] = useState(false);
  const [user, setUser] = useState(readStoredUser);

  const applySession = (token, profile) => {
    try {
      if (token) localStorage.setItem(TOKEN_KEY, token);
      else localStorage.removeItem(TOKEN_KEY);
      if (profile) localStorage.setItem(USER_KEY, JSON.stringify(profile));
      else localStorage.removeItem(USER_KEY);
    } catch {
      /* ignore quota / private mode */
    }
    setUser(profile || null);
  };

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const status = await fetch("/auth/status").then((res) =>
          res.ok ? res.json() : { auth_enabled: true }
        );
        if (cancelled) return;
        const enabled = Boolean(status.auth_enabled);
        setAuthEnabled(enabled);
        if (!enabled) {
          setEvalGate(false);
          applySession("", null);
          return;
        }
        const token = getStoredToken();
        if (!token) {
          setEvalGate(false);
          applySession("", null);
          return;
        }
        const me = await fetch("/auth/me", { headers: { Authorization: `Bearer ${token}` } });
        if (cancelled) return;
        if (!me.ok) {
          setEvalGate(false);
          applySession("", null);
          return;
        }
        const data = await me.json();
        setEvalGate(Boolean(data.eval_gate));
        setUser(data.user || null);
      } catch {
        if (!cancelled) setAuthEnabled(true);
      } finally {
        if (!cancelled) setReady(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const value = useMemo(
    () => ({
      ready,
      authEnabled,
      evalGate,
      user,
      login: async (username, password) => {
        const data = await api("/auth/login", { method: "POST", body: { username, password } });
        applySession(data.access_token, data.user);
        setEvalGate(Boolean(data.eval_gate));
        setAuthEnabled(true);
        return data.user;
      },
      register: async (username, password, displayName) => {
        const data = await api("/auth/register", {
          method: "POST",
          body: { username, password, display_name: displayName || "" },
        });
        applySession(data.access_token, data.user);
        setEvalGate(Boolean(data.eval_gate));
        setAuthEnabled(true);
        return data.user;
      },
      logout: () => {
        applySession("", null);
        setEvalGate(false);
      },
    }),
    [ready, authEnabled, evalGate, user]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
