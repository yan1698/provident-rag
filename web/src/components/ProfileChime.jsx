import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../auth";

function CatMoonAvatar({ className, gid = "chime-sky" }) {
  return (
    <svg className={className} viewBox="0 0 64 64" aria-hidden="true">
      <defs>
        <radialGradient id={gid} cx="50%" cy="40%" r="70%">
          <stop offset="0%" stopColor="#f7f4ff" />
          <stop offset="100%" stopColor="#d9cff5" />
        </radialGradient>
      </defs>
      <circle cx="32" cy="32" r="32" fill={`url(#${gid})`} />
      <circle cx="12" cy="14" r="0.7" fill="#8b6cff" opacity="0.85" />
      <circle cx="22" cy="8" r="0.55" fill="#a78bfa" />
      <circle cx="8" cy="28" r="0.5" fill="#c4b5fd" />
      <circle cx="18" cy="22" r="0.4" fill="#7c5ce4" />
      <circle cx="54" cy="48" r="0.6" fill="#a78bfa" />
      <circle cx="48" cy="56" r="0.45" fill="#7c5ce4" />
      <circle cx="58" cy="18" r="0.5" fill="#8b6cff" />
      <circle cx="44" cy="20" r="11.5" fill="#f4efe0" />
      <circle cx="47" cy="17" r="4" fill="rgba(255,255,255,0.35)" />
      <path
        fill="#05060c"
        d="M16 38c0-6.2 4.2-12 11.4-13.2 2.1-4.6 6.8-7.4 9.2-7.6.4 2.2-.2 4.4-1.4 6.2 4.4.8 7.6 4.2 8.2 8.8 1.8 1.2 3.2 3.2 3.2 5.6 0 5.8-6.2 10.4-16.2 10.4S16 43.6 16 38z"
      />
      <path fill="#05060c" d="M24.2 18.2 21 8.5l6.4 6.8zM36.8 17.6 41.6 8.2 38.2 19z" />
      <path fill="#05060c" d="M18 42c-4.2 1.2-8.4-1.6-9.6-6.2 3.8 1.2 6.6 4.2 7.4 7.8z" />
      <circle cx="30.2" cy="32.4" r="2.15" fill="#fbbf24" />
      <circle cx="37.4" cy="31.8" r="2.15" fill="#fbbf24" />
      <circle cx="30.7" cy="32" r="0.7" fill="#1a1204" />
      <circle cx="37.9" cy="31.4" r="0.7" fill="#1a1204" />
    </svg>
  );
}

function StarIcon() {
  return (
    <svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true">
      <path
        fill="currentColor"
        d="M8 1.2 9.7 5.6h4.6l-3.7 2.8 1.4 4.5L8 10.4 4 12.9l1.4-4.5L1.7 5.6h4.6z"
      />
    </svg>
  );
}

const MENU = [
  { id: "profile", label: "个人中心" },
  { id: "theme", label: "主题设置" },
  { id: "notify", label: "通知中心" },
  { id: "system", label: "系统设置" },
  { id: "logout", label: "退出登录", danger: true },
];

export default function ProfileChime() {
  const { user, authEnabled, logout } = useAuth();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState("");
  const root = useRef(null);

  const display = user?.display_name || user?.username || "本地访客";
  const handle = user?.username || (authEnabled ? "未登录" : "访客模式");

  useEffect(() => {
    if (!open) return undefined;
    const onDoc = (event) => {
      if (!root.current?.contains(event.target)) setOpen(false);
    };
    const onKey = (event) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const pick = (item) => {
    if (item.id === "logout") {
      if (!authEnabled || !user) {
        setNote("当前未开启登录，无需退出。");
        setOpen(false);
        return;
      }
      logout();
      setOpen(false);
      navigate("/login");
      return;
    }
    setNote(`${item.label}即将接入。`);
  };

  return (
    <div className={`profile-chime${open ? " is-open" : ""}`} ref={root}>
      <div className="chime-ornament">
        <span className="chime-pin" aria-hidden="true" />
        <span className="chime-cord" aria-hidden="true" />
        <svg className="chime-knot" viewBox="0 0 24 24" aria-hidden="true">
          <path
            fill="#fbbf24"
            d="M12 1.6 14.4 8.4H21l-5.4 4.1 2 6.5L12 15.2 6.4 19l2-6.5L3 8.4h6.6z"
          />
        </svg>
        <span className="chime-cord chime-cord-mid" aria-hidden="true" />
        <button
          type="button"
          className="chime-hit"
          aria-expanded={open}
          aria-haspopup="menu"
          aria-label="个人中心"
          onClick={() => {
            setOpen((cur) => !cur);
            setNote("");
          }}
        >
          <span className="chime-ring">
            <CatMoonAvatar className="chime-face" gid="chime-sky-main" />
          </span>
        </button>
        <span className="chime-cord chime-cord-tip" aria-hidden="true" />
        <svg className="chime-charm" viewBox="0 0 24 24" aria-hidden="true">
          <path fill="#f5d76e" d="M14.2 3.2a9 9 0 1 0 6.2 15.4 8.2 8.2 0 1 1-6.2-15.4z" />
        </svg>
      </div>
      {open ? (
        <div className="chime-menu" role="menu">
          <div className="chime-menu-bar" />
          <div className="chime-menu-head">
            <span className="chime-mini">
              <CatMoonAvatar gid="chime-sky-mini" />
            </span>
            <div>
              <strong>{display}</strong>
              <span>{handle}</span>
            </div>
          </div>
          {MENU.map((item) => (
            <button
              key={item.id}
              type="button"
              role="menuitem"
              className={`chime-item${item.danger ? " danger" : ""}`}
              onClick={() => pick(item)}
            >
              <StarIcon />
              {item.label}
            </button>
          ))}
          {note ? <p className="chime-note">{note}</p> : null}
        </div>
      ) : null}
    </div>
  );
}
