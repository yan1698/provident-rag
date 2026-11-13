import { NavLink, Link } from "react-router-dom";
import { useAuth } from "../auth";
import ProfileChime from "./ProfileChime";

export default function Layout({ children }) {
  const { evalGate } = useAuth();
  return (
    <>
      <div className="sky" aria-hidden="true" />
      <ProfileChime />
      <div className="app-shell">
        <header className="topbar">
          <Link className="brand" to="/agent">
            <strong>Provident</strong>
            <span>问你的文档，每句都能点回原文</span>
          </Link>
          <nav className="nav">
            <NavLink to="/agent">Agent</NavLink>
            <NavLink to="/library">知识库</NavLink>
            <NavLink to="/graph">知识图谱</NavLink>
            <NavLink to="/traces">历史</NavLink>
          </nav>
          <nav className="nav nav-aux">
            {evalGate ? <NavLink to="/eval">回归闸门</NavLink> : null}
          </nav>
        </header>
        <main className="page">{children}</main>
      </div>
    </>
  );
}
