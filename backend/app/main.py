from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from app.api.agent import router as agent_router
from app.api.auth import router as auth_router
from app.api.embed import router as embed_router
from app.api.eval import router as eval_router
from app.api.feedback import router as feedback_router
from app.api.graph import router as graph_router
from app.api.ingest import router as ingest_router
from app.api.library import router as library_router
from app.api.metrics import router as metrics_router
from app.api.query import router as query_router
from app.api.search import router as search_router
from app.api.tags import router as tags_router
from app.api.traces import router as traces_router
from app.api.verify import router as verify_router
from app.core.config import get_settings, validate_runtime_timeouts
from app.core.log import configure_logging
from app.core.request_id import RequestIdMiddleware
from app.core.startup import assert_production_ready, warn_if_insecure_defaults
from app.core.version import API_VERSION
from app.db.session import engine, init_db

settings = get_settings()


def _web_dir() -> Path:
    here = Path(__file__).resolve()
    roots = (
        here.parents[2] / "web",
        here.parents[1] / "web",
        Path("/app/web"),
    )
    for root in roots:
        dist = root / "dist"
        if (dist / "index.html").is_file():
            return dist
    return roots[0] / "dist"


def _check_postgres() -> dict[str, Any]:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            vector_row = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'")).fetchone()
        return {
            "ok": True,
            "pgvector": vector_row is not None,
        }
    except SQLAlchemyError as exc:
        return {"ok": False, "error": str(exc.__class__.__name__)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc.__class__.__name__)}


def _check_redis() -> dict[str, Any]:
    try:
        client = redis.from_url(settings.redis_connection_url, socket_connect_timeout=2)
        pong = client.ping()
        return {"ok": bool(pong)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc.__class__.__name__)}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    configure_logging()
    validate_runtime_timeouts(settings)
    assert_production_ready(settings)
    warn_if_insecure_defaults(settings)
    init_db()
    yield


app = FastAPI(
    title="Provident RAG API",
    description="Engineering-grade RAG platform + orchestrable Agents",
    version=API_VERSION,
    lifespan=lifespan,
)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(auth_router)
app.include_router(ingest_router)
app.include_router(library_router)
app.include_router(embed_router)
app.include_router(search_router)
app.include_router(query_router)
app.include_router(verify_router)
app.include_router(traces_router)
app.include_router(tags_router)
app.include_router(eval_router)
app.include_router(agent_router)
app.include_router(feedback_router)
app.include_router(graph_router)
app.include_router(metrics_router)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "provident-rag",
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health",
        "ui": "/ui",
        "ingest": "/ingest",
        "embed": "/embed",
        "auth": "/auth/login",
        "auth_enabled": settings.auth_enabled,
        "documents": "/documents",
        "search_dense": "/search/dense",
        "search_sparse": "/search/sparse",
        "search_hybrid": "/search/hybrid",
        "search_rerank": "/search/rerank",
        "query": "/query",
        "verify": "/verify",
        "traces": "/traces",
        "tags": "/tags",
        "eval_summary": "/eval/summary",
        "eval_summary_legacy": True,
        "eval_gold": "/eval/gold",
        "eval_report": "/eval/report",
        "eval_replay": "/eval/replay",
        "eval_compare": "/eval/compare",
        "eval_graph": "/eval/graph",
        "agent": "/agent/run",
        "agent_stream": "/agent/{id}/stream",
        "agent_cancel": "/agent/{id}/cancel",
        "feedback": "/feedback",
        "graph": "/graph",
        "metrics": "/metrics",
        "metrics_llm": "/metrics/llm",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    """Verify Postgres + Redis connectivity."""
    postgres = _check_postgres()
    redis_status = _check_redis()
    healthy = bool(postgres.get("ok")) and bool(redis_status.get("ok"))
    pool = engine.pool
    return {
        "status": "ok" if healthy else "degraded",
        "env": settings.app_env,
        "postgres": postgres,
        "redis": redis_status,
        "db_pool": {
            "size": int(pool.size()),
            "checked_out": int(pool.checkedout()),
            "overflow": int(pool.overflow()),
            "checked_in": int(pool.checkedin()),
        },
    }


@app.get("/ui/assets/{filename}")
def ui_asset(filename: str):
    assets = (_web_dir() / "assets").resolve()
    path = (assets / filename).resolve()
    try:
        path.relative_to(assets)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="asset not found") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="asset not found")
    media = {
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".svg": "image/svg+xml",
        ".ico": "image/x-icon",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(path.suffix.lower())
    headers = {"Cache-Control": "no-store, max-age=0"}
    return FileResponse(path, media_type=media, headers=headers)


@app.get("/ui")
@app.get("/ui/{path:path}")
def ui_spa(path: str = ""):
    index = _web_dir() / "index.html"
    if not index.is_file():
        raise HTTPException(
            status_code=404,
            detail="UI not found. Build the React app (cd web && npm run build) and mount ./web onto the API container.",
        )
    return FileResponse(
        index,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store, max-age=0"},
    )
