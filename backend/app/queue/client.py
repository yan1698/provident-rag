"""Redis list queue. No arq — stays off requirements.txt so the image is not rebuilt."""

from __future__ import annotations

import json
import threading
from typing import Any
from uuid import UUID

from app.core.config import get_settings
from app.core.log import get_logger

_log = get_logger("queue")
QUEUE_KEY = "provident:jobs"
PROCESSING_KEY = "provident:processing"
WORKER_PRESENCE_PREFIX = "provident:worker:"
RUN_OWNER_PREFIX = "provident:run-owner:"
LOCK_TTL = 7200
CANCEL_TTL = 3600

_memory_lists: dict[str, list[str]] = {}
_memory_kv: dict[str, str] = {}
_memory_hashes: dict[str, dict[str, str]] = {}
_memory_lock = threading.Lock()
_force_memory = False


def use_memory_queue() -> None:
    """Unit tests: never touch live Redis."""
    global _force_memory
    _force_memory = True
    reset_memory_queue()


def reset_memory_queue() -> None:
    with _memory_lock:
        _memory_lists.clear()
        _memory_kv.clear()
        _memory_hashes.clear()


def _owner_key(owner_id: UUID | None) -> str:
    return str(owner_id) if owner_id is not None else "anon"


def _redis(*, block: bool = False):
    if _force_memory:
        return None
    try:
        import redis

        settings = get_settings()
        client = redis.from_url(
            settings.redis_connection_url,
            socket_connect_timeout=1,
            socket_timeout=None if block else 1,
        )
        if not block:
            client.ping()
        return client
    except Exception:  # noqa: BLE001 — fail over to memory / in-process
        return None


def enqueue_agent(
    run_id: UUID,
    *,
    max_steps: int | None = None,
    planner: str | None = None,
    intent: str | None = None,
    owner_id: UUID | None = None,
) -> None:
    job = {
        "kind": "agent",
        "run_id": str(run_id),
        "max_steps": max_steps,
        "planner": planner,
        "intent": intent,
        "owner_id": str(owner_id) if owner_id is not None else None,
    }
    return _push(job)


def enqueue_embed(*, owner_id: UUID | None = None) -> bool:
    if not _acquire_lock("embed", owner_id):
        return False
    job = {"kind": "embed", "owner_id": str(owner_id) if owner_id is not None else None}
    _write_state("embed", owner_id, {"running": "1", "error": ""})
    if not _push(job):
        _run_inline(job)
    return True


def enqueue_graph(
    *,
    all_docs: bool = False,
    extract: bool = False,
    doc_ids: list | None = None,
    owner_id: UUID | None = None,
) -> bool:
    if not _acquire_lock("graph", owner_id):
        return False
    job = {
        "kind": "graph",
        "all_docs": all_docs,
        "extract": extract,
        "doc_ids": [str(item) for item in (doc_ids or [])],
        "owner_id": str(owner_id) if owner_id is not None else None,
    }
    _write_state("graph", owner_id, {"running": "1", "error": ""})
    if not _push(job):
        _run_inline(job)
    return True


def cancel_agent(run_id: UUID) -> bool:
    """Write the cancel flag. False means this process could not publish it."""
    key = f"provident:cancel:{run_id}"
    client = _redis()
    if client is not None:
        try:
            client.set(key, "1", ex=CANCEL_TTL)
        except Exception:  # noqa: BLE001 — caller must not report a cancel that was not published
            return False
        return True
    if not _force_memory:
        return False
    with _memory_lock:
        _memory_kv[key] = "1"
    return True


def is_cancelled(run_id: UUID) -> bool:
    key = f"provident:cancel:{run_id}"
    client = _redis()
    if client is not None:
        return bool(client.get(key))
    with _memory_lock:
        return key in _memory_kv


def clear_cancel(run_id: UUID) -> None:
    key = f"provident:cancel:{run_id}"
    client = _redis()
    if client is not None:
        client.delete(key)
        return
    with _memory_lock:
        _memory_kv.pop(key, None)


def job_state(kind: str, owner_id: UUID | None = None) -> dict[str, Any]:
    raw = _read_state(kind, owner_id)
    running = str(raw.get("running") or "") in {"1", "true", "True"}
    out: dict[str, Any] = {"running": running, "error": raw.get("error") or None, "note": raw.get("note") or None}
    for name in (
        "skipped_chunks",
        "extract_calls",
        "extract_failed",
        "extract_empty",
        "extract_dropped",
    ):
        try:
            out[name] = int(raw.get(name) or 0)
        except (TypeError, ValueError):
            out[name] = 0
    out["source"] = raw.get("source") or "rules"
    return out


def write_job_state(kind: str, owner_id: UUID | None, values: dict[str, Any]) -> None:
    payload = {key: "" if value is None else str(value) for key, value in values.items()}
    if "running" in payload:
        payload["running"] = "1" if str(payload["running"]).lower() in {"1", "true"} else "0"
    _write_state(kind, owner_id, payload)


def release_lock(kind: str, owner_id: UUID | None) -> None:
    key = f"provident:lock:{kind}:{_owner_key(owner_id)}"
    client = _redis()
    if client is not None:
        client.delete(key)
        return
    with _memory_lock:
        _memory_kv.pop(key, None)


def pop_job(*, timeout: int = 5) -> dict[str, Any] | None:
    """Atomically move the oldest job into the processing list."""
    client = _redis(block=True)
    if client is not None:
        try:
            raw = client.blmove(QUEUE_KEY, PROCESSING_KEY, timeout=timeout, src="RIGHT", dest="LEFT")
        except Exception:  # noqa: BLE001 — a lost connection is not an accepted job
            _log.exception("blmove failed")
            return None
        if not raw:
            return None
        return _decode_job(raw)
    with _memory_lock:
        items = _memory_lists.get(QUEUE_KEY) or []
        if not items:
            return None
        raw = items.pop()
        _memory_lists.setdefault(PROCESSING_KEY, []).insert(0, raw)
        return _decode_job(raw)


def _decode_job(raw: Any) -> dict[str, Any]:
    text_raw = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    job = json.loads(text_raw)
    job["_raw"] = raw if isinstance(raw, bytes) else text_raw
    return job


def _job_raw(job: dict[str, Any]) -> bytes | str:
    raw = job.get("_raw")
    if raw is not None:
        return raw
    visible = {key: value for key, value in job.items() if key != "_raw"}
    return json.dumps(visible, ensure_ascii=False)


def list_processing_jobs() -> list[dict[str, Any]]:
    return _list_jobs(PROCESSING_KEY)


def ack_job(job: dict[str, Any]) -> None:
    """Remove the processing record for this payload only."""
    raw = _job_raw(job)
    client = _redis()
    if client is not None:
        client.lrem(PROCESSING_KEY, 1, raw)
        return
    _memory_remove(PROCESSING_KEY, raw)


def requeue_job(job: dict[str, Any]) -> None:
    raw = _job_raw(job)
    client = _redis()
    if client is not None:
        client.lrem(PROCESSING_KEY, 1, raw)
        client.lpush(QUEUE_KEY, raw)
        return
    _memory_remove(PROCESSING_KEY, raw)
    with _memory_lock:
        _memory_lists.setdefault(QUEUE_KEY, []).insert(0, raw if isinstance(raw, str) else raw.decode("utf-8"))


def register_worker(worker_id: str, ttl: int) -> bool:
    """Refresh this process's lease. False when Redis did not store it."""
    client = _redis()
    if client is None or not worker_id:
        return False
    try:
        client.set(f"{WORKER_PRESENCE_PREFIX}{worker_id}", "1", ex=max(int(ttl), 1))
    except Exception:  # noqa: BLE001
        _log.exception("worker lease refresh failed")
        return False
    return True


def clear_worker(worker_id: str) -> None:
    client = _redis()
    if client is None or not worker_id:
        return
    try:
        client.delete(f"{WORKER_PRESENCE_PREFIX}{worker_id}")
    except Exception:  # noqa: BLE001
        _log.exception("worker lease delete failed")


def bind_run_executor(run_id: UUID, worker_id: str, ttl: int) -> bool:
    """Remember which worker claimed this run. A different holder cannot take the key."""
    client = _redis()
    if client is None or not worker_id:
        return False
    key = f"{RUN_OWNER_PREFIX}{run_id}"
    ttl = max(int(ttl), 1)
    try:
        if client.set(key, worker_id, ex=ttl, nx=True):
            return True
        current = client.get(key)
        if isinstance(current, bytes):
            current = current.decode("utf-8")
        if current == worker_id:
            client.set(key, worker_id, ex=ttl)
            return True
        if not current and client.set(key, worker_id, ex=ttl, nx=True):
            return True
    except Exception:  # noqa: BLE001
        _log.exception("run owner bind failed")
        return False
    return False


def lease_allows_write(
    owner: str | None,
    presence: bool,
    worker_id: str,
    *,
    redis_known: bool,
) -> bool:
    """Whether this process may write the run.

    Redis being down is not evidence that someone else owns it. A recorded
    owner must match, and that owner's lease must still be present.
    """
    if not redis_known:
        return True
    if not owner:
        return True
    if owner != worker_id:
        return False
    return bool(presence)


def writer_owns(run_id: UUID, worker_id: str) -> bool:
    """Redis view of lease_allows_write. Unknown Redis returns True."""
    client = _redis()
    if client is None or not worker_id:
        return lease_allows_write(None, False, worker_id, redis_known=False)
    try:
        owner = client.get(f"{RUN_OWNER_PREFIX}{run_id}")
        if isinstance(owner, bytes):
            owner = owner.decode("utf-8")
        presence = False
        if owner:
            presence = bool(client.exists(f"{WORKER_PRESENCE_PREFIX}{owner}"))
        return lease_allows_write(owner or None, presence, worker_id, redis_known=True)
    except Exception:  # noqa: BLE001
        _log.exception("lease ownership check failed")
        return True


def executor_liveness(run_id: UUID) -> str:
    """alive, dead, or unknown. Unknown means Redis has no contrary evidence."""
    client = _redis()
    if client is None:
        return "unknown"
    try:
        owner = client.get(f"{RUN_OWNER_PREFIX}{run_id}")
        if not owner:
            return "unknown"
        if isinstance(owner, bytes):
            owner = owner.decode("utf-8")
        if client.exists(f"{WORKER_PRESENCE_PREFIX}{owner}"):
            return "alive"
        return "dead"
    except Exception:  # noqa: BLE001
        _log.exception("executor liveness check failed")
        return "unknown"


def queued_run_ids() -> set[str]:
    found: set[str] = set()
    for key in (QUEUE_KEY, PROCESSING_KEY):
        for job in _list_jobs(key):
            if job.get("kind") == "agent" and job.get("run_id"):
                found.add(str(job["run_id"]))
    return found


def _list_jobs(key: str) -> list[dict[str, Any]]:
    client = _redis()
    if client is not None:
        rows = client.lrange(key, 0, -1) or []
        return [_decode_job(raw) for raw in rows]
    with _memory_lock:
        return [_decode_job(raw) for raw in list(_memory_lists.get(key) or [])]


def _memory_remove(key: str, raw: bytes | str) -> None:
    text_raw = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    with _memory_lock:
        items = _memory_lists.get(key) or []
        for index, item in enumerate(items):
            current = item.decode("utf-8") if isinstance(item, bytes) else item
            if current == text_raw:
                items.pop(index)
                break


def _push(job: dict[str, Any]) -> bool:
    payload = json.dumps(job, ensure_ascii=False)
    client = _redis()
    if client is not None:
        try:
            client.lpush(QUEUE_KEY, payload)
        except Exception:  # noqa: BLE001 — the caller keeps the committed row and reports deferred
            _log.exception("queue push failed")
            return False
        return True
    if not _force_memory:
        return False
    with _memory_lock:
        _memory_lists.setdefault(QUEUE_KEY, []).insert(0, payload)
    return True


def _acquire_lock(kind: str, owner_id: UUID | None) -> bool:
    key = f"provident:lock:{kind}:{_owner_key(owner_id)}"
    client = _redis()
    if client is not None:
        return bool(client.set(key, "1", ex=LOCK_TTL, nx=True))
    with _memory_lock:
        if key in _memory_kv:
            return False
        _memory_kv[key] = "1"
        return True


def _state_key(kind: str, owner_id: UUID | None) -> str:
    return f"provident:state:{kind}:{_owner_key(owner_id)}"


def _write_state(kind: str, owner_id: UUID | None, values: dict[str, str]) -> None:
    key = _state_key(kind, owner_id)
    client = _redis()
    if client is not None:
        if values:
            client.hset(key, mapping=values)
            client.expire(key, LOCK_TTL)
        return
    with _memory_lock:
        _memory_hashes.setdefault(key, {}).update(values)


def _read_state(kind: str, owner_id: UUID | None) -> dict[str, str]:
    key = _state_key(kind, owner_id)
    client = _redis()
    if client is not None:
        raw = client.hgetall(key) or {}
        out = {}
        for name, value in raw.items():
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            out[str(name)] = str(value)
        return out
    with _memory_lock:
        return dict(_memory_hashes.get(key) or {})


def _run_inline(job: dict[str, Any]) -> None:
    _log.info("redis unavailable, running %s in-process", job.get("kind"))
    threading.Thread(target=_inline_dispatch, args=(job,), daemon=True, name=f"job-{job.get('kind')}").start()


def _inline_dispatch(job: dict[str, Any]) -> None:
    from app.queue.worker import dispatch

    dispatch(job)


def selfcheck() -> None:
    from uuid import uuid4

    use_memory_queue()
    owner = uuid4()
    if not enqueue_embed(owner_id=owner):
        raise SystemExit("first embed should lock")
    if enqueue_embed(owner_id=owner):
        raise SystemExit("second embed should wait on lock")
    if not job_state("embed", owner).get("running"):
        raise SystemExit("embed state should be running")
    job = pop_job(timeout=0)
    if not job or job.get("kind") != "embed":
        raise SystemExit(f"expected embed job, got {job}")
    run_id = uuid4()
    if is_cancelled(run_id):
        raise SystemExit("fresh run is not cancelled")
    cancel_agent(run_id)
    if not is_cancelled(run_id):
        raise SystemExit("cancel flag should be visible")
    print("queue memory locks + cancel ok")


if __name__ == "__main__":
    selfcheck()
