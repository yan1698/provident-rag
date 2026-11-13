"""Bounded process-local LRU. Snapshots are built privately and published atomically."""

from __future__ import annotations

import os
import sys
import threading
from collections import Counter, OrderedDict
from dataclasses import dataclass
from types import MappingProxyType

from app.search.sparse_version import Uncacheable
from app.search.stages import best_effort, note_meta, stage_timer


def deep_size(value) -> int:
    """Owned Python object graph estimate, not RSS or a hard allocator limit."""
    seen = set()

    def visit(obj):
        identity = id(obj)
        if identity in seen:
            return 0
        seen.add(identity)
        size = sys.getsizeof(obj)
        if isinstance(obj, (dict, MappingProxyType)):
            # Proxy overhead plus a conservative estimate for its hidden dict.
            if isinstance(obj, MappingProxyType):
                size += sys.getsizeof(dict(obj))
            size += sum(visit(k) + visit(v) for k, v in obj.items())
        elif isinstance(obj, (tuple, list)):
            size += sum(visit(v) for v in obj)
        elif hasattr(obj, "__dict__"):
            size += visit(vars(obj))
        return size

    return visit(value)


@dataclass(frozen=True)
class SnapshotRow:
    id: object
    doc_id: object
    page: int
    start_offset: int
    end_offset: int
    snippet: str


@dataclass(frozen=True)
class SparseSnapshot:
    rows: tuple[SnapshotRow, ...]
    # Private ownership: only get_scores reads the model after publication.
    index: object
    estimated_bytes: int


@dataclass(frozen=True)
class CacheLimits:
    entries: int
    bytes: int
    wait_seconds: float


class SnapshotCache:
    def __init__(self):
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._items = OrderedDict()
        self._building = {}
        self._bytes = 0
        self._counts = Counter()

    def clear(self):
        """Administrative/test reset. Never race an active builder."""
        with self._lock:
            if self._building:
                raise RuntimeError("cannot clear while building")
            self._items.clear()
            self._bytes = 0
            self._counts.clear()

    def stats(self):
        with self._lock:
            return {
                **self._counts,
                "entries": len(self._items),
                "estimated_bytes": self._bytes,
                "builders": len(self._building),
            }

    def fallback(self, reason):
        if os.getpid() != self._pid:
            note_meta("sparse_index_cache_status", "fallback")
            note_meta("sparse_index_cache_reason", "forked_process")
            return
        with self._lock:
            self._counts["fallback"] += 1
            self._counts["fallback_" + reason] += 1
        self._diagnostics("fallback", reason)

    @best_effort
    def _diagnostics(self, status, reason=""):
        note_meta("sparse_index_cache_status", status)
        note_meta("sparse_index_cache_reason", reason)
        note_meta("sparse_index_reused", int(status == "hit"))
        for name, value in self.stats().items():
            note_meta("sparse_cache_" + name, value)

    def _remove(self, base, reason):
        _, snapshot = self._items.pop(base)
        self._bytes -= snapshot.estimated_bytes
        self._counts[reason] += 1

    def get(self, base, stamp, build, validate, limits: CacheLimits):
        if os.getpid() != self._pid:
            # Do not touch inherited locks in forked workers; safe legacy fallback.
            raise Uncacheable("forked_process")
        if limits.entries < 1 or limits.bytes < 1:
            raise Uncacheable("budget_disabled")
        leader = False
        with self._lock:
            while self._items and (len(self._items) > limits.entries or self._bytes > limits.bytes):
                self._remove(next(iter(self._items)), "eviction")
            entry = self._items.get(base)
            if entry and entry[0] != stamp:
                self._remove(base, "invalidation")
                entry = None
            if entry:
                self._items.move_to_end(base)
                self._counts["hit"] += 1
                result = entry[1]
            else:
                result = None
                pending = self._building.get(base)
                if pending is None:
                    # One cache builder per process bounds simultaneous construction.
                    if self._building:
                        raise Uncacheable("other_key_building")
                    pending = threading.Event()
                    self._building[base] = pending
                    self._counts["cold_build"] += 1
                    leader = True
                else:
                    self._counts["wait"] += 1
        if result is not None:
            self._diagnostics("hit")
            return result
        if not leader:
            with stage_timer("sparse_cache_wait_ms"):
                ready = pending.wait(max(0, limits.wait_seconds))
            if not ready:
                raise Uncacheable("wait_timeout")
            # Re-probe committed state after waiting, never trust the leader's stamp.
            fresh = validate()
            with self._lock:
                entry = self._items.get(base)
                if entry is None or entry[0] != fresh:
                    raise Uncacheable("wait_no_valid_snapshot")
                self._items.move_to_end(base)
                self._counts["hit"] += 1
                result = entry[1]
            self._diagnostics("hit")
            return result
        try:
            result = build()
            if validate() != stamp:
                raise Uncacheable("changed_during_build")
            if result.estimated_bytes > limits.bytes:
                raise Uncacheable("snapshot_over_budget")
            with self._lock:
                while self._items and (
                    len(self._items) >= limits.entries or self._bytes + result.estimated_bytes > limits.bytes
                ):
                    self._remove(next(iter(self._items)), "eviction")
                self._items[base] = (stamp, result)
                self._bytes += result.estimated_bytes
                self._counts["publish"] += 1
            self._diagnostics("cold_build")
            return result
        except BaseException:
            with self._lock:
                self._counts["build_failed"] += 1
            raise
        finally:
            with self._lock:
                self._building.pop(base, None)
                pending.set()


CACHE = SnapshotCache()


def process_rss_bytes():
    """Current process RSS/Windows working set. Never a model or remote call."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class Counters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    *[
                        (name, ctypes.c_size_t)
                        for name in (
                            "PeakWorkingSetSize",
                            "WorkingSetSize",
                            "QuotaPeakPagedPoolUsage",
                            "QuotaPagedPoolUsage",
                            "QuotaPeakNonPagedPoolUsage",
                            "QuotaNonPagedPoolUsage",
                            "PagefileUsage",
                            "PeakPagefileUsage",
                        )
                    ],
                ]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            data = Counters()
            data.cb = ctypes.sizeof(data)
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(data), data.cb):
                return int(data.WorkingSetSize)
        elif sys.platform.startswith("linux"):
            from pathlib import Path

            return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        pass
    return None
