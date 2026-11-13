"""Decision table for run-lease writes. No Redis and no database."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_backend = ROOT / "backend"
if _backend.is_dir():
    sys.path.insert(0, str(_backend))
elif (ROOT / "app").is_dir():
    sys.path.insert(0, str(ROOT))

from app.queue.client import lease_allows_write  # noqa: E402

# owner, presence, worker_id, redis_known, allowed
CASES = (
    (None, False, "worker-a", False, True),
    (None, False, "worker-a", True, True),
    ("", False, "worker-a", True, True),
    ("worker-b", True, "worker-a", True, False),
    ("worker-a", False, "worker-a", True, False),
    ("worker-a", True, "worker-a", True, True),
)


def main() -> None:
    for owner, presence, worker_id, redis_known, allowed in CASES:
        got = lease_allows_write(
            owner,
            presence,
            worker_id,
            redis_known=redis_known,
        )
        if got is not allowed:
            raise SystemExit(
                "FAIL: "
                f"owner={owner!r} presence={presence} worker={worker_id} "
                f"redis_known={redis_known} -> {got}, want {allowed}"
            )
    print("lease rules ok")


if __name__ == "__main__":
    main()
