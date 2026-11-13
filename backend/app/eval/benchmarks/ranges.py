"""Bounded, immutable byte-range Parquet sampling for controlled Chinese reranking pilots."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen

from app.eval.benchmarks.adapters import adapt, validate_cases
from app.eval.benchmarks.data import DATA_ROOT, canonical, digest, load_snapshot, registry, write_json


class RangeFile(io.RawIOBase):
    def __init__(self, url: str, size: int, folder: Path, *, max_bytes: int = 32_000_000):
        self.url, self.size, self.folder, self.max_bytes = url, size, folder, max_bytes
        self.position, self.used = 0, 0
        folder.mkdir(parents=True, exist_ok=True)
        self.manifest_path = folder / "manifest.json"
        self.manifest = (
            json.loads(self.manifest_path.read_text())
            if self.manifest_path.exists()
            else {
                "url": url,
                "official_file_bytes": size,
                "ranges": {},
                "errors": [],
                "full_file_sha256_verified": False,
            }
        )
        if self.manifest["url"] != url or self.manifest["official_file_bytes"] != size:
            raise ValueError("range cache belongs to another source snapshot")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = self.position if whence == 1 else (self.size if whence == 2 else 0)
        position = base + offset
        if whence not in (0, 1, 2) or position < 0:
            raise ValueError("invalid seek")
        self.position = position
        return position

    def read(self, n=-1):
        n = min(self.size - self.position, n if n >= 0 else self.size)
        if n <= 0:
            return b""
        if self.used + n > self.max_bytes:
            raise ValueError(f"Parquet range pilot exceeds byte cap {self.max_bytes}; no full download fallback")
        start, end = self.position, self.position + n - 1
        key = f"{start}-{end}"
        path = self.folder / f"{key}.bin"
        previous = self.manifest["ranges"].get(key)
        try:
            if previous:
                blob = path.read_bytes()
                if len(blob) != n or digest(blob) != previous["sha256"]:
                    raise ValueError("immutable range cache checksum mismatch")
            else:
                with urlopen(
                    Request(
                        self.url,
                        headers={
                            "Range": f"bytes={key}",
                            "Accept-Encoding": "identity",
                            "User-Agent": "Provident-benchmark/1",
                        },
                    ),
                    timeout=30,
                ) as response:
                    expected_range = f"bytes {key}/{self.size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError("server did not honor exact byte range; refusing full download")
                    etag = response.headers.get("ETag")
                    if self.manifest.get("etag") and etag != self.manifest["etag"]:
                        raise ValueError("remote entity changed across range reads")
                    blob = response.read(n + 1)
                    if len(blob) != n:
                        raise ValueError(f"incomplete range: received {len(blob)} of {n} bytes")
                    self.manifest["etag"] = etag
                path.write_bytes(blob)
                self.manifest["ranges"][key] = {"path": path.name, "sha256": digest(blob), "bytes": n}
                write_json(self.manifest_path, self.manifest)
        except Exception as exc:
            self.manifest["errors"].append({"range": key, "error": f"{type(exc).__name__}: {exc}"})
            write_json(self.manifest_path, self.manifest)
            raise
        self.used += n
        self.position += n
        return blob


def sample_t2(*, limit: int = 20, root: Path = DATA_ROOT, max_bytes: int = 32_000_000) -> Path:
    if not 1 <= limit <= 20:
        raise ValueError("controlled range sampling is limited to 1..20 official rows")
    import pyarrow.parquet as pq

    name = "cmteb_t2"
    spec = registry(name)
    relative = next(x for x in spec["files"] if x.endswith(".parquet"))
    expected = spec["file_checksums"][relative]
    url = spec["base_url"].format(revision=spec["revision"]) + relative + "?download=true"
    folder = root / "raw" / name / spec["revision"] / "parquet_ranges" / digest(relative.encode())
    reader = RangeFile(url, expected["bytes"], folder, max_bytes=max_bytes)
    parquet = pq.ParquetFile(reader)
    rows = next(parquet.iter_batches(batch_size=limit)).to_pylist()
    cases = adapt(name, rows, relative=relative, spec=spec, options={"limit": limit})
    validate_cases(cases)
    snapshot = {
        "schema_version": 1,
        "adapter_version": "public-benchmark-v1",
        "benchmark": name,
        "dataset_revision": spec["revision"],
        "source_file": relative,
        "license": spec["license"],
        "usage_restrictions": spec["usage_restrictions"],
        "source": {
            "url": url,
            "download_mode": "pinned_parquet_byte_ranges",
            "official_file": expected,
            "full_file_sha256_verified": False,
            "range_manifest": reader.manifest,
            "source_row_count": parquet.metadata.num_rows,
            "selected_rows": [0, len(rows)],
        },
        "options": {"limit": limit, "range_byte_cap": max_bytes},
        "cases": cases,
    }
    snapshot["snapshot_sha256"] = digest(canonical(snapshot))
    target = root / "normalized" / name / f"{snapshot['snapshot_sha256']}.json"
    if target.exists():
        previous, _ = load_snapshot(target)
        if previous != snapshot:
            raise ValueError("immutable normalized range snapshot mismatch")
    else:
        write_json(target, snapshot)
    write_json(
        root / "t2-controlled-sample.json",
        {"snapshot": str(target.relative_to(root)).replace("\\", "/"), "snapshot_sha256": snapshot["snapshot_sha256"]},
    )
    return target
