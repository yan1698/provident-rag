"""Immutable official downloads and content-addressed conversions (stdlib except Parquet)."""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[4]
REGISTRY_PATH = ROOT / "benchmarks" / "registry.json"
DATA_ROOT = ROOT / "data" / "benchmarks"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    temporary.replace(path)


def registry(name: str) -> dict:
    spec = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))[name]
    if not re.fullmatch(r"[a-f0-9]{40}", spec["revision"]):
        raise ValueError("immutable 40-character revision required")
    return spec


def validate_blob(relative: str, blob: bytes, spec: dict) -> None:
    if blob.startswith(b"version https://git-lfs.github.com/spec"):
        raise ValueError("received Git LFS pointer instead of data")
    if relative.endswith(".parquet") and (blob[:4] != b"PAR1" or blob[-4:] != b"PAR1"):
        raise ValueError("incomplete Parquet transfer: missing header/footer magic")
    expected = spec.get("file_checksums", {}).get(relative)
    if expected and (digest(blob) != expected["sha256"] or len(blob) != expected["bytes"]):
        raise ValueError("file does not match official LFS SHA256/size")


def fetch(
    name: str,
    *,
    root: Path = DATA_ROOT,
    max_bytes: int = 16_000_000,
    timeout: float = 30,
    files: list[str] | None = None,
    retry_invalid: bool = False,
) -> dict:
    spec = registry(name)
    folder = root / "raw" / name / spec["revision"]
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {
            "benchmark": name,
            "revision": spec["revision"],
            "source": spec["source"],
            "license": spec["license"],
            "usage_restrictions": spec["usage_restrictions"],
            "files": {},
        }
    )
    manifest.update(license=spec["license"], usage_restrictions=spec["usage_restrictions"])
    selected = files if files is not None else spec["files"]
    if not selected or not set(selected) <= set(spec["files"] + spec.get("optional_files", [])):
        raise ValueError("download files must be explicitly registered")
    for relative in selected:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("unsafe registered path")
        target = folder / relative
        url = spec["base_url"].format(revision=spec["revision"]) + relative
        previous = manifest["files"].get(relative)
        if target.exists():
            if not previous or previous.get("sha256") != digest(target.read_bytes()):
                raise ValueError(f"raw file checksum mismatch: {relative}; never overwritten")
            try:
                validate_blob(relative, target.read_bytes(), spec)
            except ValueError as exc:
                if not retry_invalid:
                    raise ValueError(
                        f"{relative}: {exc}; use --retry-invalid to preserve and retry failed bytes"
                    ) from exc
                archived = target.with_name(target.name + ".failed-" + previous["sha256"])
                if archived.exists():
                    raise ValueError("failed artifact already exists; inspect before another retry")
                target.rename(archived)
                manifest.setdefault("failed_artifacts", []).append(
                    {**previous, "path": str(archived.relative_to(folder)), "reason": str(exc)}
                )
            else:
                continue
        try:
            req = Request(url, headers={"User-Agent": "Provident-public-benchmark/1"})
            started = time.monotonic()
            with urlopen(req, timeout=timeout) as response:
                length = response.headers.get("Content-Length")
                if length and int(length) > max_bytes:
                    raise ValueError(f"declared file size {length} exceeds cap {max_bytes}")
                chunks, size = [], 0
                while True:
                    if time.monotonic() - started > timeout * 10:
                        raise TimeoutError("total download deadline exceeded")
                    chunk = response.read(min(64 * 1024, max_bytes + 1 - size))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError(f"download exceeds per-file cap {max_bytes}")
                blob = b"".join(chunks)
                if length and len(blob) != int(length):
                    raise ValueError(f"incomplete HTTP transfer: received {len(blob)} of {length} bytes")
            if len(blob) > max_bytes:
                raise ValueError(f"download exceeds per-file cap {max_bytes}")
            validate_blob(relative, blob, spec)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(blob)
            manifest["files"][relative] = {"url": url, "sha256": digest(blob), "bytes": len(blob)}
        except Exception as exc:
            manifest["files"][relative] = {"url": url, "error": f"{type(exc).__name__}: {exc}"}
            write_json(manifest_path, manifest)
            raise RuntimeError(f"download failed: {url}: {type(exc).__name__}: {exc}") from exc
        write_json(manifest_path, manifest)
    return manifest


def verified_raw(name: str, relative: str, root: Path = DATA_ROOT) -> tuple[Path, dict]:
    spec = registry(name)
    folder = root / "raw" / name / spec["revision"]
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    item = manifest["files"][relative]
    path = folder / relative
    if digest(path.read_bytes()) != item.get("sha256"):
        raise ValueError(f"raw checksum mismatch: {relative}")
    validate_blob(relative, path.read_bytes(), spec)
    return path, item


def read_rows(path: Path, limit: int | None = None):
    if limit is not None and limit < 1:
        raise ValueError("conversion limit must be positive")
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Parquet conversion needs benchmarks/requirements.txt; no model dependency required"
            ) from exc
        if limit is not None:
            batches = pq.ParquetFile(path).iter_batches(batch_size=limit)
            return next(batches).to_pylist()
        return pq.read_table(path).to_pylist()
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        rows = json.loads(text)
        return rows[:limit] if limit else rows
    if text.lstrip().startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            pass  # RGB is JSONL despite its .json suffix.
        else:
            return obj
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def load_snapshot(path: Path) -> tuple[dict, list[dict]]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot.get("schema_version") != 1:
        raise ValueError("unsupported snapshot schema")
    content = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    if snapshot.get("snapshot_sha256") != digest(canonical(content)):
        raise ValueError("normalized snapshot checksum mismatch")
    from app.eval.benchmarks.adapters import validate_cases

    validate_cases(snapshot["cases"])
    return snapshot, snapshot["cases"]


def convert(name: str, relative: str, *, root: Path = DATA_ROOT, **options) -> Path:
    from app.eval.benchmarks.adapters import adapt, validate_cases

    path, source = verified_raw(name, relative, root)
    spec = registry(name)
    rows = read_rows(path, limit=options.get("limit") if name != "crud" else None)
    cases = adapt(name, rows, relative=relative, spec=spec, options=options)
    auxiliary = {}
    if name == "crud":
        for prompt in sorted({c["prompt_file"] for c in cases}):
            _, auxiliary[prompt] = verified_raw("crud", prompt, root)
    if name == "cmteb_du":
        qspec = registry("cmteb_du_qrels")
        qfile = next(f for f in qspec["files"] if f.endswith(".parquet"))
        qpath, qsource = verified_raw("cmteb_du_qrels", qfile, root)
        qrels = {}
        for row in read_rows(qpath):
            qrels.setdefault(row["qid"], {})[row["pid"]] = row["score"]
        for case in cases:
            if case["sample_id"] not in qrels:
                raise ValueError("DuRetrieval query has no official relevance judgments")
            case["relevance_labels"] = qrels[case["sample_id"]]
        auxiliary["qrels"] = {**qsource, "revision": qspec["revision"], "source": qspec["source"]}
    validate_cases(cases)
    snapshot = {
        "schema_version": 1,
        "adapter_version": "public-benchmark-v1",
        "benchmark": name,
        "dataset_revision": spec["revision"],
        "source_file": relative,
        "source": source,
        "license": spec["license"],
        "usage_restrictions": spec["usage_restrictions"],
        "options": options,
        "cases": cases,
        "auxiliary_sources": auxiliary,
    }
    snapshot["snapshot_sha256"] = digest(canonical(snapshot))
    target = root / "normalized" / name / (snapshot["snapshot_sha256"] + ".json")
    if target.exists():
        existing, _ = load_snapshot(target)
        if existing != snapshot:
            raise ValueError("conversion collision")
    else:
        write_json(target, snapshot)
    return target
