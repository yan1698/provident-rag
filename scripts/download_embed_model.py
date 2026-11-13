"""Download multilingual-e5-small for offline embedding.

ModelScope is preferred. Without that package, files come from the HF mirror.
Only the PyTorch/tokenizer set is fetched; onnx and openvino builds are skipped.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = (
    ROOT
    / "data"
    / "runtime"
    / "models"
    / "modelscope"
    / "models"
    / "intfloat--multilingual-e5-small"
    / "snapshots"
    / "master"
)
MIRROR = "https://hf-mirror.com/intfloat/multilingual-e5-small/resolve/main"
MODELSCOPE = "https://www.modelscope.cn/api/v1/models/intfloat/multilingual-e5-small/repo?Revision=master&FilePath="
FILES = {
    "config.json": 655,
    "modules.json": 387,
    "sentence_bert_config.json": 57,
    "sentencepiece.bpe.model": 5069051,
    "special_tokens_map.json": 167,
    "tokenizer.json": 17082730,
    "tokenizer_config.json": 443,
    "1_Pooling/config.json": 200,
    "model.safetensors": 470641600,
}


def _open(url: str):
    """Follow 308. Python 3.9's urllib only redirects 301/302/303/307."""
    for _ in range(6):
        req = urllib.request.Request(url, headers={"User-Agent": "provident-embed-download"})
        try:
            return urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as exc:
            if exc.code not in (301, 302, 303, 307, 308):
                raise
            location = exc.headers.get("Location")
            if not location:
                raise
            url = urllib.parse.urljoin(url, location)
    raise RuntimeError(f"too many redirects: {url}")


def _download_file(url: str, dest: Path, expected: int) -> None:
    if dest.is_file() and dest.stat().st_size == expected:
        label = dest.name
        try:
            label = dest.relative_to(TARGET).as_posix()
        except ValueError:
            pass
        print(f"skip {label} ({expected} bytes)", flush=True)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    print(f"get {dest.name}", flush=True)
    written = 0
    with _open(url) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
            if written % (32 * 1024 * 1024) < 1024 * 1024:
                print(f"  {written}/{expected}", flush=True)
    if dest.stat().st_size != expected:
        raise RuntimeError(f"{dest.name} is {dest.stat().st_size} bytes, expected {expected}")
    print(f"  {expected} bytes", flush=True)


def _download_mirror() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        dest = TARGET / name
        # Large weights are served completely from ModelScope. The HF mirror
        # redirects those to huggingface.co and the transfer dies part way.
        url = MODELSCOPE + urllib.parse.quote(name) if expected > 50_000_000 else f"{MIRROR}/{name}"
        try:
            _download_file(url, dest, expected)
        except Exception:
            if url.startswith(MIRROR):
                _download_file(MODELSCOPE + urllib.parse.quote(name), dest, expected)
            else:
                raise


def main() -> int:
    try:
        from modelscope import snapshot_download
    except ImportError:
        snapshot_download = None

    if snapshot_download is not None:
        path = snapshot_download(
            "intfloat/multilingual-e5-small",
            cache_dir=str(ROOT / "data" / "runtime" / "models" / "modelscope"),
        )
        print(path)
    else:
        print("modelscope is not installed; downloading from hf-mirror")
        _download_mirror()
        print(TARGET)

    print("\nSet in .env:")
    print(f"EMBEDDING_MODEL_PATH={TARGET}")
    print('EMBEDDING_QUERY_PREFIX="query: "')
    print('EMBEDDING_PASSAGE_PREFIX="passage: "')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
