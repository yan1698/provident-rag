"""Download BAAI/bge-reranker-base via ModelScope (works better on CN networks).

Only fetches the safetensors weights + tokenizer files; skips the redundant
onnx/ and pytorch_model.bin copies that ship in the same repo to save ~2GB.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "data" / "runtime" / "models" / "modelscope"


def main() -> int:
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("Install modelscope first: pip install modelscope")
        return 1

    path = snapshot_download(
        "BAAI/bge-reranker-base",
        cache_dir=str(TARGET),
        ignore_patterns=["onnx/*", "pytorch_model.bin"],
    )
    print(path)
    print("\nSet in .env:")
    print(f"RERANK_MODEL_PATH={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
