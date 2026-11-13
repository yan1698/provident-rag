"""Day 10: 给 golden queries 标注 gold chunk 的辅助工具。

两种用法：

    python scripts/label_gold.py --offline path/to/doc.md
        不依赖数据库，直接按 Day 1 的分块规则切文件，列出「章节 -> chunk」，
        用来为使用者自行提供的文档选择 gold_sections 标签或 chunk_hash 前缀。

    python scripts/label_gold.py
        调 GET /eval/gold，检查当前评测集里的标注实际命中了哪些 chunk，
        并对写错的章节名（missing_labels）和没打标的题目给出告警。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")


def _get_json(url: str, timeout: int = 60) -> dict:
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return json.loads(body) if body else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code} {url}: {detail}") from exc
    except URLError as exc:
        raise SystemExit(f"cannot reach {url}: {exc.reason}") from exc


def show_offline(path: Path) -> None:
    from app.eval.gold import label_sections, preview
    from app.ingest.chunker import chunk_pages
    from app.ingest.parser import parse_document

    if not path.is_file():
        raise SystemExit(f"file not found: {path}")

    pages = parse_document(path.name, path.read_bytes())
    drafts = chunk_pages(pages)
    sections = label_sections([draft.text for draft in drafts])

    print(f"{path.name}: {len(drafts)} chunks")
    current = None
    for draft, labels in zip(drafts, sections):
        head = labels[0] if labels else "(preamble)"
        if head != current:
            current = head
            print(f"\n## {head}")
        extra = f"  +{list(labels[1:])}" if len(labels) > 1 else ""
        print(f"  {draft.chunk_hash[:12]}  p{draft.page}  {draft.token_count:>4}tok{extra}")
        print(f"      {preview(draft.text)}")


def show_resolved(eval_set: str = "") -> None:
    suffix = f"?eval_set={eval_set}" if eval_set else ""
    data = _get_json(f"{BASE}/eval/gold{suffix}")
    print(
        f"eval_set={data.get('eval_set')} corpus={data.get('corpus')} "
        f"chunks={data.get('corpus_chunk_count')} labeled={data.get('labeled_count')}"
    )
    print("\nsections in corpus:")
    for section in data.get("sections") or []:
        print(f"  - {section}")

    problems = 0
    print("\ngold per query:")
    for item in data.get("results") or []:
        print(f"  [{item['eval_id']}] gold_count={item['gold_count']} sections={item['gold_sections']}")
        for chunk in item.get("chunks") or []:
            print(f"      {chunk['chunk_hash'][:12]}  p{chunk['page']}  {chunk['section']}")
            print(f"          {chunk['preview']}")
        if item.get("missing_labels"):
            problems += 1
            print(f"      !! 无法解析的标签: {item['missing_labels']}")
        elif not item["gold_count"] and (item["gold_sections"] or item["gold_chunk_hashes"]):
            problems += 1
            print("      !! 标注存在但没命中任何 chunk（语料是否已 ingest？）")

    if problems:
        raise SystemExit(f"{problems} golden queries 的 gold 标注需要修正")
    print("\ngold 标注全部解析成功")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect / prepare gold chunk labels")
    parser.add_argument("--offline", default="", help="Chunk a local file instead of querying the API")
    parser.add_argument("--set", dest="eval_set", default="", help="Eval set name (default: golden-agent-notes-v1)")
    args = parser.parse_args()

    if args.offline:
        show_offline(Path(args.offline))
        return
    show_resolved(args.eval_set)


if __name__ == "__main__":
    main()
