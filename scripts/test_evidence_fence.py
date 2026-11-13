"""Evidence fence: documents cannot close the prompt or sit outside <evidence>."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.generate.fence import escape_evidence_text, format_evidence_block
from app.generate.prompt import EvidenceItem, SYSTEM_PROMPT, build_prompt


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def main() -> None:
    _assert("<evidence>" in SYSTEM_PROMPT and _TEST_MATERIALS["text_006"] in SYSTEM_PROMPT, SYSTEM_PROMPT)

    raw = _TEST_MATERIALS["text_001"]
    escaped = escape_evidence_text(raw)
    _assert("</evidence>" not in escaped, escaped)
    _assert("<evidence" not in escaped, escaped)

    block = format_evidence_block(
        _TEST_MATERIALS["text_002"],
        index=1,
        source="guide.md",
        page=2,
    )
    _assert(block.startswith('<evidence id="S1" source="guide.md" page="2">'), block)
    _assert(block.endswith("</evidence>"), block)
    _assert(_TEST_MATERIALS["text_004"] in block, block)

    item = EvidenceItem(
        index=1,
        chunk_id="c1",
        doc_id="guide.md",
        page=3,
        text=_TEST_MATERIALS["text_005"],
        chunk_hash="aaa",
    )
    prompt = build_prompt(_TEST_MATERIALS["text_003"], [item], max_context_chars=4000)
    _assert('<evidence id="S1"' in prompt, prompt)
    _assert(prompt.count("</evidence>") == 1, prompt)
    _assert("＜/evidence" in prompt, prompt)
    print("evidence fence ok")


if __name__ == "__main__":
    main()
