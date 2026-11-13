"""Prompt budget / citation whitelist contracts. No DB, network, or model inference."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import re
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import Settings
from app.generate.prompt import EvidenceItem, build_prompt, prepare_prompt
from app.generate.service import answer_from_evidence
from app.llm.base import LLMResponse, LLMUsage


def main():
    settings = Settings(_env_file=None, answer_llm_provider="stub", allow_llm_stub=True, qa_max_context_chars=6000)
    candidates = [
        EvidenceItem(index=i, chunk_id=f"c{i}", doc_id="d", page=1, text=f"Evidence {i} " + "known content " * 90)
        for i in range(1, 9)
    ]
    prepared = prepare_prompt("question", candidates, max_context_chars=6000)
    sent = [int(n) for n in re.findall(r'<evidence\s+id="S(\d+)"', prepared.text)]
    assert sent == [item.index for item in prepared.evidence]
    assert 8 not in sent
    assert build_prompt("question", candidates, max_context_chars=6000) == prepared.text

    class Capture:
        text = _TEST_MATERIALS["text_001"]

        def generate(self, **kwargs):
            assert kwargs["prompt"] == prepared.text
            return LLMResponse(text=self.text, usage=LLMUsage(provider="stub", model="contract"))

    client = Capture()
    with patch("app.generate.service.get_llm_client", return_value=client):
        result = answer_from_evidence("question", candidates, settings=settings)
        assert [item.index for item in result.evidence] == sent
        assert [item.index for item in result.citations] == [1]
        assert len(result.evidence_selection["candidate_evidence"]) == 8
        for text in ["Invented fact. [S8]", "Supported [S1] and invented [S8].", "Invalid [S0]."]:
            client.text = text
            rejected = answer_from_evidence("question", candidates, settings=settings)
            assert not rejected.citations
            assert not re.search(r"\[S\d+\]", rejected.answer), rejected.answer
            assert _TEST_MATERIALS["text_003"] in rejected.answer
            assert rejected.evidence_selection["invalid_citation_indices"]
        assert answer_from_evidence("question", [], settings=settings).evidence == []
    assert len(candidates) == 8  # caller's retrieval diagnostics remain intact
    print(_TEST_MATERIALS["text_002"])


if __name__ == "__main__":
    main()
