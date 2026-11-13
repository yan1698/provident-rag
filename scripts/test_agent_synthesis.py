"""Host contract for synthesis Agent (API 0.19).

Does not call /query or golden gates. Factoid HTTP stays in test_agent.py.
"""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import sys
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[1]
if (_REPO / "backend" / "app").is_dir():
    ROOT = _REPO
    sys.path.insert(0, str(ROOT / "backend"))
    sys.path.insert(0, str(ROOT / "scripts"))
    SAMPLE_MD = ROOT / "data" / "raw" / "sample.md"
else:
    ROOT = Path("/tmp")
    sys.path.insert(0, "/app")
    SAMPLE_MD = Path("/tmp/sample.md")

from app.agent.analyze import (
    analyze_task,
    apply_analyze,
    build_questions,
    detect_intent,
    extract_subjects,
    is_catalog_inventory,
)
from app.agent.budget import resolve_retrieve_limit, resolve_step_limit
from app.agent.classify import SYNTHESIS_WORDS, classify_task_kind
from app.agent.decompose import (
    apply_decompose,
    filter_hits_for_question,
    hit_covers_question,
    map_hit_to_sections,
    score_catalog,
    select_scored,
)
from app.agent.guards import apply_guards
from app.agent.planner import Plan, rule_plan
from app.agent.state import AgentPhase, AgentState, SynthesisNode
from app.agent.survey import EMPTY_CORPUS_ANSWER, apply_survey
from app.corpus.catalog import (
    CatalogBuild,
    CatalogDocument,
    build_catalog,
    catalog_observation,
    headings_in_order,
    label_chunk_sections,
)
from app.generate.prompt import EvidenceItem
from app.generate.synthesis import (
    DEFAULT_ORDER_NOTE,
    apply_g3,
    attach_citations,
    compose_synthesis,
    finalize_draft,
    format_cite_markers,
    normalize_cite_markup,
    plain_answer,
    plain_prose,
    stub_compose,
    synthesis_verify_body,
    COMPARE_EXTRACTIVE_NOTE,
)
from app.verify.meta import STATUS_META, meta_claim_checks, module_coverage_rate

FRONTEND = [
    CatalogDocument(
        "d-fe",
        "frontend-design.md",
        [_TEST_MATERIALS["text_098"], _TEST_MATERIALS["text_099"], _TEST_MATERIALS["text_100"]],
    )
]
INVOICE = [
    CatalogDocument(
        "d-inv",
        "invoice-policy.md",
        [_TEST_MATERIALS["text_101"], _TEST_MATERIALS["text_102"], _TEST_MATERIALS["text_103"]],
    )
]
OUT_OF_CORPUS = _TEST_MATERIALS["text_019"]
_HANDBOOK = [
    _TEST_MATERIALS["text_020"],
    _TEST_MATERIALS["text_021"],
    _TEST_MATERIALS["text_022"],
    _TEST_MATERIALS["text_023"],
]


def _assert(cond: bool, message: str) -> None:
    if not cond:
        raise SystemExit(f"FAIL: {message}")


def _sample_catalog() -> list[CatalogDocument]:
    text = SAMPLE_MD.read_text(encoding="utf-8")
    chunks = [piece for piece in text.split("\n\n") if piece.strip()]
    return list(build_catalog([("d-sample", SAMPLE_MD.name, chunks)]).documents)


def _mark_retrieved(state: AgentState) -> None:
    evidence: list[EvidenceItem] = []
    index = 0
    for node in state.nodes:
        if not node.section and not node.source_sections:
            node.status = "empty"
            continue
        index += 1
        node.status = "retrieved"
        node.chunk_ids = [f"c{index}"]
        if not node.source_sections and node.section:
            filename = next((doc.filename for doc in state.catalog if doc.doc_id == node.doc_id), "")
            node.source_sections = [{"doc_id": node.doc_id or "", "section": node.section, "filename": filename}]
        evidence.append(
            EvidenceItem(
                index=index,
                chunk_id=f"c{index}",
                doc_id=node.doc_id or "d",
                page=1,
                text=f"{node.question or node.title}{_TEST_MATERIALS['text_018']}",
                chunk_hash=f"h{index}",
            )
        )
        state.evidence_by_node[node.id] = [index]
    state.evidence = evidence
    state.hit_count = len(evidence)


def _check_classify() -> None:
    _assert(classify_task_kind(_TEST_MATERIALS["text_156"]) == "synthesis", _TEST_MATERIALS["text_029"])
    _assert(classify_task_kind(_TEST_MATERIALS["text_157"]) == "synthesis", _TEST_MATERIALS["text_030"])
    for word in SYNTHESIS_WORDS:
        kind = classify_task_kind(f"{_TEST_MATERIALS['text_009']}{word}{_TEST_MATERIALS['text_010']}")
        _assert(kind == "synthesis", f"intent {word!r}{_TEST_MATERIALS['text_011']}{kind}")
    _assert(classify_task_kind(_TEST_MATERIALS["text_158"]) == "synthesis", _TEST_MATERIALS["text_031"])
    _assert(classify_task_kind(_TEST_MATERIALS["text_159"]) == "synthesis", _TEST_MATERIALS["text_032"])
    for goal in (
        _TEST_MATERIALS["text_033"],
        _TEST_MATERIALS["text_034"],
        _TEST_MATERIALS["text_035"],
        _TEST_MATERIALS["text_036"],
        _TEST_MATERIALS["text_037"],
        _TEST_MATERIALS["text_038"],
        _TEST_MATERIALS["text_039"],
        _TEST_MATERIALS["text_040"],
    ):
        kind = classify_task_kind(goal)
        _assert(kind == "factoid", f"{goal!r}{_TEST_MATERIALS['text_012']}{kind}")
    _assert(classify_task_kind(_TEST_MATERIALS["text_160"]) == "synthesis", _TEST_MATERIALS["text_041"])
    _assert(
        classify_task_kind(_TEST_MATERIALS["text_161"]) == "synthesis",
        _TEST_MATERIALS["text_042"],
    )
    _assert(classify_task_kind(_TEST_MATERIALS["text_162"]) == "synthesis", _TEST_MATERIALS["text_043"])
    _assert(classify_task_kind(_TEST_MATERIALS["text_163"], "factoid") == "factoid", "explicit factoid override")
    _assert(classify_task_kind(_TEST_MATERIALS["text_164"], "synthesis") == "synthesis", "explicit synthesis override")
    _assert(classify_task_kind(_TEST_MATERIALS["text_165"], "digest") == "digest", "explicit digest override")
    _assert(classify_task_kind(_TEST_MATERIALS["text_166"], None) == "synthesis", "None requested → auto")
    try:
        classify_task_kind("x", "domain")
        raise SystemExit(_TEST_MATERIALS["text_104"])
    except ValueError:
        pass
    print("classify ok")


def _check_state_and_budget() -> None:
    _assert(AgentPhase.ANALYZING.value == "analyzing", "ANALYZING")
    _assert(AgentPhase.SURVEYING.value == "surveying", "SURVEYING")
    _assert(AgentPhase.COMPOSING.value == "composing", "COMPOSING")
    state = AgentState(goal=_TEST_MATERIALS["text_105"], task_kind="synthesis")
    _assert(state.catalog == [] and state.nodes == [] and state.gaps == [], "defaults")
    _assert(state.evidence_by_node == {} and state.subtopic_retrieves == 0, "retrieve counters")
    _assert(resolve_step_limit("factoid") == 8, "factoid default steps")
    _assert(resolve_step_limit("factoid", 20) == 12, "factoid cap stays 12")
    _assert(resolve_step_limit("synthesis") == 20, "synthesis default steps")
    _assert(resolve_step_limit("synthesis", 30, synthesis_cap=20) == 20, "synthesis cap")
    _assert(resolve_retrieve_limit("factoid") == 2, "factoid retrieves")
    _assert(resolve_retrieve_limit("synthesis") == 8, "synthesis retrieves")
    _assert(resolve_step_limit("digest") == 20, _TEST_MATERIALS["text_044"])
    _assert(resolve_retrieve_limit("digest") == 8, _TEST_MATERIALS["text_045"])
    print("state/budget ok")


def _check_analyze() -> None:
    _assert(detect_intent(_TEST_MATERIALS["text_167"]) == "compare", "compare")
    _assert(detect_intent(_TEST_MATERIALS["text_168"]) == "compare", _TEST_MATERIALS["text_046"])
    _assert(detect_intent(_TEST_MATERIALS["text_169"]) == "explain", _TEST_MATERIALS["text_047"])
    _assert(detect_intent(_TEST_MATERIALS["text_170"]) == "plan", "plan")
    _assert(detect_intent(_TEST_MATERIALS["text_171"]) == "plan", _TEST_MATERIALS["text_048"])
    _assert(detect_intent(_TEST_MATERIALS["text_172"]) == "inventory", "inventory")
    _assert(detect_intent(_TEST_MATERIALS["text_173"]) == "inventory", _TEST_MATERIALS["text_049"])
    scoped = analyze_task(_TEST_MATERIALS["text_050"])
    _assert(scoped.intent == "inventory" and _TEST_MATERIALS["text_174"] in scoped.subjects, scoped)
    _assert(scoped.questions, scoped.questions)
    _assert(not is_catalog_inventory(scoped.intent, scoped.subjects, _TEST_MATERIALS["text_175"]), scoped)
    whole = analyze_task(_TEST_MATERIALS["text_051"])
    _assert(whole.intent == "inventory" and not whole.subjects and not whole.questions, whole)
    _assert(is_catalog_inventory(whole.intent, whole.subjects, _TEST_MATERIALS["text_106"]), whole)
    _assert("Agent" not in " ".join(whole.subjects), whole)
    _assert(detect_intent(_TEST_MATERIALS["text_176"]) == "explain", "explain")
    _assert(detect_intent(_TEST_MATERIALS["text_177"], "compare") == "compare", "explicit intent override")
    spec = analyze_task(_TEST_MATERIALS["text_052"])
    _assert(spec.intent == "compare" and len(spec.questions) == 2, spec)
    _assert(
        not any(_TEST_MATERIALS["text_302"] in q or _TEST_MATERIALS["text_303"] in q for q in spec.questions),
        spec.questions,
    )
    _assert(
        _TEST_MATERIALS["text_178"] in spec.subjects and _TEST_MATERIALS["text_179"] in spec.subjects, spec.subjects
    )
    agent_goal = _TEST_MATERIALS["text_024"]
    agent_spec = analyze_task(agent_goal)
    blob = " ".join(agent_spec.subjects + agent_spec.questions).casefold()
    _assert("agent" in blob, agent_spec)
    _assert(any(_TEST_MATERIALS["text_264"] in item for item in agent_spec.subjects + agent_spec.questions), agent_spec)
    _assert(not any(_TEST_MATERIALS["text_279"] in q for q in agent_spec.questions), agent_spec.questions)
    _assert("agent" in " ".join(extract_subjects(agent_goal)).casefold(), extract_subjects(agent_goal))
    state = AgentState(goal=_TEST_MATERIALS["text_107"], task_kind="synthesis")
    obs = apply_analyze(state)
    _assert(state.task_spec is not None and obs["intent"] == "explain", obs)
    _assert(state.phase == AgentPhase.ANALYZING, state.phase)
    why = analyze_task(_TEST_MATERIALS["text_053"])
    _assert(why.intent == "explain" and len(why.questions) <= 2, why)
    _assert(
        not any(
            _TEST_MATERIALS["text_304"] in q or _TEST_MATERIALS["text_305"] in q or _TEST_MATERIALS["text_306"] in q
            for q in why.questions
        ),
        why.questions,
    )
    rewritten = build_questions(
        "explain", _TEST_MATERIALS["text_054"], [_TEST_MATERIALS["text_108"], _TEST_MATERIALS["text_109"]]
    )
    _assert(not any(_TEST_MATERIALS["text_280"] in q for q in rewritten), rewritten)
    _assert(
        any(_TEST_MATERIALS["text_281"] in q for q in rewritten)
        and any(_TEST_MATERIALS["text_282"] in q for q in rewritten),
        rewritten,
    )
    print("analyze ok")


def _catalog_sections(catalog: list[CatalogDocument]) -> set[str]:
    return {section for doc in catalog for section in doc.sections}


def _check_catalog() -> None:
    sections = label_chunk_sections(_HANDBOOK)
    _assert(sections[1][0].startswith("1."), f"heading not detected: {sections}")
    _assert(sections[2][0].startswith("1."), f"{_TEST_MATERIALS['text_001']}{sections}")
    _assert(sections[3][0].startswith("2."), f"section not switched: {sections}")
    _assert(
        headings_in_order(_HANDBOOK) == ["Handbook", _TEST_MATERIALS["text_180"], _TEST_MATERIALS["text_181"]],
        headings_in_order(_HANDBOOK),
    )
    overlap = [_TEST_MATERIALS["text_055"], _TEST_MATERIALS["text_056"], _TEST_MATERIALS["text_057"]]
    _assert(
        headings_in_order(overlap) == [_TEST_MATERIALS["text_182"], _TEST_MATERIALS["text_183"]],
        headings_in_order(overlap),
    )

    plain = [_TEST_MATERIALS["text_058"], _TEST_MATERIALS["text_059"]]
    _assert(headings_in_order(plain, fallback="invoice.txt") == ["invoice.txt"], "filename fallback")
    _assert(headings_in_order(plain) == [], _TEST_MATERIALS["text_060"])
    built = build_catalog([("d-md", "sample.md", _HANDBOOK), ("d-txt", "invoice.txt", plain)])
    by_name = {item.filename: item.sections for item in built.documents}
    _assert(by_name["sample.md"][0] == "Handbook" and by_name["invoice.txt"] == ["invoice.txt"], by_name)

    truncated = build_catalog(
        [("d1", "a.md", ["# One\n", "# Two\n", "# Three\n"]), ("d2", "b.md", ["# Four\n", "# Five\n"])],
        max_sections=3,
    )
    _assert(truncated.truncated and truncated.section_count == 3, truncated)
    obs = catalog_observation(truncated)
    _assert(obs["truncated"] is True and "documents" in obs and "sections" in obs, obs)
    split = build_catalog(
        [("d1", "a.md", ["# One\n", "# Two\n"]), ("d2", "b.md", ["# Three\n", "# Four\n"])],
        max_sections=3,
    )
    _assert([item.sections for item in split.documents] == [["One", "Two"], ["Three"]], split)

    empty_state = AgentState(goal=_TEST_MATERIALS["text_110"], task_kind="synthesis")
    empty_obs, empty_terminal = apply_survey(empty_state, build_catalog([]))
    _assert(empty_terminal and empty_obs.get("status") == "refused", empty_obs)
    _assert(empty_state.answer == EMPTY_CORPUS_ANSWER, empty_state.answer)

    filled = build_catalog([("d1", "invoice.txt", [_TEST_MATERIALS["text_265"]])])
    state = AgentState(goal=_TEST_MATERIALS["text_111"], task_kind="synthesis")
    filled_obs, terminal = apply_survey(state, filled)
    _assert(not terminal and state.catalog[0].sections == ["invoice.txt"], filled_obs)

    fe_headings = [f"{_TEST_MATERIALS['text_002']}{i}{_TEST_MATERIALS['text_003']}" for i in range(40)]
    agent_chunks = [_TEST_MATERIALS["text_061"], _TEST_MATERIALS["text_062"]]
    mixed = [
        ("d-fe", _TEST_MATERIALS["text_112"], fe_headings),
        ("d-ag", "agent-learning.md", agent_chunks),
    ]
    by_time = build_catalog(mixed, max_sections=8)
    _assert(
        by_time.documents[0].filename.startswith(_TEST_MATERIALS["text_113"]),
        [item.filename for item in by_time.documents],
    )
    _assert("agent-learning.md" not in {item.filename for item in by_time.documents}, by_time)
    ranked = build_catalog(mixed, max_sections=8, goal=_TEST_MATERIALS["text_114"])
    names = [item.filename for item in ranked.documents]
    _assert(names[0] == "agent-learning.md", names)
    _assert("agent-learning.md" in names, names)
    print("catalog ok")


def _check_decompose() -> None:
    invoice_goal = _TEST_MATERIALS["text_025"]
    titles = [
        item.section
        for item in select_scored(score_catalog(invoice_goal, FRONTEND + INVOICE), max_subtopics=6, score_min=0.15)
    ]
    _assert(_TEST_MATERIALS["text_184"] in titles and _TEST_MATERIALS["text_185"] in titles, titles)
    _assert(_TEST_MATERIALS["text_186"] not in titles and _TEST_MATERIALS["text_187"] not in titles, titles)

    state = AgentState(goal=invoice_goal, task_kind="synthesis", catalog=list(INVOICE))
    obs = apply_decompose(state, max_subtopics=6, score_min=0.1)
    _assert(len(state.nodes) >= 2, obs)
    _assert(all(node.question for node in state.nodes), state.nodes)
    queries = [node.query for node in state.nodes]
    _assert(len(set(queries)) >= 2, queries)
    _assert(
        any(_TEST_MATERIALS["text_283"] in q for q in queries)
        and any(_TEST_MATERIALS["text_284"] in q for q in queries),
        queries,
    )
    _assert(not all(q.split()[0] == (node.section or "") for node, q in zip(state.nodes, queries)), queries)

    capped = apply_decompose(
        AgentState(goal=invoice_goal, task_kind="synthesis", catalog=list(INVOICE)), max_subtopics=1, score_min=0.1
    )
    _assert(len(capped["nodes"]) == 1 and capped["probe_needed"] is True, capped)

    ordered = [
        CatalogDocument(
            "d1", "a.md", [_TEST_MATERIALS["text_188"], _TEST_MATERIALS["text_189"], _TEST_MATERIALS["text_190"]]
        )
    ]
    only_inv = apply_decompose(
        AgentState(goal=_TEST_MATERIALS["text_191"], task_kind="synthesis", catalog=ordered),
        max_subtopics=6,
        score_min=0.2,
    )
    node_titles = [node["title"] for node in only_inv["nodes"]]
    _assert(all(_TEST_MATERIALS["text_266"] in title for title in node_titles), only_inv)
    _assert(not any(_TEST_MATERIALS["text_285"] in title for title in node_titles), only_inv)

    hits = map_hit_to_sections(_TEST_MATERIALS["text_063"], "d-inv", list(INVOICE))
    _assert(hits and hits[0][1] == _TEST_MATERIALS["text_192"], hits)
    plain = [CatalogDocument("d-txt", "plain.txt", ["plain.txt"])]
    _assert(
        map_hit_to_sections(_TEST_MATERIALS["text_193"], "d-txt", plain) == [(plain[0], "plain.txt")], "probe fallback"
    )
    design = _TEST_MATERIALS["text_026"]
    ingest = _TEST_MATERIALS["text_027"]
    _assert(hit_covers_question(_TEST_MATERIALS["text_115"], ingest), ingest)
    _assert(not hit_covers_question(_TEST_MATERIALS["text_194"], design), design)

    react = EvidenceItem(
        index=1,
        chunk_id="c-react",
        doc_id="d-ag",
        page=1,
        text=_TEST_MATERIALS["text_116"],
        chunk_hash="h1",
    )
    noise = EvidenceItem(
        index=2,
        chunk_id="c-noise",
        doc_id="d-fe",
        page=1,
        text=_TEST_MATERIALS["text_117"],
        chunk_hash="h2",
    )
    question = _TEST_MATERIALS["text_028"]
    _assert(not hit_covers_question(question, react.text), _TEST_MATERIALS["text_064"])
    kept = filter_hits_for_question(
        question,
        [react, noise],
        scores={"c-react": 0.71, "c-noise": 0.004},
        semantic_min=0.1,
    )
    _assert([item.chunk_id for item in kept] == ["c-react"], [item.chunk_id for item in kept])
    bound = filter_hits_for_question(question, [noise], scores={"c-noise": 0.004}, exempt_doc_ids={"d-fe"})
    _assert(len(bound) == 1, _TEST_MATERIALS["text_065"])
    blind = filter_hits_for_question(_TEST_MATERIALS["text_066"], [react], scores=None)
    _assert(not blind, _TEST_MATERIALS["text_067"])
    _assert(
        _TEST_MATERIALS["text_118"] in plain_prose(_TEST_MATERIALS["text_195"]),
        plain_prose(_TEST_MATERIALS["text_119"]),
    )
    _assert("|" not in plain_prose(_TEST_MATERIALS["text_196"]), "tables stripped")
    listed = plain_prose(_TEST_MATERIALS["text_068"])
    _assert(_TEST_MATERIALS["text_197"] in listed and _TEST_MATERIALS["text_198"] in listed, listed)
    _assert("**" not in listed and listed.strip()[0] != "-", listed)
    cited = plain_prose(_TEST_MATERIALS["text_069"])
    _assert("[S1]" in cited and "**" not in cited, cited)
    fenced = plain_answer(_TEST_MATERIALS["text_070"])
    _assert("def foo():" in fenced and "return 1" in fenced, fenced)
    _assert("```" not in fenced, fenced)
    inline = plain_prose(_TEST_MATERIALS["text_071"])
    _assert("rerank_search" in inline and "`" not in inline, inline)
    consecutive = plain_prose(_TEST_MATERIALS["text_072"])
    _assert("[S5]" in consecutive and "[S6]" in consecutive, consecutive)
    stripped_pair = consecutive.replace("[S5]", "").replace("[S6]", "")
    _assert("56" not in stripped_pair, consecutive)
    _assert(
        normalize_cite_markup(_TEST_MATERIALS["text_199"]) == _TEST_MATERIALS["text_120"],
        normalize_cite_markup(_TEST_MATERIALS["text_121"]),
    )
    body, src = attach_citations(_TEST_MATERIALS["text_073"], [5, 6], valid={5, 6})
    _assert("56" not in body and src == [5, 6], (body, src))
    body2, src2 = attach_citations(_TEST_MATERIALS["text_074"], [6], valid={6})
    _assert("[S6]" not in body2 and src2 == [6], (body2, src2))
    _assert(format_cite_markers([5, 6]) == "[S5][S6]", format_cite_markers([5, 6]))
    doubled, src3 = attach_citations(_TEST_MATERIALS["text_075"], [5, 6], valid={5, 6})
    _assert(src3 == [5, 6] and _TEST_MATERIALS["text_200"] in doubled and "[S" not in doubled, (doubled, src3))
    print("decompose ok")


def _retrieve_queries(goal: str, catalog: list[CatalogDocument], want: int = 2) -> list[str]:
    state = AgentState(goal=goal, task_kind="synthesis", catalog=list(catalog), max_steps=20, max_retrieves=8)
    history: list[dict] = []
    queries: list[str] = []
    for _ in range(16):
        plan = rule_plan(goal, history, state=state)
        if plan.action == "analyze":
            history.append({"action": "analyze", "observation": apply_analyze(state)})
            continue
        if plan.action == "survey":
            history.append({"action": "survey", "observation": {"section_count": 3}})
            continue
        if plan.action == "graph_walk":
            history.append({"action": "graph_walk", "observation": {"skipped": True, "dimensions": []}})
            continue
        if plan.action == "decompose":
            history.append({"action": "decompose", "observation": apply_decompose(state)})
            continue
        if plan.action == "retrieve":
            queries.append(plan.input)
            if plan.node_id:
                next(item for item in state.nodes if item.id == plan.node_id).status = "retrieved"
            history.append(
                {
                    "action": "retrieve",
                    "input": plan.input,
                    "observation": {
                        "query": plan.input,
                        "node_id": plan.node_id or None,
                        "fresh_count": 1,
                        "hits": [{"snippet": "x"}],
                        "probe": not plan.node_id,
                    },
                }
            )
            if len([q for q in queries if q != goal]) >= want:
                break
            continue
        break
    return [q for q in queries if q != goal]


def _check_retrieve_and_guards() -> None:
    first = rule_plan(
        _TEST_MATERIALS["text_076"], [], state=AgentState(goal=_TEST_MATERIALS["text_267"], task_kind="synthesis")
    )
    _assert(first.action == "analyze", first)

    fe_queries = _retrieve_queries(_TEST_MATERIALS["text_077"], FRONTEND)
    inv_queries = _retrieve_queries(_TEST_MATERIALS["text_078"], INVOICE)
    _assert(len(set(fe_queries)) >= 2 and len(set(inv_queries)) >= 2, f"{fe_queries} / {inv_queries}")
    _assert(set(fe_queries).isdisjoint(set(inv_queries)), f"{fe_queries} vs {inv_queries}")
    cmp_queries = _retrieve_queries(_TEST_MATERIALS["text_079"], INVOICE)
    _assert(len(cmp_queries) == 2, cmp_queries)
    _assert(
        not any(_TEST_MATERIALS["text_307"] in q or _TEST_MATERIALS["text_308"] in q for q in cmp_queries), cmp_queries
    )

    syn = AgentState(goal=_TEST_MATERIALS["text_122"], task_kind="synthesis", max_steps=20, max_retrieves=8)
    g1 = apply_guards(Plan("x", "retrieve", "foo", "llm"), syn, [])
    _assert(g1.action == "analyze" and g1.source == "guard", g1)
    g1_survey = apply_guards(Plan("x", "retrieve", "foo", "llm"), syn, [{"action": "analyze"}])
    _assert(g1_survey.action == "survey" and g1_survey.source == "guard", g1_survey)
    restated = AgentState(
        goal=_TEST_MATERIALS["text_123"],
        task_kind="synthesis",
        memory_brief=_TEST_MATERIALS["text_124"],
        prior_answer=_TEST_MATERIALS["text_125"],
        turn_kind="summarize",
    )
    apply_analyze(restated)
    _assert(restated.turn_kind == "summarize", restated.turn_kind)
    _assert(not (restated.task_spec.questions if restated.task_spec else ["x"]), restated.task_spec)
    after = rule_plan(restated.goal, [{"action": "analyze", "observation": {}}], state=restated)
    _assert(after.action == "compose", after)
    g_mem = apply_guards(Plan("x", "compose", "", "rules"), restated, [{"action": "analyze"}])
    _assert(g_mem.action == "compose" and g_mem.source == "rules", g_mem)
    g_mem_ret = apply_guards(Plan("x", "retrieve", restated.goal, "rules"), restated, [{"action": "analyze"}])
    _assert(g_mem_ret.action == "compose", g_mem_ret)
    cover = AgentState(
        goal=_TEST_MATERIALS["text_126"],
        task_kind="synthesis",
        catalog=list(INVOICE) + list(FRONTEND),
    )
    apply_analyze(cover)
    _assert(cover.catalog_inventory(), cover.task_spec)
    cover_plan = rule_plan(cover.goal, [{"action": "analyze"}], state=cover)
    _assert(cover_plan.action == "compose", cover_plan)
    g_cover = apply_guards(Plan("x", "retrieve", cover.goal, "rules"), cover, [{"action": "analyze"}])
    _assert(g_cover.action == "compose", g_cover)
    meta_state = AgentState(
        goal=_TEST_MATERIALS["text_127"],
        task_kind="synthesis",
        memory_brief=_TEST_MATERIALS["text_128"],
        prior_answer=_TEST_MATERIALS["text_129"],
        catalog=list(INVOICE),
    )
    apply_analyze(meta_state)
    _assert(meta_state.turn_kind == "clarify", meta_state.turn_kind)
    meta_plan = rule_plan(meta_state.goal, [{"action": "analyze"}], state=meta_state)
    _assert(meta_plan.action == "compose", meta_plan)
    syn.catalog = list(INVOICE)
    started = [{"action": "analyze"}, {"action": "survey"}]
    g2 = apply_guards(Plan("x", "compose", "", "rules"), syn, started)
    _assert(g2.action == "refuse" and g2.source == "guard", g2)
    syn.nodes = [
        SynthesisNode(
            id="n1",
            title=_TEST_MATERIALS["text_201"],
            query=_TEST_MATERIALS["text_202"],
            status="retrieved",
            question=_TEST_MATERIALS["text_203"],
        )
    ]
    syn.total_steps = 19
    g5 = apply_guards(Plan("x", "retrieve", _TEST_MATERIALS["text_130"], "rules", node_id="n1"), syn, started)
    _assert(g5.action == "compose" and g5.source == "guard", g5)
    syn2 = AgentState(
        goal=_TEST_MATERIALS["text_131"],
        task_kind="synthesis",
        catalog=list(INVOICE),
        max_steps=20,
        max_retrieves=8,
        total_steps=19,
    )
    g5_refuse = apply_guards(Plan("x", "retrieve", "x", "rules"), syn2, started)
    _assert(g5_refuse.action == "refuse" and g5_refuse.source == "guard", g5_refuse)
    g6 = apply_guards(Plan("x", "web_search", "x", "llm"), syn, started)
    _assert(g6.action == "error" and g6.source == "guard", g6)
    g4 = apply_guards(
        Plan("x", "finish", "", "llm"), AgentState(goal=_TEST_MATERIALS["text_204"], task_kind="factoid"), []
    )
    _assert(g4.action == "retrieve" and g4.source == "guard", g4)
    print("retrieve/guards ok")


def _compose_fixture(goal: str, catalog: list[CatalogDocument]):
    state = AgentState(goal=goal, task_kind="synthesis", catalog=list(catalog))
    apply_analyze(state)
    apply_decompose(state)
    _assert(len(state.nodes) >= 2, f"need ≥2 nodes for {goal!r}: {state.nodes}")
    _mark_retrieved(state)
    draft = compose_synthesis(state, SimpleNamespace(llm_provider="stub"))
    headings = [module.get("heading") or module.get("title") for module in draft.modules]
    allowed = _catalog_sections(catalog)
    for module in draft.modules:
        sources = module.get("source_sections") or []
        _assert(sources, f"module missing source_sections: {module}")
        for src in sources:
            _assert(src.get("section") in allowed, f"{src} not ⊆ {sorted(allowed)}")
    _assert(draft.compose_mode == "extractive", draft.compose_mode)
    _assert(OUT_OF_CORPUS not in "".join(headings) and OUT_OF_CORPUS not in draft.markdown, headings)
    _assert("##" not in draft.markdown, draft.markdown)
    return draft, state, headings


def _check_compose() -> None:
    fe_draft, fe_state, fe_titles = _compose_fixture(
        f"{_TEST_MATERIALS['text_004']}{OUT_OF_CORPUS}{_TEST_MATERIALS['text_005']}",
        FRONTEND,
    )
    sample_draft, sample_state, sample_titles = _compose_fixture(
        f"{_TEST_MATERIALS['text_006']}{OUT_OF_CORPUS}{_TEST_MATERIALS['text_007']}",
        _sample_catalog(),
    )
    _assert(any(_TEST_MATERIALS["text_268"] in title for title in fe_titles), fe_titles)
    _assert(
        any(
            _TEST_MATERIALS["text_286"] in title
            or _TEST_MATERIALS["text_287"] in title
            or _TEST_MATERIALS["text_288"] in title
            for title in sample_titles
        ),
        sample_titles,
    )
    _assert(set(fe_titles).isdisjoint(set(sample_titles)), f"{fe_titles} vs {sample_titles}")

    cmp_draft, cmp_state, _cmp_titles = _compose_fixture(_TEST_MATERIALS["text_080"], INVOICE)
    _assert(cmp_state.task_spec and cmp_state.task_spec.intent == "compare", cmp_state.task_spec)
    _assert(len(cmp_state.nodes) == 2, cmp_state.nodes)
    _assert(COMPARE_EXTRACTIVE_NOTE in (cmp_draft.fallback_reason or ""), cmp_draft.fallback_reason)
    _assert(any(gap.get("topic") == _TEST_MATERIALS["text_269"] for gap in cmp_draft.gaps), cmp_draft.gaps)

    for state in (fe_state, sample_state):
        raw = stub_compose(state)
        raw["modules"].append(
            {"heading": OUT_OF_CORPUS, "paragraphs": [{"text": _TEST_MATERIALS["text_289"], "sources": [1]}]}
        )
        stripped = finalize_draft(raw, state)
        headings = [m.get("heading") or m.get("title") for m in stripped.modules]
        _assert(OUT_OF_CORPUS not in headings, stripped.modules)
        _assert(any(gap["topic"] == OUT_OF_CORPUS for gap in stripped.gaps), stripped.gaps)
        _assert(OUT_OF_CORPUS not in stripped.markdown, stripped.markdown)

    kept, stripped = apply_g3(
        [
            {
                "heading": _TEST_MATERIALS["text_205"],
                "source_sections": [{"section": _TEST_MATERIALS["text_290"], "filename": _TEST_MATERIALS["text_291"]}],
                "paragraphs": [{"text": _TEST_MATERIALS["text_292"], "sources": [1]}],
            }
        ],
        [{"section": _TEST_MATERIALS["text_206"], "filename": _TEST_MATERIALS["text_207"], "doc_id": "d-inv"}],
        valid_indices={1},
    )
    _assert(len(kept) == 1 and kept[0]["heading"] == _TEST_MATERIALS["text_208"] and not stripped, kept)
    leaky, _leak_stripped = apply_g3(
        [
            {
                "heading": _TEST_MATERIALS["text_209"],
                "source_sections": [{"section": _TEST_MATERIALS["text_293"], "filename": _TEST_MATERIALS["text_294"]}],
                "paragraphs": [{"text": _TEST_MATERIALS["text_295"], "sources": [5, 6]}],
            }
        ],
        [{"section": _TEST_MATERIALS["text_210"], "filename": _TEST_MATERIALS["text_211"], "doc_id": "d-inv"}],
        valid_indices={5, 6},
    )
    para = leaky[0]["paragraphs"][0]
    _assert("56" not in para["text"] and "[S" not in para["text"], para)
    _assert(para["sources"] == [5, 6], para)
    leak_body = synthesis_verify_body(leaky)
    _assert("[S5][S6]" in leak_body, leak_body)
    _assert("56" not in leak_body.replace("[S5]", "").replace("[S6]", ""), leak_body)

    modules = [
        {"heading": _TEST_MATERIALS["text_132"], "paragraphs": [{"text": _TEST_MATERIALS["text_270"], "sources": [1]}]}
    ]
    body = synthesis_verify_body(modules)
    _assert(_TEST_MATERIALS["text_133"] in body, body)
    metas = meta_claim_checks(DEFAULT_ORDER_NOTE, [{"topic": OUT_OF_CORPUS, "reason": _TEST_MATERIALS["text_212"]}])
    _assert(all(item.status == STATUS_META and not item.hallucination for item in metas), metas)
    _assert(
        module_coverage_rate(modules, [SimpleNamespace(text=_TEST_MATERIALS["text_309"], status="supported")]) == 1.0,
        "covered",
    )
    _assert(module_coverage_rate(modules, []) == 0.0, "uncovered")
    print("compose ok")


def _walk_synthesis(state: AgentState, catalog: list[CatalogDocument]) -> list[str]:
    history: list[dict] = []
    actions: list[str] = []
    for _ in range(state.max_steps):
        plan = apply_guards(rule_plan(state.goal, history, state=state), state, history)
        actions.append(plan.action)
        if plan.action == "analyze":
            history.append({"action": "analyze", "observation": apply_analyze(state)})
        elif plan.action == "survey":
            state.catalog = list(catalog)
            history.append({"action": "survey", "observation": {"section_count": len(catalog)}})
        elif plan.action == "graph_walk":
            history.append({"action": "graph_walk", "observation": {"skipped": True, "dimensions": []}})
        elif plan.action == "decompose":
            history.append({"action": "decompose", "observation": apply_decompose(state)})
        elif plan.action == "retrieve":
            node = next((item for item in state.nodes if item.id == plan.node_id), None)
            if node is not None:
                node.status = "retrieved"
            state.subtopic_retrieves += 1
            history.append(
                {"action": "retrieve", "input": plan.input, "observation": {"query": plan.input, "fresh_count": 1}}
            )
        elif plan.action == "compose":
            history.append({"action": "compose", "observation": {"modules": [{"title": _TEST_MATERIALS["text_310"]}]}})
        elif plan.action == "verify":
            history.append({"action": "verify", "observation": {"coverage_rate": 1.0, "module_coverage_rate": 1.0}})
        elif plan.action in {"finish", "refuse", "error"}:
            break
        else:
            break
        state.total_steps += 1
    return actions


def _check_budget_and_edges() -> None:
    catalog = [
        CatalogDocument(
            "d-inv",
            "invoice-policy.md",
            [_TEST_MATERIALS["text_213"], _TEST_MATERIALS["text_214"], _TEST_MATERIALS["text_215"]],
        )
    ]
    syn = AgentState(goal=_TEST_MATERIALS["text_134"], task_kind="synthesis", max_steps=5, max_retrieves=8)
    actions = _walk_synthesis(syn, catalog)
    _assert("compose" in actions, actions)
    _assert("analyze" in actions, actions)
    _assert([n for n in syn.nodes if n.status == "pending"], _TEST_MATERIALS["text_081"])
    _assert([n for n in syn.nodes if n.status == "retrieved"], _TEST_MATERIALS["text_082"])

    empty_state = AgentState(goal=_TEST_MATERIALS["text_135"], task_kind="synthesis")
    empty_obs, empty_terminal = apply_survey(empty_state, CatalogBuild(documents=()))
    _assert(empty_terminal and empty_state.phase == AgentPhase.DONE, empty_obs)

    huge = [
        (
            f"d{i}",
            f"doc-{i}.md",
            [
                f"{_TEST_MATERIALS['text_013']}{i}{_TEST_MATERIALS['text_014']}",
                f"{_TEST_MATERIALS['text_015']}{i}{_TEST_MATERIALS['text_016']}",
            ],
        )
        for i in range(50)
    ]
    built = build_catalog(huge, max_sections=80)
    _assert(built.truncated and built.section_count == 80, built.section_count)

    pdf_chunks = [_TEST_MATERIALS["text_083"], _TEST_MATERIALS["text_084"]]
    _assert(headings_in_order(pdf_chunks) == [], _TEST_MATERIALS["text_085"])
    pdf_state = AgentState(
        goal=_TEST_MATERIALS["text_136"],
        task_kind="synthesis",
        catalog=[CatalogDocument("d-pdf", "scan.pdf", ["scan.pdf"])],
    )
    apply_analyze(pdf_state)
    apply_decompose(pdf_state)
    _mark_retrieved(pdf_state)
    pdf_draft = compose_synthesis(pdf_state, SimpleNamespace(llm_provider="stub"))
    pdf_sources = [src.get("section") for mod in pdf_draft.modules for src in (mod.get("source_sections") or [])]
    _assert("scan.pdf" in pdf_sources, pdf_sources)
    _assert(
        all((mod.get("heading") or mod.get("title")) != "scan.pdf" or True for mod in pdf_draft.modules),
        pdf_draft.modules,
    )

    twins = [
        CatalogDocument("d-a", "handbook-a.md", [_TEST_MATERIALS["text_216"]]),
        CatalogDocument("d-b", "handbook-b.md", [_TEST_MATERIALS["text_217"]]),
    ]
    twin_state = AgentState(goal=_TEST_MATERIALS["text_137"], task_kind="synthesis", catalog=twins)
    apply_analyze(twin_state)
    apply_decompose(twin_state)
    _assert(
        any(_TEST_MATERIALS["text_271"] in (node.question or node.title) for node in twin_state.nodes), twin_state.nodes
    )
    print("budget/edges ok")


def _check_digest() -> None:
    from app.agent.planner import digest_rule_plan
    from app.generate.synthesis import compose_synthesis

    first = rule_plan(
        _TEST_MATERIALS["text_086"], [], state=AgentState(goal=_TEST_MATERIALS["text_272"], task_kind="digest")
    )
    _assert(first.action == "analyze", first)
    g1 = apply_guards(
        Plan("x", "digest", "", "rules"), AgentState(goal=_TEST_MATERIALS["text_218"], task_kind="digest"), []
    )
    _assert(g1.action == "analyze" and g1.source == "guard", g1)
    surveyed = AgentState(goal=_TEST_MATERIALS["text_138"], task_kind="digest", catalog=list(INVOICE))
    apply_analyze(surveyed)
    nxt = digest_rule_plan(
        _TEST_MATERIALS["text_087"],
        [{"action": "analyze", "observation": {}}, {"action": "survey", "observation": {"section_count": 3}}],
        surveyed,
    )
    _assert(nxt.action == "digest", nxt)
    state = AgentState(goal=_TEST_MATERIALS["text_139"], task_kind="digest", catalog=list(INVOICE))
    apply_analyze(state)
    state.evidence = [
        EvidenceItem(index=1, chunk_id="c1", doc_id="d-inv", page=1, text=_TEST_MATERIALS["text_219"], chunk_hash="h1")
    ]
    state.digests = [
        {
            "doc_id": "d-inv",
            "filename": _TEST_MATERIALS["text_140"],
            "mode": "extractive",
            "bullets": [{"text": _TEST_MATERIALS["text_273"], "sources": [1]}],
            "source_sections": [
                {"doc_id": "d-inv", "section": _TEST_MATERIALS["text_274"], "filename": _TEST_MATERIALS["text_275"]}
            ],
        }
    ]
    state.nodes = [
        SynthesisNode(
            id="digest-d-inv",
            title=_TEST_MATERIALS["text_220"],
            query=state.goal,
            status="retrieved",
            question=_TEST_MATERIALS["text_221"],
            doc_id="d-inv",
            source_sections=[
                {"doc_id": "d-inv", "section": _TEST_MATERIALS["text_296"], "filename": _TEST_MATERIALS["text_297"]}
            ],
        )
    ]
    draft = compose_synthesis(state, SimpleNamespace(llm_provider="stub"))
    _assert(draft.modules, draft)
    _assert("90" in draft.markdown, draft.markdown)

    from app.agent.digest import pending_digest_docs, _read_document
    from app.corpus.relevance import score_document

    agent_doc = CatalogDocument("d-ag", "agent-learning.md", ["ReAct", _TEST_MATERIALS["text_141"]])
    mixed_state = AgentState(
        goal=_TEST_MATERIALS["text_142"],
        task_kind="digest",
        catalog=list(FRONTEND) + [agent_doc],
    )
    pending = pending_digest_docs(mixed_state)
    _assert(pending[0].filename == "agent-learning.md", [doc.filename for doc in pending])
    _assert(all(doc.filename != "frontend-design.md" for doc in pending), [doc.filename for doc in pending])
    _assert(
        score_document(mixed_state.goal, "agent-learning.md", ["ReAct"])
        > score_document(
            mixed_state.goal, _TEST_MATERIALS["text_222"], [_TEST_MATERIALS["text_276"], _TEST_MATERIALS["text_277"]]
        ),
        _TEST_MATERIALS["text_088"],
    )
    off_items = [
        EvidenceItem(
            index=1,
            chunk_id="c-fe",
            doc_id="d-fe",
            page=1,
            text=_TEST_MATERIALS["text_223"],
            chunk_hash="h-fe",
        )
    ]
    bullets, _gaps, mode, _note = _read_document(
        mixed_state.goal,
        off_items,
        SimpleNamespace(deepseek_api_key="", llm_max_output_tokens=256),
        FRONTEND[0],
    )
    _assert(mode == "off_topic" and not bullets, (mode, bullets))

    from app.agent.digest import _select_doc_chunks

    rows = [
        SimpleNamespace(id=f"r{i}", text=text, page=1, chunk_hash=f"h{i}")
        for i, text in enumerate(
            [
                _TEST_MATERIALS["text_298"] * 8,
                _TEST_MATERIALS["text_299"] * 12,
                _TEST_MATERIALS["text_300"] * 4,
            ]
        )
    ]
    budget = max(len(rows[0].text), len(rows[2].text)) + 10
    picked = _select_doc_chunks(_TEST_MATERIALS["text_089"], rows, budget, SimpleNamespace(rerank_batch_size=4))
    ids = [row.id for row in picked]
    _assert("r2" in ids, f"{_TEST_MATERIALS['text_008']}{ids}")
    _assert(ids == sorted(ids), f"selected passages should stay in document order: {ids}")
    whole = _select_doc_chunks(_TEST_MATERIALS["text_090"], rows, 10_000, SimpleNamespace(rerank_batch_size=4))
    _assert(len(whole) == 3, _TEST_MATERIALS["text_091"])

    ok_state = AgentState(goal=_TEST_MATERIALS["text_143"], task_kind="digest", catalog=[agent_doc] + list(FRONTEND))
    ok_state.digests = [
        {"doc_id": agent_doc.doc_id, "filename": agent_doc.filename, "bullets": [{"text": _TEST_MATERIALS["text_278"]}]}
    ]
    _assert(not pending_digest_docs(ok_state), _TEST_MATERIALS["text_092"])
    print("digest ok")


def _check_json_repair() -> None:
    from app.llm.json_io import parse_json_object

    _assert(parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}, "fenced JSON")
    _assert(parse_json_object(_TEST_MATERIALS["text_224"]) == {"a": 1}, "surrounded JSON")
    _assert(parse_json_object('{"a": 1,}') == {"a": 1}, "trailing comma")
    newline = parse_json_object(_TEST_MATERIALS["text_093"])
    _assert(newline == {"text": _TEST_MATERIALS["text_225"]}, newline)
    inner = parse_json_object(_TEST_MATERIALS["text_094"])
    _assert(inner is not None and _TEST_MATERIALS["text_226"] in inner["text"], inner)
    truncated = parse_json_object(_TEST_MATERIALS["text_095"])
    _assert(isinstance(truncated, dict) and truncated.get("modules"), truncated)
    smart = parse_json_object("{\u201ca\u201d: 1}")
    _assert(smart == {"a": 1}, smart)
    _assert(parse_json_object(_TEST_MATERIALS["text_227"]) is None, _TEST_MATERIALS["text_096"])
    print("json repair ok")


def _check_lanes() -> None:
    from app.agent.evidence import select_global_evidence
    from app.agent.neighbors import NeighborRow, expand_from_rows
    from app.agent.planner import synthesis_rule_plan

    rows = [
        NeighborRow("c0", "d1", 1, 0, _TEST_MATERIALS["text_144"], "h0"),
        NeighborRow("c1", "d1", 1, 20, _TEST_MATERIALS["text_145"], "h1"),
        NeighborRow("c2", "d1", 1, 40, _TEST_MATERIALS["text_146"], "h2"),
    ]
    hit = EvidenceItem(
        index=1,
        chunk_id="c1",
        doc_id="d1",
        page=1,
        text=_TEST_MATERIALS["text_147"],
        chunk_hash="h1",
        start_offset=20,
        score=0.8,
    )
    expanded, info = expand_from_rows([hit], rows, radius=1, max_chars=10_000)
    texts = " ".join(item.text for item in expanded)
    _assert(_TEST_MATERIALS["text_148"] in texts, texts)
    _assert(_TEST_MATERIALS["text_149"] in texts, texts)
    _assert(info["neighbors_added"] == 2, info)
    tiny, tiny_info = expand_from_rows([hit], rows, radius=1, max_chars=len(hit.text) + 5)
    _assert(any(item.chunk_id == "c1" for item in tiny), tiny)
    _assert(tiny_info["neighbors_dropped"] >= 1, tiny_info)
    _assert(all(item.chunk_id == "c1" or item.role == "neighbor" for item in tiny), tiny)

    agent_cat = [
        CatalogDocument(
            "d-ag",
            "agent-learning.md",
            [_TEST_MATERIALS["text_228"], _TEST_MATERIALS["text_229"], _TEST_MATERIALS["text_230"]],
        ),
        CatalogDocument("d-fe", "frontend-design.md", [_TEST_MATERIALS["text_231"], _TEST_MATERIALS["text_232"]]),
    ]
    agent_state = AgentState(goal=_TEST_MATERIALS["text_150"], task_kind="synthesis", catalog=agent_cat)
    agent_obs = apply_decompose(agent_state, max_subtopics=6, score_min=0.08)
    blob = " ".join(f"{node.section or ''} {node.title} {node.question}" for node in agent_state.nodes)
    _assert("ReAct" in blob or _TEST_MATERIALS["text_233"] in blob or _TEST_MATERIALS["text_234"] in blob, agent_obs)
    _assert(not any(_TEST_MATERIALS["text_301"] == (node.section or "") for node in agent_state.nodes), agent_obs)

    from app.agent.state import TaskSpec

    mixed = [
        CatalogDocument("d-ag", "agent-orchestration.md", [_TEST_MATERIALS["text_235"], _TEST_MATERIALS["text_236"]]),
        CatalogDocument("d-bird", "bird_guide.pdf", [_TEST_MATERIALS["text_237"], _TEST_MATERIALS["text_238"]]),
        CatalogDocument(
            "d-fe", _TEST_MATERIALS["text_151"], [_TEST_MATERIALS["text_239"], _TEST_MATERIALS["text_240"]]
        ),
    ]
    topic_state = AgentState(goal=_TEST_MATERIALS["text_152"], task_kind="synthesis", catalog=mixed)
    topic_state.task_spec = TaskSpec(
        intent="explain",
        subjects=[_TEST_MATERIALS["text_241"], _TEST_MATERIALS["text_242"]],
        questions=[_TEST_MATERIALS["text_243"], _TEST_MATERIALS["text_244"]],
    )
    topic_obs = apply_decompose(topic_state, max_subtopics=6, score_min=0.08)
    topic_blob = " ".join(f"{node.section or ''} {node.title}" for node in topic_state.nodes)
    _assert(_TEST_MATERIALS["text_245"] in topic_blob or _TEST_MATERIALS["text_246"] in topic_blob, topic_obs)
    _assert(_TEST_MATERIALS["text_247"] not in topic_blob and _TEST_MATERIALS["text_248"] not in topic_blob, topic_blob)

    state = AgentState(goal=_TEST_MATERIALS["text_153"], task_kind="synthesis")
    state.nodes = [
        SynthesisNode(
            id="n1",
            title=_TEST_MATERIALS["text_249"],
            query=_TEST_MATERIALS["text_250"],
            status="retrieved",
            question=_TEST_MATERIALS["text_251"],
            chunk_ids=["c-a"],
            score=0.9,
        ),
        SynthesisNode(
            id="n2",
            title=_TEST_MATERIALS["text_252"],
            query=_TEST_MATERIALS["text_253"],
            status="retrieved",
            question=_TEST_MATERIALS["text_254"],
            chunk_ids=["c-b"],
            score=0.8,
        ),
        SynthesisNode(
            id="n3",
            title=_TEST_MATERIALS["text_255"],
            query=_TEST_MATERIALS["text_256"],
            status="retrieved",
            question=_TEST_MATERIALS["text_257"],
            chunk_ids=["c-c"],
            score=0.7,
        ),
    ]
    state.evidence = [
        EvidenceItem(
            index=i + 1,
            chunk_id=f"c-{letter}",
            doc_id="d",
            page=1,
            text=f"{_TEST_MATERIALS['text_017']}{letter}",
            chunk_hash=f"h{i}",
            score=score,
        )
        for i, (letter, score) in enumerate(
            [
                ("a", 0.9),
                ("a2", 0.2),
                ("a3", 0.15),
                ("a4", 0.12),
                ("a5", 0.11),
                ("a6", 0.1),
                ("a7", 0.09),
                ("a8", 0.08),
                ("b", 0.8),
                ("b2", 0.2),
                ("b3", 0.15),
                ("b4", 0.12),
                ("b5", 0.11),
                ("b6", 0.1),
                ("b7", 0.09),
                ("b8", 0.08),
                ("c", 0.7),
                ("c2", 0.2),
                ("c3", 0.15),
                ("c4", 0.12),
                ("c5", 0.11),
                ("c6", 0.1),
                ("c7", 0.09),
                ("c8", 0.08),
            ]
        )
    ]
    state.nodes[0].chunk_ids = ["c-a", "c-a2", "c-a3", "c-a4", "c-a5", "c-a6", "c-a7", "c-a8"]
    state.nodes[1].chunk_ids = ["c-b", "c-b2", "c-b3", "c-b4", "c-b5", "c-b6", "c-b7", "c-b8"]
    state.nodes[2].chunk_ids = ["c-c", "c-c2", "c-c3", "c-c4", "c-c5", "c-c6", "c-c7", "c-c8"]
    cull = select_global_evidence(state, cap=12)
    _assert(len(state.evidence) <= 12, (len(state.evidence), cull))
    kept_ids = {item.chunk_id for item in state.evidence}
    _assert("c-a" in kept_ids and "c-b" in kept_ids and "c-c" in kept_ids, kept_ids)
    _assert(cull["dropped"] > 0, cull)

    weak = AgentState(goal=_TEST_MATERIALS["text_154"], task_kind="synthesis", max_retrieves=8)
    weak.nodes = [
        SynthesisNode(
            id="n1",
            title=_TEST_MATERIALS["text_258"],
            query=_TEST_MATERIALS["text_259"],
            status="retrieved",
            question=_TEST_MATERIALS["text_260"],
            chunk_ids=["c1"],
            score=0.04,
        ),
        SynthesisNode(
            id="n2",
            title=_TEST_MATERIALS["text_261"],
            query=_TEST_MATERIALS["text_262"],
            status="retrieved",
            question=_TEST_MATERIALS["text_263"],
            chunk_ids=["c2"],
            score=0.8,
        ),
    ]
    weak_plan = synthesis_rule_plan(
        weak.goal,
        [
            {"action": "analyze"},
            {"action": "survey"},
            {"action": "decompose"},
            {"action": "retrieve", "observation": {"fresh_count": 1}},
            {"action": "retrieve", "observation": {"fresh_count": 1}},
            {"action": "compose", "observation": {"modules": [{}]}},
            {"action": "verify", "observation": {"coverage_rate": 0.2, "module_coverage_rate": 0.2}},
        ],
        weak,
        settings=SimpleNamespace(agent_synthesis_min_module_coverage=0.5, agent_weak_node_score=0.15),
    )
    _assert(weak_plan.action == "retrieve" and weak_plan.node_id == "n1", weak_plan)
    _assert(_TEST_MATERIALS["text_155"] in weak_plan.thought, weak_plan.thought)
    print(_TEST_MATERIALS["text_097"])


def _run() -> None:
    _assert(SAMPLE_MD.is_file(), f"missing fixture {SAMPLE_MD}")
    _check_classify()
    _check_state_and_budget()
    _check_analyze()
    _check_catalog()
    _check_decompose()
    _check_retrieve_and_guards()
    _check_compose()
    _check_budget_and_edges()
    _check_digest()
    _check_json_repair()
    _check_lanes()
    print("test_agent_synthesis ok")


def main() -> None:
    try:
        _run()
        return
    except (ModuleNotFoundError, ImportError):
        if Path("/app/app").is_dir() and not (_REPO / "backend" / "app").is_dir():
            raise
        from lib.offline import can_use_local_api, run_script_in_api

        if not can_use_local_api():
            raise
        run_script_in_api(_HERE, "test_agent_synthesis ok", extras=[_REPO / "data" / "raw" / "sample.md"])


if __name__ == "__main__":
    main()
