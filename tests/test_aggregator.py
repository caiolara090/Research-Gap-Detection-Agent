from datetime import date

import pytest

import research_gap_agent.nodes.aggregator as aggregator_module
from research_gap_agent.schemas import (
    ExtractedInsights,
    FinalReport,
    GapEvidence,
    GapIdentificationResult,
    GapWarning,
    GraphInsight,
    IdentifiedGap,
    Paper,
    SearchQuery,
)
from research_gap_agent.state import GraphState


def _paper(paper_id: str, source: str) -> Paper:
    return Paper(
        id=paper_id,
        source=source,
        title=f"Paper {paper_id}",
        abstract="Abstract",
        published_date=date(2025, 1, 1),
        url=f"https://example.com/{paper_id}",
        pdf_url=f"https://example.com/{paper_id}.pdf",
    )


def _insight(paper_id: str) -> ExtractedInsights:
    return ExtractedInsights(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        published_date=date(2025, 1, 1),
    )


def _gap(**overrides) -> IdentifiedGap:
    payload = {
        "research_question": (
            "How do candidate effects change over time?"
        ),
        "description": "Longitudinal effects remain insufficiently studied.",
        "evidence_strength": 84,
        "evidence": [
            GapEvidence(
                paper_id="paper-1",
                evidence_type="stated_limitations",
                description="The study calls for longitudinal follow-up.",
            )
        ],
        "rationale": (
            "The available corpus repeatedly identifies this scope."
        ),
    }
    payload.update(overrides)
    return IdentifiedGap(**payload)


def _graph_insight() -> GraphInsight:
    return GraphInsight(
        summary="This summary must not be used by the aggregator prompt.",
        disconnected_pairs=[("summary-only", "pair")],
        raw={
            "ranked_hypotheses": [
                {
                    "concepts": [
                        "longitudinal effects",
                        "AI agents",
                    ],
                    "missing_links": [
                        ["longitudinal effects", "AI agents"]
                    ],
                    "score": 0.91,
                }
            ],
            "graph_nodes": 123,
        },
    )


def _state(gap_result: GapIdentificationResult | None) -> GraphState:
    return GraphState(
        initial_topic="Longitudinal effects of AI agents",
        queries=[
            SearchQuery(
                text="AI agents longitudinal effects",
                rationale="Find longitudinal studies.",
            )
        ],
        raw_papers=[
            _paper("paper-1", "arxiv"),
            _paper("paper-2", "openalex"),
            _paper("paper-3", "semantic_scholar"),
        ],
        ranked_papers=[
            _paper("paper-1", "arxiv"),
            _paper("paper-2", "openalex"),
            _paper("paper-3", "semantic_scholar"),
        ],
        extracted=[_insight("paper-1")],
        graph_insight=_graph_insight(),
        gap_identification=gap_result,
    )


class FakeStructuredLLM:
    def __init__(self, response: FinalReport):
        self.response = response
        self.invoke_calls: list[list[tuple[str, str]]] = []

    def invoke(self, messages: list[tuple[str, str]]) -> FinalReport:
        self.invoke_calls.append(messages)
        return self.response


class FakeLLM:
    def __init__(self, response: FinalReport):
        self.structured_llm = FakeStructuredLLM(response)
        self.structured_output_calls: list[type] = []

    def with_structured_output(
        self,
        schema: type,
        **kwargs,
    ) -> FakeStructuredLLM:
        self.structured_output_calls.append(schema)
        return self.structured_llm


def _report(gaps: list[IdentifiedGap], warnings=None) -> FinalReport:
    return FinalReport(
        topic="Longitudinal effects of AI agents",
        cutoff_date=date(2025, 4, 3),
        warnings=warnings or [],
        gaps=gaps,
        summary="LLM-produced final report.",
        methodology_note="Candidate gaps through the cutoff date.",
        sources_used=["arxiv"],
        papers_considered=1,
    )


def test_aggregator_calls_llm_for_structured_final_report(monkeypatch):
    textual_gap = _gap()
    final_gap = textual_gap.model_copy(
        deep=True,
        update={
            "origin": "textual_and_graph",
            "matched_graph_hypothesis": (
                _graph_insight().raw["ranked_hypotheses"][0]
            ),
            "graph_refinement": (
                "The graph niche sharpens the question around AI agents."
            ),
        },
    )
    llm_response = _report(
        gaps=[final_gap],
        warnings=[
            GapWarning(
                code="invalid_counter_evidence_reference",
                message="Earlier warning was preserved.",
            )
        ],
    )
    fake_llm = FakeLLM(llm_response)
    requested_roles: list[str] = []

    def fake_get_llm(role: str) -> FakeLLM:
        requested_roles.append(role)
        return fake_llm

    monkeypatch.setattr(aggregator_module, "get_llm", fake_get_llm)
    state = _state(
        GapIdentificationResult(
            cutoff_date=date(2025, 4, 3),
            warnings=[
                GapWarning(
                    code="invalid_counter_evidence_reference",
                    message="Earlier warning was preserved.",
                )
            ],
            gaps=[textual_gap],
        )
    )

    output = aggregator_module.aggregator_node(state)

    assert requested_roles == ["aggregator"]
    assert fake_llm.structured_output_calls == [FinalReport]
    assert len(fake_llm.structured_llm.invoke_calls) == 1
    assert output == {"final_report": llm_response}


def test_aggregator_uses_state_for_operational_report_metadata(monkeypatch):
    final_gap = _gap(
        origin="textual_only",
        matched_graph_hypothesis=None,
        graph_refinement=None,
    )
    llm_response = FinalReport(
        topic="Wrong LLM topic",
        cutoff_date=date(1999, 1, 1),
        warnings=[],
        gaps=[final_gap],
        summary="LLM-produced final report.",
        methodology_note="LLM-produced methodology.",
        sources_used=[],
        papers_considered=999,
    )
    fake_llm = FakeLLM(llm_response)
    monkeypatch.setattr(
        aggregator_module,
        "get_llm",
        lambda role: fake_llm,
    )
    state = _state(
        GapIdentificationResult(
            cutoff_date=date(2025, 4, 3),
            gaps=[_gap()],
        )
    )

    report = aggregator_module.aggregator_node(state)["final_report"]

    assert report.topic == "Longitudinal effects of AI agents"
    assert report.cutoff_date == date(2025, 4, 3)
    assert report.sources_used == ["arxiv"]
    assert report.papers_considered == 1
    assert report.summary == "LLM-produced final report."


def test_aggregator_calls_llm_even_when_textual_gap_list_is_empty(monkeypatch):
    llm_response = _report(gaps=[])
    fake_llm = FakeLLM(llm_response)
    monkeypatch.setattr(
        aggregator_module,
        "get_llm",
        lambda role: fake_llm,
    )
    state = _state(
        GapIdentificationResult(
            cutoff_date=None,
            warnings=[
                GapWarning(
                    code="no_extracted_insights",
                    message="No structured insights were available.",
                )
            ],
            gaps=[],
        )
    )

    output = aggregator_module.aggregator_node(state)

    assert output["final_report"].gaps == []
    assert output["final_report"].cutoff_date is None
    assert len(fake_llm.structured_llm.invoke_calls) == 1


def test_aggregator_returns_empty_report_when_graph_hypotheses_are_missing(
    monkeypatch,
):
    def fail_if_called(role: str):
        raise AssertionError(f"LLM should not be called for {role}")

    monkeypatch.setattr(aggregator_module, "get_llm", fail_if_called)
    state = _state(
        GapIdentificationResult(
            cutoff_date=date(2025, 4, 3),
            gaps=[_gap()],
        )
    ).model_copy(
        deep=True,
        update={
            "graph_insight": GraphInsight(
                summary="No hypotheses.",
                raw={"ranked_hypotheses": []},
            )
        },
    )

    report = aggregator_module.aggregator_node(state)["final_report"]

    assert report.gaps == []
    assert report.cutoff_date == date(2025, 4, 3)
    assert [
        warning.code for warning in report.warnings
    ] == ["missing_ranked_graph_hypotheses"]
    assert "ranked graph hypotheses" in report.summary
    assert report.sources_used == ["arxiv"]
    assert report.papers_considered == 1


def test_aggregator_preserves_existing_warnings_in_missing_graph_fallback(
    monkeypatch,
):
    monkeypatch.setattr(
        aggregator_module,
        "get_llm",
        lambda role: pytest.fail("LLM should not be called"),
    )
    existing_warning = GapWarning(
        code="missing_evidence",
        message="A textual candidate was discarded.",
    )
    state = _state(
        GapIdentificationResult(
            cutoff_date=date(2025, 4, 3),
            warnings=[existing_warning],
            gaps=[],
        )
    ).model_copy(deep=True, update={"graph_insight": None})

    report = aggregator_module.aggregator_node(state)["final_report"]

    assert report.warnings[0] == existing_warning
    assert report.warnings[1].code == "missing_ranked_graph_hypotheses"


def test_aggregator_skips_when_gap_identifier_has_not_finished(monkeypatch):
    def fail_if_called(role: str):
        raise AssertionError(f"LLM should not be called for {role}")

    monkeypatch.setattr(aggregator_module, "get_llm", fail_if_called)

    assert aggregator_module.aggregator_node(_state(None)) == {}
