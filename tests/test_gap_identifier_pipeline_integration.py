from datetime import date
from types import SimpleNamespace

import research_gap_agent.nodes.gap_identifier as gap_identifier_module
import research_gap_agent.nodes.paper_extractor as paper_extractor_module
from research_gap_agent.cli import render_report
from research_gap_agent.nodes.aggregator import aggregator_node
from research_gap_agent.schemas import (
    CounterEvidence,
    GapEvidence,
    GapIdentificationResult,
    GraphInsight,
    IdentifiedGap,
    Paper,
    SearchQuery,
)
from research_gap_agent.state import GraphState


def _paper(
    paper_id: str,
    source: str,
    title: str,
    published_date: date,
) -> Paper:
    return Paper(
        id=paper_id,
        source=source,
        title=title,
        abstract=f"Abstract for {title}",
        authors=["Researcher"],
        published_date=published_date,
        url=f"https://example.com/{paper_id}",
        pdf_url=f"https://example.com/{paper_id}.pdf",
    )


def _initial_state() -> GraphState:
    papers = [
        _paper(
            "paper-alpha",
            "arxiv",
            "Reliable Agents in Short Deployments",
            date(2025, 3, 14),
        ),
        _paper(
            "paper-beta",
            "openalex",
            "Evaluating Agents in Production",
            date(2026, 2, 20),
        ),
        _paper(
            "paper-gamma",
            "semantic_scholar",
            "Agent Reliability Benchmarks",
            date(2026, 4, 8),
        ),
    ]
    return GraphState(
        initial_topic="Reliability of AI agents in long deployments",
        queries=[
            SearchQuery(
                text="AI agent reliability long deployments",
                rationale="Find evidence about deployment duration.",
            )
        ],
        raw_papers=papers,
        ranked_papers=papers,
    )


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        yaml=SimpleNamespace(
            document_converter=SimpleNamespace(
                provider_name="fake",
                use_arxiv_html=False,
            )
        )
    )


def _merge_state(state: GraphState, node_output: dict) -> GraphState:
    return GraphState.model_validate(
        {**state.model_dump(), **node_output}
    )


class FakeConverter:
    def __init__(self, results: list[str | None]):
        self.results = results
        self.convert_batch_calls: list[list[str]] = []

    def convert_batch(self, sources: list[str]) -> list[str | None]:
        self.convert_batch_calls.append(sources)
        return self.results


class RecordingStructuredLLM:
    def __init__(self, response: GapIdentificationResult):
        self.response = response
        self.structured_output_calls: list[type] = []
        self.invoke_calls: list[list[tuple[str, str]]] = []

    def with_structured_output(
        self,
        schema: type,
        **kwargs,
    ) -> "RecordingStructuredLLM":
        self.structured_output_calls.append(schema)
        return self

    def invoke(
        self,
        messages: list[tuple[str, str]],
    ) -> GapIdentificationResult:
        self.invoke_calls.append(messages)
        return self.response


def test_offline_pipeline_preserves_state_and_renders_identified_gap(
    monkeypatch,
):
    initial_state = _initial_state()
    converter = FakeConverter(
        [
            "# Reliable Agents\n\nShort deployment study.",
            "",
            None,
        ]
    )
    monkeypatch.setattr(
        paper_extractor_module,
        "load_settings",
        _settings,
    )
    monkeypatch.setattr(
        paper_extractor_module,
        "get_converter",
        lambda provider: converter,
    )

    extractor_output = paper_extractor_module.paper_extractor_node(
        initial_state
    )
    extracted_state = _merge_state(initial_state, extractor_output)

    assert converter.convert_batch_calls == [
        [paper.pdf_url for paper in initial_state.ranked_papers]
    ]
    assert [
        document.id for document in extracted_state.extracted_documents
    ] == ["paper-alpha"]
    assert [
        document.title for document in extracted_state.extracted_documents
    ] == ["Reliable Agents in Short Deployments"]
    assert [
        document.published_date
        for document in extracted_state.extracted_documents
    ] == [date(2025, 3, 14)]
    assert [
        insight.paper_id for insight in extracted_state.extracted
    ] == ["paper-alpha"]
    assert [
        insight.title for insight in extracted_state.extracted
    ] == ["Reliable Agents in Short Deployments"]
    assert [
        insight.published_date for insight in extracted_state.extracted
    ] == [date(2025, 3, 14)]
    assert all(
        insight.questions_answered == []
        and insight.methodologies == []
        and insight.not_addressed == []
        and insight.stated_limitations == []
        for insight in extracted_state.extracted
    )
    assert all(
        "full_text" not in insight.model_dump()
        for insight in extracted_state.extracted
    )
    assert all(
        document.full_text
        for document in extracted_state.extracted_documents
    )

    extracted_snapshot = [
        insight.model_copy(deep=True)
        for insight in extracted_state.extracted
    ]
    identified_gap = IdentifiedGap(
        research_question=(
            "Which deployment conditions remain understudied?"
        ),
        description=(
            "Long-duration reliability remains a candidate gap."
        ),
        evidence_strength=86,
        evidence=[
            GapEvidence(
                paper_id="paper-alpha",
                evidence_type="contrast",
                description=(
                    "Paper Alpha reports limited longitudinal evaluation."
                ),
            )
        ],
        rationale="The corpus focuses on shorter evaluation windows.",
        counter_evidence=[
            CounterEvidence(
                paper_id="unknown-paper",
                description="This invalid reference must be removed.",
            )
        ],
    )
    fake_llm = RecordingStructuredLLM(
        GapIdentificationResult(
            cutoff_date=max(
                insight.published_date
                for insight in extracted_state.extracted
            ),
            gaps=[identified_gap],
        )
    )
    requested_roles: list[str] = []

    def fake_get_llm(role: str) -> RecordingStructuredLLM:
        requested_roles.append(role)
        return fake_llm

    monkeypatch.setattr(
        gap_identifier_module,
        "get_llm",
        fake_get_llm,
    )

    gap_output = gap_identifier_module.gap_identifier_node(
        extracted_state
    )
    gap_state = _merge_state(extracted_state, gap_output)
    graph_state = gap_state.model_copy(
        update={
            "graph_insight": GraphInsight(
                summary="Two evaluation themes remain disconnected."
            )
        }
    )
    final_state = _merge_state(
        graph_state,
        aggregator_node(graph_state),
    )

    assert requested_roles == ["gap_identifier"]
    assert fake_llm.structured_output_calls == [GapIdentificationResult]
    assert len(fake_llm.invoke_calls) == 1
    human_messages = [
        content
        for role, content in fake_llm.invoke_calls[0]
        if role == "human"
    ]
    assert len(human_messages) == 1
    assert all(
        insight.paper_id in human_messages[0]
        for insight in extracted_state.extracted
    )
    assert "paper-beta" not in human_messages[0]
    assert "paper-gamma" not in human_messages[0]

    assert final_state.extracted == extracted_snapshot
    assert final_state.final_report is not None
    assert final_state.final_report.cutoff_date == date(2025, 3, 14)
    assert final_state.final_report.gaps == [
        identified_gap.model_copy(update={"counter_evidence": []})
    ]
    assert [
        warning.code for warning in final_state.final_report.warnings
    ] == ["invalid_counter_evidence_reference"]
    assert final_state.final_report.sources_used == ["arxiv"]
    assert final_state.final_report.papers_considered == 1
    assert (
        "analyzed 1 structured insight from 3 ranked papers"
        in final_state.final_report.methodology_note
    )

    rendered = render_report(final_state.final_report)
    assert "Which deployment conditions remain understudied" in rendered
    assert "Paper Alpha reports limited longitudinal evaluation" in rendered
    assert "_Cutoff date: 2025-03-14_" in rendered


def test_empty_conversion_skips_llm_and_reports_analysis_not_performed(
    monkeypatch,
):
    initial_state = _initial_state()
    converter = FakeConverter(["", None, ""])
    monkeypatch.setattr(
        paper_extractor_module,
        "load_settings",
        _settings,
    )
    monkeypatch.setattr(
        paper_extractor_module,
        "get_converter",
        lambda provider: converter,
    )
    llm_roles: list[str] = []

    def fail_if_llm_is_requested(role: str):
        llm_roles.append(role)
        raise AssertionError("LLM must not be requested without insights")

    monkeypatch.setattr(
        gap_identifier_module,
        "get_llm",
        fail_if_llm_is_requested,
    )

    extracted_state = _merge_state(
        initial_state,
        paper_extractor_module.paper_extractor_node(initial_state),
    )
    gap_state = _merge_state(
        extracted_state,
        gap_identifier_module.gap_identifier_node(extracted_state),
    )
    graph_state = gap_state.model_copy(
        update={
            "graph_insight": GraphInsight(
                summary="No textual documents were converted."
            )
        }
    )
    final_state = _merge_state(
        graph_state,
        aggregator_node(graph_state),
    )

    assert converter.convert_batch_calls == [
        [paper.pdf_url for paper in initial_state.ranked_papers]
    ]
    assert extracted_state.extracted_documents == []
    assert extracted_state.extracted == []
    assert llm_roles == []
    assert final_state.final_report is not None
    assert final_state.final_report.gaps == []
    assert final_state.final_report.cutoff_date is None
    assert [
        warning.code for warning in final_state.final_report.warnings
    ] == ["no_extracted_insights"]
    assert final_state.final_report.sources_used == []
    assert final_state.final_report.papers_considered == 0
    assert (
        "analyzed 0 structured insights from 3 ranked papers"
        in final_state.final_report.methodology_note
    )
    assert final_state.final_report.summary.startswith(
        "Gap identification was not performed because no structured "
        "article insights were available."
    )
    assert (
        "Analysis was not performed because no structured article insights "
        "were available."
        in render_report(final_state.final_report)
    )
