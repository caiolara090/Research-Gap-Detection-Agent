from datetime import date

import pytest

import research_gap_agent.nodes.gap_identifier as gap_identifier_module
from research_gap_agent.schemas import (
    CounterEvidence,
    ExtractedInsights,
    GapEvidence,
    GapIdentificationResult,
    GapWarning,
    IdentifiedGap,
)
from research_gap_agent.state import GraphState


def _insight(
    paper_id: str,
    title: str,
    published_date: date,
) -> ExtractedInsights:
    return ExtractedInsights(
        paper_id=paper_id,
        title=title,
        published_date=published_date,
        questions_answered=[f"Question answered by {paper_id}"],
        methodologies=[f"Method used by {paper_id}"],
        not_addressed=[f"Scope omitted by {paper_id}"],
        stated_limitations=[f"Limitation stated by {paper_id}"],
    )


class FakeStructuredLLM:
    def __init__(self, response: GapIdentificationResult):
        self.response = response
        self.invoke_calls: list[list[tuple[str, str]]] = []

    def invoke(
        self,
        messages: list[tuple[str, str]],
    ) -> GapIdentificationResult:
        self.invoke_calls.append(messages)
        return self.response


class FakeLLM:
    def __init__(self, response: GapIdentificationResult):
        self.structured_llm = FakeStructuredLLM(response)
        self.structured_output_calls: list[type] = []

    def with_structured_output(
        self,
        schema: type,
        **kwargs,
    ) -> FakeStructuredLLM:
        self.structured_output_calls.append(schema)
        return self.structured_llm


def test_empty_input_skips_llm_and_returns_structured_warning(monkeypatch):
    def fail_if_called(role: str):
        raise AssertionError(f"get_llm must not be called for role {role}")

    monkeypatch.setattr(gap_identifier_module, "get_llm", fail_if_called)
    state = GraphState(initial_topic="AI agent reliability")

    result = gap_identifier_module.gap_identifier_node(state)

    assert result == {
        "gap_identification": GapIdentificationResult(
            cutoff_date=None,
            warnings=[
                GapWarning(
                    code="no_extracted_insights",
                    message=(
                        "No structured article insights were available for "
                        "gap identification."
                    ),
                )
            ],
            gaps=[],
        )
    }


def test_non_empty_input_uses_one_structured_request_for_the_complete_corpus(
    monkeypatch,
):
    insights = [
        _insight("paper-1", "Paper One", date(2025, 1, 2)),
        _insight("paper-2", "Paper Two", date(2025, 2, 3)),
    ]
    expected_gap = IdentifiedGap(
        research_question="How reliable are agents over long deployments?",
        description="Long deployments remain underexplored in this corpus.",
        evidence_strength=88,
        evidence=[
            GapEvidence(
                paper_id="paper-1",
                evidence_type="stated_limitations",
                description="Paper One identifies a follow-up limitation.",
            )
        ],
        rationale="The corpus contains direct textual support.",
        counter_evidence=[],
    )
    llm_result = GapIdentificationResult(
        cutoff_date=date(2025, 2, 3),
        gaps=[expected_gap],
    )
    fake_llm = FakeLLM(llm_result)
    requested_roles: list[str] = []

    def fake_get_llm(role: str) -> FakeLLM:
        requested_roles.append(role)
        return fake_llm

    monkeypatch.setattr(gap_identifier_module, "get_llm", fake_get_llm)
    state = GraphState(
        initial_topic="AI agent reliability",
        extracted=insights,
    )

    result = gap_identifier_module.gap_identifier_node(state)

    assert requested_roles == ["gap_identifier"]
    assert fake_llm.structured_output_calls == [GapIdentificationResult]
    assert len(fake_llm.structured_llm.invoke_calls) == 1
    messages = fake_llm.structured_llm.invoke_calls[0]
    human_messages = [content for role, content in messages if role == "human"]
    assert len(human_messages) == 1
    assert "AI agent reliability" in human_messages[0]
    assert "paper-1" in human_messages[0]
    assert "paper-2" in human_messages[0]
    assert result == {"gap_identification": llm_result}


def test_non_empty_input_rejects_missing_cutoff_after_one_invoke(monkeypatch):
    insight = _insight("paper-1", "Paper One", date(2025, 1, 2))
    fake_llm = FakeLLM(
        GapIdentificationResult(
            cutoff_date=None,
            gaps=[],
        )
    )
    monkeypatch.setattr(
        gap_identifier_module,
        "get_llm",
        lambda role: fake_llm,
    )

    with pytest.raises(
        ValueError,
        match=(
            "^cutoff_date is required when extracted insights are available$"
        ),
    ):
        gap_identifier_module.gap_identifier_node(
            GraphState(
                initial_topic="AI agent reliability",
                extracted=[insight],
            )
        )

    assert len(fake_llm.structured_llm.invoke_calls) == 1


def test_non_empty_input_rejects_inexact_cutoff_after_one_invoke(monkeypatch):
    insights = [
        _insight("paper-1", "Paper One", date(2025, 1, 2)),
        _insight("paper-2", "Paper Two", date(2025, 2, 3)),
    ]
    fake_llm = FakeLLM(
        GapIdentificationResult(
            cutoff_date=date(2025, 1, 2),
            gaps=[],
        )
    )
    monkeypatch.setattr(
        gap_identifier_module,
        "get_llm",
        lambda role: fake_llm,
    )

    with pytest.raises(
        ValueError,
        match=(
            "^cutoff_date must equal the latest extracted insight "
            "publication date$"
        ),
    ):
        gap_identifier_module.gap_identifier_node(
            GraphState(
                initial_topic="AI agent reliability",
                extracted=insights,
            )
        )

    assert len(fake_llm.structured_llm.invoke_calls) == 1


def test_node_stores_validated_result(monkeypatch):
    insight = _insight("paper-1", "Paper One", date(2025, 1, 2))
    raw_gap = IdentifiedGap(
        research_question="What remains underexplored?",
        description="A candidate gap.",
        evidence_strength=75,
        evidence=[
            GapEvidence(
                paper_id="paper-1",
                evidence_type="contrast",
                description="A neighboring question remains unanswered.",
            )
        ],
        rationale="The structured insight supports the candidate.",
        counter_evidence=[
            CounterEvidence(
                paper_id="unknown-paper",
                description="Invalid counterevidence.",
            )
        ],
    )
    fake_llm = FakeLLM(
        GapIdentificationResult(
            cutoff_date=date(2025, 1, 2),
            gaps=[raw_gap],
        )
    )
    monkeypatch.setattr(
        gap_identifier_module,
        "get_llm",
        lambda role: fake_llm,
    )

    result = gap_identifier_module.gap_identifier_node(
        GraphState(
            initial_topic="AI agent reliability",
            extracted=[insight],
        )
    )

    stored = result["gap_identification"]
    assert stored.gaps[0].counter_evidence == []
    assert stored.warnings == [
        GapWarning(
            code="invalid_counter_evidence_reference",
            message=(
                "Removed counterevidence from candidate gap 'What remains "
                "underexplored?' because it referenced unknown paper_id "
                "'unknown-paper'."
            ),
        )
    ]
