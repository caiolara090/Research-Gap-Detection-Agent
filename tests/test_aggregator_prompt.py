from datetime import date

from research_gap_agent.prompts.aggregator import build_aggregator_messages
from research_gap_agent.schemas import (
    ExtractedInsights,
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


def test_aggregator_prompt_contains_textual_gaps_and_ranked_graph_hypotheses_only():
    ranked_hypothesis = {
        "concepts": ["longitudinal evaluation", "agent reliability"],
        "missing_links": [["longitudinal evaluation", "agent reliability"]],
        "score": 0.91,
        "diversified_score": 0.84,
    }
    state = GraphState(
        initial_topic="Long-term reliability of AI agents",
        queries=[
            SearchQuery(
                text="AI agents longitudinal reliability",
                rationale="Find long-horizon studies.",
            )
        ],
        raw_papers=[
            _paper("paper-1", "arxiv"),
            _paper("paper-2", "openalex"),
        ],
        ranked_papers=[
            _paper("paper-1", "arxiv"),
            _paper("paper-2", "openalex"),
        ],
        extracted=[
            ExtractedInsights(
                paper_id="paper-1",
                title="Paper paper-1",
                published_date=date(2025, 1, 1),
            )
        ],
        graph_insight=GraphInsight(
            summary="DO NOT INCLUDE GRAPH SUMMARY PROSE",
            disconnected_pairs=[("unused-source", "unused-target")],
            raw={
                "ranked_hypotheses": [ranked_hypothesis],
                "graph_nodes": 999,
                "graph_edges": 888,
            },
        ),
        gap_identification=GapIdentificationResult(
            cutoff_date=date(2025, 1, 1),
            warnings=[
                GapWarning(
                    code="invalid_counter_evidence_reference",
                    message="Earlier warning must be visible.",
                )
            ],
            gaps=[
                IdentifiedGap(
                    research_question=(
                        "How reliable are agents over long deployments?"
                    ),
                    description=(
                        "Long deployments remain underexplored."
                    ),
                    evidence_strength=88,
                    evidence=[
                        GapEvidence(
                            paper_id="paper-1",
                            evidence_type="stated_limitations",
                            description=(
                                "The paper asks for longer follow-up."
                            ),
                        )
                    ],
                    rationale="The textual corpus supports the candidate.",
                )
            ],
        ),
    )

    messages = build_aggregator_messages(state)

    assert [role for role, _ in messages] == ["system", "human"]
    system_message = messages[0][1]
    human_message = messages[1][1]

    for expected_rule in [
        "No graph_only",
        "textual_only",
        "textual_and_graph",
        "ranked_graph_hypotheses",
        "FinalReport",
        "JSON",
    ]:
        assert expected_rule in system_message

    for expected_payload in [
        "Long-term reliability of AI agents",
        "How reliable are agents over long deployments?",
        "invalid_counter_evidence_reference",
        "longitudinal evaluation",
        "agent reliability",
        "missing_links",
        "methodology_context",
        "sources_used",
        "arxiv",
    ]:
        assert expected_payload in human_message

    assert "DO NOT INCLUDE GRAPH SUMMARY PROSE" not in human_message
    assert "unused-source" not in human_message
    assert "graph_nodes" not in human_message
