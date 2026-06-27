"""Prompt assembly for final LLM-driven gap aggregation."""

import json
from typing import Any

from research_gap_agent.state import GraphState


AGGREGATOR_SYSTEM = """\
You produce the final research gap report by merging textual candidate gaps
with ranked graph hypotheses.

Domain rules:
- The final output contains candidate research gaps, not proven gaps.
- No graph_only final gaps are allowed.
- The graph can refine or strengthen a textual candidate gap, but cannot
  create a final gap by itself.
- Preserve every valid textual candidate unless it is a semantic duplicate of
  a stronger candidate or the corpus clearly answers it.

Fusion rules:
- Compare each textual candidate against ranked_graph_hypotheses.
- A valid graph match requires overlap or equivalence between central
  concepts, compatibility with the underexplored graph relation, and a short
  explanation of how the niche refines the textual question.
- Use origin="textual_and_graph" when a ranked graph hypothesis refines a
  textual candidate. Fill matched_graph_hypothesis with the exact full graph
  hypothesis object and graph_refinement with the refinement explanation.
- Use origin="textual_only" when no ranked graph hypothesis matches. Set
  matched_graph_hypothesis and graph_refinement to null.
- Never add graph hypotheses that have no matching textual candidate.

Ranking rules:
- Rank textual_and_graph gaps before textual_only gaps.
- Within each origin group, sort by textual evidence_strength descending.
- Graph strength may break ties, but must not replace textual evidence.

Output rules:
- Return exactly one JSON object matching FinalReport.
- Every gap in FinalReport.gaps must be an IdentifiedGap with origin,
  matched_graph_hypothesis, and graph_refinement set coherently.
- Preserve textual-stage warnings in FinalReport.warnings and add any
  aggregation warnings when needed.
- Mention that gaps are candidates observed in the analyzed corpus up to the
  cutoff date.
- Return JSON only, without Markdown or explanatory text.
"""


def ranked_graph_hypotheses(state: GraphState) -> list[dict[str, Any]]:
    """Return the only graph signal allowed into the aggregator prompt."""
    if state.graph_insight is None:
        return []
    hypotheses = state.graph_insight.raw.get("ranked_hypotheses")
    if not isinstance(hypotheses, list):
        return []
    return [
        hypothesis
        for hypothesis in hypotheses
        if isinstance(hypothesis, dict)
    ]


def sources_used(state: GraphState) -> list[str]:
    ranked_papers_by_id = {
        paper.id: paper for paper in state.ranked_papers
    }
    return sorted(
        {
            ranked_papers_by_id[insight.paper_id].source
            for insight in state.extracted
            if insight.paper_id in ranked_papers_by_id
        }
    )


def methodology_context(state: GraphState) -> dict[str, Any]:
    return {
        "sources_used": sources_used(state),
        "query_count": len(state.queries),
        "raw_papers": len(state.raw_papers),
        "ranked_papers": len(state.ranked_papers),
        "structured_insights": len(state.extracted),
    }


def build_aggregator_messages(state: GraphState) -> list[tuple[str, str]]:
    if state.gap_identification is None:
        raise ValueError(
            "gap_identification result is required to build aggregator prompt"
        )

    payload = {
        "topic": state.initial_topic,
        "cutoff_date": (
            state.gap_identification.cutoff_date.isoformat()
            if state.gap_identification.cutoff_date
            else None
        ),
        "textual_warnings": [
            warning.model_dump(mode="json")
            for warning in state.gap_identification.warnings
        ],
        "textual_candidate_gaps": [
            gap.model_dump(mode="json")
            for gap in state.gap_identification.gaps
        ],
        "ranked_graph_hypotheses": ranked_graph_hypotheses(state),
        "methodology_context": methodology_context(state),
    }

    return [
        ("system", AGGREGATOR_SYSTEM),
        (
            "human",
            "Aggregator input (JSON):\n"
            + json.dumps(payload, ensure_ascii=True, indent=2),
        ),
    ]
