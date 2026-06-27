"""
Aggregator node (owner: Vitor).
"""

import logging

from research_gap_agent.llm import get_llm
from research_gap_agent.prompts.aggregator import (
    build_aggregator_messages,
    methodology_context,
    ranked_graph_hypotheses,
)
from research_gap_agent.schemas import FinalReport, GapWarning
from research_gap_agent.state import GraphState


logger = logging.getLogger(__name__)


def _missing_graph_hypotheses_report(state: GraphState) -> FinalReport:
    gap_identification = state.gap_identification
    if gap_identification is None:
        raise ValueError(
            "Aggregator integration error: gap_identification result is "
            "required; use an empty GapIdentificationResult for a valid "
            "empty analysis."
        )

    context = methodology_context(state)
    warnings = [
        warning.model_copy(deep=True)
        for warning in gap_identification.warnings
    ]
    warnings.append(
        GapWarning(
            code="missing_ranked_graph_hypotheses",
            message=(
                "Aggregator did not run the LLM fusion step because no "
                "ranked graph hypotheses were available."
            ),
        )
    )

    return FinalReport(
        topic=state.initial_topic,
        cutoff_date=gap_identification.cutoff_date,
        warnings=warnings,
        gaps=[],
        summary=(
            "Aggregation was not performed because ranked graph hypotheses "
            "were unavailable."
        ),
        methodology_note=(
            "Candidate gaps are reported only after textual candidates can "
            "be compared with ranked graph hypotheses. "
            f"Searched {len(context['sources_used'])} sources with "
            f"{context['query_count']} rewritten queries; ranked top "
            f"{context['ranked_papers']} of {context['raw_papers']} papers; "
            f"analyzed {context['structured_insights']} structured insights."
        ),
        sources_used=context["sources_used"],
        papers_considered=context["structured_insights"],
    )


def _with_deterministic_metadata(
    report: FinalReport,
    state: GraphState,
) -> FinalReport:
    gap_identification = state.gap_identification
    if gap_identification is None:
        return report

    context = methodology_context(state)
    return FinalReport.model_validate(
        {
            **report.model_dump(mode="python"),
            "topic": state.initial_topic,
            "cutoff_date": gap_identification.cutoff_date,
            "sources_used": context["sources_used"],
            "papers_considered": context["structured_insights"],
        }
    )


def aggregator_node(state: GraphState) -> dict:
    gap_identification = state.gap_identification
    if gap_identification is None:
        logger.info(
            "aggregator skipped because gap_identifier has not written "
            "gap_identification yet for topic=%r",
            state.initial_topic,
        )
        return {}

    if not ranked_graph_hypotheses(state):
        report = _missing_graph_hypotheses_report(state)
        return {"final_report": report}

    messages = build_aggregator_messages(state)
    llm = get_llm("aggregator").with_structured_output(
        FinalReport,
        method="function_calling",
    )
    report = llm.invoke(messages)
    report = _with_deterministic_metadata(report, state)

    logger.info(
        "aggregator produced %d final candidates for topic=%r",
        len(report.gaps),
        state.initial_topic,
    )

    return {"final_report": report}
