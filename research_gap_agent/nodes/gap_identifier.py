"""Identify corpus-relative research gap candidates with one LLM request."""

import logging

from research_gap_agent.llm import get_llm
from research_gap_agent.prompts.gap_identifier import (
    build_gap_identifier_messages,
)
from research_gap_agent.schemas import (
    ExtractedInsights,
    GapIdentificationResult,
    GapWarning,
)
from research_gap_agent.state import GraphState


logger = logging.getLogger(__name__)


def validate_gap_identification(
    result: GapIdentificationResult,
    extracted_insights: list[ExtractedInsights],
) -> GapIdentificationResult:
    """Remove invalid references without mutating or rescoring LLM output."""
    if extracted_insights and result.cutoff_date is None:
        raise ValueError(
            "cutoff_date is required when extracted insights are available"
        )
    if extracted_insights and result.cutoff_date != max(
        insight.published_date for insight in extracted_insights
    ):
        raise ValueError(
            "cutoff_date must equal the latest extracted insight "
            "publication date"
        )

    valid_paper_ids = {
        insight.paper_id
        for insight in extracted_insights
    }
    warnings = []
    valid_gaps = []

    for gap in result.gaps:
        if not gap.evidence:
            warnings.append(
                GapWarning(
                    code="missing_evidence",
                    message=(
                        f"Discarded candidate gap "
                        f"'{gap.research_question}' because it contained no "
                        "evidence."
                    ),
                )
            )
            continue

        invalid_evidence_ids = list(
            dict.fromkeys(
                evidence.paper_id
                for evidence in gap.evidence
                if evidence.paper_id not in valid_paper_ids
            )
        )
        if invalid_evidence_ids:
            if len(invalid_evidence_ids) == 1:
                reference_description = (
                    "unknown paper_id "
                    f"'{invalid_evidence_ids[0]}'"
                )
            else:
                rendered_ids = ", ".join(
                    f"'{paper_id}'"
                    for paper_id in invalid_evidence_ids
                )
                reference_description = (
                    f"unknown paper_ids {rendered_ids}"
                )

            warnings.append(
                GapWarning(
                    code="invalid_evidence_reference",
                    message=(
                        f"Discarded candidate gap "
                        f"'{gap.research_question}' because evidence "
                        f"referenced {reference_description}."
                    ),
                )
            )
            continue

        valid_counter_evidence = []
        for counter_evidence in gap.counter_evidence:
            if counter_evidence.paper_id in valid_paper_ids:
                valid_counter_evidence.append(
                    counter_evidence.model_copy(deep=True)
                )
                continue

            warnings.append(
                GapWarning(
                    code="invalid_counter_evidence_reference",
                    message=(
                        f"Removed counterevidence from candidate gap "
                        f"'{gap.research_question}' because it referenced "
                        f"unknown paper_id '{counter_evidence.paper_id}'."
                    ),
                )
            )

        if len(valid_counter_evidence) == len(gap.counter_evidence):
            valid_gaps.append(gap.model_copy(deep=True))
        else:
            valid_gaps.append(
                gap.model_copy(
                    deep=True,
                    update={
                        "counter_evidence": valid_counter_evidence,
                    }
                )
            )

    return result.model_copy(
        deep=True,
        update={
            "warnings": warnings,
            "gaps": valid_gaps,
        }
    )


def gap_identifier_node(state: GraphState) -> dict:
    """Identify and validate candidate gaps from all extracted insights."""
    if not state.extracted:
        return {
            "gap_identification": GapIdentificationResult(
                cutoff_date=None,
                warnings=[
                    GapWarning(
                        code="no_extracted_insights",
                        message=(
                            "No structured article insights were available "
                            "for gap identification."
                        ),
                    )
                ],
                gaps=[],
            )
        }

    messages = build_gap_identifier_messages(
        state.initial_topic,
        state.extracted,
    )
    llm = get_llm("gap_identifier").with_structured_output(
        GapIdentificationResult,
        method="function_calling",
    )
    raw_result = llm.invoke(messages)
    validated_result = validate_gap_identification(
        raw_result,
        state.extracted,
    )

    logger.info(
        "gap_identifier produced %d validated candidates for topic=%r",
        len(validated_result.gaps),
        state.initial_topic,
    )
    return {"gap_identification": validated_result}
