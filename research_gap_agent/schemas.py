"""Data models shared by all nodes in the pipeline.

These Pydantic models define the contracts between nodes. Whenever a node
reads or writes a piece of structured data, it should use one of the models
below so that we get validation for free at every step.
"""

from datetime import date
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


SourceName = Literal["arxiv", "openalex", "semantic_scholar"]


class SearchQuery(BaseModel):
    """A single search query produced by the query rewriter."""

    text: str
    rationale: str


class Paper(BaseModel):
    """A normalized academic paper.

    Every Paper that flows past the search node is guaranteed to be open
    access and to have a working pdf_url, so downstream nodes can download
    the full text without worrying about paywalls.
    """

    id: str
    source: SourceName
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    published_date: date
    url: str
    pdf_url: str
    is_open_access: bool = True
    oa_status: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    categories: list[str] = Field(default_factory=list)
    full_text: Optional[str] = None


class ExtractedInsights(BaseModel):
    """Structured extraction from a single paper.

    Produced by the paper_extractor node. The four fields mirror the
    questions defined in the project spec.
    """

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    title: str
    published_date: date
    questions_answered: list[str] = Field(default_factory=list)
    methodologies: list[str] = Field(default_factory=list)
    not_addressed: list[str] = Field(default_factory=list)
    stated_limitations: list[str] = Field(default_factory=list)


class GraphInsight(BaseModel):
    """Output of the graph analyzer.

    TODO(caio): may need to change.
    """

    summary: str
    disconnected_pairs: list[tuple[str, str]] = Field(default_factory=list)
    raw: dict = Field(default_factory=dict)


EvidenceType = Literal[
    "stated_limitations",
    "recurring_not_addressed",
    "contrast",
]

FusionOrigin = Literal["textual_only", "textual_and_graph"]


class GapWarning(BaseModel):
    """A stable warning emitted while identifying gaps."""

    model_config = ConfigDict(extra="forbid")

    code: str
    message: str


class CounterEvidence(BaseModel):
    """Evidence that weakens or qualifies a candidate gap."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    description: str


class GapEvidence(BaseModel):
    """Traceable textual evidence supporting a candidate gap."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    evidence_type: EvidenceType
    description: str

    @field_validator("evidence_type", mode="before")
    @classmethod
    def normalize_evidence_type(cls, value):
        if value == "not_addressed":
            return "recurring_not_addressed"
        return value


class IdentifiedGap(BaseModel):
    """A candidate research gap built from the extracted insights."""

    model_config = ConfigDict(extra="forbid")

    research_question: str
    description: str
    evidence_strength: int = Field(..., ge=70, le=100)
    evidence: list[GapEvidence] = Field(default_factory=list)
    rationale: str
    counter_evidence: list[CounterEvidence] = Field(default_factory=list)
    origin: FusionOrigin | None = None
    matched_graph_hypothesis: dict[str, Any] | None = None
    graph_refinement: str | None = None

    @model_validator(mode="after")
    def validate_fusion_fields(self) -> "IdentifiedGap":
        has_graph_match = self.matched_graph_hypothesis is not None
        has_graph_refinement = bool(self.graph_refinement)

        if self.origin is None:
            if has_graph_match or has_graph_refinement:
                raise ValueError(
                    "graph fusion fields require an origin"
                )
            return self

        if self.origin == "textual_only":
            if has_graph_match or has_graph_refinement:
                raise ValueError(
                    "textual_only gaps must not include graph fusion fields"
                )
            return self

        if not has_graph_match or not has_graph_refinement:
            raise ValueError(
                "textual_and_graph gaps require matched_graph_hypothesis "
                "and graph_refinement"
            )
        return self


class GapIdentificationResult(BaseModel):
    """Complete output of the gap identification stage."""

    model_config = ConfigDict(extra="forbid")

    cutoff_date: date | None
    warnings: list[GapWarning] = Field(default_factory=list)
    gaps: list[IdentifiedGap] = Field(default_factory=list)


class FinalReport(BaseModel):
    """What the user sees at the end of the pipeline."""

    topic: str
    cutoff_date: date | None
    warnings: list[GapWarning] = Field(default_factory=list)
    gaps: list[IdentifiedGap]
    summary: str
    methodology_note: str
    sources_used: list[SourceName]
    papers_considered: int

    @model_validator(mode="after")
    def require_origin_on_final_gaps(self) -> "FinalReport":
        missing_origin = [
            gap.research_question
            for gap in self.gaps
            if gap.origin is None
        ]
        if missing_origin:
            rendered = ", ".join(f"'{question}'" for question in missing_origin)
            raise ValueError(
                "FinalReport gaps must declare origin; missing for "
                f"{rendered}"
            )
        return self
