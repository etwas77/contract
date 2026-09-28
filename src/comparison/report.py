from __future__ import annotations

from comparison.models import (
    ComparisonReportData,
    ComparisonStatus,
    ConstraintResult,
)

STATUS_ICON = {
    ComparisonStatus.SATISFIED: "PASS",
    ComparisonStatus.VIOLATED: "FAIL",
    ComparisonStatus.UNKNOWN: "UNKNOWN",
    ComparisonStatus.NOT_APPLICABLE: "N/A",
}


def render_report(data: ComparisonReportData) -> str:
    summary = data.summary
    lines = [
        "# Candidate and company offer comparison",
        "",
        f"**Overall outcome:** `{summary.overall_outcome.value}`",
        "",
        "This report is a structured comparison aid, not legal advice.",
        "",
        "## Summary",
        "",
        "| Priority | Satisfied | Violated | Unknown | Not applicable |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Mandatory | {summary.mandatory_satisfied} | "
            f"{summary.mandatory_violated} | {summary.mandatory_unknown} | "
            f"{summary.mandatory_not_applicable} |"
        ),
        (
            f"| Preferred | {summary.preferred_satisfied} | "
            f"{summary.preferred_violated} | {summary.preferred_unknown} | "
            f"{summary.preferred_not_applicable} |"
        ),
        "",
        "## Constraint results",
        "",
        "| Status | Priority | Requirement | Actual offer term | Method | Source |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(_result_row(result) for result in data.results)
    unknown = [result for result in data.results if result.status == ComparisonStatus.UNKNOWN]
    lines.extend(["", "## Unknowns requiring clarification", ""])
    if unknown:
        lines.extend(
            f"- **{_escape(result.requirement)}:** {_escape(result.rationale)}"
            for result in unknown
        )
    else:
        lines.append("- None.")

    lines.extend(["", "## Candidate-role assessment", ""])
    assessment = data.candidate_role_assessment
    if assessment is None:
        lines.append("No semantic candidate-role assessment was required.")
    else:
        lines.extend(
            [
                assessment.summary,
                "",
                "**Matched skills:** "
                + (", ".join(assessment.matched_skills) or "None established."),
                "",
                "**Possible gaps:** "
                + (", ".join(assessment.possible_gaps) or "None established."),
            ]
        )

    lines.extend(
        [
            "",
            "## Method and provenance",
            "",
            f"- Comparison version: `{data.comparison_version}`",
            f"- Model: `{data.model or 'not used'}`",
            f"- Conservative OpenAI cost reserved: `${data.reserved_cost_usd}`",
            "- Deterministic results were not overridable by the semantic agents.",
            "- Missing contract information was treated as unknown, not compliant.",
            "",
            "### Input hashes",
            "",
        ]
    )
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in data.input_hashes.items())
    if data.warnings:
        lines.extend(["", "### Warnings", ""])
        lines.extend(f"- {_escape(warning)}" for warning in data.warnings)
    return "\n".join(lines) + "\n"


def _result_row(result: ConstraintResult) -> str:
    source = _source_label(result)
    return (
        f"| {STATUS_ICON[result.status]} | {result.priority.value} | "
        f"{_escape(result.requirement)} | {_escape(result.actual or result.rationale)} | "
        f"{result.method.value} | {_escape(source)} |"
    )


def _source_label(result: ConstraintResult) -> str:
    if not result.company_evidence:
        return "No offer evidence"
    source = result.company_evidence[0].source
    if source.page is not None:
        return f"offer page {source.page}"
    return f"offer lines {source.line_start}-{source.line_end or source.line_start}"


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
