"""Structured prompt construction for radiology VLM inference."""

from __future__ import annotations

from data.retriever import RetrievedCase

MEDICAL_DISCLAIMER = (
    "DISCLAIMER: This analysis is for educational and research purposes only. "
    "It is NOT a substitute for professional medical advice, diagnosis, or "
    "treatment. Always consult a qualified healthcare provider for medical "
    "decisions."
)

SYSTEM_INSTRUCTION = (
    "You are a board-certified radiologist AI assistant specializing in "
    "chest X-ray interpretation. Analyze the provided chest X-ray image "
    "and give a structured diagnostic report.\n\n"
    "Your report must include:\n"
    "1. **Findings**: Describe observable abnormalities or normal anatomy.\n"
    "2. **Diagnosis**: List the most likely diagnoses based on the image.\n"
    "3. **Confidence**: For each diagnosis, indicate your confidence level "
    "(High / Moderate / Low) and briefly explain why.\n"
    "4. **Recommendations**: Suggest follow-up imaging or clinical actions "
    "if appropriate.\n\n"
    f"{MEDICAL_DISCLAIMER}"
)


def _format_similar_case(case: RetrievedCase, index: int) -> str:
    """Format a single retrieved case as a few-shot example line."""
    findings = case.positive_findings
    if not findings:
        findings_str = "No Finding"
    else:
        findings_str = ", ".join(findings)
    return (
        f"  Similar Case {index}: "
        f"Image={case.image_id}, "
        f"Similarity={case.similarity:.3f}, "
        f"Diagnosis=[{findings_str}]"
    )


def build_prompt(similar_cases: list[RetrievedCase]) -> str:
    """Build the full prompt for VLM inference.

    Parameters
    ----------
    similar_cases : list[RetrievedCase]
        Top-k retrieved cases from the dataset.

    Returns
    -------
    str
        Fully assembled prompt string.
    """
    sections: list[str] = [SYSTEM_INSTRUCTION, ""]

    # Few-shot context from retrieved similar cases.
    if similar_cases:
        sections.append(
            "The following similar cases were found in the reference dataset "
            "based on image embedding similarity. Use them as supporting "
            "context (not as the sole basis) for your analysis:"
        )
        for i, case in enumerate(similar_cases, start=1):
            sections.append(_format_similar_case(case, i))
        sections.append("")

    # User instruction.
    sections.append(
        "Now analyze the attached chest X-ray image. Provide your "
        "structured diagnostic report following the format above."
    )

    return "\n".join(sections)
