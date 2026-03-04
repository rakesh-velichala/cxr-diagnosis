"""Structured prompt construction for VLM-based chest X-ray diagnosis."""

from __future__ import annotations

from models.base import DISEASE_LABELS

MEDICAL_DISCLAIMER = (
    "DISCLAIMER: This analysis is for educational and research purposes only. "
    "It is NOT a substitute for professional medical advice, diagnosis, or "
    "treatment. Always consult a qualified healthcare provider for medical "
    "decisions."
)

_DISEASE_LIST = "\n".join(f"  - {label}" for label in DISEASE_LABELS)

_DIAGNOSIS_PROMPT = f"""You are a board-certified radiologist AI assistant specializing in chest X-ray interpretation.

Analyze the provided chest X-ray image and identify the most likely diagnosis.

**Valid diagnoses** (choose ONLY from this list):
{_DISEASE_LIST}

**Instructions:**
1. Examine the chest X-ray carefully.
2. Select the 1-2 most likely diagnoses from the valid list above.
3. For each diagnosis, assign a confidence level: "High", "Moderate", or "Low".
4. Order diagnoses by descending probability (most likely first).
5. If the X-ray appears normal, use "No Finding".

**Output format — respond with ONLY this JSON, no other text:**
```json
{{
  "diagnoses": [
    {{"disease": "<disease name from list>", "confidence": "<High|Moderate|Low>"}},
    {{"disease": "<disease name from list>", "confidence": "<High|Moderate|Low>"}}
  ]
}}
```

Respond with the JSON only. Do not include any explanation, findings, or recommendations."""


def build_diagnosis_prompt() -> str:
    """Return the structured diagnosis prompt for VLM inference.

    Returns
    -------
    str
        Prompt string that instructs the VLM to return JSON with
        1-2 diagnoses from the canonical 20-label set.
    """
    return _DIAGNOSIS_PROMPT
