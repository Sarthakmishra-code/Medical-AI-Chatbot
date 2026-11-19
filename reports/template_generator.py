
from typing import Dict, List

PATHOLOGIES = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion"
]


def generate_report(
    predictions: Dict[str, float],
    threshold: float = 0.5
) -> Dict:

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    findings: List[str] = []

    for pathology in PATHOLOGIES:
        probability = predictions.get(pathology)

        if probability is None:
            continue

        if not 0 <= probability <= 1:
            raise ValueError(
                f"Invalid probability for {pathology}"
            )

        if probability >= threshold:
            findings.append(
                f"{pathology}: model score {probability:.2f}"
            )

    if findings:
        findings_text = "; ".join(findings)
    else:
        findings_text = (
            "No target pathology exceeded the configured "
            "model threshold."
        )

    return {
        "title": "AI-Assisted Chest X-ray Draft",
        "findings": findings_text,
        "model_predictions": predictions,
        "disclaimer": (
            "Automated model output, not a confirmed diagnosis. "
            "Requires review by a qualified clinician."
        )
    }
