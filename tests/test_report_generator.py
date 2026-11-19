
from reports.template_generator import generate_report


def test_report_generation():
    predictions = {
        "Cardiomegaly": 0.82,
        "Edema": 0.15,
        "Consolidation": 0.73,
        "Atelectasis": 0.20,
        "Pleural Effusion": 0.91
    }

    report = generate_report(predictions)

    assert "Cardiomegaly" in report["findings"]
    assert "Consolidation" in report["findings"]
    assert "Pleural Effusion" in report["findings"]
    assert "Edema" not in report["findings"]

    assert report["title"] == "AI-Assisted Chest X-ray Draft"

    assert "clinician" in report["disclaimer"].lower()
