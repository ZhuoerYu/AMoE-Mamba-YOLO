import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / "results" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_reported_results_use_only_the_paper_coco_evaluator():
    rows = read_csv("paper_coco_metrics.csv")
    by_id = {row["ID"]: row for row in rows}

    assert len(rows) == 14
    assert float(by_id["s0-m01"]["AP50_95_%"]) == 14.176
    assert float(by_id["s0-m01"]["AR500_%"]) == 29.820
    assert float(by_id["ablation-m01-axis-top4-s0"]["AP50_95_%"]) == 14.228


def test_results_directory_contains_only_the_paper_accuracy_table():
    assert {path.name for path in (ROOT / "results").glob("*.csv")} == {"paper_coco_metrics.csv"}
