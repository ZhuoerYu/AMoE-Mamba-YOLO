import csv
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / "results" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_main_results_contain_the_three_matched_seeds():
    rows = read_csv("visdrone_main.csv")
    proposed = [row for row in rows if row["model"] == "AMoE-Mamba-YOLO-N"]
    baseline = [row for row in rows if row["model"] == "Baseline-Conv-YOLO-N"]

    assert [int(row["seed"]) for row in proposed] == [20260821, 20260822, 20260823]
    assert round(mean(float(row["ap50_95"]) for row in proposed), 6) == 13.485607
    assert round(mean(float(row["ap50_95"]) for row in baseline), 6) == 12.532589


def test_ablation_results_identify_the_sparse_and_dense_variants():
    rows = read_csv("visdrone_ablations.csv")
    by_name = {row["variant"]: row for row in rows}

    assert float(by_name["Axis-Top2"]["ap50_95"]) == 13.663710
    assert float(by_name["Axis-Top4"]["ap50_95"]) == 13.876960
    assert float(by_name["Axis-Top2"]["forward_latency_ms"]) == 8.940232
    assert float(by_name["Axis-Top4"]["forward_latency_ms"]) == 11.068575
