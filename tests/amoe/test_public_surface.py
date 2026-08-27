from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"


def test_readme_has_academic_repository_structure():
    text = README.read_text(encoding="utf-8")
    expected = [
        "# AMoE-Mamba-YOLO",
        "Zhuoer Yu",
        "Guangyu Wu",
        "## Model Zoo",
        "## Getting started",
        "### 1. Installation",
        "### 2. Data preparation",
        "### 3. Training",
        "### 4. Evaluation",
        "### 5. Inference",
        "## Acknowledgement",
        "## Citation",
    ]
    assert all(item in text for item in expected)


def test_public_documentation_and_assets_are_present():
    required = [
        "assets/architecture.png",
        "assets/amoe_block.png",
        "docs/DATA.md",
        "docs/MODEL_ZOO.md",
        "docs/REPRODUCTION.md",
        "requirements/base.txt",
        "requirements/cuda.txt",
        "THIRD_PARTY_NOTICES.md",
        "CITATION.cff",
    ]
    assert all((ROOT / relative).is_file() for relative in required)


def test_citation_lists_the_two_current_authors_in_order():
    citation = yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    assert [(author["given-names"], author["family-names"]) for author in citation["authors"]] == [
        ("Zhuoer", "Yu"),
        ("Guangyu", "Wu"),
    ]


def test_package_version_matches_the_public_release():
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.1.0"' in project
    assert 'dynamic = ["version"]' not in project


def test_public_text_has_no_machine_specific_paths_or_unverified_venue_claims():
    files = [README, *sorted((ROOT / "docs").glob("*.md")), ROOT / "CITATION.cff"]
    text = "\n".join(path.read_text(encoding="utf-8") for path in files)
    prohibited = [
        "/" + "Users/",
        "/" + "data/jyj/",
        "5090" + "ubuntu",
        "f" + "rp",
        "AAAI" + "2025",
        "accepted" + " at",
    ]
    assert all(item not in text for item in prohibited)
