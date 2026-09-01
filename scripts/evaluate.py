#!/usr/bin/env python3
"""Evaluate an AMoE-Mamba-YOLO checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
from paper_eval import PAPER_PROTOCOL, run_evaluation


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/visdrone.yaml")
    parser.add_argument("--imgsz", type=int, choices=[640], default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/eval")
    parser.add_argument("--name", default="amoe-mamba-yolo-n")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_evaluation_config(args: argparse.Namespace) -> dict[str, object]:
    """Return the immutable protocol used to produce the manuscript tables."""
    return {
        **PAPER_PROTOCOL,
        "image_size": args.imgsz,
        "weights": args.weights,
        "data": args.data,
        "batch": args.batch,
        "device": args.device,
        "output": str(Path(args.project) / args.name),
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_evaluation_config(args)
    if args.dry_run:
        print(json.dumps(config, indent=2))
        return
    result = run_evaluation(
        weights=args.weights,
        data=Path(args.data),
        output=Path(args.project) / args.name,
        batch=args.batch,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
