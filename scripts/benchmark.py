#!/usr/bin/env python3
"""Measure validation speed for an AMoE-Mamba-YOLO checkpoint."""

from __future__ import annotations

import argparse
import json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/visdrone.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/benchmark")
    parser.add_argument("--name", default="amoe-mamba-yolo-n")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    kwargs = {
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "plots": False,
    }
    if args.dry_run:
        print(json.dumps({"weights": args.weights, **kwargs}, indent=2))
        return
    from ultralytics import YOLO

    metrics = YOLO(args.weights, task="detect").val(**kwargs)
    print(json.dumps(metrics.speed, indent=2))


if __name__ == "__main__":
    main()
