#!/usr/bin/env python3
"""Run AMoE-Mamba-YOLO inference on images or video."""

from __future__ import annotations

import argparse
import json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/predict")
    parser.add_argument("--name", default="amoe-mamba-yolo-n")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    kwargs = {
        "source": args.source,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "save": True,
    }
    if args.dry_run:
        print(json.dumps({"weights": args.weights, **kwargs}, indent=2))
        return
    from ultralytics import YOLO

    YOLO(args.weights, task="detect").predict(**kwargs)


if __name__ == "__main__":
    main()
