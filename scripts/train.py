#!/usr/bin/env python3
"""Train AMoE-Mamba-YOLO with the reported VisDrone protocol."""

from __future__ import annotations

import argparse
import json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ultralytics/cfg/models/amoe/amoe-mamba-yolo-n.yaml")
    parser.add_argument("--data", default="configs/visdrone.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.000714)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--nbs", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default=None)
    parser.add_argument("--exist-ok", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_train_kwargs(args: argparse.Namespace) -> dict:
    name = args.name or f"amoe-mamba-yolo-n-seed{args.seed}"
    return {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "nbs": args.nbs,
        "amp": True,
        "deterministic": True,
        "pretrained": False,
        "patience": 0,
        "workers": args.workers,
        "seed": args.seed,
        "device": args.device,
        "project": args.project,
        "name": name,
        "exist_ok": args.exist_ok,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    kwargs = build_train_kwargs(args)
    if args.dry_run:
        print(json.dumps({"model": args.model, **kwargs}, indent=2))
        return
    from ultralytics import YOLO

    YOLO(args.model, task="detect").train(**kwargs)


if __name__ == "__main__":
    main()
