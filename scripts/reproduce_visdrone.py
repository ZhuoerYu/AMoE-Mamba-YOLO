#!/usr/bin/env python3
"""Run a YAML experiment manifest sequentially."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def load_tasks(path: str | Path) -> list[dict]:
    manifest = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    tasks = []
    for seed in manifest["seeds"]:
        for model in manifest["models"]:
            tasks.append(
                {
                    "name": model["name"],
                    "model": model["config"],
                    "data": manifest["dataset"],
                    "output": manifest["output"],
                    "fixed": dict(manifest["fixed"]),
                    "seed": seed,
                }
            )
    return tasks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main_experiments.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--only", nargs="*", help="Run only the named model entries")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_kwargs(task: dict, device: str) -> dict:
    return {
        "data": task["data"],
        **task["fixed"],
        "seed": task["seed"],
        "device": device,
        "project": task["output"],
        "name": f"{task['name'].lower()}-seed{task['seed']}",
        "exist_ok": False,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tasks = load_tasks(args.config)
    if args.only:
        selected = set(args.only)
        tasks = [task for task in tasks if task["name"] in selected]
    if args.dry_run:
        print(
            json.dumps(
                [{"model": task["model"], **build_kwargs(task, args.device)} for task in tasks],
                indent=2,
            )
        )
        return
    from ultralytics import YOLO

    for task in tasks:
        print(f"Training {task['name']} with seed {task['seed']}")
        YOLO(task["model"], task="detect").train(**build_kwargs(task, args.device))


if __name__ == "__main__":
    main()
