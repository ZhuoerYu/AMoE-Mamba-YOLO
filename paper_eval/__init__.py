"""Frozen COCO evaluation protocol used by the AMoE-Mamba-YOLO paper."""

from .evaluator import PAPER_PROTOCOL, evaluate_detections, run_evaluation

__all__ = ["PAPER_PROTOCOL", "evaluate_detections", "run_evaluation"]

