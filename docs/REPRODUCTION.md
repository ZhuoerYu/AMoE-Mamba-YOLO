# Reproduction

## Reported training protocol

| Setting | Value |
|---|---|
| Dataset | VisDrone2019-DET |
| Epochs | 100 |
| Image size | 640 |
| Batch size | 8 |
| Optimizer | AdamW |
| Initial learning rate | 0.000714 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| Nominal batch size | 64 |
| AMP | enabled |
| Pretrained weights | disabled |
| Seeds | 20260821, 20260822, 20260823 |

The defaults in `scripts/train.py`, `configs/main_experiments.yaml`, and `configs/ablations.yaml` encode this protocol.

## Main experiments

```bash
python scripts/reproduce_visdrone.py \
  --config configs/main_experiments.yaml \
  --device 0
```

This runs the standard convolutional baseline, the proposed AMoE-Mamba fusion model, the Backbone-MoE control, and the combined control for each seed.

## Ablations

```bash
python scripts/reproduce_visdrone.py \
  --config configs/ablations.yaml \
  --device 0
```

The ablation manifest covers Single-SS2D, homogeneous experts, Global Top-2, GAP Top-2, Axis Top-1, and Axis Top-4. The proposed Axis Top-2 configuration is the `AMoE-Mamba-YOLO-N` run from the main manifest.

Use `--dry-run` to inspect every resolved task before allocating a GPU. Use `--only MODEL_NAME` to select entries from a manifest.

## Evaluation and latency

```bash
python scripts/evaluate.py \
  --weights amoe-mamba-yolo-n-visdrone-seed20260821.pt \
  --data configs/visdrone.local.yaml \
  --device 0

python scripts/benchmark.py \
  --weights amoe-mamba-yolo-n-visdrone-seed20260821.pt \
  --data configs/visdrone.local.yaml \
  --device 0
```

The published latency values were measured with native FP16 inference on an RTX 5090 at batch size 1 and image size 640. Compare latency only under the same device, precision, batch, and input-size protocol.
