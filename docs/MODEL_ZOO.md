# Model Zoo

## AMoE-Mamba-YOLO-N checkpoints

The three checkpoints were trained from scratch with identical settings and matched random seeds. The paper's unified COCO evaluation uses the seed-20260821 checkpoint. The remaining checkpoints are provided to support repeated training studies; no alternate evaluator metrics are reported here.

| Seed | File | SHA-256 |
|---:|---|---|
| 20260821 | `amoe-mamba-yolo-n-visdrone-seed20260821.pt` | `6a55dad1ecf952f13281817a1c01c51b59bf079581a9e3924f8ac275f102f942` |
| 20260822 | `amoe-mamba-yolo-n-visdrone-seed20260822.pt` | `ba7beb870f54ac20f2d60befc90a27e7b0070ad2e6c3b4c4d91416a6c1a3a566` |
| 20260823 | `amoe-mamba-yolo-n-visdrone-seed20260823.pt` | `1f0be9f8ed609a0b4ba804189ef70f780a49db1e6ccf6f985fc9c5953adc5424` |

Weights are available from the [v0.1.0 release](https://github.com/ZhuoerYu/AMoE-Mamba-YOLO/releases/tag/v0.1.0). Verify downloaded files with the release `SHA256SUMS` manifest.

## Controlled results

All paper accuracy values are in [`results/paper_coco_metrics.csv`](../results/paper_coco_metrics.csv). They use `pycocotools.COCOeval` on all 548 validation images with confidence 0.001 and `maxDets=[1, 10, 100, 500]`. AMoE-Mamba-YOLO-N obtains 14.176 AP; the Top-4 accuracy reference obtains 14.228 AP.
