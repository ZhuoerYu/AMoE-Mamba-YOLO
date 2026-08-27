# Model Zoo

## AMoE-Mamba-YOLO-N checkpoints

The three checkpoints were trained from scratch with identical settings and matched random seeds. AP values are Ultralytics AP50:95 on the complete 548-image VisDrone2019-DET validation split.

| Seed | AP50:95 | File | SHA-256 |
|---:|---:|---|---|
| 20260821 | 13.663710 | `amoe-mamba-yolo-n-visdrone-seed20260821.pt` | `6a55dad1ecf952f13281817a1c01c51b59bf079581a9e3924f8ac275f102f942` |
| 20260822 | 13.275759 | `amoe-mamba-yolo-n-visdrone-seed20260822.pt` | `ba7beb870f54ac20f2d60befc90a27e7b0070ad2e6c3b4c4d91416a6c1a3a566` |
| 20260823 | 13.517352 | `amoe-mamba-yolo-n-visdrone-seed20260823.pt` | `1f0be9f8ed609a0b4ba804189ef70f780a49db1e6ccf6f985fc9c5953adc5424` |

Weights are available from the [v0.1.0 release](https://github.com/ZhuoerYu2-c/AMoE-Mamba-YOLO/releases/tag/v0.1.0). Verify downloaded files with the release `SHA256SUMS` manifest.

## Controlled results

The four placement configurations use the same training budget and three matched seeds. AMoE-Mamba-YOLO-N improves the baseline mean AP50:95 from 12.532589 to 13.485607. Seed-level precision, recall, and AP values are available in [`results/visdrone_main.csv`](../results/visdrone_main.csv).

The sparse routing table is available in [`results/visdrone_ablations.csv`](../results/visdrone_ablations.csv). All ablations use seed 20260821 and the complete validation split.
