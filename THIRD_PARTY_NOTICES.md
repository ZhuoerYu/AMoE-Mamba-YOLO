# Third-Party Notices

AMoE-Mamba-YOLO includes or adapts source from the projects below. The repository as a whole is distributed under the GNU Affero General Public License v3.0. Original notices remain in the corresponding source files and license copies are retained in `LICENSES/`.

## Ultralytics

- Source: https://github.com/ultralytics/ultralytics
- Runtime version: 8.4.101
- License: GNU Affero General Public License v3.0
- Use: model parser, detection models, training, validation, inference, losses, data loading, and utilities under `ultralytics/`.

## YOLO-Master

- Source: https://github.com/Tencent/YOLO-Master
- License: GNU Affero General Public License v3.0
- Use: dynamic convolutional expert controls and the global-average-pooling Top-K routing control.
- License copy: `LICENSES/YOLO-Master.LICENSE`.

## Mamba-YOLO

- Source: https://github.com/HZAI-ZJNU/Mamba-YOLO
- Pinned revision: `b26cbda230dfa217f96faee8dc7020db3962f3df`
- License: GNU Affero General Public License v3.0
- Use: XSSBlock shell, LSBlock, RGBlock, two-dimensional scan ordering, state-space parameterization, and CUDA selective-scan extension.
- License copy: `LICENSES/Mamba-YOLO.LICENSE`.

## Selective scan

The source in `selective_scan/` is taken from the pinned Mamba-YOLO revision above. Its setup metadata also credits Tri Dao and Albert Gu and links to https://github.com/state-spaces/mamba. Those original copyright and attribution lines are retained.
