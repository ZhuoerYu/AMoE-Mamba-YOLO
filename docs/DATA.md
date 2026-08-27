# VisDrone2019-DET

The experiments use the official VisDrone2019-DET training and validation splits. The dataset is not redistributed by this repository.

## Raw layout

Place the downloaded splits beneath one directory:

```text
VisDrone/raw/
|-- VisDrone2019-DET-train/
|   |-- images/
|   `-- annotations/
`-- VisDrone2019-DET-val/
    |-- images/
    `-- annotations/
```

## Conversion

```bash
python scripts/prepare_visdrone.py \
  --source VisDrone/raw \
  --output datasets/VisDrone2019-DET \
  --yaml configs/visdrone.local.yaml
```

The converter creates symbolic links to images by default and writes YOLO labels for the ten foreground categories. Pass `--copy` to copy images. An existing output directory is left unchanged unless `--overwrite` is supplied.

The reported split contains 6,471 training images and 548 validation images. Before training, verify that `configs/visdrone.local.yaml` points to the converted dataset.
