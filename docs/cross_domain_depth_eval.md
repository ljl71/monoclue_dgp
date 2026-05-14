# Cross-domain depth error evaluation

This experiment evaluates a model trained on KITTI train without using target-domain
nuScenes data.  The evaluator reports matched-object depth MAE in three ranges:
0-20m, 20-40m, and 40-80m.

## Data format

Both KITTI val and nuScenes frontal val should be arranged in KITTI label format:

```text
data/nuscenes_front_kitti/
  ImageSets/
    val.txt
  training/
    image_2/
      000000.jpg
    label_2/
      000000.txt
    calib/
      000000.txt
```

`label_2/*.txt` must use KITTI object labels:

```text
type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry
```

For nuScenes frontal, use only `CAM_FRONT` images and convert boxes into the
front camera coordinate system before writing `x y z ry`.

## Run model inference and depth evaluation

```bash
python tools/eval_cross_domain_depth.py \
  --config configs/monoclue.yaml \
  --checkpoint outputs/monoclue/checkpoint.pth \
  --dataset KITTI,/path/to/kitti,val,.png \
  --dataset nuScenes-frontal,/path/to/nuscenes_front_kitti,val,.jpg \
  --classes Car \
  --bins 0,20,40,80
```

The script writes predictions and `depth_errors.csv` under:

```text
outputs/monoclue/cross_domain_depth/<timestamp>/
```

## Evaluate existing prediction files

Use this when comparing multiple methods whose predictions are already saved in
KITTI text format:

```bash
python tools/eval_depth_error.py \
  --label-dir /path/to/nuscenes_front_kitti/training/label_2 \
  --split-file /path/to/nuscenes_front_kitti/ImageSets/val.txt \
  --dataset-name nuScenes-frontal \
  --result-dir MonoDGP=/path/to/monodgp/results/data \
  --result-dir Ours=/path/to/ours/results/data \
  --classes Car \
  --bins 0,20,40,80 \
  --output-csv outputs/depth_error_compare.csv
```

Depth error is computed after class-consistent 2D IoU matching.  The default
matching threshold is 0.5.  The table also reports coverage, so a lower MAE with
very low matched coverage should not be over-interpreted.
