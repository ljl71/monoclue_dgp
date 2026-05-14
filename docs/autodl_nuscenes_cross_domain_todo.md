# AutoDL nuScenes Frontal 跨域深度误差实验续跑记录

本文档记录当前 AutoDL 环境、已有数据、已执行步骤，以及晚上继续执行的命令。目标是复现实验：

> 所有模型仅在 KITTI train 上训练，不使用 nuScenes 训练数据；分别在 KITTI val 和 nuScenes frontal val 上统计 Car 类在 0-20m、20-40m、40-80m 三个距离段的深度误差。

## 当前条件

AutoDL 项目目录：

```bash
/root/autodl-tmp/monoclue_dgp
```

Conda 环境：

```bash
conda activate monoclue
```

磁盘状态，已扩容：

```text
/root/autodl-tmp 总容量: 200G
已用: 43G
可用: 158G
```

已有 KITTI 数据：

```bash
/root/autodl-tmp/monoclue_dgp/data/kitti
```

大小：

```text
29G
```

AutoDL 公共 nuScenes 数据：

```bash
/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval
```

内容为压缩包：

```text
v1.0-trainval01_blobs.tgz
v1.0-trainval02_blobs.tgz
...
v1.0-trainval10_blobs.tgz
v1.0-trainval_meta.tgz
```

公共 nuScenes trainval 压缩包总大小：

```text
294G
```

注意：公共数据不占 `/root/autodl-tmp` 空间；只有解压到 `/root/autodl-tmp` 的内容会占用自己的磁盘。

## 当前已经执行到的位置

已经进入项目目录并激活环境：

```bash
cd /root/autodl-tmp/monoclue_dgp
conda activate monoclue
```

已经创建 nuScenes 解压目录：

```bash
mkdir -p /root/autodl-tmp/data/nuscenes
NUSC=/root/autodl-tmp/data/nuscenes
SRC=/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval
```

已经解压 metadata：

```bash
tar -xzf $SRC/v1.0-trainval_meta.tgz -C $NUSC
```

检查结果为：

```text
/root/autodl-tmp/data/nuscenes/maps
/root/autodl-tmp/data/nuscenes/v1.0-trainval
```

当前正在执行只提取 `CAM_FRONT` 和 `LIDAR_TOP` 的命令：

```bash
for f in $SRC/v1.0-trainval*_blobs.tgz; do
  echo "extracting $f"
  tar -xzf "$f" -C "$NUSC" \
    --wildcards 'samples/CAM_FRONT/*' 'samples/LIDAR_TOP/*'
done
```

当前输出停在：

```text
extracting /root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval01_blobs.tgz
```

这是正常的。每个 `.tgz` 很大，`tar` 需要扫描压缩包，可能很久才输出下一个文件名。

## 今晚继续执行步骤

### 1. 先确认解压是否完成

如果原终端还在执行，不要关，不要重复运行，继续等待它依次输出到：

```text
v1.0-trainval10_blobs.tgz
```

可以另开一个终端监控：

```bash
watch -n 30 'df -h /root/autodl-tmp; du -sh /root/autodl-tmp/data/nuscenes; echo CAM_FRONT; ls /root/autodl-tmp/data/nuscenes/samples/CAM_FRONT 2>/dev/null | wc -l; echo LIDAR_TOP; ls /root/autodl-tmp/data/nuscenes/samples/LIDAR_TOP 2>/dev/null | wc -l'
```

理想文件数大约：

```text
CAM_FRONT: 34149 左右
LIDAR_TOP: 34149 左右
```

解压完成后检查：

```bash
du -sh /root/autodl-tmp/data/nuscenes
ls /root/autodl-tmp/data/nuscenes/samples/CAM_FRONT | head
ls /root/autodl-tmp/data/nuscenes/samples/LIDAR_TOP | head
df -h /root/autodl-tmp
```

如果可用空间低于 `30G`，先暂停后续转换。

### 2. 安装并检查 nuScenes devkit

```bash
cd /root/autodl-tmp/monoclue_dgp
conda activate monoclue
pip show nuscenes-devkit || pip install nuscenes-devkit
```

建立 nuScenes 官方默认路径软链接：

```bash
mkdir -p /data/sets
ln -sfn /root/autodl-tmp/data/nuscenes /data/sets/nuscenes
```

测试能否读取：

```bash
python - <<'PY'
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-trainval', dataroot='/data/sets/nuscenes', verbose=True)
print("sample count:", len(nusc.sample))
PY
```

正常情况下 `sample count` 应该约为：

```text
34149
```

### 3. 导出 nuScenes frontal val 为 KITTI 格式

执行：

```bash
python -m nuscenes.scripts.export_kitti nuscenes_gt_to_kitti \
  --nusc_kitti_dir /root/autodl-tmp/data/nuscenes_kitti_raw \
  --nusc_version v1.0-trainval \
  --split val \
  --cam_name CAM_FRONT \
  --image_count 6019
```

导出后应出现：

```bash
/root/autodl-tmp/data/nuscenes_kitti_raw/val
```

检查：

```bash
find /root/autodl-tmp/data/nuscenes_kitti_raw/val -maxdepth 2 -type d
find /root/autodl-tmp/data/nuscenes_kitti_raw/val -maxdepth 2 -type f | head
```

通常会包含：

```text
image_2
label_2
calib
velodyne
```

### 4. 整理成当前模型需要的 KITTI-like 目录

当前模型读取目录格式：

```text
data/nuscenes_front_kitti/
  ImageSets/val.txt
  training/image_2/
  training/label_2/
  training/calib/
```

执行：

```bash
RAW=/root/autodl-tmp/data/nuscenes_kitti_raw/val
OUT=/root/autodl-tmp/monoclue_dgp/data/nuscenes_front_kitti

mkdir -p $OUT/ImageSets $OUT/training/image_2 $OUT/training/label_2 $OUT/training/calib

python - <<'PY'
import os, glob, shutil

raw = "/root/autodl-tmp/data/nuscenes_kitti_raw/val"
out = "/root/autodl-tmp/monoclue_dgp/data/nuscenes_front_kitti"

cls_map = {
    "car": "Car",
    "pedestrian": "Pedestrian",
    "bicycle": "Cyclist",
    "motorcycle": "Cyclist",
}

ids = []
imgs = sorted(glob.glob(os.path.join(raw, "image_2", "*.png")))

for i, img in enumerate(imgs):
    old_id = os.path.splitext(os.path.basename(img))[0]
    new_id = f"{i:06d}"
    ids.append(new_id)

    dst_img = os.path.join(out, "training/image_2", new_id + ".png")
    if not os.path.exists(dst_img):
        os.symlink(img, dst_img)

    shutil.copyfile(
        os.path.join(raw, "calib", old_id + ".txt"),
        os.path.join(out, "training/calib", new_id + ".txt")
    )

    src_label = os.path.join(raw, "label_2", old_id + ".txt")
    dst_label = os.path.join(out, "training/label_2", new_id + ".txt")

    with open(dst_label, "w") as fw:
        if os.path.exists(src_label):
            for line in open(src_label):
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] not in cls_map:
                    continue
                parts[0] = cls_map[parts[0]]
                fw.write(" ".join(parts) + "\n")

with open(os.path.join(out, "ImageSets/val.txt"), "w") as f:
    f.write("\n".join(ids) + "\n")

print("converted", len(ids), "CAM_FRONT images")
PY
```

检查：

```bash
wc -l /root/autodl-tmp/monoclue_dgp/data/nuscenes_front_kitti/ImageSets/val.txt
find /root/autodl-tmp/monoclue_dgp/data/nuscenes_front_kitti -maxdepth 3 -type f | head
head /root/autodl-tmp/monoclue_dgp/data/nuscenes_front_kitti/training/label_2/000000.txt
df -h /root/autodl-tmp
```

`val.txt` 行数应接近或等于：

```text
6019
```

## 跑跨域深度误差实验

前提：本地代码改动已经同步到 AutoDL，至少应有以下文件：

```text
tools/eval_cross_domain_depth.py
tools/eval_depth_error.py
lib/datasets/kitti/depth_error_eval.py
docs/cross_domain_depth_eval.md
```

如果还没有同步，需要先上传/复制这些文件到 AutoDL 的：

```bash
/root/autodl-tmp/monoclue_dgp
```

确认 checkpoint 路径。当前可能使用：

```bash
outputs/monoclue/checkpoint.pth
```

如果你的实际 checkpoint 是 `checkpoint_best.pth` 或在 `/root/autodl-tmp/checkpoints`，需要替换命令里的路径。

运行：

```bash
cd /root/autodl-tmp/monoclue_dgp
conda activate monoclue

python tools/eval_cross_domain_depth.py \
  --config configs/monoclue.yaml \
  --checkpoint outputs/monoclue/checkpoint.pth \
  --dataset KITTI,/root/autodl-tmp/monoclue_dgp/data/kitti,val,.png \
  --dataset nuScenes-frontal,/root/autodl-tmp/monoclue_dgp/data/nuscenes_front_kitti,val,.png \
  --classes Car \
  --bins 0,20,40,80
```

输出位置：

```bash
outputs/monoclue/cross_domain_depth/<timestamp>/
```

里面会有：

```text
depth_errors.csv
KITTI/data/*.txt
nuScenes-frontal/data/*.txt
eval.log
```

## 如果只想评估已有预测结果

如果其他方法已经生成 KITTI 格式预测结果，可以不重新跑模型，直接比较：

```bash
python tools/eval_depth_error.py \
  --label-dir /root/autodl-tmp/monoclue_dgp/data/nuscenes_front_kitti/training/label_2 \
  --split-file /root/autodl-tmp/monoclue_dgp/data/nuscenes_front_kitti/ImageSets/val.txt \
  --dataset-name nuScenes-frontal \
  --result-dir Ours=/path/to/ours/results/data \
  --classes Car \
  --bins 0,20,40,80 \
  --output-csv outputs/depth_error_compare.csv
```

可以重复加多个方法：

```bash
--result-dir MonoDGP=/path/to/monodgp/results/data
--result-dir Ours=/path/to/ours/results/data
```

## 常见问题

### 1. 解压时看起来卡住

正常。`.tgz` 很大，`tar` 过滤特定目录也需要顺序扫描整个压缩包。

### 2. 如果解压被中断

可以重新执行同一个循环命令。它会重新扫描压缩包并覆盖/补齐文件，但会比较耗时。

### 3. 如果 `python -m nuscenes.scripts.export_kitti` 报找不到模块

先执行：

```bash
pip install nuscenes-devkit
```

### 4. 如果 nuScenes devkit 读取失败

重点检查：

```bash
ls /root/autodl-tmp/data/nuscenes/v1.0-trainval
ls /root/autodl-tmp/data/nuscenes/samples/CAM_FRONT | head
ls /root/autodl-tmp/data/nuscenes/samples/LIDAR_TOP | head
ls -l /data/sets/nuscenes
```

### 5. 如果深度误差全是空或匹配数为 0

检查 label 类别名是否是 KITTI 风格：

```bash
head /root/autodl-tmp/monoclue_dgp/data/nuscenes_front_kitti/training/label_2/000000.txt
```

应该看到：

```text
Car ...
```

而不是：

```text
car ...
```

### 6. 如果 checkpoint 路径不对

查找：

```bash
find /root/autodl-tmp/monoclue_dgp/outputs -name "*.pth"
find /root/autodl-tmp/checkpoints -name "*.pth" 2>/dev/null
```

然后把 `--checkpoint` 改成真实路径。
