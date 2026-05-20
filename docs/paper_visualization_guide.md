# MonoCLUE-DGP 论文可视化运行说明

本文档说明如何使用 `tools/visualize_paper_figures.py` 生成论文实验部分需要的可视化图。脚本默认适配你的重训练模型：

```bash
/root/autodl-tmp/monoclue_dgp/outputs/monoclue_dgp_rerun/checkpoint_best.pth
```

## 1. 基本运行命令

在 AutoDL 服务器上进入项目目录：

```bash
cd /root/autodl-tmp/monoclue_dgp
```

生成 6 个困难样本的完整可视化：

```bash
python tools/visualize_paper_figures.py \
  --config configs/monoclue_dgp_rerun.yaml \
  --checkpoint /root/autodl-tmp/monoclue_dgp/outputs/monoclue_dgp_rerun/checkpoint_best.pth \
  --output-dir paper_figures/visual_analysis_rerun \
  --split val \
  --num-samples 6 \
  --auto-filter hard \
  --threshold 0.25 \
  --workers 2 \
  --panel-mode both \
  --detection-views 2d,3d,bev
```

输出目录结构如下：

```text
paper_figures/visual_analysis_rerun/
  01_label_supervision/          # 二维框监督 vs SAM 可见前景/深度监督
  02_response_maps/              # 前景概率图、深度响应图、前景加权深度响应
  03_prototype_relocalization/   # 前景原型、背景参照、原型相似性/重定位响应
  04_detection_bev/              # 图像视角 2D 框、3D 框 + BEV 定位对比
  05_ablation_summary/           # 消融趋势和原型数量敏感性图
  06_depth_predictor_outputs/     # DepthPredictor 真实模型输出：logits 分布与 soft-argmin 深度
```

其中 `04_detection_bev/split/` 下会保存可直接放入论文的单独图片：

- `*_01_image_view_2d_boxes.png`：2D 检测效果图；
- `*_02_image_view_3d_boxes.png`：图像视角 3D 检测框；
- `*_03_bev_localization.png`：BEV 俯视定位效果图。

## 2. 指定样本出图

如果你已经挑好了 KITTI 验证集图片编号，可以指定 `--sample-ids`：

```bash
python tools/visualize_paper_figures.py \
  --config configs/monoclue_dgp_rerun.yaml \
  --checkpoint /root/autodl-tmp/monoclue_dgp/outputs/monoclue_dgp_rerun/checkpoint_best.pth \
  --output-dir paper_figures/visual_analysis_selected \
  --split val \
  --sample-ids 000123,001234,003456 \
  --threshold 0.25
```

建议优先挑这些样本：

- 远距离车辆明显的样本；
- 遮挡或截断车辆明显的样本；
- 相邻车辆密集、容易出现背景干扰的样本；
- MonoDGP 容易漏检或 BEV 偏差较大的样本。

## 3. 只生成某一类图

`--make` 可以控制生成图的类型：

```bash
--make label
--make response
--make prototype
--make detection
--make depth
--make ablation
```

也可以组合：

```bash
--make label,response,prototype
```

例如只生成消融趋势图：

```bash
python tools/visualize_paper_figures.py \
  --config configs/monoclue_dgp_rerun.yaml \
  --make ablation \
  --output-dir paper_figures/ablation_only
```

如果只想生成论文中的 2D/3D 检测效果图，可以这样运行：

```bash
python tools/visualize_paper_figures.py \
  --config configs/monoclue_dgp_rerun.yaml \
  --checkpoint /root/autodl-tmp/monoclue_dgp/outputs/monoclue_dgp_rerun/checkpoint_best.pth \
  --output-dir paper_figures/detection_effects \
  --split val \
  --make detection \
  --panel-mode split \
  --detection-views 2d,3d,bev \
  --num-samples 6 \
  --auto-filter hard \
  --threshold 0.25 \
  --workers 2
```

`--detection-views` 可以写成 `2d,3d`、`3d,bev` 或单独的 `2d`，用于控制输出哪几类检测面板。

## 4. 生成 DepthPredictor 真实模型输出图

这类图不是手画结构图，而是加载 checkpoint 做真实前向推理后，从模型输出里读取：

- `pred_depth_map_logits`：DepthPredictor 的深度分类 logits；
- softmax 后的深度 bin 分布；
- `weighted_depth`：soft-argmin 得到的加权深度图；
- `pred_region_prob`：前景/分割响应，用来选取代表性像素点。

运行命令：

```bash
python tools/visualize_paper_figures.py \
  --config configs/monoclue_dgp_rerun.yaml \
  --checkpoint /root/autodl-tmp/monoclue_dgp/outputs/monoclue_dgp_rerun/checkpoint_best.pth \
  --output-dir /root/autodl-tmp/monoclue_dgp/paper_figures/visual_candidates \
  --split val \
  --sample-ids 000152,000201,000415,000427,000985 \
  --make depth \
  --panel-mode split \
  --threshold 0.25 \
  --workers 2 \
  --device cuda
```

会生成：

```text
/root/autodl-tmp/monoclue_dgp/paper_figures/visual_candidates/06_depth_predictor_outputs/
```

每个样本的 `split/` 子目录下包含：

- `*_01_input_and_sampled_point.png`：输入图与自动选取的代表性像素点；
- `*_02_predicted_foreground_response.png`：真实前景响应；
- `*_03_depth_logit_confidence.png`：深度 logits softmax 后的最高置信度图；
- `*_04_depth_bin_distribution.png`：该像素点的真实深度 bin 分布；
- `*_05_weighted_depth_soft_argmin.png`：soft-argmin 加权深度图。

## 5. 加入 MonoDGP 基线结果对比

如果你已经有 MonoDGP 的 KITTI 格式预测结果，例如：

```text
/root/autodl-tmp/MonoDGP/outputs/monodgp/outputs/data/
```

可以在检测与 BEV 图里同时画出基线预测：

```bash
python tools/visualize_paper_figures.py \
  --config configs/monoclue_dgp_rerun.yaml \
  --checkpoint /root/autodl-tmp/monoclue_dgp/outputs/monoclue_dgp_rerun/checkpoint_best.pth \
  --output-dir paper_figures/visual_analysis_with_baseline \
  --split val \
  --num-samples 6 \
  --auto-filter hard \
  --baseline-result-dir /root/autodl-tmp/MonoDGP/outputs/monodgp/outputs/data \
  --threshold 0.25
```

颜色约定：

- 绿色：GT；
- 红色：本文方法；
- 蓝色：基线方法。

如果没有传入 `--baseline-result-dir`，脚本仍会正常生成 GT 与本文方法的对比图。

## 6. 自动筛选策略

当不指定 `--sample-ids` 时，脚本会从验证集按顺序筛选样本。

可用筛选条件：

```bash
--auto-filter all       # 不筛选
--auto-filter hard      # 优先遮挡、截断、远距离或 Hard 样本
--auto-filter far       # 含 35m 以上车辆
--auto-filter occluded  # 含遮挡车辆
```

论文中建议使用：

```bash
--auto-filter hard
```

这样更容易得到能说明 Hard 难度和 BEV 定位优势的图。

## 7. 推荐放入论文的图

建议最终保留 4 到 5 张实验可视化：

1. `01_label_supervision` 中选择 1 张：说明二维框监督包含背景，SAM 可见前景和前景深度标签更精细。
2. `02_response_maps` 中选择 1 张：说明本文方法的前景概率和深度响应更集中于目标可见区域。
3. `03_prototype_relocalization` 中选择 1 张：说明前景原型、有效背景参照和相似性重定位确实在工作。
4. `04_detection_bev` 中选择 1 到 2 张：展示遮挡、远距离或密集车辆场景下 3D 框和 BEV 定位更准确。
5. `05_ablation_summary` 中选择 1 张：把表 2、表 4、表 5 的趋势图形化，突出模块协同和 10 个原型最优。

## 8. 常见问题

如果提示找不到 SAM 标签，确认数据集下存在：

```text
/root/autodl-tmp/data/kitti/training/label_sam/region/
/root/autodl-tmp/data/kitti/training/label_sam/depth/
```

如果提示 CUDA 扩展或 `MSDeformAttn` 相关错误，先按照项目原 README 编译 deformable attention：

```bash
cd lib/models/monoclue/ops
sh make.sh
cd /root/autodl-tmp/monoclue_dgp
```

如果显存紧张，把样本数量和 workers 调小：

```bash
--num-samples 2 --workers 0
```
