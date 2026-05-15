import argparse
import os
import re
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from lib.datasets.kitti.kitti_utils import get_objects_from_label


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PALETTE = np.array([
    [0, 0, 0],
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 190],
    [0, 128, 128],
    [230, 190, 255],
    [170, 110, 40],
    [255, 250, 200],
], dtype=np.uint8)


class PrintLogger:
    def info(self, message):
        print(message)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate paper visualizations for MonoCLUE-DGP.")
    parser.add_argument("--config", default="configs/monoclue_dgp_rerun.yaml")
    parser.add_argument(
        "--checkpoint",
        default="/root/autodl-tmp/monoclue_dgp/outputs/monoclue_dgp_rerun/checkpoint_best.pth")
    parser.add_argument("--model-type", default="monoclue", choices=["monoclue", "monodgp"],
                        help="Architecture used by --checkpoint.")
    parser.add_argument("--method-name", default="MonoCLUE-DGP",
                        help="Display/output name for --checkpoint.")
    parser.add_argument("--baseline-checkpoint", default=None,
                        help="Optional baseline checkpoint. When set, figures are generated for both models.")
    parser.add_argument("--baseline-config", default=None,
                        help="Optional config for --baseline-checkpoint. Defaults to --config.")
    parser.add_argument("--baseline-model-type", default="monodgp", choices=["monoclue", "monodgp"],
                        help="Architecture used by --baseline-checkpoint.")
    parser.add_argument("--baseline-name", default="MonoDGP",
                        help="Display/output name for --baseline-checkpoint.")
    parser.add_argument("--output-dir", default="paper_figures/visual_analysis")
    parser.add_argument("--split", default=None, help="Override dataset.test_split, e.g. val.")
    parser.add_argument("--sample-ids", default=None,
                        help="Comma-separated ids, e.g. 000123,000456. If omitted, samples are selected from the split.")
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--auto-filter", default="all",
                        choices=["all", "hard", "far", "occluded"],
                        help="Automatic sample filter when --sample-ids is omitted.")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-result-dir", default=None,
                        help="Optional KITTI-format baseline results folder, e.g. MonoDGP outputs/data.")
    parser.add_argument("--make", default="all",
                        help="Comma-separated: all,label,response,prototype,detection,ablation.")
    parser.add_argument("--panel-mode", default="combined", choices=["combined", "split", "both"],
                        help="combined: old multi-panel figures; split: each panel as an image; both: save both.")
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_id_set(value):
    if not value:
        return None
    ids = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        ids.append(int(item))
    return set(ids)


def image_id_name(img_id):
    return f"{int(img_id):06d}"


def load_cfg(path, args):
    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    if args.split:
        cfg["dataset"]["test_split"] = args.split
    cfg["dataset"]["batch_size"] = 1
    cfg["dataset"]["load_sam"] = True
    cfg["dataset"]["random_mixup3d"] = 0.0
    cfg["tester"]["threshold"] = args.threshold
    cfg["tester"]["topk"] = args.topk
    if args.device:
        cfg["model"]["device"] = args.device
    return cfg


def build_model_by_type(cfg, model_type):
    if model_type == "monodgp":
        from lib.models.monodgp import build_monodgp
        return build_monodgp(cfg)
    if model_type == "monoclue":
        from lib.models.monoclue import build_monoclue
        return build_monoclue(cfg)
    raise ValueError(f"Unsupported model type: {model_type}")


def checkpoint_model_keys(checkpoint):
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    state = checkpoint_data.get("model_state", checkpoint_data)
    return set(state.keys())


def adapt_monodgp_cfg_to_checkpoint(cfg, checkpoint):
    model_cfg = cfg["model"] if "model" in cfg else cfg
    keys = checkpoint_model_keys(checkpoint)
    has_fem = any(key.startswith("backbone.0.fem_") for key in keys)
    has_sam_gate = any(key.startswith("depth_predictor.sam_gate.") for key in keys)
    has_se_attention = any(key.startswith("region_head.attention.0.fc1.") for key in keys)
    has_eca_attention = any(key.startswith("region_head.attention.0.conv.") for key in keys)

    model_cfg["use_fem"] = has_fem
    model_cfg["use_sam_gate"] = has_sam_gate
    if has_se_attention:
        model_cfg["region_attention"] = "se"
    elif has_eca_attention:
        model_cfg["region_attention"] = "eca"
    else:
        model_cfg["region_attention"] = "none"

    print(
        "[INFO] MonoDGP checkpoint-compatible options: "
        f"use_fem={model_cfg['use_fem']}, "
        f"use_sam_gate={model_cfg['use_sam_gate']}, "
        f"region_attention={model_cfg['region_attention']}")


def build_and_load_model(cfg, checkpoint, device, model_type="monoclue"):
    from lib.helpers.save_helper import load_checkpoint

    if model_type == "monodgp":
        adapt_monodgp_cfg_to_checkpoint(cfg, checkpoint)

    model, _ = build_model_by_type(cfg["model"], model_type)
    model = model.to(device)
    load_checkpoint(
        model=model,
        optimizer=None,
        filename=checkpoint,
        map_location=device,
        logger=PrintLogger())
    model.eval()
    return model


def tensor_image_to_rgb(tensor):
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * STD + MEAN, 0.0, 1.0)
    return (img * 255).astype(np.uint8)


def resize_float_map(arr, out_hw, interpolation=cv2.INTER_LINEAR):
    arr = np.asarray(arr, dtype=np.float32)
    return cv2.resize(arr, (out_hw[1], out_hw[0]), interpolation=interpolation)


def normalize_map(arr, valid=None):
    arr = np.asarray(arr, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(arr)
    if valid.sum() == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo = np.percentile(arr[valid], 2)
    hi = np.percentile(arr[valid], 98)
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def colorize_heat(arr, valid=None, cmap_name="magma"):
    norm = normalize_map(arr, valid)
    colored = plt.get_cmap(cmap_name)(norm)[..., :3]
    colored = (colored * 255).astype(np.uint8)
    if valid is not None:
        colored[~valid] = 0
    return colored


def overlay_mask(rgb, mask, color=(255, 80, 30), alpha=0.42):
    out = rgb.copy()
    mask = mask.astype(bool)
    color_arr = np.array(color, dtype=np.float32)
    out[mask] = np.clip((1 - alpha) * out[mask] + alpha * color_arr, 0, 255)
    return out.astype(np.uint8)


def slugify(value):
    value = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower())
    return value.strip("_") or "panel"


def safe_dir_name(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "model"


def display_model_name(value):
    normalized = re.sub(r"[^a-z0-9]+", "", value.strip().lower())
    if normalized == "monodgp":
        return "baseline"
    if normalized in {"monocluedgp", "monoclue"}:
        return "ours"
    return value


def save_rgb_image(image, out_path):
    ensure_dir(Path(out_path).parent)
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        cv2.imwrite(str(out_path), image)
    else:
        cv2.imwrite(str(out_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_panel(images, titles, out_path, ncols=None, dpi=180):
    if ncols is None:
        ncols = len(images)
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 2.4 * nrows), dpi=dpi)
    axes = np.array(axes).reshape(-1)
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def split_dir_for(out_path):
    out_path = Path(out_path)
    return out_path.parent / "split" / out_path.stem


def save_figure_outputs(images, titles, out_path, panel_mode="combined", ncols=None, dpi=180):
    out_path = Path(out_path)
    if panel_mode in {"combined", "both"}:
        ensure_dir(out_path.parent)
        save_panel(images, titles, out_path, ncols=ncols, dpi=dpi)
    if panel_mode in {"split", "both"}:
        split_dir = split_dir_for(out_path)
        ensure_dir(split_dir)
        for idx, (image, title) in enumerate(zip(images, titles), start=1):
            save_rgb_image(image, split_dir / f"{out_path.stem}_{idx:02d}_{slugify(title)}.png")


def save_side_by_side(images, titles, model_names, out_path, panel_mode="combined", dpi=180):
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    save_panel(images, model_names, out_path, ncols=len(images), dpi=dpi)
    if panel_mode == "both":
        split_dir = split_dir_for(out_path)
        ensure_dir(split_dir)
        for image, model_name in zip(images, model_names):
            save_rgb_image(image, split_dir / f"{out_path.stem}_{safe_dir_name(model_name)}.png")


def figure_to_rgb(fig):
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[..., :3].copy()


def masks_to_label_map(masks):
    masks = np.asarray(masks)
    if masks.ndim == 4:
        masks = masks[0]
    if masks.size == 0 or masks.shape[0] == 0:
        return np.zeros(masks.shape[-2:] if masks.ndim >= 2 else (1, 1), dtype=np.int32)
    labels = np.zeros(masks.shape[-2:], dtype=np.int32)
    for idx, mask in enumerate(masks):
        labels[mask > 0.5] = idx + 1
    return labels


def colorize_labels(labels):
    labels = labels.astype(np.int32)
    colors = PALETTE[labels % len(PALETTE)]
    colors[labels == 0] = 0
    return colors


def build_box_depth_map(target, hw):
    boxes = target["boxes_2d"][0].detach().cpu().numpy()
    depths = target["depth"][0].detach().cpu().numpy().reshape(-1)
    depth_map = np.zeros(hw, dtype=np.float32)
    for box, depth in zip(boxes, depths):
        if depth <= 0:
            continue
        x1, y1, x2, y2 = np.round(box).astype(int)
        x1, x2 = np.clip([x1, x2], 0, hw[1] - 1)
        y1, y2 = np.clip([y1, y2], 0, hw[0] - 1)
        if x2 <= x1 or y2 <= y1:
            continue
        patch = depth_map[y1:y2, x1:x2]
        depth_map[y1:y2, x1:x2] = np.where((patch > 0) & (patch < depth), patch, depth)
    return depth_map


def build_label_supervision_images(inputs, target):
    rgb = tensor_image_to_rgb(inputs[0])
    hw = rgb.shape[:2]
    box_region = target["obj_region"][0].detach().cpu().numpy().astype(bool)
    sam_region = target["sam_region"][0].detach().cpu().numpy() > 0.5
    sam_depth = target["depth_map"][0].detach().cpu().numpy().astype(np.float32)
    box_depth = build_box_depth_map(target, hw)

    images = [
        rgb,
        overlay_mask(rgb, box_region, color=(240, 120, 35)),
        overlay_mask(rgb, sam_region, color=(30, 180, 90)),
        colorize_heat(box_depth, valid=box_depth > 0),
        colorize_heat(sam_depth, valid=sam_depth > 0),
    ]
    titles = ["Input", "2D-box region", "SAM-visible region", "2D-box depth label", "SAM depth label"]
    return images, titles


def make_label_supervision_figure(inputs, target, out_path, panel_mode="combined"):
    images, titles = build_label_supervision_images(inputs, target)
    save_figure_outputs(images, titles, out_path, panel_mode=panel_mode)


def build_response_images(inputs, outputs):
    rgb = tensor_image_to_rgb(inputs[0])
    hw = rgb.shape[:2]
    region_prob = outputs["pred_region_prob"][0][0, 0].detach().cpu().numpy()
    region_prob = resize_float_map(region_prob, hw)

    if "visual_debug" in outputs and outputs["visual_debug"].get("weighted_depth") is not None:
        weighted_depth = outputs["visual_debug"]["weighted_depth"][0].detach().cpu().numpy()
    else:
        logits = outputs["pred_depth_map_logits"]
        depth_probs = torch.softmax(logits, dim=1)
        depth_values = torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype)
        weighted_depth = (depth_probs * depth_values.view(1, -1, 1, 1)).sum(1)[0].detach().cpu().numpy()
    weighted_depth = resize_float_map(weighted_depth, hw)
    fg_depth = weighted_depth * normalize_map(region_prob)

    images = [
        rgb,
        colorize_heat(region_prob, cmap_name="viridis"),
        colorize_heat(weighted_depth, valid=weighted_depth > 0, cmap_name="magma"),
        colorize_heat(fg_depth, valid=region_prob > 0.35, cmap_name="magma"),
        overlay_mask(rgb, region_prob > 0.5, color=(30, 180, 90)),
    ]
    titles = ["Input", "Predicted foreground", "Predicted depth response", "Foreground-weighted depth", "Foreground overlay"]
    return images, titles


def make_response_figure(inputs, outputs, out_path, panel_mode="combined"):
    images, titles = build_response_images(inputs, outputs)
    save_figure_outputs(images, titles, out_path, panel_mode=panel_mode)


def corr_to_map(corr, spatial_shape):
    corr = np.asarray(corr)
    if corr.ndim == 3:
        corr = corr[0]
    corr = corr.squeeze(-1)
    h, w = int(spatial_shape[0]), int(spatial_shape[1])
    if corr.size != h * w:
        return np.zeros((h, w), dtype=np.float32)
    return corr.reshape(h, w).astype(np.float32)


def build_prototype_images(inputs, outputs):
    debug = outputs.get("visual_debug", {}).get("query_initializer", None)
    if not debug or not debug.get("fg_cluster_masks"):
        return None, None

    rgb = tensor_image_to_rgb(inputs[0])
    hw = rgb.shape[:2]
    spatial_shapes = np.asarray(debug["spatial_shapes"])

    region_prob = outputs["pred_region_prob"][0][0, 0].detach().cpu().numpy()
    region_prob = resize_float_map(region_prob, hw)

    fg_labels = masks_to_label_map(np.asarray(debug["fg_cluster_masks"][0]))
    fg_labels = cv2.resize(fg_labels.astype(np.int32), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    fg_overlay = overlay_mask(rgb, fg_labels > 0, color=(255, 255, 255), alpha=0.10)
    fg_colors = colorize_labels(fg_labels)
    fg_overlay[fg_labels > 0] = np.clip(0.52 * fg_overlay[fg_labels > 0] + 0.48 * fg_colors[fg_labels > 0], 0, 255)

    if debug.get("bg_cluster_masks"):
        bg_labels = masks_to_label_map(np.asarray(debug["bg_cluster_masks"][0]))
        bg_labels = cv2.resize(bg_labels.astype(np.int32), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
        bg_overlay = overlay_mask(rgb, bg_labels > 0, color=(40, 170, 220), alpha=0.45)
    else:
        bg_overlay = rgb.copy()

    corr_map = corr_to_map(np.asarray(debug["corr_maps"][0]), spatial_shapes[0])
    corr_map = resize_float_map(corr_map, hw)

    images = [
        rgb,
        colorize_heat(region_prob, cmap_name="viridis"),
        fg_overlay.astype(np.uint8),
        bg_overlay,
        colorize_heat(corr_map, cmap_name="inferno"),
    ]
    titles = ["Input", "High-confidence foreground", "Foreground prototypes", "Effective background prototypes", "Prototype similarity"]
    return images, titles


def make_prototype_figure(inputs, outputs, out_path, panel_mode="combined"):
    images, titles = build_prototype_images(inputs, outputs)
    if images is None:
        return False
    save_figure_outputs(images, titles, out_path, panel_mode=panel_mode)
    return True


def compute_3d_corners(h, w, l, x, y, z, yaw):
    rot = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ])
    x_c = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_c = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_c = np.array([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])
    corners = rot @ np.vstack([x_c, y_c, z_c])
    corners += np.array([[x], [y], [z]])
    return corners.T


def project_box_corners(calib, corners):
    if hasattr(calib, "rect_to_img"):
        pts, depth = calib.rect_to_img(corners)
    else:
        pts = calib.project_rect_to_image(corners)
        depth = corners[:, 2]
    return np.asarray(pts, dtype=np.float32), np.asarray(depth, dtype=np.float32)


def draw_projected_box(image, pts, color, thickness=2, depth=None,
                       min_depth=0.1, pad_ratio=0.10, max_edge_ratio=0.75):
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (8, 2) or not np.isfinite(pts).all():
        return image
    if depth is not None:
        depth = np.asarray(depth, dtype=np.float32)
        if depth.shape[0] != 8 or np.any(depth <= min_depth):
            return image

    h, w = image.shape[:2]
    pad = max(h, w) * pad_ratio
    in_padded_image = (
        (pts[:, 0] >= -pad) & (pts[:, 0] <= w + pad) &
        (pts[:, 1] >= -pad) & (pts[:, 1] <= h + pad)
    )
    max_edge_len = np.hypot(w, h) * max_edge_ratio
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    rect = (0, 0, w, h)
    for i, j in edges:
        if not (in_padded_image[i] and in_padded_image[j]):
            continue
        if np.linalg.norm(pts[i] - pts[j]) > max_edge_len:
            continue
        p1 = tuple(np.round(pts[i]).astype(np.int32))
        p2 = tuple(np.round(pts[j]).astype(np.int32))
        ok, p1, p2 = cv2.clipLine(rect, p1, p2)
        if ok:
            cv2.line(image, p1, p2, color, thickness, lineType=cv2.LINE_AA)
    return image


def draw_objects_on_image(rgb, calib, gt_objects, pred_objects, baseline_objects=None,
                          pred_color=(255, 40, 40), baseline_color=(30, 130, 255)):
    out = rgb.copy()
    for obj in gt_objects:
        corners = obj.generate_corners3d()
        pts, depth = project_box_corners(calib, corners)
        out = draw_projected_box(out, pts, (60, 220, 60), 2, depth=depth)
    if baseline_objects:
        for pred in baseline_objects:
            corners = compute_3d_corners(*pred["dims"], *pred["loc"], pred["ry"])
            pts, depth = project_box_corners(calib, corners)
            out = draw_projected_box(out, pts, baseline_color, 2, depth=depth)
    for pred in pred_objects:
        corners = compute_3d_corners(*pred["dims"], *pred["loc"], pred["ry"])
        pts, depth = project_box_corners(calib, corners)
        out = draw_projected_box(out, pts, pred_color, 2, depth=depth)
    return out


def pred_row_to_dict(row):
    return {
        "cls_id": int(row[0]),
        "alpha": float(row[1]),
        "bbox": [float(x) for x in row[2:6]],
        "dims": [float(x) for x in row[6:9]],
        "loc": [float(x) for x in row[9:12]],
        "ry": float(row[12]),
        "score": float(row[13]),
    }


def read_baseline_result(result_dir, img_id, class_name="Car", threshold=0.0):
    if not result_dir:
        return []
    path = os.path.join(result_dir, image_id_name(img_id) + ".txt")
    if not os.path.exists(path):
        return []
    preds = []
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 16 or parts[0] != class_name:
                continue
            score = float(parts[15])
            if score < threshold:
                continue
            preds.append({
                "dims": [float(parts[8]), float(parts[9]), float(parts[10])],
                "loc": [float(parts[11]), float(parts[12]), float(parts[13])],
                "ry": float(parts[14]),
                "score": score,
            })
    return preds


def draw_bev(ax, gt_objects, pred_objects, baseline_objects=None,
             pred_label="ours", baseline_label="baseline",
             pred_color="#e53e3e", baseline_color="#3182ce"):
    def draw_corners(corners, color, label=None, lw=1.8):
        pts = corners[:4, [0, 2]]
        pts = np.vstack([pts, pts[0]])
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=lw, label=label)

    labels = set()
    for obj in gt_objects:
        draw_corners(obj.generate_corners3d(), "#38a169", "GT" if "GT" not in labels else None, 2.0)
        labels.add("GT")
    if baseline_objects:
        for pred in baseline_objects:
            corners = compute_3d_corners(*pred["dims"], *pred["loc"], pred["ry"])
            draw_corners(corners, baseline_color, baseline_label if baseline_label not in labels else None, 1.5)
            labels.add(baseline_label)
    for pred in pred_objects:
        corners = compute_3d_corners(*pred["dims"], *pred["loc"], pred["ry"])
        draw_corners(corners, pred_color, pred_label if pred_label not in labels else None, 1.8)
        labels.add(pred_label)

    ax.set_xlim(-20, 20)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.45)
    ax.set_xlabel("x / m")
    ax.set_ylabel("z / m")
    if labels:
        ax.legend(loc="upper right", fontsize=8)


def decode_pred_objects(dataset, outputs, info, threshold, topk):
    from lib.helpers.decode_helper import decode_detections, extract_dets_from_outputs

    img_id = int(info["img_id"][0].detach().cpu().item())
    dets = extract_dets_from_outputs(outputs=outputs, K=dataset.max_objs, topk=topk).detach().cpu().numpy()
    info_np = {key: val.detach().cpu().numpy() for key, val in info.items()}
    calibs = [dataset.get_calib(img_id)]
    decoded = decode_detections(dets, info_np, calibs, dataset.cls_mean_size, threshold=threshold)
    return [pred_row_to_dict(row) for row in decoded.get(img_id, []) if int(row[0]) == dataset.cls2id.get("Car", 1)]


def get_detection_context(dataset, info):
    img_id = int(info["img_id"][0].detach().cpu().item())
    img_name = image_id_name(img_id)
    img_path = os.path.join(dataset.image_dir, img_name + dataset.image_ext)
    rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    calib = dataset.get_calib(img_id)

    gt_path = os.path.join(dataset.label_dir, img_name + ".txt")
    gt_objects = [
        obj for obj in get_objects_from_label(gt_path)
        if obj.cls_type == "Car" and obj.level_str != "DontCare"
    ]
    return img_id, rgb, calib, gt_objects


def render_bev_image(gt_objects, pred_objects, baseline_objects=None,
                     pred_label="ours", baseline_label="baseline",
                     pred_color="#e53e3e", baseline_color="#3182ce"):
    fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=180)
    draw_bev(ax, gt_objects, pred_objects, baseline_objects,
             pred_label=pred_label,
             baseline_label=baseline_label,
             pred_color=pred_color,
             baseline_color=baseline_color)
    fig.tight_layout()
    image = figure_to_rgb(fig)
    plt.close(fig)
    return image


def build_detection_images(dataset, outputs, info, threshold, topk, baseline_result_dir=None,
                           pred_label="ours", baseline_label="baseline",
                           pred_image_color=(255, 40, 40), baseline_image_color=(30, 130, 255),
                           pred_bev_color="#e53e3e", baseline_bev_color="#3182ce",
                           pred_objects_override=None, baseline_objects_override=None):
    img_id, rgb, calib, gt_objects = get_detection_context(dataset, info)
    pred_objects = pred_objects_override
    if pred_objects is None:
        pred_objects = decode_pred_objects(dataset, outputs, info, threshold, topk)
    baseline_objects = baseline_objects_override
    if baseline_objects is None:
        baseline_objects = read_baseline_result(baseline_result_dir, img_id, threshold=threshold)

    drawn = draw_objects_on_image(
        rgb, calib, gt_objects, pred_objects, baseline_objects,
        pred_color=pred_image_color,
        baseline_color=baseline_image_color)
    bev_image = render_bev_image(
        gt_objects, pred_objects, baseline_objects,
        pred_label=pred_label,
        baseline_label=baseline_label,
        pred_color=pred_bev_color,
        baseline_color=baseline_bev_color)
    return [drawn, bev_image], ["Image-view 3D boxes", "BEV localization"], pred_objects, gt_objects


def make_detection_figure(dataset, inputs, outputs, info, out_path, threshold, topk,
                          baseline_result_dir=None, panel_mode="combined",
                          pred_label="ours",
                          pred_image_color=(255, 40, 40),
                          pred_bev_color="#e53e3e"):
    images, titles, _, _ = build_detection_images(
        dataset, outputs, info, threshold, topk,
        baseline_result_dir=baseline_result_dir,
        pred_label=pred_label,
        pred_image_color=pred_image_color,
        pred_bev_color=pred_bev_color)
    save_figure_outputs(images, titles, out_path, panel_mode=panel_mode, ncols=2)


def make_detection_comparison_figure(dataset, info, baseline_outputs, method_outputs,
                                     out_path, threshold, topk, baseline_name,
                                     method_name, panel_mode="combined"):
    baseline_pred = decode_pred_objects(dataset, baseline_outputs, info, threshold, topk)
    method_pred = decode_pred_objects(dataset, method_outputs, info, threshold, topk)
    images, titles, _, _ = build_detection_images(
        dataset, method_outputs, info, threshold, topk,
        pred_label=method_name,
        baseline_label=baseline_name,
        pred_image_color=(255, 40, 40),
        baseline_image_color=(30, 130, 255),
        pred_bev_color="#e53e3e",
        baseline_bev_color="#3182ce",
        pred_objects_override=method_pred,
        baseline_objects_override=baseline_pred)
    save_figure_outputs(images, titles, out_path, panel_mode=panel_mode, ncols=2)


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(val, device) for key, val in value.items()}
    if isinstance(value, list):
        return [move_to_device(val, device) for val in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(val, device) for val in value)
    return value


def run_model_forward(model, model_type, inputs_gpu, calibs_gpu, target, info, device,
                      return_visual_debug=False):
    img_sizes = info["img_size"].to(device)
    if model_type == "monodgp":
        target_gpu = move_to_device(target, device)
        return model(inputs_gpu, calibs_gpu, target_gpu, img_sizes, dn_args=0)
    return model(
        inputs_gpu, calibs_gpu, img_sizes,
        dn_args=0, return_visual_debug=return_visual_debug)


def save_response_comparison(inputs, baseline_outputs, method_outputs, out_dir, name,
                             baseline_name, method_name, panel_mode):
    baseline_images, titles = build_response_images(inputs, baseline_outputs)
    method_images, _ = build_response_images(inputs, method_outputs)
    for idx, title in enumerate(titles, start=1):
        out_path = Path(out_dir) / f"{name}_{idx:02d}_{slugify(title)}_compare.png"
        save_side_by_side(
            [baseline_images[idx - 1], method_images[idx - 1]],
            [title, title],
            [baseline_name, method_name],
            out_path,
            panel_mode=panel_mode,
            dpi=180)


def sample_matches_filter(dataset, img_id, mode):
    if mode == "all":
        return True
    label_path = os.path.join(dataset.label_dir, image_id_name(img_id) + ".txt")
    if not os.path.exists(label_path):
        return False
    objects = [obj for obj in get_objects_from_label(label_path) if obj.cls_type == "Car"]
    if mode == "far":
        return any(obj.pos[2] >= 35 for obj in objects)
    if mode == "occluded":
        return any(obj.occlusion >= 1 for obj in objects)
    if mode == "hard":
        return any(obj.level_str == "Hard" or obj.pos[2] >= 35 or obj.occlusion >= 1 or obj.trucation > 0.1
                   for obj in objects)
    return True


def plot_ablation_summary(out_path, panel_mode="combined"):
    main = {
        "A": [28.62, 21.09, 18.80],
        "B": [29.86, 21.61, 19.47],
        "C": [30.43, 22.51, 19.39],
        "D": [30.51, 22.03, 19.87],
        "E": [31.33, 22.60, 20.33],
    }
    proto = {
        "None": [30.51, 22.03, 19.87],
        "FG": [30.72, 22.20, 19.76],
        "FG+BG": [30.95, 22.31, 19.92],
        "FG+BG+Mem": [31.08, 22.42, 20.05],
        "Full": [31.33, 22.60, 20.33],
    }
    nums = {
        "5": [30.48, 21.51, 19.52],
        "10": [31.33, 22.60, 20.33],
        "15": [30.36, 21.56, 18.99],
    }
    series = [("Easy", 0, "#386cb0"), ("Mod.", 1, "#2e8b57"), ("Hard", 2, "#a64e4e")]

    plots = [
        (main, "(a) Main module ablation", "Experiment ID"),
        (proto, "(b) Prototype components", "Component setting"),
        (nums, "(c) Prototype number sensitivity", "Foreground prototypes"),
    ]

    def draw_ablation_plot(ax, data, title, xlabel, with_legend=False):
        keys = list(data.keys())
        x = np.arange(len(keys))
        for name, idx, color in series:
            ax.plot(x, [data[k][idx] for k in keys], marker="o", linewidth=2.0, color=color, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=15 if len(keys) > 3 else 0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$AP_{3D}$ (%)")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if with_legend:
            ax.legend(loc="lower right", ncol=3, fontsize=8)

    out_path = Path(out_path)
    if panel_mode in {"combined", "both"}:
        ensure_dir(out_path.parent)
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.0), dpi=180)
        for ax, (data, title, xlabel) in zip(axes, plots):
            draw_ablation_plot(ax, data, title, xlabel)
        axes[0].legend(loc="lower right", ncol=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

    if panel_mode in {"split", "both"}:
        split_dir = split_dir_for(out_path)
        ensure_dir(split_dir)
        for idx, (data, title, xlabel) in enumerate(plots, start=1):
            fig, ax = plt.subplots(1, 1, figsize=(5.3, 4.0), dpi=180)
            draw_ablation_plot(ax, data, title, xlabel, with_legend=True)
            fig.tight_layout()
            fig.savefig(split_dir / f"{out_path.stem}_{idx:02d}_{slugify(title)}.png", bbox_inches="tight")
            plt.close(fig)


def main():
    args = parse_args()
    make_set = set(["label", "response", "prototype", "detection", "ablation"]
                   if args.make == "all" else [x.strip() for x in args.make.split(",") if x.strip()])

    cfg = load_cfg(args.config, args)
    output_dir = Path(args.output_dir)
    dir_names = {
        "label": "01_label_supervision",
        "response": "02_response_maps",
        "prototype": "03_prototype_relocalization",
        "detection": "04_detection_bev",
        "ablation": "05_ablation_summary",
    }
    compare_mode = args.baseline_checkpoint is not None

    if compare_mode:
        shared_subdirs = {
            "label": output_dir / "shared" / dir_names["label"],
            "ablation": output_dir / "shared" / dir_names["ablation"],
        }
        model_subdirs = {}
        for model_name in [args.baseline_name, args.method_name]:
            model_root = output_dir / safe_dir_name(model_name)
            model_subdirs[model_name] = {
                key: model_root / dirname for key, dirname in dir_names.items()
            }
        comparison_subdirs = {
            "response": output_dir / "comparison" / dir_names["response"],
            "detection": output_dir / "comparison" / dir_names["detection"],
        }
        for key in make_set:
            if key in shared_subdirs:
                ensure_dir(shared_subdirs[key])
            if key in comparison_subdirs:
                ensure_dir(comparison_subdirs[key])
            for model_name in model_subdirs:
                if key in {"response", "prototype", "detection"}:
                    ensure_dir(model_subdirs[model_name][key])
    else:
        subdirs = {
            key: output_dir / dirname for key, dirname in dir_names.items()
        }
        for key in make_set:
            ensure_dir(subdirs[key])

    dataloader_needed = bool(make_set & {"label", "response", "prototype", "detection"})
    model_needed = bool(make_set & {"response", "prototype", "detection"})
    if dataloader_needed:
        from lib.helpers.dataloader_helper import build_test_dataloader

        device = torch.device(args.device if model_needed and torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    else:
        build_test_dataloader = None
        device = None

    if not dataloader_needed:
        if "ablation" in make_set:
            if compare_mode:
                plot_ablation_summary(shared_subdirs["ablation"] / "fig_ablation_summary.png", panel_mode=args.panel_mode)
            else:
                plot_ablation_summary(subdirs["ablation"] / "fig_ablation_summary.png", panel_mode=args.panel_mode)
        print(f"Saved figures to {output_dir}")
        return

    dataloader = build_test_dataloader(cfg["dataset"], workers=args.workers, SAM=True)
    dataset = dataloader.dataset

    if "ablation" in make_set:
        if compare_mode:
            plot_ablation_summary(shared_subdirs["ablation"] / "fig_ablation_summary.png", panel_mode=args.panel_mode)
        else:
            plot_ablation_summary(subdirs["ablation"] / "fig_ablation_summary.png", panel_mode=args.panel_mode)

    model_specs = [{
        "name": args.method_name,
        "display_name": display_model_name(args.method_name),
        "type": args.model_type,
        "cfg": cfg,
        "checkpoint": args.checkpoint,
        "pred_image_color": (255, 40, 40),
        "pred_bev_color": "#e53e3e",
    }]
    if compare_mode:
        baseline_cfg = load_cfg(args.baseline_config or args.config, args)
        model_specs.insert(0, {
            "name": args.baseline_name,
            "display_name": display_model_name(args.baseline_name),
            "type": args.baseline_model_type,
            "cfg": baseline_cfg,
            "checkpoint": args.baseline_checkpoint,
            "pred_image_color": (30, 130, 255),
            "pred_bev_color": "#3182ce",
        })

    if model_needed:
        for spec in model_specs:
            spec["model"] = build_and_load_model(spec["cfg"], spec["checkpoint"], device, spec["type"])

    sample_ids = parse_id_set(args.sample_ids)
    saved = 0

    with torch.no_grad():
        for inputs, calibs, target, info in dataloader:
            img_id = int(info["img_id"][0].item())
            if sample_ids is not None:
                if img_id not in sample_ids:
                    continue
            elif not sample_matches_filter(dataset, img_id, args.auto_filter):
                continue

            name = image_id_name(img_id)
            if "label" in make_set:
                label_dir = shared_subdirs["label"] if compare_mode else subdirs["label"]
                make_label_supervision_figure(
                    inputs, target,
                    label_dir / f"{name}_label_supervision.png",
                    panel_mode=args.panel_mode)

            outputs_by_name = {}
            if model_needed:
                inputs_gpu = inputs.to(device)
                calibs_gpu = calibs.to(device)
                for spec in model_specs:
                    return_debug = spec["type"] == "monoclue" and ("prototype" in make_set or "response" in make_set)
                    outputs = run_model_forward(
                        spec["model"], spec["type"],
                        inputs_gpu, calibs_gpu, target, info, device,
                        return_visual_debug=return_debug)
                    outputs_by_name[spec["name"]] = outputs

                    active_subdirs = model_subdirs[spec["name"]] if compare_mode else subdirs
                    safe_name = safe_dir_name(spec["name"])
                    if "response" in make_set:
                        make_response_figure(
                            inputs, outputs,
                            active_subdirs["response"] / f"{name}_{safe_name}_response_maps.png",
                            panel_mode=args.panel_mode)
                    if "prototype" in make_set:
                        ok = make_prototype_figure(
                            inputs, outputs,
                            active_subdirs["prototype"] / f"{name}_{safe_name}_prototype_relocalization.png",
                            panel_mode=args.panel_mode)
                        if not ok:
                            print(f"[WARN] prototype debug data is unavailable for {name} ({spec['name']})")
                    if "detection" in make_set:
                        make_detection_figure(
                            dataset, inputs, outputs, info,
                            active_subdirs["detection"] / f"{name}_{safe_name}_detection_bev.png",
                            threshold=args.threshold,
                            topk=args.topk,
                            baseline_result_dir=None if compare_mode else args.baseline_result_dir,
                            panel_mode=args.panel_mode,
                            pred_label=spec["display_name"],
                            pred_image_color=spec["pred_image_color"],
                            pred_bev_color=spec["pred_bev_color"])

                if compare_mode and "response" in make_set:
                    save_response_comparison(
                        inputs,
                        outputs_by_name[args.baseline_name],
                        outputs_by_name[args.method_name],
                        comparison_subdirs["response"] / name,
                        name,
                        display_model_name(args.baseline_name),
                        display_model_name(args.method_name),
                        panel_mode=args.panel_mode)
                if compare_mode and "detection" in make_set:
                    make_detection_comparison_figure(
                        dataset, info,
                        outputs_by_name[args.baseline_name],
                        outputs_by_name[args.method_name],
                        comparison_subdirs["detection"] / f"{name}_detection_bev_compare.png",
                        threshold=args.threshold,
                        topk=args.topk,
                        baseline_name=display_model_name(args.baseline_name),
                        method_name=display_model_name(args.method_name),
                        panel_mode=args.panel_mode)

            saved += 1
            print(f"[{saved}] saved visualizations for {name}")
            if sample_ids is None and saved >= args.num_samples:
                break

    print(f"Saved {saved} sample(s) to {output_dir}")


if __name__ == "__main__":
    main()
