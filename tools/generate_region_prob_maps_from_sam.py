import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np


NUM_LEVELS = 4


INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "area": cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate P1-P4 multi-scale soft foreground probability maps from a "
            "black-background, white-foreground SAM mask."
        )
    )
    parser.add_argument("--mask", required=True, help="Path to the SAM mask image.")
    parser.add_argument(
        "--output-dir",
        default="paper_figures/region_prob_000985",
        help="Directory for generated probability-map PNG files.",
    )
    parser.add_argument("--prefix", default="P", help="Output filename prefix.")
    parser.add_argument(
        "--colormap",
        default="viridis",
        help="Matplotlib colormap name, e.g. viridis, plasma, magma.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Foreground threshold after mask normalization.",
    )
    parser.add_argument(
        "--downsample-ratios",
        default="1.0,0.5,0.25,0.125",
        help="Comma-separated spatial ratios for P1-P4 before upsampling.",
    )
    parser.add_argument(
        "--blur-kernels",
        default="5,11,21,35",
        help="Comma-separated Gaussian blur kernels for P1-P4. 0 disables blur.",
    )
    parser.add_argument(
        "--blur-sigmas",
        default="0,0,0,0",
        help="Comma-separated Gaussian sigma values for P1-P4. 0 lets OpenCV infer it.",
    )
    parser.add_argument(
        "--gammas",
        default="1.0,0.85,0.70,0.58",
        help="Comma-separated contrast gammas for P1-P4. Smaller values lift soft responses.",
    )
    parser.add_argument(
        "--edge-widths",
        default="1.5,2.5,4.0,6.0",
        help="Comma-separated foreground edge-decay widths for P1-P4 in pixels.",
    )
    parser.add_argument(
        "--edge-min",
        type=float,
        default=0.72,
        help="Lowest multiplier applied to foreground boundaries for soft sigmoid-like edges.",
    )
    parser.add_argument(
        "--background-floor",
        type=float,
        default=0.015,
        help="Small non-zero background response so viridis stays dark purple instead of pure black.",
    )
    parser.add_argument(
        "--foreground-ceil",
        type=float,
        default=0.98,
        help="Maximum response value after soft processing.",
    )
    parser.add_argument(
        "--close-kernel",
        type=int,
        default=5,
        help="Morphological closing kernel for cleaning fragmented SAM regions. 0 disables it.",
    )
    parser.add_argument(
        "--dilate-kernel",
        type=int,
        default=3,
        help="Light dilation kernel for keeping foreground regions coherent. 0 disables it.",
    )
    parser.add_argument(
        "--dilate-iterations",
        type=int,
        default=1,
        help="Number of light dilation iterations after closing.",
    )
    parser.add_argument(
        "--upsample-modes",
        default="linear,linear,nearest,nearest",
        help="Comma-separated upsampling modes for P1-P4: nearest, linear, area, cubic.",
    )
    parser.add_argument(
        "--component-boosts",
        default="0.0,0.03,0.07,0.12",
        help=(
            "Comma-separated component-center boosts for P1-P4. This helps low levels "
            "keep distant small vehicles as bright response spots."
        ),
    )
    parser.add_argument(
        "--component-radius-scales",
        default="0.10,0.14,0.20,0.28",
        help="Comma-separated ellipse radius scales for component-center boosts.",
    )
    parser.add_argument(
        "--component-min-area",
        type=int,
        default=4,
        help="Ignore connected foreground components smaller than this many pixels.",
    )
    return parser.parse_args()


def parse_csv(value, cast, name, expected=NUM_LEVELS):
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value.")
    parsed = [cast(item) for item in values]
    if len(parsed) == 1:
        parsed = parsed * expected
    if len(parsed) != expected:
        raise ValueError(f"{name} must contain {expected} values, got {len(parsed)}.")
    return parsed


def odd_kernel(value):
    value = int(value)
    if value <= 0:
        return 0
    return value if value % 2 == 1 else value + 1


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_gray_image(path):
    path = Path(path)
    if not path.is_file():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def load_mask(path, threshold):
    mask = read_gray_image(path)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask image: {path}")

    mask = mask.astype(np.float32)
    min_value = float(mask.min())
    max_value = float(mask.max())
    if max_value <= min_value:
        raise ValueError(f"Mask image has no usable foreground variation: {path}")

    mask = (mask - min_value) / (max_value - min_value)
    binary = (mask >= threshold).astype(np.uint8)
    if int(binary.sum()) == 0:
        raise ValueError(
            "No foreground pixels found. Check --threshold or whether the SAM mask uses white foreground."
        )
    return mask, binary


def make_ellipse_kernel(size):
    size = odd_kernel(size)
    if size <= 1:
        return None
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def clean_binary_mask(binary, close_kernel, dilate_kernel, dilate_iterations):
    cleaned = (binary > 0).astype(np.uint8)

    close = make_ellipse_kernel(close_kernel)
    if close is not None:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close)

    dilate = make_ellipse_kernel(dilate_kernel)
    if dilate is not None and dilate_iterations > 0:
        cleaned = cv2.dilate(cleaned, dilate, iterations=dilate_iterations)

    return cleaned.astype(np.float32)


def resize_to_level(mask, ratio, output_shape, upsample_mode):
    h, w = output_shape
    if ratio >= 0.999:
        return mask.copy()

    small_w = max(1, int(round(w * ratio)))
    small_h = max(1, int(round(h * ratio)))
    low = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(low, (w, h), interpolation=INTERPOLATION[upsample_mode])


def apply_gaussian(prob, kernel_size, sigma):
    kernel_size = odd_kernel(kernel_size)
    if kernel_size <= 1:
        return prob
    return cv2.GaussianBlur(prob, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def foreground_edge_decay(binary, edge_width, edge_min):
    if edge_width <= 0:
        return np.ones_like(binary, dtype=np.float32)

    fg = (binary > 0).astype(np.uint8)
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 5).astype(np.float32)
    ramp = 1.0 - np.exp(-dist / max(edge_width, 1e-6))
    decay = edge_min + (1.0 - edge_min) * ramp
    return np.where(fg > 0, decay, 1.0).astype(np.float32)


def component_center_prior(binary, strength, radius_scale, blur_kernel, min_area):
    if strength <= 0:
        return np.zeros_like(binary, dtype=np.float32)

    fg = (binary > 0).astype(np.uint8)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    prior = np.zeros_like(binary, dtype=np.float32)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[label]
        axis_x = max(2, int(round(w * radius_scale)))
        axis_y = max(2, int(round(h * radius_scale)))
        center = (int(round(cx)), int(round(cy)))
        cv2.ellipse(prior, center, (axis_x, axis_y), 0, 0, 360, 1.0, -1)

    prior = apply_gaussian(prior, max(3, blur_kernel), 0)
    max_value = float(prior.max())
    if max_value > 0:
        prior = prior / max_value
    return np.clip(prior * strength, 0.0, 1.0)


def make_probability_map(
    clean_mask,
    clean_binary,
    ratio,
    blur_kernel,
    blur_sigma,
    gamma,
    edge_width,
    edge_min,
    background_floor,
    foreground_ceil,
    upsample_mode,
    component_boost,
    component_radius_scale,
    component_min_area,
):
    h, w = clean_mask.shape[:2]
    prob = resize_to_level(clean_mask, ratio, (h, w), upsample_mode)
    prob = apply_gaussian(prob, blur_kernel, blur_sigma)
    prob = np.clip(prob, 0.0, 1.0)

    decay = foreground_edge_decay(clean_binary, edge_width, edge_min)
    prob = prob * decay

    if gamma > 0:
        prob = np.power(np.clip(prob, 0.0, 1.0), gamma)

    prior = component_center_prior(
        clean_binary,
        component_boost,
        component_radius_scale,
        blur_kernel,
        component_min_area,
    )
    prob = np.maximum(prob, prior)

    prob = background_floor + (foreground_ceil - background_floor) * np.clip(prob, 0.0, 1.0)
    return np.clip(prob, 0.0, 1.0).astype(np.float32)


def colorize_probability(prob, colormap_name):
    try:
        colormap = matplotlib.colormaps.get_cmap(colormap_name)
    except AttributeError:
        try:
            colormap = cm.get_cmap(colormap_name)
        except ValueError as exc:
            raise ValueError(f"Unknown matplotlib colormap: {colormap_name}") from exc
    except ValueError as exc:
        raise ValueError(f"Unknown matplotlib colormap: {colormap_name}") from exc

    rgba = colormap(np.clip(prob, 0.0, 1.0))
    rgb = (rgba[..., :3] * 255.0).round().astype(np.uint8)
    return rgb


def save_rgb(path, rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(Path(path).suffix or ".png", bgr)
    if not ok:
        raise IOError(f"Failed to write image: {path}")
    encoded.tofile(str(path))


def main():
    args = parse_args()

    downsample_ratios = parse_csv(args.downsample_ratios, float, "--downsample-ratios")
    blur_kernels = parse_csv(args.blur_kernels, int, "--blur-kernels")
    blur_sigmas = parse_csv(args.blur_sigmas, float, "--blur-sigmas")
    gammas = parse_csv(args.gammas, float, "--gammas")
    edge_widths = parse_csv(args.edge_widths, float, "--edge-widths")
    upsample_modes = parse_csv(args.upsample_modes, str, "--upsample-modes")
    component_boosts = parse_csv(args.component_boosts, float, "--component-boosts")
    component_radius_scales = parse_csv(
        args.component_radius_scales, float, "--component-radius-scales"
    )

    for mode in upsample_modes:
        if mode not in INTERPOLATION:
            valid = ", ".join(sorted(INTERPOLATION))
            raise ValueError(f"Unsupported upsample mode: {mode}. Use one of: {valid}.")

    _, binary = load_mask(args.mask, args.threshold)
    clean = clean_binary_mask(
        binary,
        args.close_kernel,
        args.dilate_kernel,
        args.dilate_iterations,
    )
    clean_binary = (clean > 0).astype(np.uint8)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    outputs = []
    for level in range(NUM_LEVELS):
        prob = make_probability_map(
            clean,
            clean_binary,
            downsample_ratios[level],
            blur_kernels[level],
            blur_sigmas[level],
            gammas[level],
            edge_widths[level],
            args.edge_min,
            args.background_floor,
            args.foreground_ceil,
            upsample_modes[level],
            component_boosts[level],
            component_radius_scales[level],
            args.component_min_area,
        )
        rgb = colorize_probability(prob, args.colormap)
        out_path = output_dir / f"{args.prefix}{level + 1}.png"
        save_rgb(out_path, rgb)
        outputs.append(out_path)

    print("Generated region probability maps:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
