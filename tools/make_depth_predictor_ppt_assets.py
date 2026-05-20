import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


BLUE = (93, 151, 215, 255)
DARK_BLUE = (34, 88, 156, 255)
LIGHT_BLUE = (226, 240, 255, 255)
GREEN = (112, 186, 94, 255)
DARK_GREEN = (42, 128, 48, 255)
GRAY = (78, 84, 92, 255)
PAPER = (247, 248, 251, 255)
BLACK = (20, 24, 30, 255)
WHITE = (255, 255, 255, 255)
TRANSPARENT = (255, 255, 255, 0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate small transparent PPT assets for the Depth Predictor block.")
    parser.add_argument(
        "--output-dir",
        default="paper_figures/visual_candidates/depth_predictor_ppt_assets",
        help="Directory for generated PNG/SVG assets.")
    return parser.parse_args()


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def alpha_blend(src, color, alpha):
    out = src.copy()
    color_arr = np.array(color[:3], dtype=np.float32)
    out[..., :3] = (1 - alpha) * out[..., :3] + alpha * color_arr
    out[..., 3] = 255
    return out.astype(np.uint8)


def heat_color(value):
    value = float(np.clip(value, 0.0, 1.0))
    stops = [
        (0.00, np.array([35, 57, 161])),
        (0.25, np.array([35, 145, 213])),
        (0.48, np.array([80, 205, 135])),
        (0.68, np.array([252, 218, 72])),
        (0.84, np.array([241, 116, 35])),
        (1.00, np.array([174, 35, 35])),
    ]
    for (x0, c0), (x1, c1) in zip(stops[:-1], stops[1:]):
        if x0 <= value <= x1:
            t = (value - x0) / (x1 - x0)
            rgb = (1 - t) * c0 + t * c1
            return tuple(rgb.astype(np.uint8).tolist()) + (255,)
    return tuple(stops[-1][1].astype(np.uint8).tolist()) + (255,)


def draw_road_depth_heatmap(size=(760, 420)):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yn = yy / max(1, h - 1)
    xn = xx / max(1, w - 1)

    road_width = 0.12 + 0.78 * yn
    center = 0.50 + 0.035 * np.sin(yn * math.pi * 1.2)
    road_mask = np.abs(xn - center) < road_width / 2
    depth_response = 1.0 - yn
    heat = 0.18 + 0.75 * yn
    heat += 0.10 * np.sin(18 * xn + 5 * yn)
    heat = np.clip(heat, 0, 1)

    arr = np.zeros((h, w, 4), dtype=np.uint8)
    sky = np.stack([
        36 + 45 * (1 - yn),
        72 + 80 * (1 - yn),
        150 + 70 * (1 - yn),
    ], axis=-1)
    for y in range(h):
        for x in range(w):
            base = heat_color(heat[y, x] if road_mask[y, x] else depth_response[y, x] * 0.45)
            if road_mask[y, x]:
                arr[y, x] = base
            else:
                arr[y, x, :3] = sky[y, x]
                arr[y, x, 3] = 255

    image = Image.fromarray(arr, "RGBA")
    draw = ImageDraw.Draw(image, "RGBA")

    # Road boundaries and lane markings, with perspective.
    for side in [-1, 1]:
        points = []
        for t in np.linspace(0.05, 0.98, 40):
            road_w = 0.12 + 0.78 * t
            cx = 0.50 + 0.035 * math.sin(t * math.pi * 1.2)
            points.append((int((cx + side * road_w / 2) * w), int(t * h)))
        draw.line(points, fill=(255, 255, 255, 145), width=3)

    for lane in [-0.16, 0.16]:
        for k in range(7):
            t0 = 0.20 + k * 0.115
            t1 = min(t0 + 0.055, 0.96)
            pts = []
            for t in [t0, t1]:
                road_w = 0.12 + 0.78 * t
                cx = 0.50 + 0.035 * math.sin(t * math.pi * 1.2)
                pts.append((int((cx + lane * road_w) * w), int(t * h)))
            draw.line(pts, fill=(255, 255, 255, 160), width=max(2, int(5 * t0)))

    draw_vehicle(draw, 0.30 * w, 0.63 * h, 1.0, color=(36, 42, 52, 235), accent=(235, 72, 60, 230))
    draw_vehicle(draw, 0.56 * w, 0.50 * h, 0.68, color=(35, 43, 55, 225), accent=(240, 240, 238, 210))
    draw_vehicle(draw, 0.70 * w, 0.39 * h, 0.48, color=(38, 45, 55, 215), accent=(82, 145, 210, 210))

    return image


def draw_vehicle(draw, cx, cy, scale=1.0, color=(30, 30, 35, 255), accent=(230, 70, 60, 230)):
    bw, bh = 96 * scale, 42 * scale
    roof_w, roof_h = 54 * scale, 24 * scale
    x0, y0 = cx - bw / 2, cy - bh / 2
    x1, y1 = cx + bw / 2, cy + bh / 2
    draw.rounded_rectangle([x0, y0, x1, y1], radius=8 * scale, fill=color)
    draw.polygon([
        (cx - roof_w / 2, y0 + 7 * scale),
        (cx + roof_w / 2, y0 + 7 * scale),
        (cx + roof_w * 0.38, y0 - roof_h * 0.50),
        (cx - roof_w * 0.38, y0 - roof_h * 0.50),
    ], fill=accent)
    draw.rectangle([x0 + 10 * scale, y1 - 9 * scale, x0 + 24 * scale, y1 - 2 * scale],
                   fill=(255, 220, 80, 220))
    draw.rectangle([x1 - 24 * scale, y1 - 9 * scale, x1 - 10 * scale, y1 - 2 * scale],
                   fill=(255, 220, 80, 220))
    for wx in [x0 + 22 * scale, x1 - 22 * scale]:
        draw.ellipse([wx - 11 * scale, y1 - 7 * scale, wx + 11 * scale, y1 + 15 * scale],
                     fill=(8, 10, 14, 255))


def make_weighted_depth_asset(out_dir):
    image = Image.new("RGBA", (860, 500), TRANSPARENT)
    inner = draw_road_depth_heatmap((760, 420))
    image.alpha_composite(inner, (50, 38))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle([50, 38, 810, 458], radius=18, outline=BLUE, width=4)
    image.save(out_dir / "weighted_depth_vehicle_heatmap.png")


def draw_feature_stack_asset(out_dir):
    image = Image.new("RGBA", (640, 400), TRANSPARENT)
    draw = ImageDraw.Draw(image, "RGBA")
    base_x, base_y = 130, 118
    for i, (dx, dy) in enumerate([(92, 74), (62, 48), (32, 24), (0, 0)]):
        x, y = base_x + dx, base_y + dy
        poly = [
            (x, y + 42), (x + 275, y + 42), (x + 318, y),
            (x + 45, y), (x, y + 42),
        ]
        draw.polygon([(px + 10, py + 10) for px, py in poly], fill=(0, 0, 0, 45))
        draw.polygon(poly, fill=LIGHT_BLUE, outline=DARK_BLUE)
        draw.line([poly[0], poly[1], poly[2], poly[3], poly[0]], fill=DARK_BLUE, width=3)
        car_x = x + 132 + i * 8
        car_y = y + 14
        draw.rounded_rectangle([car_x, car_y, car_x + 44, car_y + 18], radius=4,
                               fill=(37, 43, 55, 210))
        draw.rectangle([car_x + 7, car_y - 7, car_x + 31, car_y + 7],
                       fill=(230, 72, 60, 210))
    for i in range(5):
        x = 92 + i * 42
        draw.rounded_rectangle([x, 292, x + 25, 346], radius=4,
                               fill=(160, 222, 145, 255), outline=DARK_GREEN, width=3)
    image.save(out_dir / "feature_stack_with_seg_tokens.png")


def draw_logits_asset(out_dir):
    image = Image.new("RGBA", (660, 420), TRANSPARENT)
    draw = ImageDraw.Draw(image, "RGBA")
    axis_x, axis_y = 80, 332
    draw.line([axis_x, axis_y, 590, axis_y], fill=GRAY, width=4)
    draw.line([axis_x, 76, axis_x, axis_y], fill=GRAY, width=4)
    values = [0.12, 0.16, 0.22, 0.31, 0.50, 0.78, 0.92, 0.74, 0.56, 0.34, 0.22, 0.14]
    for i, val in enumerate(values):
        x = axis_x + 32 + i * 38
        bar_h = int(val * 230)
        color = (42, 115, 210, 255) if i != 6 else (236, 91, 54, 255)
        draw.rounded_rectangle([x, axis_y - bar_h, x + 24, axis_y], radius=4,
                               fill=color, outline=DARK_BLUE, width=2)
    draw.line([axis_x + 32 + 6 * 38 + 12, 92, axis_x + 32 + 6 * 38 + 12, axis_y],
              fill=(236, 91, 54, 210), width=4)
    image.save(out_dir / "depth_logits_distribution.png")


def draw_softargmin_asset(out_dir):
    image = Image.new("RGBA", (520, 220), TRANSPARENT)
    draw = ImageDraw.Draw(image, "RGBA")
    for i in range(7):
        x = 72 + i * 48
        h = [26, 38, 55, 92, 70, 44, 30][i]
        draw.rounded_rectangle([x, 150 - h, x + 30, 150], radius=4,
                               fill=(75, 135, 210, 235), outline=DARK_BLUE, width=2)
    draw.line([92, 158, 380, 158], fill=GRAY, width=3)
    draw.line([235, 46, 235, 168], fill=(236, 91, 54, 235), width=4)
    draw.polygon([(235, 42), (224, 60), (246, 60)], fill=(236, 91, 54, 235))
    draw.rounded_rectangle([395, 76, 486, 136], radius=14,
                           fill=(236, 245, 255, 245), outline=BLUE, width=3)
    for i, color in enumerate([(45, 91, 181, 255), (68, 160, 214, 255),
                               (250, 218, 72, 255), (220, 73, 45, 255)]):
        draw.rectangle([412 + i * 14, 104, 424 + i * 14, 125], fill=color)
    image.save(out_dir / "softargmin_weighted_depth_icon.png")


def write_svg_assets(out_dir):
    arrow_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="360" height="80" viewBox="0 0 360 80">
<defs><marker id="arrow" markerWidth="12" markerHeight="12" refX="9" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L9,3 z" fill="#2b63c6"/></marker></defs>
<path d="M20 40 C110 40, 170 40, 330 40" fill="none" stroke="#2b63c6" stroke-width="5" stroke-linecap="round" marker-end="url(#arrow)"/>
</svg>
"""
    tokens_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="360" height="120" viewBox="0 0 360 120">
<g fill="#a0de91" stroke="#2a8030" stroke-width="4">
<rect x="36" y="34" width="44" height="58" rx="5"/>
<rect x="102" y="34" width="44" height="58" rx="5"/>
<rect x="168" y="34" width="44" height="58" rx="5"/>
<rect x="234" y="34" width="44" height="58" rx="5"/>
</g>
</svg>
"""
    (out_dir / "blue_arrow.svg").write_text(arrow_svg, encoding="utf-8")
    (out_dir / "seg_embed_tokens.svg").write_text(tokens_svg, encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)
    make_weighted_depth_asset(out_dir)
    draw_feature_stack_asset(out_dir)
    draw_logits_asset(out_dir)
    draw_softargmin_asset(out_dir)
    write_svg_assets(out_dir)
    print(f"Saved DepthPredictor PPT assets to {out_dir}")


if __name__ == "__main__":
    main()
