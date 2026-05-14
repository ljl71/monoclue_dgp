import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MAIN_MODULES = {
    "labels": ["A", "B", "C", "D", "E"],
    "easy": [28.62, 29.43, 30.43, 30.34, 31.99],
    "mod": [21.09, 21.94, 22.51, 23.06, 23.18],
    "hard": [18.80, 19.52, 19.39, 19.62, 19.90],
}

PROTO_COMPONENTS = {
    "labels": ["None", "FG", "FG+BG", "FG+BG+Mem", "Full"],
    "easy": [30.34, 30.95, 31.26, 31.58, 31.99],
    "mod": [23.06, 23.09, 23.12, 23.15, 23.18],
    "hard": [19.62, 19.66, 19.75, 19.84, 19.90],
}

PROTO_NUMBERS = {
    "labels": [5, 10, 15],
    "easy": [31.61, 31.99, 31.78],
    "mod": [23.03, 23.18, 23.11],
    "hard": [19.72, 19.90, 19.81],
}


def plot_metric(ax, x, data, title, xlabel):
    colors = {
        "easy": "#3B6EA8",
        "mod": "#2E8B57",
        "hard": "#9A4F4F",
    }
    labels = {
        "easy": "Easy",
        "mod": "Mod.",
        "hard": "Hard",
    }
    for key in ["easy", "mod", "hard"]:
        ax.plot(
            x,
            data[key],
            marker="o",
            linewidth=1.8,
            markersize=4.5,
            color=colors[key],
            label=labels[key],
        )
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(r"$AP_{3D}$ (%)", fontsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(axis="both", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def grouped_bars(ax, categories, values, title, xlabel, ylabel):
    colors = ["#3B6EA8", "#2E8B57", "#9A4F4F"]
    labels = ["Easy", "Mod.", "Hard"]
    x = np.arange(len(categories))
    width = 0.24
    for i, (label, color) in enumerate(zip(labels, colors)):
        offset = (i - 1) * width
        ax.bar(x + offset, values[i], width=width, label=label, color=color, alpha=0.9)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.tick_params(axis="y", labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def save_gain_summary(output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.3), constrained_layout=True)

    main_gain = np.array(
        [
            np.array(MAIN_MODULES["easy"]) - MAIN_MODULES["easy"][0],
            np.array(MAIN_MODULES["mod"]) - MAIN_MODULES["mod"][0],
            np.array(MAIN_MODULES["hard"]) - MAIN_MODULES["hard"][0],
        ]
    )
    grouped_bars(
        axes[0],
        MAIN_MODULES["labels"],
        main_gain,
        "(a) Gain of main modules",
        "Experiment ID",
        r"$\Delta AP_{3D}$ (%)",
    )
    axes[0].set_ylim(-0.2, 3.8)

    proto_gain = np.array(
        [
            np.array(PROTO_COMPONENTS["easy"]) - PROTO_COMPONENTS["easy"][0],
            np.array(PROTO_COMPONENTS["mod"]) - PROTO_COMPONENTS["mod"][0],
            np.array(PROTO_COMPONENTS["hard"]) - PROTO_COMPONENTS["hard"][0],
        ]
    )
    grouped_bars(
        axes[1],
        PROTO_COMPONENTS["labels"],
        proto_gain,
        "(b) Gain of prototype components",
        "Component setting",
        r"$\Delta AP_{3D}$ (%)",
    )
    axes[1].set_xticklabels(PROTO_COMPONENTS["labels"], rotation=18, ha="right", fontsize=8)
    axes[1].set_ylim(-0.1, 1.9)

    best_idx = PROTO_NUMBERS["labels"].index(10)
    proto_num_delta = np.array(
        [
            np.array(PROTO_NUMBERS["easy"]) - PROTO_NUMBERS["easy"][best_idx],
            np.array(PROTO_NUMBERS["mod"]) - PROTO_NUMBERS["mod"][best_idx],
            np.array(PROTO_NUMBERS["hard"]) - PROTO_NUMBERS["hard"][best_idx],
        ]
    )
    grouped_bars(
        axes[2],
        [str(v) for v in PROTO_NUMBERS["labels"]],
        proto_num_delta,
        "(c) Sensitivity to prototype number",
        "Foreground prototypes",
        r"$\Delta AP_{3D}$ vs. 10 (%)",
    )
    axes[2].set_ylim(-0.45, 0.12)

    axes[2].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    png_path = output_dir / "fig_ablation_gain_summary.png"
    pdf_path = output_dir / "fig_ablation_gain_summary.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot paper ablation figures.")
    parser.add_argument(
        "--output-dir",
        default="paper_figures",
        help="Directory for generated figures.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.dpi": 150,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.4), constrained_layout=True)

    x0 = np.arange(len(MAIN_MODULES["labels"]))
    plot_metric(axes[0], x0, MAIN_MODULES, "(a) Main module ablation", "Experiment ID")
    axes[0].set_xticks(x0)
    axes[0].set_xticklabels(MAIN_MODULES["labels"])
    axes[0].set_ylim(18.0, 32.8)

    x1 = np.arange(len(PROTO_COMPONENTS["labels"]))
    plot_metric(axes[1], x1, PROTO_COMPONENTS, "(b) Prototype components", "Component setting")
    axes[1].set_xticks(x1)
    axes[1].set_xticklabels(PROTO_COMPONENTS["labels"], rotation=20, ha="right")
    axes[1].set_ylim(19.0, 32.5)

    x2 = np.array(PROTO_NUMBERS["labels"])
    plot_metric(axes[2], x2, PROTO_NUMBERS, "(c) Prototype number sensitivity", "Foreground prototypes")
    axes[2].set_xticks(x2)
    axes[2].set_ylim(19.3, 32.5)

    axes[2].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
        fontsize=8,
    )

    png_path = output_dir / "fig_ablation_trends.png"
    pdf_path = output_dir / "fig_ablation_trends.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    save_gain_summary(output_dir)


if __name__ == "__main__":
    main()
