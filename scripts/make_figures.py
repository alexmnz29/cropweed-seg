"""Generate every matplotlib figure for the Medium article from runs/ CSVs.

Reads per-epoch metrics from runs/<name>/metrics.csv and writes PNGs to
docs/img/. Each figure is skipped with a warning if its run dirs are missing,
so the script runs even while some experiments are absent. Best-epoch values
are always selected by max validation mIoU, matching the checkpoint rule
(ADR 0002).

Run from the repo root: uv run scripts/make_figures.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "docs" / "img"

# Run directory names. Adjust to whatever runs/ actually contains.
RUN_CHAMPION = "focal_dice_s42"
RUN_LOSSES = {
    "weighted CE": "baseline_e25",
    "focal": "focal",
    "dice": "dice_e25",
    "focal + dice": "focal_dice_s42",
}
RUN_AUGMENTED = "unet_focal_dice_full_e35"
RUN_FRACTIONS = {  # learning curve (ADR 0008)
    25: "unet_focal_dice_p25_e100",
    50: "unet_focal_dice_p50_e50",
    75: "unet_focal_dice_p75_e35",
    100: "focal_dice_s42",
}
FRACTION_IMAGES = {25: 70, 50: 140, 75: 210, 100: 280}

# Palette, matching the overlay colors in make_demo_frames.py
C_WEED = "#e132a0"
C_CROP = "#46aa46"
C_BG = "#8a8a8a"
C_NEUTRAL = "#444444"
LOSS_COLORS = ["#8a8a8a", "#e6a23c", "#4a90d9", "#e132a0"]

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
    }
)


def read_metrics(run_name: str) -> list[dict] | None:
    """Rows of runs/<name>/metrics.csv as dicts with float values, or None."""
    path = RUNS / run_name / "metrics.csv"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return None
    with path.open() as f:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def best_row(rows: list[dict]) -> dict:
    """The row of the best-mIoU epoch, the checkpoint selection rule."""
    return max(rows, key=lambda r: r["miou"])


def fig_loss_curves() -> None:
    """Per-epoch weed IoU for the four losses. The dice dip should be visible."""
    print("fig 1: loss study curves")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for (label, run), color in zip(RUN_LOSSES.items(), LOSS_COLORS):
        rows = read_metrics(run)
        if rows is None:
            continue
        ax.plot(
            [r["epoch"] for r in rows],
            [r["iou_weed"] for r in rows],
            label=label,
            color=color,
            linewidth=2 if label == "focal + dice" else 1.5,
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation weed IoU")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(OUT / "loss_study_weed_iou.png")
    plt.close(fig)


def fig_convergence_trap() -> None:
    """Focal vs focal+dice, vertical line at epoch 15 where the wrong
    conclusion lived."""
    print("fig 2: convergence trap (focal vs focal+dice)")
    focal = read_metrics(RUN_LOSSES["focal"])
    combo = read_metrics(RUN_LOSSES["focal + dice"])
    if focal is None or combo is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        [r["epoch"] for r in focal],
        [r["iou_weed"] for r in focal],
        label="focal",
        color="#e6a23c",
        linewidth=1.5,
    )
    ax.plot(
        [r["epoch"] for r in combo],
        [r["iou_weed"] for r in combo],
        label="focal + dice",
        color=C_WEED,
        linewidth=2,
    )
    ax.axvline(15, color=C_NEUTRAL, linestyle=":", linewidth=1)
    ax.annotate(
        "epoch 15: they look tied.\nNeither has converged.",
        xy=(15.3, 0.45),
        fontsize=9,
        color=C_NEUTRAL,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation weed IoU")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(OUT / "convergence_trap.png")
    plt.close(fig)


def fig_champion_vs_augmented() -> None:
    """Champion vs full augmentation stack, both plateaued on the same value."""
    print("fig 3: champion vs augmented")
    champ = read_metrics(RUN_CHAMPION)
    aug = read_metrics(RUN_AUGMENTED)
    if champ is None or aug is None:
        return
    champ_best = best_row(champ)["iou_weed"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        [r["epoch"] for r in champ],
        [r["iou_weed"] for r in champ],
        label="champion (no augmentation, 25 ep)",
        color=C_NEUTRAL,
        linewidth=1.5,
    )
    ax.plot(
        [r["epoch"] for r in aug],
        [r["iou_weed"] for r in aug],
        label="flips + color jitter (35 ep)",
        color=C_WEED,
        linewidth=2,
    )
    ax.axhline(champ_best, color=C_NEUTRAL, linestyle="--", linewidth=0.8)
    ax.annotate(
        f"champion best: {champ_best:.3f}",
        xy=(1, champ_best + 0.008),
        fontsize=9,
        color=C_NEUTRAL,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation weed IoU")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(OUT / "champion_vs_augmented.png")
    plt.close(fig)


def fig_learning_curve() -> None:
    """weed IoU vs train size, crop and background as flat references."""
    print("fig 4: learning curve")
    points = {}
    for frac, run in RUN_FRACTIONS.items():
        rows = read_metrics(run)
        if rows is None:
            return
        points[frac] = best_row(rows)

    fracs = sorted(points)
    xs = [FRACTION_IMAGES[f] for f in fracs]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, color, label in [
        ("iou_background", C_BG, "background"),
        ("iou_crop", C_CROP, "crop"),
        ("iou_weed", C_WEED, "weed"),
    ]:
        ys = [points[f][key] for f in fracs]
        ax.plot(
            xs,
            ys,
            marker="o",
            color=color,
            label=label,
            linewidth=2 if key == "iou_weed" else 1.2,
            markersize=6 if key == "iou_weed" else 4,
        )
        if key == "iou_weed":
            for x, y in zip(xs, ys):
                ax.annotate(
                    f"{y:.3f}",
                    xy=(x, y),
                    xytext=(0, -16),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                    color=C_WEED,
                )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{f}%\n({FRACTION_IMAGES[f]} images)" for f in fracs])
    ax.set_xlabel("fraction of the train split (equal optimizer-step budget)")
    ax.set_ylabel("validation IoU at best epoch")
    ax.set_ylim(0.55, 1.0)
    ax.legend(frameon=False, loc="center right")
    fig.savefig(OUT / "learning_curve.png")
    plt.close(fig)


def fig_class_balance() -> None:
    """The two opposite imbalances, per pixel and per image (notebook 01)."""
    print("fig 5: class balance")
    classes = ["background", "crop", "weed"]
    colors = [C_BG, C_CROP, C_WEED]
    per_pixel = [84.0, 12.3, 3.7]
    per_image = [100.0, 70.5, 100.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, values, title in [
        (ax1, per_pixel, "share of pixels (%)"),
        (ax2, per_image, "present in images (%)"),
    ]:
        bars = ax.bar(classes, values, color=colors, width=0.6)
        ax.bar_label(bars, fmt="%.1f", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, 108)
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "weed: rare per pixel, universal per image. crop: the opposite.",
        fontsize=11,
        y=1.02,
    )
    fig.savefig(OUT / "class_balance.png")
    plt.close(fig)


def fig_four_levers() -> None:
    """Schematic: four refuted levers converging on the 0.67 wall, the
    learning curve pointing at the exit."""
    print("fig 6: four levers diagram")
    from matplotlib.patches import Rectangle

    levers = [
        ("loss study\n(best: focal+dice)", 0.670),
        ("architecture\n(DeepLabV3+)", 0.655),
        ("input resolution\n(crop 720)", 0.659),
        ("augmentation\n(flips + jitter)", 0.671),
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.grid(False)

    # the wall; label vertically centered in the bar
    ax.add_patch(Rectangle((6.6, 0.6), 0.5, 8.8, color=C_NEUTRAL, alpha=0.9, zorder=2))
    ax.text(
        6.85,
        5.0,
        "weed IoU ~0.67",
        ha="center",
        va="center",
        rotation=90,
        fontsize=11,
        fontweight="bold",
        color="white",
        zorder=3,
    )

    # four blocked levers, smaller arrows
    ys = [8.3, 6.5, 4.7, 2.9]
    for (label, value), y in zip(levers, ys):
        ax.annotate(
            "",
            xy=(6.55, y),
            xytext=(2.7, y),
            arrowprops=dict(
                arrowstyle="-|>",
                color=C_NEUTRAL,
                linewidth=1.2,
                mutation_scale=12,
            ),
        )
        ax.text(2.55, y, label, ha="right", va="center", fontsize=10)
        ax.text(
            4.6,
            y + 0.3,
            f"{value:.3f}",
            ha="center",
            fontsize=9.5,
            color=C_NEUTRAL,
        )

    # the one arrow that goes through the wall
    ax.annotate(
        "",
        xy=(9.7, 7.4),
        xytext=(6.1, 7.4),
        arrowprops=dict(
            arrowstyle="-|>",
            color=C_WEED,
            linewidth=1.8,
            mutation_scale=16,
        ),
        zorder=4,
    )
    ax.text(
        8.6,
        8.15,
        "learning curve:\nstill climbing at 400 images",
        ha="center",
        va="center",
        fontsize=9.5,
        color=C_WEED,
    )
    ax.text(
        8.6,
        6.55,
        "more (and more\ndiverse) data",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=C_WEED,
    )
    fig.savefig(OUT / "four_levers.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig_loss_curves()
    fig_convergence_trap()
    fig_champion_vs_augmented()
    fig_learning_curve()
    fig_class_balance()
    fig_four_levers()
    print(f"\nfigures in {OUT}")


if __name__ == "__main__":
    main()
