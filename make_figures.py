"""
make_figures.py

Generates three IEEE-conference-quality figures from the outputs of
train.py, for use in a 1-page report:

  1. figures/loss_curves.png      -- training + validation loss, all
                                      trained variants overlaid
  2. figures/val_accuracy.png     -- validation accuracy over epochs,
                                      all trained variants overlaid,
                                      with final test accuracy marked
  3. figures/confidence_histogram.png
                                   -- distribution of per-image softmax
                                      max-probability (prediction
                                      confidence) on the CIFAR-10 TEST
                                      set, all trained variants overlaid

Run this AFTER train.py has produced results_<variant>.json and
checkpoint_<variant>.pt for at least one variant (works with 1, 2, or 3
of original/modified/phasegrid -- whatever you've trained so far):

    python make_figures.py

All three figures are saved into ./figures/ as both .png (300 DPI, for
Word/PowerPoint) and .pdf (vector, for LaTeX/IEEE templates).

Figures 1-2 only need the saved results_*.json files (fast, no GPU
needed). Figure 3 needs to re-run inference over the CIFAR-10 test set
using the saved checkpoint_*.pt files, so it will use the GPU if
available and takes roughly as long as one evaluation pass did during
training (seconds to low minutes, not another full training run).
"""

import json
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

from train import (
    build_test_loader, CIFAR_MEAN, CIFAR_STD, NUM_CLASSES,
)
from vit_original import build_vit_original
from vit_modified import build_vit_modified
from vit_phasegrid import build_vit_phasegrid


# ---------------------------------------------------------------------------
# IEEE-style matplotlib configuration
# ---------------------------------------------------------------------------
IEEE_RC = {
    "font.family": "serif",       # Times-like serif, matches IEEE body text
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.3,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,           # embed TrueType (not Type 3) fonts in PDF
    "ps.fonttype": 42,
}
mpl.rcParams.update(IEEE_RC)

# IEEE double-column figure width is ~3.5in for a single sub-figure;
# these plots are meant to sit in a 1-page report, so a slightly wider
# single-column-ish width reads well without looking stretched.
FIG_WIDTH_IN = 3.5
FIG_HEIGHT_IN = 2.6

# Colorblind-safe, print-safe palette (Okabe-Ito), distinct in grayscale too
COLORS = {
    "original": "#0072B2",   # blue
    "modified": "#D55E00",   # vermillion
    "phasegrid": "#009E73",  # bluish green
}
LINESTYLES = {
    "original": "-",
    "modified": "--",
    "phasegrid": "-.",
}
LABELS = {
    "original": "Original",
    "modified": "Modified (RoPE)",
    "phasegrid": "PhaseGrid",
}

VARIANTS_ORDER = ["original", "modified", "phasegrid"]
OUT_DIR = "figures"


def load_available_results():
    """Returns {variant: results_dict} for every results_<variant>.json
    that exists in the current directory."""
    results = {}
    for v in VARIANTS_ORDER:
        path = f"results_{v}.json"
        if os.path.exists(path):
            with open(path) as f:
                results[v] = json.load(f)
        else:
            print(f"  (skipping '{v}': {path} not found)")
    if not results:
        raise SystemExit(
            "No results_*.json files found. Run train.py for at least one "
            "variant before running make_figures.py."
        )
    return results


def save_both_formats(fig, name: str):
    os.makedirs(OUT_DIR, exist_ok=True)
    png_path = os.path.join(OUT_DIR, f"{name}.png")
    pdf_path = os.path.join(OUT_DIR, f"{name}.pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    print(f"  saved {png_path}")
    print(f"  saved {pdf_path}")


# ---------------------------------------------------------------------------
# Figure 1: training + validation loss curves
# ---------------------------------------------------------------------------
def make_loss_curve_figure(results: dict):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    for variant, res in results.items():
        epochs = range(1, len(res["history"]["train_loss"]) + 1)
        color = COLORS[variant]
        ax.plot(epochs, res["history"]["train_loss"],
                color=color, linestyle=LINESTYLES[variant],
                label=f"{LABELS[variant]} (train)")
        ax.plot(epochs, res["history"]["val_loss"],
                color=color, linestyle=LINESTYLES[variant], alpha=0.55,
                marker="none")
        # Distinguish train/val with alpha rather than a second legend
        # entry per variant, to keep the legend compact for a 1-page report.

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training (solid, opaque) / Validation (same style, faded) Loss")
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9,
              fontsize=7, borderaxespad=0.4)
    fig.tight_layout()
    save_both_formats(fig, "loss_curves")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: validation accuracy curves + final test accuracy markers
# ---------------------------------------------------------------------------
def make_val_accuracy_figure(results: dict):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    for variant, res in results.items():
        epochs = range(1, len(res["history"]["val_acc"]) + 1)
        color = COLORS[variant]
        ax.plot(epochs, [a * 100 for a in res["history"]["val_acc"]],
                color=color, linestyle=LINESTYLES[variant],
                label=f"{LABELS[variant]}")
        ax.axhline(res["test_acc"] * 100, color=color, linestyle=":",
                   linewidth=0.9, alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title("Validation Accuracy (dotted line = final test accuracy)")
    ax.grid(True)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    save_both_formats(fig, "val_accuracy")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: prediction-confidence histogram on the CIFAR-10 test set
# ---------------------------------------------------------------------------
BUILDERS = {
    "original": build_vit_original,
    "modified": build_vit_modified,
    "phasegrid": build_vit_phasegrid,
}


@torch.no_grad()
def collect_confidences(variant: str, results: dict, device, test_loader):
    """Loads checkpoint_<variant>.pt, runs it over the test set, and
    returns a 1D numpy array of per-image softmax max-probability
    (i.e. the model's confidence in its top prediction, whether or not
    that prediction was correct)."""
    ckpt_path = f"checkpoint_{variant}.pt"
    if not os.path.exists(ckpt_path):
        print(f"  (skipping confidence histogram for '{variant}': "
              f"{ckpt_path} not found)")
        return None

    args = results[variant]["args"]
    model = BUILDERS[variant](
        img_size=args["img_size"], patch_size=args["patch_size"],
        num_classes=NUM_CLASSES, embed_dim=args["embed_dim"],
        depth=args["depth"], num_heads=args["num_heads"],
        mlp_ratio=args["mlp_ratio"], drop_rate=args["drop_rate"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    confidences = []
    for x, _y in test_loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        max_prob, _ = probs.max(dim=1)
        confidences.append(max_prob.cpu().numpy())

    return np.concatenate(confidences)


def make_confidence_histogram_figure(results: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  (running test-set inference on {device} for the confidence histogram)")

    # Any one trained variant's args determine img_size/batch_size for the
    # shared CIFAR-10 test loader (all variants share the same img_size
    # by construction of train.py's controlled comparison). This only
    # touches the test set -- no train/val split logic involved, so it
    # works regardless of how large the training set was.
    any_args = next(iter(results.values()))["args"]
    test_loader = build_test_loader(any_args["img_size"], batch_size=512)

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

    bins = np.linspace(1.0 / NUM_CLASSES, 1.0, 30)
    any_plotted = False
    for variant in results:
        conf = collect_confidences(variant, results, device, test_loader)
        if conf is None:
            continue
        ax.hist(conf, bins=bins, density=True, histtype="step",
                linewidth=1.4, color=COLORS[variant],
                linestyle=LINESTYLES[variant], label=LABELS[variant])
        any_plotted = True

    if not any_plotted:
        print("  (no checkpoint_*.pt files found -- skipping confidence histogram)")
        plt.close(fig)
        return

    ax.axvline(1.0 / NUM_CLASSES, color="gray", linestyle=":", linewidth=0.8)
    ax.annotate("random\nchance", xy=(1.0 / NUM_CLASSES, 0), xytext=(0, 6),
                textcoords="offset points", fontsize=6, color="gray",
                ha="center", va="bottom")

    ax.set_xlabel("Prediction confidence (max softmax probability)")
    ax.set_ylabel("Density")
    ax.set_title("Test-Set Prediction Confidence Distribution")
    ax.set_xlim(1.0 / NUM_CLASSES - 0.02, 1.02)
    ax.grid(True)
    # Legend placed OUTSIDE the axes (below the plot) rather than "best"
    # inside it -- with real trained models, confidence mass can land
    # anywhere along the x-axis (well-calibrated models spread across the
    # full [1/K, 1] range), so any in-axes corner can end up overlapping
    # data. A below-plot legend is layout-independent of where the
    # histograms actually fall.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3,
              frameon=False, fontsize=7, columnspacing=1.0, handlelength=1.5)
    fig.tight_layout()
    save_both_formats(fig, "confidence_histogram")
    plt.close(fig)


if __name__ == "__main__":
    print("Loading results...")
    results = load_available_results()
    print(f"Found results for: {list(results.keys())}")

    print("\nFigure 1/3: loss curves")
    make_loss_curve_figure(results)

    print("\nFigure 2/3: validation accuracy")
    make_val_accuracy_figure(results)

    print("\nFigure 3/3: confidence histogram (requires checkpoints + test inference)")
    make_confidence_histogram_figure(results)

    print(f"\nAll figures saved to ./{OUT_DIR}/")
