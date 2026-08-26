"""
compare_results.py

Reads results_original.json, results_modified.json, and
results_phasegrid.json (produced by train.py) and generates the
quantitative deliverables:
  1. A printed top-1 test accuracy comparison table (all three variants).
  2. A single combined training/validation loss + validation accuracy
     plot (loss_curves_comparison.png), all three variants overlaid.

Run after all three of:
    python train.py --variant original  --epochs 60
    python train.py --variant modified  --epochs 60
    python train.py --variant phasegrid --epochs 60
have completed. (If you only trained original + modified, this script
still works and degrades gracefully to a two-way comparison.)
"""

import json
import matplotlib.pyplot as plt

COLORS = {
    "original": "#1f77b4",
    "modified": "#d62728",
    "phasegrid": "#2ca02c",
}
LABELS = {
    "original": "Original",
    "modified": "Modified (RoPE)",
    "phasegrid": "PhaseGrid (novel)",
}


def load_results(variant: str) -> dict:
    with open(f"results_{variant}.json") as f:
        return json.load(f)


def print_comparison_table(results: dict):
    print("=" * 68)
    print(f'{"Variant":<20}{"Test Top-1 Acc":>16}{"Test Loss":>14}{"Extra Params":>18}')
    print("-" * 68)
    for variant, res in results.items():
        print(f'{LABELS[variant]:<20}{res["test_acc"]*100:>15.2f}%{res["test_loss"]:>14.4f}'
              f'{res["positional_params"]:>18,}')
    print("=" * 68)
    if "original" in results:
        base_acc = results["original"]["test_acc"]
        for variant, res in results.items():
            if variant == "original":
                continue
            delta = (res["test_acc"] - base_acc) * 100
            print(f'Delta ({LABELS[variant]} - Original): {delta:+.2f} pp')


def plot_comparison(results: dict, out_path: str = "loss_curves_comparison.png"):
    any_variant = next(iter(results.values()))
    epochs = range(1, len(any_variant["history"]["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    for variant, res in results.items():
        color = COLORS[variant]
        axes[0].plot(epochs, res["history"]["train_loss"],
                     label=f"{LABELS[variant]} — train", color=color)
        axes[0].plot(epochs, res["history"]["val_loss"],
                     label=f"{LABELS[variant]} — val", color=color, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    for variant, res in results.items():
        color = COLORS[variant]
        axes[1].plot(epochs, [a * 100 for a in res["history"]["val_acc"]],
                     label=f"{LABELS[variant]} — val acc", color=color)
        axes[1].axhline(res["test_acc"] * 100, color=color, linestyle=":", alpha=0.7,
                         label=f'{LABELS[variant]} — test ({res["test_acc"]*100:.2f}%)')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val accuracy (%)")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    # Include "phasegrid" only if you trained it -- this script degrades
    # gracefully to a two-way comparison if results_phasegrid.json is absent.
    variants_to_compare = ["original", "modified", "phasegrid"]
    results = {}
    for v in variants_to_compare:
        try:
            results[v] = load_results(v)
        except FileNotFoundError:
            print(f"(results_{v}.json not found -- skipping {v})")

    if len(results) < 2:
        raise SystemExit("Need at least 2 trained variants' results to compare.")

    print_comparison_table(results)
    plot_comparison(results)
