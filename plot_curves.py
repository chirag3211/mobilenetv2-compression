"""
plot_curves.py  —  Generate loss and accuracy curves from training_history.json (Q1c).

Usage:
    python plot_curves.py                                      # default paths
    python plot_curves.py --history training_history.json --out curves.png
"""

import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")          # non-interactive backend; works without a display
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_curves(history: dict, save_path: str = "training_curves.png"):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("MobileNet-v2 on CIFAR-10  —  Training Curves", fontsize=13)

    # ── Loss ──────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train loss",  linewidth=1.8)
    ax.plot(epochs, history["test_loss"],  label="Test loss",   linewidth=1.8, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (cross-entropy)")
    ax.set_title("Loss")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train acc",  linewidth=1.8)
    ax.plot(epochs, history["test_acc"],  label="Test acc",   linewidth=1.8, linestyle="--")

    best_acc   = max(history["test_acc"])
    best_epoch = history["test_acc"].index(best_acc) + 1
    ax.axhline(best_acc, color="grey", linestyle=":", linewidth=1.2,
               label=f"Best test: {best_acc:.2f}% (epoch {best_epoch})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.set_title("Accuracy")
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Curves saved → {save_path}")
    print(f"Best test accuracy: {best_acc:.2f}%  (epoch {best_epoch})")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--history", type=str, default="training_history.json",
                        help="Path to the JSON file saved by train.py")
    parser.add_argument("--out",     type=str, default="training_curves.png",
                        help="Output image path")
    args = parser.parse_args()

    try:
        with open(args.history) as f:
            history = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: '{args.history}' not found. Run train.py first.")
        sys.exit(1)

    required = {"train_loss", "train_acc", "test_loss", "test_acc"}
    if not required.issubset(history.keys()):
        print(f"ERROR: history file is missing keys {required - set(history.keys())}")
        sys.exit(1)

    plot_curves(history, save_path=args.out)


if __name__ == "__main__":
    main()
