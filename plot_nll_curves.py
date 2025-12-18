import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


def plot_nll_curves(mode: str):
    """
    从 ./logs/nll_{mode}.npz 读取 Train / Val NLL 曲线，
    并画成 PNG：./logs/nll_{mode}.png
    """
    logs_dir = "./logs"
    os.makedirs(logs_dir, exist_ok=True)

    npz_path = os.path.join(logs_dir, f"nll_{mode}.npz")
    if not os.path.exists(npz_path):
        print(f"[ERROR] File not found: {npz_path}")
        print("请先运行 main.py 训练一次，对应的 data_mode 要和这里的 mode 一致。")
        return

    data = np.load(npz_path)
    train_nll = data["train_nll"]
    val_nll = data["val_nll"]

    epochs = np.arange(1, len(train_nll) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_nll, marker="o", label="Train NLL")
    plt.plot(epochs, val_nll, marker="s", label="Val NLL")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Average NLL", fontsize=12)
    plt.title(f"GC-TPP Continuous NLL Curves ({mode})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    out_path = os.path.join(logs_dir, f"nll_{mode}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"[INFO] Saved NLL curve figure to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Train/Val NLL curves for GC-TPP experiments."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="icews_real",
        help="Which npz to load: toy / icews_real / icews_toy (if exists).",
    )
    args = parser.parse_args()

    plot_nll_curves(args.mode)


if __name__ == "__main__":
    main()
