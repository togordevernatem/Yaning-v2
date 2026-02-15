# tools/collect_typed_vs_struct.py
"""
S6-A: 在统一协议下，对比 Struct (untyped) vs Typed 的 Seen/OOD 泛化表现。

从 logs/ 下读取：
- gc_tpp_struct_{data_mode}.npz
- gc_tpp_struct_typed_{data_mode}.npz

并输出一张 Markdown 表格，列出：
- test_nll / test_nll_seen / test_nll_ood
- test_rmse / test_rmse_seen / test_rmse_ood
- test_mae / test_mae_seen / test_mae_ood
"""

import os
import numpy as np


def _load_metrics(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    data = np.load(path)
    # 尝试安全读取，缺哪个字段就写 NaN
    def get(name, default=np.nan):
        return float(data[name]) if name in data else default

    return {
        "test_nll": get("test_nll"),
        "test_rmse": get("test_rmse"),
        "test_mae": get("test_mae"),
        "test_nll_seen": get("test_nll_seen"),
        "test_rmse_seen": get("test_rmse_seen"),
        "test_mae_seen": get("test_mae_seen"),
        "test_nll_ood": get("test_nll_ood"),
        "test_rmse_ood": get("test_rmse_ood"),
        "test_mae_ood": get("test_mae_ood"),
    }


def collect_typed_vs_struct(
    data_mode: str = "icews_real_topk500_K500",
    struct_prefix: str = "logs/gc_tpp_struct_",
    typed_prefix: str = "logs/gc_tpp_struct_typed_",
):
    """
    读取 Struct 与 Typed 的 npz，打印 Markdown 表（两行：Struct / Typed）。
    """
    struct_path = f"{struct_prefix}{data_mode}.npz"
    typed_path = f"{typed_prefix}{data_mode}.npz"

    print(f"[INFO] Loading Struct metrics from: {struct_path}")
    struct_m = _load_metrics(struct_path)

    print(f"[INFO] Loading Typed metrics from:  {typed_path}")
    typed_m = _load_metrics(typed_path)

    # 打印 Markdown 表头
    print("\n### Struct vs Typed (data_mode = {})\n".format(data_mode))
    print("| Model  | test_nll | test_rmse | test_mae | test_nll_seen | test_rmse_seen | test_mae_seen | test_nll_ood | test_rmse_ood | test_mae_ood |")
    print("|--------|----------|-----------|----------|----------------|----------------|---------------|--------------|----------------|--------------|")

    def row(name: str, m: dict) -> str:
        return (
            f"| {name} "
            f"| {m['test_nll']:.4f} "
            f"| {m['test_rmse']:.4f} "
            f"| {m['test_mae']:.4f} "
            f"| {m['test_nll_seen']:.4f} "
            f"| {m['test_rmse_seen']:.4f} "
            f"| {m['test_mae_seen']:.4f} "
            f"| {m['test_nll_ood']:.4f} "
            f"| {m['test_rmse_ood']:.4f} "
            f"| {m['test_mae_ood']:.4f} |"
        )

    print(row("Struct", struct_m))
    print(row("Typed", typed_m))
    print()  # 结尾空行，方便复制到 Markdown


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect Struct vs Typed metrics for Stage S6-A.")
    parser.add_argument(
        "--data_mode",
        type=str,
        default="icews_real_topk500_K500",
        help="Data mode, e.g., icews_real_topk500_K500",
    )
    args = parser.parse_args()

    collect_typed_vs_struct(data_mode=args.data_mode)
