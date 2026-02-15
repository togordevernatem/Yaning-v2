import argparse
import json
from pathlib import Path

import numpy as np


def load_metrics_for_model(runs_dir: Path, model_name: str):
    """从 runs_dir 下收集指定 model 的所有 seed 的 metrics.json"""
    model_dir = runs_dir / model_name
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    metrics_list = []
    for seed_dir in sorted(model_dir.glob("seed*/metrics.json")):
        with seed_dir.open("r", encoding="utf-8") as f:
            m = json.load(f)
        metrics_list.append(m["summary"]["test"])
    if not metrics_list:
        raise RuntimeError(f"No metrics.json found under {model_dir}")
    return metrics_list


def mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs_dir",
        type=str,
        default="logs/table_runs/protB",
        help="root dir under which model/seedX/metrics.json are stored",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gc_tpp_struct_typed",
        help="model name folder (e.g., gc_tpp_struct_typed)",
    )
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    metrics_list = load_metrics_for_model(runs_dir, args.model_name)

    all_nll = []
    all_rmse = []
    all_mae = []

    ood_nll = []
    ood_rmse = []
    ood_mae = []

    for m in metrics_list:
        all_nll.append(m["all"]["time_nll"])
        all_rmse.append(m["all"]["logdt_rmse"])
        all_mae.append(m["all"]["logdt_mae"])

        ood_nll.append(m["ood"]["time_nll"])
        ood_rmse.append(m["ood"]["logdt_rmse"])
        ood_mae.append(m["ood"]["logdt_mae"])

    all_nll_mean, all_nll_std = mean_std(all_nll)
    all_rmse_mean, all_rmse_std = mean_std(all_rmse)
    all_mae_mean, all_mae_std = mean_std(all_mae)

    ood_nll_mean, ood_nll_std = mean_std(ood_nll)
    ood_rmse_mean, ood_rmse_std = mean_std(ood_rmse)
    ood_mae_mean, ood_mae_std = mean_std(ood_mae)

    print("=== Ours (+Profile Injection) over {} seeds ===".format(len(metrics_list)))
    print("All:")
    print("  Time NLL      = {:.4f} ± {:.4f}".format(all_nll_mean, all_nll_std))
    print("  logΔt RMSE    = {:.4f} ± {:.4f}".format(all_rmse_mean, all_rmse_std))
    print("  logΔt MAE     = {:.4f} ± {:.4f}".format(all_mae_mean, all_mae_std))
    print()
    print("OOD:")
    print("  Time NLL      = {:.4f} ± {:.4f}".format(ood_nll_mean, ood_nll_std))
    print("  logΔt RMSE    = {:.4f} ± {:.4f}".format(ood_rmse_mean, ood_rmse_std))
    print("  logΔt MAE     = {:.4f} ± {:.4f}".format(ood_mae_mean, ood_mae_std))


if __name__ == "__main__":
    main()
