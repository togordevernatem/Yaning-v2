# tools/eval_topk.py

import os
import sys

# ---- 关键步骤：把项目根目录加入 sys.path ----
# 当前文件路径： .../gctpp_temporal_toy/tools/eval_topk.py
# 项目根目录   ：.../gctpp_temporal_toy
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"[DEBUG] Added project root to sys.path: {PROJECT_ROOT}")
# ---------------------------------------------

import time
from typing import List, Dict, Any

import pandas as pd
import numpy as np

from models.gc_tpp_continuous import run_gc_tpp_continuous
from models.gc_tpp_struct import run_gc_tpp_struct


def run_eval_topk(
    model_name: str,               # "core" 或 "struct"
    data_mode_base: str = "icews_real_topk500",
    K_list: List[int] = [100, 500, 1000],
    out_csv: str = "logs/topk_core_struct.csv",
) -> None:
    """
    在不同 Top-K 规模下运行同一个模型（Core 或 Struct），
    记录训练时间与 Test/Seen/OOD 指标，生成一个 CSV 方便画图/写表。

    特点：具备“断点续跑”能力
    --------------------------------
    - 如果 logs/ 下已经存在对应的 .npz 结果文件，则跳过训练，
      直接从 .npz 中读取 test 指标写入 CSV。
    - 仅对还没有 .npz 的 K 重新训练一次。
    """
    rows: List[Dict[str, Any]] = []

    for K in K_list:
        # 约定：data_mode 里编码 K，例如 "icews_real_topk500_K100"
        if K is None:
            data_mode = data_mode_base
        else:
            data_mode = f"{data_mode_base}_K{K}"

        if model_name == "core":
            runner = run_gc_tpp_continuous
            model_tag = "gc_tpp_core"
            npz_name = f"gc_tpp_core_{data_mode}.npz"
        elif model_name == "struct":
            runner = run_gc_tpp_struct
            model_tag = "gc_tpp_struct"
            npz_name = f"gc_tpp_struct_{data_mode}.npz"
        else:
            raise ValueError(f"Unknown model_name={model_name}")

        npz_path = os.path.join("logs", npz_name)

        # ---- 1) 如果已有 npz，就跳过训练，直接读取结果 ----
        if os.path.exists(npz_path):
            print(f"[TOP-K] Found existing npz, skip training: {npz_path}")
            data = np.load(npz_path)

            # 字段名要和 run_gc_tpp_* 里保存的一致
            test_nll       = float(data["test_nll"])
            test_nll_seen  = float(data["test_nll_seen"])
            test_nll_ood   = float(data["test_nll_ood"])
            test_rmse_ood  = float(data["test_rmse_ood"])
            test_mae_ood   = float(data["test_mae_ood"])
            train_time_sec = float("nan")  # 原始训练时间未知，用 NaN 占位

        else:
            # ---- 2) 否则，正常训练一次 ----
            print(f"[TOP-K] Running {model_tag} on {data_mode} ...")
            t0 = time.time()
            res = runner(data_mode=data_mode)   # 你的 run_* 已返回 dict
            train_time_sec = time.time() - t0

            test_nll       = float(res["test_nll"])
            test_nll_seen  = float(res["test_nll_seen"])
            test_nll_ood   = float(res["test_nll_ood"])
            test_rmse_ood  = float(res["test_rmse_ood"])
            test_mae_ood   = float(res["test_mae_ood"])

        rows.append({
            "model": model_tag,
            "data_mode": data_mode,
            "K": K,
            "train_time_sec": train_time_sec,
            "test_nll": test_nll,
            "test_nll_seen": test_nll_seen,
            "test_nll_ood": test_nll_ood,
            "test_rmse_ood": test_rmse_ood,
            "test_mae_ood": test_mae_ood,
        })

    # 写 CSV（每次覆盖，但包含所有 rows）
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[TOP-K] Saved Top-K sweep results to {out_csv}")


if __name__ == "__main__":
    # 先跑 Core，再跑 Struct；已存在的 npz 会被自动跳过
    run_eval_topk("core",   data_mode_base="icews_real_topk500")
    run_eval_topk("struct", data_mode_base="icews_real_topk500")