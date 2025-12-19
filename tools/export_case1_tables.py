import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

def load_npz_result(path: str) -> Dict[str, Any]:
    data = np.load(path)
    return {k: data[k].tolist() for k in data.files}

def export_case1_table(
    logs_dir: str = "logs",
    out_csv: str = "logs/case1_core_struct.csv"
):
    """
    读取 Core/Struct 在 icews_real_topk500 下的 npz 结果，
    输出一个 CSV 表格，可直接用于论文 Case-1 表。
    """
    rows: List[Dict[str, Any]] = []
    for model in ["gc_tpp_core", "gc_tpp_struct"]:
        path = os.path.join(logs_dir, f"{model}_icews_real_topk500.npz")
        d = load_npz_result(path)
        rows.append({
            "model": model,
            "test_nll": d["test_nll"],
            "test_nll_seen": d["test_nll_seen"],
            "test_nll_ood": d["test_nll_ood"],
            "test_rmse": d["test_rmse"],
            "test_rmse_ood": d["test_rmse_ood"],
            "test_mae_ood": d["test_mae_ood"],
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Case-1 table saved to {out_csv}")

if __name__ == "__main__":
    export_case1_table()