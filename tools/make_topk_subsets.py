# tools/make_topk_subsets.py
"""
用途
----
从 `icews_real_topk500.csv` 中，按照某个字段（默认 `event_type`）的频次，
构造不同 Top-K 子集，并输出多份新的 CSV 文件，例如：

    - icews_real_topk500_K100.csv
    - icews_real_topk500_K500.csv
    - icews_real_topk500_K1000.csv

这些 CSV 将被 `data/dataset_toy.py` 中的 GC_TPP_Dataset 在如下模式下使用：

    - mode="icews_real_topk500_K100"
    - mode="icews_real_topk500_K500"
    - mode="icews_real_topk500_K1000"

配合 `_load_from_icews_real_topk500_csv` 中“按文件名设定不同 debug 截断上限”的逻辑，
可以保证：

    1. 不同 K 对应的事件类型集合不同（Top‑K by frequency）；
    2. 不同 K 对应的**事件条数也不同**，从而在训练时间和 Seen/OOD 统计上
       产生可观测的差异，为 S4（Top-K 实验）提供真正有信息量的结果。
"""

import os
from typing import List

import pandas as pd


def make_topk_subsets(
    events_dir: str = "data/events",
    base_csv: str = "icews_real_topk500.csv",
    K_list: List[int] = [100, 500, 1000],
    freq_col: str = "event_type",   # 用哪个列来统计“频次”决定 Top-K（默认按事件类型）
    factor: int = 50,               # 每个 type 期望保留的平均事件数，用于让总行数随 K 变化
):
    """
    从 base_csv（例如 icews_real_topk500.csv）里，按某个字段的频次抽取不同 Top-K 子集，
    输出多份新的 CSV：icews_real_topk500_K100.csv / icews_real_topk500_K500.csv / ...

    设计目标
    --------
    - 在原始 topk500 的基础上：
        * 选出频次最高的前 K 个 `freq_col` 取值（默认：event_type）；
        * 仅保留这些取值对应的事件；
        * 再通过 `limit = K * factor` 控制总事件条数，保证：
              K 越大 → 事件条数整体越多。
    - 这样一来，后续在 GC_TPP_Dataset 中按不同 K 加载数据时，
      既能感受到「Top-K 过滤掉了一部分事件类型」，也能感受到「整体规模变大」。

    参数
    ----
    events_dir : str
        事件 CSV 所在目录，一般是 data/events。
    base_csv : str
        原始 Top-500 CSV 文件名，例如 "icews_real_topk500.csv"。
    K_list : list[int]
        希望构造的 Top-K 列表，例如 [100, 500, 1000]。
    freq_col : str
        以哪个列来统计频次。简单起见，默认用 "event_type"：
          - 表示选取“出现频次最高的 K 种事件类型”；
          - 当然也可以换成 "src" / "dst" 做“前 K 个国家节点”之类的变体。
    factor : int
        控制总事件条数的缩放：limit = K * factor。
        这一步的目的：
          - 避免所有 K 都落在同一个「统一 debug 截断」上（例如统一 8000 条）；
          - 同时控制整体规模不要太大，以免训练过慢。
        可根据实际训练时间调整（例如 30/50/100 等）。
    """
    base_path = os.path.join(events_dir, base_csv)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"[TOP-K] Base CSV not found: {base_path}")

    print(f"[TOP-K] Loading base CSV: {base_path}")
    df = pd.read_csv(base_path)

    if freq_col not in df.columns:
        raise ValueError(
            f"[TOP-K] Column '{freq_col}' not in CSV columns: {df.columns.tolist()}"
        )

    # 1) 统计 freq_col 的频次（例如 event_type 的出现次数）
    freq = df[freq_col].value_counts()
    print(f"[TOP-K] Base CSV rows = {len(df)}, "
          f"unique {freq_col} = {len(freq)}")

    # 2) 为了后续对每个 K 子集截断时不打乱时间顺序，先按 time 排序
    if "time" in df.columns:
        df = df.sort_values("time")

    # 3) 针对每一个 K 构造对应的子集
    for K in K_list:
        print(f"[TOP-K] Building subset for K={K} ...")

        # 3.1 选出频次最高的前 K 个取值
        topk_values = freq.head(K).index.tolist()
        df_k = df[df[freq_col].isin(topk_values)].copy()

        # 3.2 让总行数随 K 变化：limit = K * factor
        #     这样 K 越大，理论上保留的行数越多。
        limit = K * factor if factor is not None else None
        if limit is not None and len(df_k) > limit:
            before = len(df_k)
            df_k = df_k.iloc[:limit].copy()
            print(
                f"[TOP-K]  Truncated K={K} subset to first {limit} rows "
                f"(original={before})."
            )
        else:
            print(
                f"[TOP-K]  K={K} subset rows = {len(df_k)} "
                f"(no extra truncation by factor)."
            )

        # 3.3 写出新 CSV
        out_name = f"icews_real_topk500_K{K}.csv"
        out_path = os.path.join(events_dir, out_name)
        df_k.to_csv(out_path, index=False)
        print(f"[TOP-K] Saved Top-{K} subset to {out_path} (rows={len(df_k)})")


if __name__ == "__main__":
    make_topk_subsets()
