import os
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_npz_for_mode(
    model_tag: str,
    data_mode: str,
    logs_dir: str = "logs",
) -> Dict[str, Any]:
    """
    加载 S4 已经保存好的 npz 文件。

    约定文件名形如：
        logs/gc_tpp_core_icews_real_topk500_K100.npz
        logs/gc_tpp_struct_icews_real_topk500_K500.npz
    """
    fname = f"{model_tag}_{data_mode}.npz"
    fpath = os.path.join(logs_dir, fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"[TOP-K-UNC] npz not found: {fpath}")

    data = np.load(fpath, allow_pickle=True)
    return dict(data)


def eval_topk_uncertainty_event_level(
    model_name: str = "core",
    data_mode_base: str = "icews_real_topk500",
    K_list: List[int] = (100, 500, 1000),
    logs_dir: str = "logs",
    out_csv: str = "logs/topk_uncertainty_event_core.csv",
    out_fig_dir: str = "logs/figs_topk_unc",
    bins: int = 30,
) -> None:
    """
    事件级 S4.5：在已经完成的 Top-K 实验基础上，
    读取每条 Test 事件的 NLL（nll_test_* 向量），
    分别统计 Seen / OOD 的 OOD Score（这里先直接用 NLL），
    输出：
      - 一个 CSV：不同 K 下 Seen/OOD Score 的均值/方差/样本数；
      - 若干直方图 PNG：展示 Seen vs OOD Score 的分布差异。

    要求：
    ----
    对应的 npz 中已经存在以下键（我们在 gc_tpp_continuous.py 中新增的）：
        nll_test_all : [N_test_effective]
        nll_test_seen: [N_seen_test]
        nll_test_ood : [N_ood_test]
    """
    if model_name == "core":
        model_tag = "gc_tpp_core"
    elif model_name == "struct":
        model_tag = "gc_tpp_struct"
    else:
        raise ValueError(f"[TOP-K-UNC] Unknown model_name={model_name}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(out_fig_dir, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for K in K_list:
        data_mode = f"{data_mode_base}_K{K}"
        print(f"[TOP-K-UNC] (event-level) Processing {model_tag} / {data_mode} ...")

        npz_dict = _load_npz_for_mode(model_tag, data_mode, logs_dir=logs_dir)

        # 事件级 NLL 向量（S4.5 的核心）
        if "nll_test_all" not in npz_dict:
            raise KeyError(
                f"[TOP-K-UNC] nll_test_all not found in {model_tag}_{data_mode}.npz\n"
                "请确认已使用带 per-event NLL 的 gc_tpp_continuous.py 重新生成 npz。"
            )

        nll_all = np.array(npz_dict["nll_test_all"], dtype=np.float32)
        nll_seen = np.array(npz_dict.get("nll_test_seen", []), dtype=np.float32)
        nll_ood = np.array(npz_dict.get("nll_test_ood", []), dtype=np.float32)

        # 当前版本：Score = NLL_time
        score_seen = nll_seen
        score_ood = nll_ood

        def _stats(x: np.ndarray) -> Dict[str, float]:
            if x.size == 0:
                return {"mean": np.nan, "std": np.nan, "count": 0}
            return {
                "mean": float(x.mean()),
                "std": float(x.std()),
                "count": int(x.size),
            }

        stats_seen = _stats(score_seen)
        stats_ood = _stats(score_ood)

        rows.append({
            "model": model_tag,
            "data_mode": data_mode,
            "K": K,
            "score_seen_mean": stats_seen["mean"],
            "score_seen_std": stats_seen["std"],
            "score_seen_count": stats_seen["count"],
            "score_ood_mean": stats_ood["mean"],
            "score_ood_std": stats_ood["std"],
            "score_ood_count": stats_ood["count"],
            "score_gap_ood_minus_seen": (
                stats_ood["mean"] - stats_seen["mean"]
                if np.isfinite(stats_seen["mean"]) and np.isfinite(stats_ood["mean"])
                else np.nan
            ),
        })

        # 画直方图（Seen vs OOD），只有两边都有样本时才画
        if stats_seen["count"] > 0 and stats_ood["count"] > 0:
            plt.figure(figsize=(6, 4))
            plt.hist(
                score_seen,
                bins=bins,
                alpha=0.5,
                label=f"Seen (n={stats_seen['count']})",
                density=True,
            )
            plt.hist(
                score_ood,
                bins=bins,
                alpha=0.5,
                label=f"OOD (n={stats_ood['count']})",
                density=True,
            )
            plt.xlabel("OOD Score (per-event NLL)")
            plt.ylabel("Density")
            plt.title(f"{model_tag} / {data_mode}")
            plt.legend()
            fig_path = os.path.join(out_fig_dir, f"{model_tag}_{data_mode}.png")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"[TOP-K-UNC] Saved histogram to {fig_path}")
        else:
            print(f"[TOP-K-UNC] Skip histogram for {model_tag}/{data_mode}: "
                  f"seen_count={stats_seen['count']}, ood_count={stats_ood['count']}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[TOP-K-UNC] Saved event-level uncertainty stats to {out_csv}")


if __name__ == "__main__":
    # 默认先跑 core 的事件级不确定性
    eval_topk_uncertainty_event_level(
        model_name="core",
        data_mode_base="icews_real_topk500",
        K_list=[100, 500, 1000],
        logs_dir="logs",
        out_csv="logs/topk_uncertainty_event_core.csv",
        out_fig_dir="logs/figs_topk_unc",
    )
    # 以后给 struct 也加了 per-event NLL 后，可以再开下面这一段：
    # eval_topk_uncertainty_event_level(
    #     model_name="struct",
    #     data_mode_base="icews_real_topk500",
    #     K_list=[100, 500, 1000],
    #     logs_dir="logs",
    #     out_csv="logs/topk_uncertainty_event_struct.csv",
    #     out_fig_dir="logs/figs_topk_unc_struct",
    # )
