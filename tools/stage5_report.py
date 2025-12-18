import argparse
import os
import numpy as np

def _safe_get(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default

def load_npz(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing npz: {path}")
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}

def pick_metrics(z: dict, prefix_candidates):
    out = {}
    out["test_nll_all"]  = float(_safe_get(z, [f"{p}test_nll_all"  for p in prefix_candidates] + ["test_nll_all","nll_all"], np.nan))
    out["test_nll_seen"] = float(_safe_get(z, [f"{p}test_nll_seen" for p in prefix_candidates] + ["test_nll_seen","nll_seen"], np.nan))
    out["test_nll_ood"]  = float(_safe_get(z, [f"{p}test_nll_ood"  for p in prefix_candidates] + ["test_nll_ood","nll_ood"], np.nan))

    out["test_rmse_all"]  = float(_safe_get(z, [f"{p}test_rmse_all"  for p in prefix_candidates] + ["test_rmse_all","rmse_all"], np.nan))
    out["test_rmse_seen"] = float(_safe_get(z, [f"{p}test_rmse_seen" for p in prefix_candidates] + ["test_rmse_seen","rmse_seen"], np.nan))
    out["test_rmse_ood"]  = float(_safe_get(z, [f"{p}test_rmse_ood"  for p in prefix_candidates] + ["test_rmse_ood","rmse_ood"], np.nan))

    out["test_mae_all"]  = float(_safe_get(z, [f"{p}test_mae_all"  for p in prefix_candidates] + ["test_mae_all","mae_all"], np.nan))
    out["test_mae_seen"] = float(_safe_get(z, [f"{p}test_mae_seen" for p in prefix_candidates] + ["test_mae_seen","mae_seen"], np.nan))
    out["test_mae_ood"]  = float(_safe_get(z, [f"{p}test_mae_ood"  for p in prefix_candidates] + ["test_mae_ood","mae_ood"], np.nan))

    out["seen_sum"] = int(_safe_get(z, ["seen_sum","seen_test_sum","n_seen"], -1))
    out["ood_sum"]  = int(_safe_get(z, ["ood_sum","ood_test_sum","n_ood"], -1))
    return out

def fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.4f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="icews_real_topk500")
    ap.add_argument("--logdir", default="logs")
    args = ap.parse_args()

    core_path   = os.path.join(args.logdir, f"gc_tpp_core_{args.mode}.npz")
    struct_path = os.path.join(args.logdir, f"gc_tpp_struct_{args.mode}.npz")

    core_z   = load_npz(core_path)
    struct_z = load_npz(struct_path)

    core_m   = pick_metrics(core_z,   prefix_candidates=["core_",""])
    struct_m = pick_metrics(struct_z, prefix_candidates=["struct_",""])

    print("\n# Stage-5 Report (Core vs Struct)")
    print(f"- mode: {args.mode}")
    print(f"- core_npz:   {core_path}")
    print(f"- struct_npz: {struct_path}\n")

    headers = [
        "Model",
        "Test NLL(all/Seen/OOD)",
        "RMSE(logΔt) all/Seen/OOD",
        "MAE(logΔt) all/Seen/OOD",
        "Seen/OOD (if saved)"
    ]

    def row(name, m):
        return [
            name,
            f"{fmt(m['test_nll_all'])} / {fmt(m['test_nll_seen'])} / {fmt(m['test_nll_ood'])}",
            f"{fmt(m['test_rmse_all'])} / {fmt(m['test_rmse_seen'])} / {fmt(m['test_rmse_ood'])}",
            f"{fmt(m['test_mae_all'])} / {fmt(m['test_mae_seen'])} / {fmt(m['test_mae_ood'])}",
            f"{m['seen_sum']} / {m['ood_sum']}" if (m["seen_sum"] >= 0 and m["ood_sum"] >= 0) else "NA",
        ]

    rows = [row("Core / continuous", core_m), row("Struct", struct_m)]

    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        print("| " + " | ".join(r) + " |")

    print("\n[DEBUG] If you want to see npz keys:")
    print(f"  python - <<'PY'\\nimport numpy as np\\nz=np.load(r'{core_path}', allow_pickle=True)\\nprint(z.files)\\nPY")

if __name__ == "__main__":
    main()
