import argparse, math, json
import numpy as np

def mean_rmse_mae(log_dt, pred):
    diff = log_dt - float(pred)
    rmse = math.sqrt(float(np.mean(diff * diff)))
    mae = float(np.mean(np.abs(diff)))
    return rmse, mae

def lognormal_nll_repo(log_dt, mu, log_sigma):
    # 对齐你仓库口径：0.5*((x-mu)/sigma)^2 + x + log_sigma
    sigma = math.exp(float(log_sigma))
    return 0.5 * ((log_dt - float(mu)) / sigma) ** 2 + log_dt + float(log_sigma)

def masked_mean(x, m):
    x = np.asarray(x)
    m = np.asarray(m).astype(bool)
    if m.sum() == 0:
        return float("nan")
    return float(x[m].mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="baseline_pack.npz")
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--mode", choices=["global_median"], default="global_median")
    ap.add_argument("--category", default="Statistical")
    ap.add_argument("--model", default="Global Median")
    args = ap.parse_args()

    z = np.load(args.pack, allow_pickle=True)
    dt_tr = z["dt_train"].astype(np.float64)
    dt_te = z["dt_test"].astype(np.float64)
    seen_te = z["seen_test"].astype(bool)
    ood_te  = z["ood_test"].astype(bool)

    # clamp + log
    dt_tr = np.maximum(dt_tr, args.eps)
    dt_te = np.maximum(dt_te, args.eps)
    log_tr = np.log(dt_tr)
    log_te = np.log(dt_te)

    # Global Median baseline（按你这次脚本：mu=median(logdt_train), sigma=MAD*1.4826, log_sigma=log(sigma)）
    mu = float(np.median(log_tr))
    mad = float(np.median(np.abs(log_tr - mu)))
    sigma = max(mad * 1.4826, 1e-12)
    log_sigma = math.log(sigma)

    # NLL per-event (repo-style)
    nll_all = np.array([lognormal_nll_repo(x, mu, log_sigma) for x in log_te], dtype=np.float64)

    # metrics
    all_nll = float(nll_all.mean())
    seen_nll = masked_mean(nll_all, seen_te)
    ood_nll  = masked_mean(nll_all, ood_te)

    all_rmse, all_mae = mean_rmse_mae(log_te, mu)
    seen_rmse, seen_mae = mean_rmse_mae(log_te[seen_te], mu)
    ood_rmse,  ood_mae  = mean_rmse_mae(log_te[ood_te],  mu)

    # 你表里要 mean±std；但你当前 split 是确定性的，所以 std=0
    def fmt(x): return f"{x:.6f}±0.000000"

    row = [
        fmt(all_nll), fmt(all_rmse), fmt(all_mae),
        fmt(seen_nll), fmt(seen_rmse), fmt(seen_mae),
        fmt(ood_nll), fmt(ood_rmse), fmt(ood_mae),
    ]

    print("[PASTE-TO-TABLE]\t" + "\t".join([args.category, args.model] + row))

    out = {
        "category": args.category, "model": args.model,
        "mu": mu, "sigma": sigma, "log_sigma": log_sigma,
        "all": {"nll": all_nll, "rmse": all_rmse, "mae": all_mae},
        "seen": {"nll": seen_nll, "rmse": seen_rmse, "mae": seen_mae},
        "ood": {"nll": ood_nll, "rmse": ood_rmse, "mae": ood_mae},
    }
    print("[INFO] params:", json.dumps({"mu": mu, "sigma": sigma}, ensure_ascii=False))

if __name__ == "__main__":
    main()
