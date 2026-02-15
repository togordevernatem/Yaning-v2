#!/usr/bin/env python3
# tools/build_splits_and_global_median.py
import os, sys, math, glob, pathlib
import numpy as np
import torch

DATA = "icews_real_topk500_K500"
PROT = "protB"
SEED = 0
EPS = 1e-8   # 与论文保持一致；你也可改为 1e-6
OUT_DIR = f"logs"
os.makedirs(OUT_DIR, exist_ok=True)

# --- 1. import dataset class (try common paths used in this repo) ---
ds = None
try_paths = [
    "data.dataset_toy", "data.dataset", "dataset.dataset_toy", "dataset.dataset"
]
for p in try_paths:
    try:
        mod = __import__(p, fromlist=['GC_TPP_Dataset'])
        GC_TPP_Dataset = getattr(mod, "GC_TPP_Dataset")
        ds = GC_TPP_Dataset(
            snapshots_dir="data/snapshots",
            events_dir="data/events",
            T=20, N=10, F_in=3,
            device=torch.device("cpu"),
            save_to_disk=True,
            mode=DATA,
            truncate_icews_real_to=5000
        )
        print(f"[OK] GC_TPP_Dataset loaded via {p}")
        break
    except Exception as e:
        # print(f"[DEBUG] import {p} failed: {e}")
        continue

if ds is None:
    print("ERROR: cannot import GC_TPP_Dataset from tried paths. Paste the import error here.")
    sys.exit(2)

# --- 2. get dt and full length, build simple splits (repo uses 70/15/15 in logs) ---
# ds.dt exists per your logs
dt_arr = np.array(ds.dt).astype(float).flatten()
total_len = len(dt_arr)
print(f"[INFO] dt len = {total_len}, zeros (<=0) = {(dt_arr<=0).sum()}")

# simple split (this matches your repo usage/printing)
n = total_len
tr = int(n * 0.7)
va = int(n * 0.15)
idx_train = np.arange(0, tr, dtype=np.int64)
idx_val   = np.arange(tr, tr + va, dtype=np.int64)
idx_test  = np.arange(tr + va, n, dtype=np.int64)
print(f"[INFO] split sizes: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

# --- 3. try to obtain seen/ood flags from dataset ---
seen_mask = None
ood_mask = None

# 3a: preferred: ds.get_seen_ood_flags(...) -> dict with 'seen_test','ood_test' masks
if hasattr(ds, "get_seen_ood_flags"):
    try:
        flags = ds.get_seen_ood_flags(torch.tensor(idx_train), torch.tensor(idx_val), torch.tensor(idx_test))
        if isinstance(flags, dict):
            if "seen_test" in flags:
                seen_mask = np.asarray(flags["seen_test"]).astype(bool)
            if "ood_test" in flags:
                ood_mask = np.asarray(flags["ood_test"]).astype(bool)
        print("[INFO] get_seen_ood_flags returned; using those masks if present.")
    except Exception as e:
        print("[WARN] ds.get_seen_ood_flags failed:", e)

# 3b: fallback: ds.local.flags (some code prints local.flags.*)
if (seen_mask is None or ood_mask is None) and hasattr(ds, "local"):
    try:
        lf = getattr(ds, "local")
        if hasattr(lf, "flags"):
            flags_obj = lf.flags
            # try common names
            for name in ("seen_test","ood_test","seen_val","ood_val"):
                if hasattr(flags_obj, name):
                    val = getattr(flags_obj, name)
                    val = np.asarray(val)
                    # if mask aligns with test split length, use accordingly
                    if val.shape[0] == len(idx_test):
                        if name == "seen_test": seen_mask = val.astype(bool)
                        if name == "ood_test":  ood_mask  = val.astype(bool)
                    elif val.shape[0] == total_len:
                        # map to test portion
                        remap = val[idx_test]
                        if name == "seen_test": seen_mask = remap.astype(bool)
                        if name == "ood_test":  ood_mask  = remap.astype(bool)
            print("[INFO] obtained flags from ds.local.flags")
    except Exception as e:
        print("[WARN] ds.local.flags access failed:", e)

# 3c: last resort: if flags are still None, assume all test are OOD (like before)
if seen_mask is None:
    print("[WARN] seen_mask not found from dataset; defaulting to all False")
    seen_mask = np.zeros(len(idx_test), dtype=bool)
if ood_mask is None:
    print("[WARN] ood_mask not found from dataset; defaulting to all True")
    ood_mask = np.ones(len(idx_test), dtype=bool)

# sanity: ensure masks lengths match idx_test
if len(seen_mask) != len(idx_test):
    # try to reshape if equals total_len
    if len(seen_mask) == total_len:
        seen_mask = seen_mask[idx_test]
        print("[INFO] remapped seen_mask from full len -> test len")
    else:
        # truncate/pad
        seen_mask = np.resize(seen_mask, len(idx_test))
        print("[WARN] resized seen_mask to match test len")

if len(ood_mask) != len(idx_test):
    if len(ood_mask) == total_len:
        ood_mask = ood_mask[idx_test]
        print("[INFO] remapped ood_mask from full len -> test len")
    else:
        ood_mask = np.resize(ood_mask, len(idx_test))
        print("[WARN] resized ood_mask to match test len")

seen_idx = idx_test[seen_mask]
ood_idx  = idx_test[ood_mask]

# --- 4. save splits file (npz) for later reuse ---
outpath = os.path.join(OUT_DIR, f"splits_{DATA}_{PROT}_seed{SEED}.npz")
np.savez(outpath, train_idx=idx_train, val_idx=idx_val, test_idx=idx_test,
         seen_idx=seen_idx, ood_idx=ood_idx, seen_mask=seen_mask, ood_mask=ood_mask)
print(f"[SAVED] {outpath} (train/val/test/seen/ood indices)")

# --- 5. compute Global Median baseline metrics (use train logdt to compute median mu) ---
dt_valid = dt_arr.copy()
# ignore non-positive values for log
valid_mask = dt_valid > 0
if valid_mask.sum() == 0:
    print("ERROR: no positive dt values found; cannot compute baseline.")
    sys.exit(3)

logdt = np.log(dt_valid[valid_mask] + EPS)

# we need logdt array indexed by full positions; create full_logdt with nan for <=0
full_logdt = np.full_like(dt_valid, np.nan, dtype=float)
full_logdt[valid_mask] = np.log(dt_valid[valid_mask] + EPS)

# median mu computed on train valid entries
train_mask_valid = np.isin(idx_train, np.where(valid_mask)[0])
train_valid_idx = idx_train[train_mask_valid]
if len(train_valid_idx) == 0:
    print("WARN: no positive dt in train split; using global median over all valid dt")
    mu = float(np.median(logdt))
    sigma = float(np.std(logdt, ddof=0)) + 1e-12
else:
    mu = float(np.median(full_logdt[train_valid_idx]))
    sigma = float(np.nanstd(full_logdt[train_valid_idx], ddof=0)) + 1e-12

def compute_metrics(indices):
    # indices relative to full sequence
    sel = [i for i in indices if valid_mask[i]]
    if len(sel) == 0:
        return (float("nan"), float("nan"), float("nan"))
    y = full_logdt[sel]
    rmse = float(np.sqrt(np.mean((y - mu)**2)))
    mae = float(np.mean(np.abs(y - mu)))
    # lognormal NLL per event: 0.5*((ln x - mu)/sigma)^2 + ln(x*sigma*sqrt(2pi))
    # but ln(x) = y, so nll = 0.5*((y-mu)/sigma)^2 + y + ln(sigma*sqrt(2pi))
    const = math.log(sigma * math.sqrt(2*math.pi) + 1e-30)
    nlls = 0.5 * ((y - mu)**2) / (sigma**2 + 1e-30) + y + const
    mean_nll = float(np.mean(nlls))
    return (mean_nll, rmse, mae)

nll_all, rm_all, ma_all = compute_metrics(idx_test)
nll_seen, rm_seen, ma_seen = compute_metrics(seen_idx)
nll_ood,  rm_ood,  ma_ood  = compute_metrics(ood_idx)

def fmt(x): 
    if math.isnan(x): return "nan±0.000000"
    return f"{x:.6f}±0.000000"

# print paste-to-table line (match your desired columns)
# Category row: Global Median -> All Time NLL, All RMSE, All MAE, Seen Time NLL, Seen RMSE, Seen MAE, OOD Time NLL, OOD RMSE, OOD MAE
row = [
    "Global Median",
    fmt(nll_all), fmt(rm_all), fmt(ma_all),
    fmt(nll_seen), fmt(rm_seen), fmt(ma_seen),
    fmt(nll_ood), fmt(rm_ood), fmt(ma_ood)
]
print("\n[PASTE-TO-TABLE]")
print("\t".join(row))

# debug print
print(f"\n[DEBUG] counts: train={len(idx_train)} val={len(idx_val)} test={len(idx_test)} seen_test={len(seen_idx)} ood_test={len(ood_idx)}")
print(f"[DEBUG] mu={mu:.6f} sigma={sigma:.6f} dt_positive_count={valid_mask.sum()}")

print("\n[INFO] Done.")
