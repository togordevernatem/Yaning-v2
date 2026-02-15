import os, glob, pickle
import numpy as np

PACK = "baselines/packs/icews_real_topk500_K500_tr0.70_va0.15/baseline_pack.npz"
OUT_DIR = "baselines/easytpp/data/icews_real_topk500_K500_seqnode"
os.makedirs(OUT_DIR, exist_ok=True)

z = np.load(PACK, allow_pickle=True)

def must(k):
    if k not in z.files:
        raise SystemExit(f"[FATAL] missing key in pack: {k}")
    return z[k]

idx_train = must("idx_train").astype(np.int64).reshape(-1)
idx_val   = must("idx_val").astype(np.int64).reshape(-1)
idx_test  = must("idx_test").astype(np.int64).reshape(-1)

t_train = must("event_time_train").astype(np.float64).reshape(-1)
t_val   = must("event_time_val").astype(np.float64).reshape(-1)
t_test  = must("event_time_test").astype(np.float64).reshape(-1)

m_train = must("mark_train").astype(np.int64).reshape(-1)
m_val   = must("mark_val").astype(np.int64).reshape(-1)
m_test  = must("mark_test").astype(np.int64).reshape(-1)

N_needed = int(max(idx_train.max(), idx_val.max(), idx_test.max()) + 1)
print("[OK] pack loaded:", PACK)
print("[OK] N_needed (max idx + 1) =", N_needed)

# --------- 自动找全局事件表（src/dst/time） ----------
def score_npz(path):
    try:
        zz = np.load(path, allow_pickle=True)
        keys = set(zz.files)
        # 常见命名
        src_keys = [k for k in keys if k.lower() in ("src","u","head","sub","sender","node_u","event_src")]
        dst_keys = [k for k in keys if k.lower() in ("dst","v","tail","obj","receiver","node_v","event_dst")]
        time_keys= [k for k in keys if k.lower() in ("t","ts","time","timestamp","event_time","event_time_all","times")]
        if not src_keys or not dst_keys:
            return (-1, None)
        # 取第一个候选
        s0, d0 = src_keys[0], dst_keys[0]
        src = np.asarray(zz[s0]).reshape(-1)
        dst = np.asarray(zz[d0]).reshape(-1)
        if len(src) < N_needed or len(dst) < N_needed:
            return (-1, None)
        # time 可选，但有更好
        sc = 0
        sc += 10
        sc += 5 if time_keys else 0
        # 更像事件表：长度接近 N_needed
        sc += 3 if (len(src) <= N_needed*2) else 0
        return (sc, {"path": path, "src_k": s0, "dst_k": d0, "time_k": (time_keys[0] if time_keys else None)})
    except Exception:
        return (-1, None)

cands = []
for p in glob.glob("**/*.npz", recursive=True):
    low = p.lower()
    # 优先缩小范围：topk500/k500/icews 相关
    if ("topk500" not in low and "icews" not in low and "k500" not in low):
        continue
    sc, meta = score_npz(p)
    if sc > 0:
        cands.append((sc, meta))

cands.sort(key=lambda x: x[0], reverse=True)
if not cands:
    raise SystemExit("[FATAL] cannot find any global event table (.npz) that contains src/dst with length >= N_needed.\n"
                     "You likely have it as .pkl/.pt or under different naming. We'll extend search after you show files in baselines/packs/...")

best = cands[0][1]
print("[OK] picked global table:", best["path"])
print("     src key =", best["src_k"])
print("     dst key =", best["dst_k"])
print("     time key=", best["time_k"])

g = np.load(best["path"], allow_pickle=True)
SRC_ALL = np.asarray(g[best["src_k"]]).reshape(-1).astype(np.int64)
DST_ALL = np.asarray(g[best["dst_k"]]).reshape(-1).astype(np.int64)
T_ALL   = None
if best["time_k"] is not None:
    T_ALL = np.asarray(g[best["time_k"]]).reshape(-1).astype(np.float64)

# 如果全局表没有 time，就用 pack 里的 split time（不会影响 node 分组，只影响组内排序）
def build_split(idx, t_split, m_split, name):
    # node 用 src（优先），没有就用 dst
    src = SRC_ALL[idx]
    dst = DST_ALL[idx]
    t   = t_split.astype(np.float64)
    m   = m_split.astype(np.int64)
    eid = np.arange(len(idx), dtype=np.int64)  # 这是“split 内顺序锚点”，必须保留

    # 分组：node-level（先用 src；如果你以后想用 dst 或 pair，可改这里）
    groups = {}
    for i, node in enumerate(src):
        groups.setdefault(int(node), []).append(i)

    seqs = []
    valid_eids = []
    for node, pos_list in groups.items():
        if len(pos_list) < 2:
            continue
        pos_list = sorted(pos_list, key=lambda i: float(t[i]))
        t0 = float(t[pos_list[0]])
        prev = t0
        seq = []
        for j, i in enumerate(pos_list):
            ti = float(t[i])
            dti = (ti - prev) if j > 0 else 0.0
            prev = ti
            ev = {
                "time_since_start": ti - t0,
                "time_since_last_event": dti,
                "type_event": int(m[i]),
                "eid": int(eid[i]),
            }
            seq.append(ev)
            if j > 0:
                valid_eids.append(int(eid[i]))
        seqs.append(seq)

    valid_eids = np.array(sorted(set(valid_eids)), dtype=np.int64)

    dim_process = int(max(m_train.max(), m_val.max(), m_test.max()) + 1)
    out = {"dim_process": dim_process, name: seqs}
    with open(f"{OUT_DIR}/{name}.pkl", "wb") as f:
        pickle.dump(out, f)
    np.save(f"{OUT_DIR}/valid_eids_{name}.npy", valid_eids)

    lens = [len(s) for s in seqs]
    print(f"[OK] {name}: seqs={len(seqs)}  valid_eids={len(valid_eids)}  dim_process={dim_process}")
    if lens:
        lens_sorted = sorted(lens)
        print(f"     len_min={lens_sorted[0]}  len_p50={lens_sorted[len(lens_sorted)//2]}  len_max={lens_sorted[-1]}")
    else:
        print("     [WARN] no sequences built (all groups <2). Try grouping by dst or pair.")

build_split(idx_train, t_train, m_train, "train")
build_split(idx_val,   t_val,   m_val,   "valid")
build_split(idx_test,  t_test,  m_test,  "test")
print("[DONE] wrote:", OUT_DIR)
