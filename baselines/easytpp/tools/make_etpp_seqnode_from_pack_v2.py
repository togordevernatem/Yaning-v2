import os, pickle
import numpy as np

PACK="baselines/packs/icews_real_topk500_K500_tr0.70_va0.15_v2/baseline_pack.npz"
OUT_DIR="baselines/easytpp/data/icews_real_topk500_K500_seqnode"
os.makedirs(OUT_DIR, exist_ok=True)

z=np.load(PACK, allow_pickle=True)
need=["src","dst","ev_type","event_times","dt","idx_train","idx_val","idx_test"]
for k in need:
    if k not in z.files:
        raise SystemExit(f"[FATAL] missing key in v2 pack: {k}")

src_all=z["src"].astype(np.int64).reshape(-1)
dst_all=z["dst"].astype(np.int64).reshape(-1)
typ_all=z["ev_type"].astype(np.int64).reshape(-1)
t_all  =z["event_times"].astype(np.float64).reshape(-1)

idx_tr=z["idx_train"].astype(np.int64).reshape(-1)
idx_va=z["idx_val"].astype(np.int64).reshape(-1)
idx_te=z["idx_test"].astype(np.int64).reshape(-1)

dim_process=int(typ_all.max()+1)

def build_split(name, idx):
    # split 内的“事件顺序锚点”：eid = 0..len(idx)-1（用于后续对齐评估）
    eid=np.arange(len(idx), dtype=np.int64)

    s=src_all[idx]
    t=t_all[idx]
    m=typ_all[idx]

    # node-level: 用 src 分组（最常用、最稳）
    groups={}
    for i,node in enumerate(s):
        groups.setdefault(int(node), []).append(i)

    seqs=[]
    valid=[]
    for node, pos_list in groups.items():
        if len(pos_list)<2:
            continue
        pos_list=sorted(pos_list, key=lambda i: float(t[i]))
        t0=float(t[pos_list[0]])
        prev=t0
        seq=[]
        for j,i in enumerate(pos_list):
            ti=float(t[i])
            dti=ti-prev if j>0 else 0.0
            prev=ti
            ev={
                "time_since_start": ti-t0,
                "time_since_last_event": dti,
                "type_event": int(m[i]),
                "eid": int(eid[i]),
            }
            seq.append(ev)
            if j>0:
                valid.append(int(eid[i]))
        seqs.append(seq)

    valid=np.array(sorted(set(valid)), dtype=np.int64)
    out={"dim_process": dim_process, name: seqs}
    with open(f"{OUT_DIR}/{name}.pkl","wb") as f:
        pickle.dump(out,f)
    np.save(f"{OUT_DIR}/valid_eids_{name}.npy", valid)

    lens=sorted([len(s) for s in seqs])
    print(f"[OK] {name}: seqs={len(seqs)} valid_eids={len(valid)} dim_process={dim_process}")
    if lens:
        print(f"     len_min={lens[0]} len_p50={lens[len(lens)//2]} len_max={lens[-1]}")
    else:
        print("     [WARN] no sequences built (all groups <2)")

build_split("train", idx_tr)
build_split("valid", idx_va)
build_split("test",  idx_te)
print("[DONE] wrote:", OUT_DIR)
