import os, argparse, pickle, json
import numpy as np

def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}

def pick(pack, candidates):
    for k in candidates:
        if k in pack:
            return pack[k], k
    raise KeyError(f"Cannot find any of keys: {candidates}. Available={list(pack.keys())}")

def to_seq_list(x, *, flat_mode=False, dtype=np.float64):
    """
    flat_mode=True: 1D array of length N 代表 N 个样本 -> 生成 N 条长度=1的序列
    flat_mode=False:
      - object array: list of sequences
      - ndim==2: each row is a sequence
      - ndim==1: treat as ONE sequence（本项目这里不用）
    """
    if isinstance(x, np.ndarray) and x.dtype == object:
        return [np.asarray(e, dtype=dtype) for e in x.tolist()]

    x = np.asarray(x)
    if x.ndim == 0:
        return [x.astype(dtype)]

    if x.ndim == 1:
        if flat_mode:
            return [np.asarray([x[i]], dtype=dtype) for i in range(len(x))]
        else:
            return [x.astype(dtype)]  # one long sequence

    if x.ndim == 2:
        return [row.astype(dtype) for row in x]

    # fallback
    return [np.asarray(e, dtype=dtype) for e in list(x)]

def ensure_dt(seq_list, key):
    if "log" in key.lower():
        return [np.exp(s.astype(np.float64)) for s in seq_list]
    return [s.astype(np.float64) for s in seq_list]

def build_seq(dt_seq, mark_seq, eps=1e-8, time_mode="dt"):
    seq=[]
    t=0.0
    for d,m in zip(dt_seq, mark_seq):
        d=float(d); d=max(d, eps)
        d_use = float(np.log(d)) if time_mode=="log_dt" else d
        t += d_use
        seq.append({"time_since_start": float(t),
                    "time_since_last_event": float(d_use),
                    "type_event": int(m)})
    return seq

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pack_npz", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--time_mode", choices=["dt","log_dt"], default="dt")
    args=ap.parse_args()

    pack=load_npz(args.pack_npz)

    dt_train_raw, k_dt_train = pick(pack, ["dt_train","dts_train","time_gap_train","delta_t_train","log_dt_train"])
    dt_val_raw,   k_dt_val   = pick(pack, ["dt_val","dts_val","time_gap_val","delta_t_val","log_dt_val"])
    dt_test_raw,  k_dt_test  = pick(pack, ["dt_test","dts_test","time_gap_test","delta_t_test","log_dt_test"])

    mk_train_raw, k_m_train = pick(pack, ["mark_train","marks_train","type_train","event_type_train"])
    mk_val_raw,   k_m_val   = pick(pack, ["mark_val","marks_val","type_val","event_type_val"])
    mk_test_raw,  k_m_test  = pick(pack, ["mark_test","marks_test","type_test","event_type_test"])

    # 关键：本 pack 是 1D 的“样本列表”，长度与 seen_* 一致 => flat_mode=True
    def is_flat(split_name, x):
        key = f"seen_{split_name}"
        return (isinstance(x, np.ndarray) and x.ndim==1 and x.dtype!=object and key in pack and len(pack[key])==len(x))

    flat_train = is_flat("train", dt_train_raw)
    flat_val   = is_flat("val",   dt_val_raw)
    flat_test  = is_flat("test",  dt_test_raw)

    dt_train = ensure_dt(to_seq_list(dt_train_raw, flat_mode=flat_train, dtype=np.float64), k_dt_train)
    dt_val   = ensure_dt(to_seq_list(dt_val_raw,   flat_mode=flat_val,   dtype=np.float64), k_dt_val)
    dt_test  = ensure_dt(to_seq_list(dt_test_raw,  flat_mode=flat_test,  dtype=np.float64), k_dt_test)

    mk_train = to_seq_list(mk_train_raw, flat_mode=flat_train, dtype=np.int64)
    mk_val   = to_seq_list(mk_val_raw,   flat_mode=flat_val,   dtype=np.int64)
    mk_test  = to_seq_list(mk_test_raw,  flat_mode=flat_test,  dtype=np.int64)

    assert len(dt_train)==len(mk_train), (len(dt_train), len(mk_train))
    assert len(dt_val)==len(mk_val), (len(dt_val), len(mk_val))
    assert len(dt_test)==len(mk_test), (len(dt_test), len(mk_test))

    mx=0
    for s in mk_train+mk_val+mk_test:
        if len(s)>0:
            mx=max(mx, int(np.max(s)))
    num_event_types = mx + 1
    pad_token_id = num_event_types

    train=[build_seq(d,m,eps=args.eps,time_mode=args.time_mode) for d,m in zip(dt_train,mk_train)]
    dev  =[build_seq(d,m,eps=args.eps,time_mode=args.time_mode) for d,m in zip(dt_val,mk_val)]
    test =[build_seq(d,m,eps=args.eps,time_mode=args.time_mode) for d,m in zip(dt_test,mk_test)]

    os.makedirs(args.out_dir, exist_ok=True)
    for name,obj in [("train.pkl",train),("dev.pkl",dev),("test.pkl",test)]:
        with open(os.path.join(args.out_dir,name),"wb") as f:
            pickle.dump(obj,f)

    with open(os.path.join(args.out_dir,"data_spec.json"),"w") as f:
        json.dump({"num_event_types": int(num_event_types), "pad_token_id": int(pad_token_id)}, f, indent=2)

    meta = {
        "time_mode": args.time_mode,
        "eps": float(args.eps),
        "flat_mode": {"train": flat_train, "val": flat_val, "test": flat_test},
        "keys": {"dt": [k_dt_train,k_dt_val,k_dt_test], "mark":[k_m_train,k_m_val,k_m_test]},
        "lens": {"train": len(train), "dev": len(dev), "test": len(test),
                 "len_train0": len(train[0]) if train else None,
                 "len_test0": len(test[0]) if test else None}
    }
    with open(os.path.join(args.out_dir,"meta_pack2pkl.json"),"w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] wrote pkls -> {args.out_dir}")
    print(f"     flat_mode(train/val/test)={flat_train}/{flat_val}/{flat_test}")
    print(f"     lens(seq) train/dev/test = {len(train)}/{len(dev)}/{len(test)} ; first_len(train)={len(train[0]) if train else 0}")
    print(f"     keys(dt/mark): {k_dt_train}/{k_m_train} | {k_dt_val}/{k_m_val} | {k_dt_test}/{k_m_test}")

if __name__=="__main__":
    main()
