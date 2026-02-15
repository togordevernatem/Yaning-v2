import os, argparse, pickle
import numpy as np

def load_npz(path):
    z=np.load(path, allow_pickle=True)
    return {k:z[k] for k in z.files}

def pick(pack, candidates):
    for k in candidates:
        if k in pack:
            return pack[k], k
    raise KeyError(f"Cannot find {candidates}. Available={list(pack.keys())}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--pack_npz", required=True)
    ap.add_argument("--data_dir", required=True)
    args=ap.parse_args()

    pack=load_npz(args.pack_npz)
    seen,k1 = pick(pack, ["seen_test","mask_seen_test","is_seen_test"])
    ood, k2 = pick(pack, ["ood_test","mask_ood_test","is_ood_test"])
    seen=np.asarray(seen).astype(bool)
    ood =np.asarray(ood).astype(bool)

    with open(os.path.join(args.data_dir,"test.pkl"),"rb") as f:
        test=pickle.load(f)

    assert len(test)==len(seen)==len(ood), f"len mismatch test={len(test)} seen={len(seen)} ood={len(ood)} keys={k1},{k2}"

    test_seen=[test[i] for i in range(len(test)) if seen[i]]
    test_ood =[test[i] for i in range(len(test)) if ood[i]]

    with open(os.path.join(args.data_dir,"test_seen.pkl"),"wb") as f:
        pickle.dump(test_seen,f)
    with open(os.path.join(args.data_dir,"test_ood.pkl"),"wb") as f:
        pickle.dump(test_ood,f)

    print(f"[OK] split keys={k1},{k2} -> test_seen={len(test_seen)} test_ood={len(test_ood)}")

if __name__=="__main__":
    main()
