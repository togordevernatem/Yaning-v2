import os, json, copy, argparse
import numpy as np
import yaml

from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

def pick_metric(d: dict, keys):
    for k in keys:
        if k in d:
            return float(d[k])
    raise KeyError(f"Cannot find metric in {list(d.keys())}, tried {keys}")

def run_once(cfg_path, exp_id, seed, test_pkl=None, force_stage=None, model_dir=None):
    # load yaml -> override seed/test/stage -> write tmp -> run -> return (runner, results)
    with open(cfg_path, "r") as f:
        y = yaml.safe_load(f)

    exp = y[exp_id]
    exp["trainer_config"]["seed"] = int(seed)
    if force_stage is not None:
        exp["base_config"]["stage"] = force_stage
    if test_pkl is not None:
        dsid = exp["base_config"]["dataset_id"]
        y["data"][dsid]["test_dir"] = test_pkl

    tmp = f"/tmp/easytpp_{exp_id}_seed{seed}_{os.getpid()}.yaml"
    with open(tmp, "w") as f:
        yaml.safe_dump(y, f, sort_keys=False)

    cfg = Config.build_from_yaml_file(tmp, exp_id)
    runner = Runner.build_from_config(cfg.runner_config)

    if model_dir is not None:
        runner.set_model_dir(model_dir)

    model, results = runner.run()
    return runner, results

def extract_test_metrics(results: dict):
    # results 里通常有 train/valid/test 三段；我们取 test 段
    # 兼容不同版本：优先 results['test']，否则在 results 里找包含 'test' 的key
    if isinstance(results, dict) and "test" in results and isinstance(results["test"], dict):
        m = results["test"]
    else:
        # fallback: pick first dict-like
        m = None
        for k,v in results.items():
            if isinstance(v, dict) and ("rmse" in v or "mae" in v or "loss" in v or "nll" in v):
                m = v
                break
        if m is None:
            raise ValueError(f"Unrecognized results format: {results}")

    nll  = pick_metric(m, ["nll","loss","neg_log_likelihood","negative_log_likelihood"])
    rmse = pick_metric(m, ["rmse","time_rmse"])
    mae  = pick_metric(m, ["mae","time_mae"])
    return nll, rmse, mae

def mean_std(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return float(xs.mean()), float(xs.std(ddof=0))

def fmt(m, s):
    return f"{m:.6f}±{s:.6f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--exp_id", required=True)
    ap.add_argument("--data_dir", required=True)  # contains test.pkl/test_seen.pkl/test_ood.pkl
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()!=""]

    test_all  = os.path.join(args.data_dir, "test.pkl")
    test_seen = os.path.join(args.data_dir, "test_seen.pkl")
    test_ood  = os.path.join(args.data_dir, "test_ood.pkl")

    per_seed = []
    for sd in seeds:
        print(f"\n===== [SEED {sd}] train =====")
        runner_tr, res_tr = run_once(args.config, args.exp_id, sd, test_pkl=test_all, force_stage="train")
        model_dir = runner_tr.get_model_dir()
        print(f"[INFO] model_dir = {model_dir}")

        # eval all/seen/ood with the same checkpoint
        print(f"===== [SEED {sd}] eval ALL =====")
        _, res_all  = run_once(args.config, args.exp_id, sd, test_pkl=test_all,  force_stage="eval", model_dir=model_dir)
        print(f"===== [SEED {sd}] eval SEEN =====")
        _, res_seen = run_once(args.config, args.exp_id, sd, test_pkl=test_seen, force_stage="eval", model_dir=model_dir)
        print(f"===== [SEED {sd}] eval OOD =====")
        _, res_ood  = run_once(args.config, args.exp_id, sd, test_pkl=test_ood,  force_stage="eval", model_dir=model_dir)

        all_nll, all_rmse, all_mae = extract_test_metrics(res_all)
        sn_nll,  sn_rmse,  sn_mae  = extract_test_metrics(res_seen)
        od_nll,  od_rmse,  od_mae  = extract_test_metrics(res_ood)

        row = {
            "seed": sd,
            "all":  {"nll": all_nll, "rmse": all_rmse, "mae": all_mae},
            "seen": {"nll": sn_nll,  "rmse": sn_rmse,  "mae": sn_mae},
            "ood":  {"nll": od_nll,  "rmse": od_rmse,  "mae": od_mae},
            "model_dir": model_dir,
        }
        per_seed.append(row)

        with open(os.path.join(args.out_dir, f"seed{sd}.json"), "w") as f:
            json.dump(row, f, indent=2)

    # summarize
    def collect(part, key):
        return [r[part][key] for r in per_seed]

    summ = {}
    for part in ["all","seen","ood"]:
        summ[part] = {}
        for key in ["nll","rmse","mae"]:
            m,s = mean_std(collect(part,key))
            summ[part][key] = {"mean": m, "std": s}

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"seeds": seeds, "summary": summ}, f, indent=2)

    cells = [
        fmt(summ["all"]["nll"]["mean"],  summ["all"]["nll"]["std"]),
        fmt(summ["all"]["rmse"]["mean"], summ["all"]["rmse"]["std"]),
        fmt(summ["all"]["mae"]["mean"],  summ["all"]["mae"]["std"]),
        fmt(summ["seen"]["nll"]["mean"],  summ["seen"]["nll"]["std"]),
        fmt(summ["seen"]["rmse"]["mean"], summ["seen"]["rmse"]["std"]),
        fmt(summ["seen"]["mae"]["mean"],  summ["seen"]["mae"]["std"]),
        fmt(summ["ood"]["nll"]["mean"],  summ["ood"]["nll"]["std"]),
        fmt(summ["ood"]["rmse"]["mean"], summ["ood"]["rmse"]["std"]),
        fmt(summ["ood"]["mae"]["mean"],  summ["ood"]["mae"]["std"]),
    ]
    print("\n[PASTE-TO-TABLE]\tBaseline\tEasyTPP-RMTPP\t" + "\t".join(cells))

if __name__ == "__main__":
    main()
