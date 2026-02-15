
import os, json, argparse, sys
sys.path.insert(0, "baselines/easytpp/EasyTemporalPointProcess")

from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

def build_cfg(cfg_yaml: str, exp_id: str):
    # 最稳：两种参数名一起传；不行再降级
    try:
        return Config.build_from_yaml_file(cfg_yaml, exp_id=exp_id, experiment_id=exp_id)
    except TypeError:
        try:
            return Config.build_from_yaml_file(cfg_yaml, experiment_id=exp_id)
        except TypeError:
            return Config.build_from_yaml_file(cfg_yaml, exp_id=exp_id)

def pick_loader(runner, split: str):
    dl = runner._data_loader
    if split == "train":
        return dl.train_loader()
    if split == "dev":
        return dl.valid_loader()
    if split == "test":
        return dl.test_loader()
    raise SystemExit(f"[FATAL] unknown split={split}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_yaml", required=True)
    ap.add_argument("--experiment_id", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--split", required=True, choices=["train","dev","test"])
    args = ap.parse_args()

    cfg = build_cfg(args.config_yaml, args.experiment_id)
    runner = Runner.build_from_config(cfg, unique_model_dir=args.run_dir)

    ckpt = os.path.join(args.run_dir, "models", "saved_model")
    if not os.path.isfile(ckpt):
        raise SystemExit(f"[FATAL] missing ckpt file: {ckpt}")

    runner._load_model(ckpt)

    loader = pick_loader(runner, args.split)
    metrics = runner._evaluate_model(loader)   # OrderedDict(loglike, num_events)

    if "loglike" in metrics and "num_events" in metrics:
        ll = float(metrics["loglike"])
        n = float(metrics["num_events"])
        metrics["nll_per_event"] = (-ll / n) if n > 0 else None

    out = os.path.join(args.run_dir, f"eval_{args.split}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[OK] wrote:", out)
    print(metrics)

if __name__ == "__main__":
    main()
