
import os, json, argparse, sys

sys.path.insert(0, "baselines/easytpp/EasyTemporalPointProcess")

from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

def build_cfg(cfg_yaml: str, exp_id: str):
    # 兼容不同版本的参数名
    try:
        return Config.build_from_yaml_file(cfg_yaml, experiment_id=exp_id)
    except TypeError:
        return Config.build_from_yaml_file(cfg_yaml, exp_id=exp_id)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_yaml", required=True)
    ap.add_argument("--experiment_id", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train","dev","test"])
    args = ap.parse_args()

    model_dir = os.path.join(args.run_dir, "models")
    ckpt_path = os.path.join(model_dir, "saved_model")
    if not os.path.isfile(ckpt_path):
        raise SystemExit(f"[FATAL] missing checkpoint file: {ckpt_path}")

    cfg = build_cfg(args.config_yaml, args.experiment_id)

    # 尽量把输出固定在同一个 run_dir，避免你又“迷路”
    try:
        cfg.base_config.base_dir = args.run_dir
    except Exception:
        pass

    runner = Runner.build_from_config(cfg, unique_model_dir=args.run_dir)

    # 你的 EasyTPP：_load_model 需要“文件路径”，不是目录
    runner._load_model(ckpt_path)

    # 关键：不要用 runner.evaluate() 的返回值（它会强行 metric['rmse']）
    # 我们自己取 loader，然后调用内部 _evaluate_model 得到 metric dict
    dl = runner._data_loader
    if args.split == "train":
        loader = dl.train_loader()
    elif args.split == "dev":
        loader = dl.valid_loader()
    else:
        loader = dl.test_loader()

    metric = runner._evaluate_model(loader)

    out_path = os.path.join(args.run_dir, f"eval_{args.split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metric, f, ensure_ascii=False, indent=2)

    print("[OK] wrote:", out_path)
    print("[METRIC]", metric)

if __name__ == "__main__":
    main()
