
import os, json, argparse, sys, yaml
sys.path.insert(0, "baselines/easytpp/EasyTemporalPointProcess")

from easy_tpp.config_factory.runner_config import RunnerConfig
from easy_tpp.runner import Runner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_yaml", required=True)
    ap.add_argument("--experiment_id", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--split", default="test", choices=["train","dev","test"])
    args = ap.parse_args()

    model_dir = os.path.join(args.run_dir, "models")
    if not os.path.exists(os.path.join(model_dir, "saved_model")):
        raise SystemExit(f"[FATAL] missing {model_dir}/saved_model")

    raw = yaml.safe_load(open(args.out_yaml, "r"))

    # bypass Config.build_from_yaml_file (pipeline_config_id often missing)
    cfg = RunnerConfig.parse_from_yaml_config(raw, exp_id=args.experiment_id)

    runner = Runner.build_from_config(cfg, unique_model_dir=args.run_dir)

    # this EasyTPP requires model_dir argument
    runner._load_model(model_dir)

    # Some versions ignore split arg; still ok.
    try:
        res = runner.evaluate(split=args.split)
    except TypeError:
        res = runner.evaluate()

    out_path = os.path.join(args.run_dir, f"eval_{args.split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("[OK] wrote:", out_path)
    print("[METRICS]", res)

if __name__ == "__main__":
    main()
