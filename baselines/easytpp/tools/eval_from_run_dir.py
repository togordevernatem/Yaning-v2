import os, json, glob, argparse, sys

sys.path.insert(0, "baselines/easytpp/EasyTemporalPointProcess")

from easy_tpp.config_factory.runner_config import RunnerConfig
from easy_tpp.runner import Runner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    ys = glob.glob(os.path.join(args.run_dir, "*_output.yaml"))
    if not ys:
        raise SystemExit(f"[FATAL] no *_output.yaml in {args.run_dir}")
    out_yaml = sorted(ys)[0]
    print("[USING out_yaml]", out_yaml)

    model_dir = os.path.join(args.run_dir, "models")
    if not os.path.exists(os.path.join(model_dir, "saved_model")):
        raise SystemExit(f"[FATAL] no models/saved_model under {args.run_dir}")
    print("[USING model_dir]", model_dir)

    runner_cfg = RunnerConfig.build_from_yaml_file(out_yaml)
    runner = Runner.build_from_config(runner_cfg, unique_model_dir=args.run_dir)

    # IMPORTANT: this version requires model_dir argument
    runner._load_model(model_dir)

    res = runner.evaluate()
    out_path = os.path.join(args.run_dir, "eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"result": res}, f, ensure_ascii=False, indent=2)

    print("[OK] wrote", out_path)
    print("[RESULT]", res)

if __name__ == "__main__":
    main()
