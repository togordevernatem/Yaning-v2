
import os, json, argparse, sys, yaml
sys.path.insert(0, "baselines/easytpp/EasyTemporalPointProcess")

from easy_tpp.config_factory.runner_config import RunnerConfig
from easy_tpp.runner import Runner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_yaml", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    print(f"[INFO] Processing Run Dir: {args.run_dir}")

    # 1. 检查模型文件位置
    model_dir = os.path.join(args.run_dir, "models")
    if not os.path.exists(os.path.join(model_dir, "saved_model")):
        if os.path.exists(os.path.join(args.run_dir, "saved_model")):
            model_dir = args.run_dir
        else:
            print(f"[FATAL] No saved_model found in {model_dir} or {args.run_dir}")
            sys.exit(1)
            
    # 2. 读取 YAML
    with open(args.out_yaml, "r") as f:
        raw_config = yaml.safe_load(f)
    
    # 3. 【核心修复】智能适配 YAML 结构
    # 情况A: YAML里包含 ExpID (标准输入格式)
    if args.experiment_id in raw_config:
        print("[INFO] YAML structure: Nested (Standard)")
        final_config_dict = raw_config
    # 情况B: YAML是扁平的 (EasyTPP 输出格式) -> 我们手动套壳！
    elif 'model_config' in raw_config:
        print(f"[INFO] YAML structure: Flat (Output Format). Wrapping with ID: {args.experiment_id}")
        final_config_dict = {args.experiment_id: raw_config}
    else:
        print(f"[ERROR] YAML format unrecognized. Keys: {list(raw_config.keys())}")
        sys.exit(1)

    # 4. 解析配置
    cfg = RunnerConfig.parse_from_yaml_config(final_config_dict, exp_id=args.experiment_id)

    # 5. 构建 Runner
    runner = Runner.build_from_config(cfg, unique_model_dir=args.run_dir)
    
    # 6. 加载模型
    print(f"[INFO] Loading model from {model_dir}")
    runner._load_model(model_dir)

    # 7. 评估
    print(f"[INFO] Evaluating split: {args.split}")
    try:
        res = runner.evaluate(split=args.split)
    except TypeError:
        res = runner.evaluate()

    # 8. 保存结果
    out_path = os.path.join(args.run_dir, f"eval_{args.split}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(f"[OK] Saved metrics to: {out_path}")
    print("[RESULT]", res)

if __name__ == "__main__":
    main()
