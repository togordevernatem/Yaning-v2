
import os, json, argparse, sys
import torch
import numpy as np

# 引入 EasyTPP 路径
sys.path.insert(0, "baselines/easytpp/EasyTemporalPointProcess")

from easy_tpp.config_factory import Config
from easy_tpp.runner import Runner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--experiment_id", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    print(f"[INFO] 1. Config: {args.config_yaml}")
    
    # 1. 加载配置
    config = Config.build_from_yaml_file(args.config_yaml, experiment_id=args.experiment_id)

    # 2. 构建 Runner
    runner = Runner.build_from_config(config, unique_model_dir=args.run_dir)

    # 3. 定位 saved_model 文件
    p1 = os.path.join(args.run_dir, "models", "saved_model")
    p2 = os.path.join(args.run_dir, "saved_model")
    
    model_file_path = None
    if os.path.isfile(p1): model_file_path = p1
    elif os.path.isfile(p2): model_file_path = p2
    
    if not model_file_path:
        print(f"[FATAL] Cannot find 'saved_model' FILE.")
        sys.exit(1)

    print(f"[INFO] 2. Loading model from: {model_file_path}")
    runner.model_wrapper.restore(model_file_path)

    # 4. 【核心修复】绕过 runner.evaluate() 的 KeyError 坑
    print(f"[INFO] 3. Evaluating split: {args.split}")
    
    # A. 手动获取 DataLoader
    data_loader = runner.get_loader(args.split)
    
    # B. 直接调用内部评估函数 (返回 {'loglike': ..., 'num_events': ...})
    # 这样就不会触发 base_runner.py 里的 return metric['rmse'] 报错了
    metrics = runner._evaluate_model(data_loader)
    
    # 5. 后处理结果 (计算平均 NLL)
    # LogLike 通常是负的对数似然总和
    # NLL (per event) = -LogLike / NumEvents
    if 'loglike' in metrics and 'num_events' in metrics:
        ll = metrics['loglike']
        n = metrics['num_events']
        metrics['nll_per_event'] = -ll / n if n > 0 else 0.0
        print(f"\n>>> CALCULATED NLL: {metrics['nll_per_event']:.4f}")

    # 6. 保存结果
    out_json = os.path.join(args.run_dir, f"eval_{args.split}.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[SUCCESS] Metrics saved to: {out_json}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
