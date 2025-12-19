import argparse
from models.gc_tpp_continuous import run_gc_tpp_continuous
from models.gc_tpp_struct import run_gc_tpp_struct


def parse_args():
    parser = argparse.ArgumentParser(description="GC-TPP Temporal Toy / ICEWS Runner")

    parser.add_argument(
        "--model",
        type=str,
        default="gc_tpp_continuous",
        choices=["gc_tpp_continuous", "gc_tpp_struct"],
        help=(
            "选择要运行的模型："
            "gc_tpp_continuous（Core 连续版） 或 "
            "gc_tpp_struct（Struct 结构版）"
        ),
    )

    parser.add_argument(
        "--data_mode",
        type=str,
        default="toy",
        choices=["toy", "icews_real", "icews_real_topk500"],
        help="数据模式：toy 或 icews_real 或 icews_real_topk500",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.model == "gc_tpp_continuous":
        print("============================================================")
        print(f"[INFO] Selected model: gc_tpp_continuous (Core)")
        print(f"[INFO] Data mode: {args.data_mode}")
        print("============================================================")
        run_gc_tpp_continuous(data_mode=args.data_mode)

    elif args.model == "gc_tpp_struct":
        print("============================================================")
        print(f"[INFO] Selected model: gc_tpp_struct (Struct)")
        print(f"[INFO] Data mode: {args.data_mode}")
        print("============================================================")
        run_gc_tpp_struct(data_mode=args.data_mode)

    else:
        raise ValueError(f"Unknown model: {args.model}")


if __name__ == "__main__":
    main()
