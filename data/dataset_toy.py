import os
from typing import Tuple, Dict
import torch
import pandas as pd

from data.event_type_mapping import map_event_type_to_coarse, COARSE_LABELS
from data.icews0515_loader import load_icews0515_txt_dir


class GC_TPP_Dataset:
    """
    统一管理 GC-TPP 用到的 toy / ICEWS 事件数据集。

    支持多种 mode:
      - "toy"                        : data/events/toy_events.csv
      - "icews_toy"                  : data/events/icews_events_toy.csv
      - "icews_real"                 : data/events/icews_real.csv  （由 2010_2011_AllProtests 清洗而来）
      - "icews_real_topk500"         : 频次前 500 的事件（Top-K 数据，原始版本）
      - "icews_real_topk500_K100"    : 在 icews_real_topk500 基础上按频次抽取 Top-100 子集
      - "icews_real_topk500_K500"    : 在 icews_real_topk500 基础上按频次抽取 Top-500 子集
      - "icews_real_topk500_K1000"   : 在 icews_real_topk500 基础上按频次抽取 Top-1000 子集
      - "icews0515"                  : 从 data/events/icews0515/*.txt 读取事件时间与三元组

    注意：后面三个 *_KXXX 模式依赖 tools/make_topk_subsets.py 预先生成的
          icews_real_topk500_K100.csv 等文件。

    ✅ 关键说明（本文件最重要的口径）：
    - self.ev_type: 细粒度事件类型（CAMEO 整数 code）
    - self.coarse_types: coarse 类型（Other / Protest / Violence / Diplomacy / Economic ...）
    - Seen/OOD 的判定：按论文的 (i, j, c) —— 其中 c 必须是 coarse_type
      所以 flags 使用 (src, dst, coarse_type) 三元组，而不是 (src, dst, ev_type)。
    """

    def __init__(
        self,
        snapshots_dir: str,
        events_dir: str,
        T: int,
        N: int,
        F_in: int,
        device: torch.device,
        save_to_disk: bool = True,
        mode: str = "toy",
        truncate_icews_real_to: int = 5000,
    ):
        self.snapshots_dir = snapshots_dir
        self.events_dir = events_dir
        self.T = T
        self.N = N
        self.F_in = F_in
        self.device = device
        self.save_to_disk = save_to_disk
        self.mode = mode
        # 对 icews_real 生效；对 icews_real_topk500 / *_KXXX 我们在对应 loader 里单独控制
        self.truncate_icews_real_to = truncate_icews_real_to

        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.events_dir, exist_ok=True)

        # 1) 构建或加载图快照
        self.X_list, self.edge_index = self._build_or_load_snapshots()

        # 2) 构建或加载事件时间序列
        self.event_times, self.dt = self._build_or_load_events()

        # 3) 构建或加载每条事件的 (src, dst, ev_type)
        self.src, self.dst, self.ev_type = self._build_or_load_triplets()

        # 4) 基于 fine 类型构建 coarse 类型
        #    现在 self.ev_type 直接是 CAMEO 整数 code。
        self.coarse_types = self._build_coarse_types()

        # 5) Seen/OOD 标记（延迟生成）
        self._seen_ood_flags = None

    # =========================
    # 数据集切分
    # =========================
    def get_train_val_test_split(self, train_ratio=0.7, val_ratio=0.1):
        """
        划分事件数据集为训练、验证和测试集。
        返回值顺序与 gc_tpp_struct / gc_tpp_continuous 中的 build_*_with_flags 一致。
        """
        total = self.event_times.size(0)
        train_end = int(train_ratio * total)
        val_end = int((train_ratio + val_ratio) * total)
        indices = torch.arange(total, device=self.device)

        idx_train = indices[:train_end]
        idx_val = indices[train_end:val_end]
        idx_test = indices[val_end:]

        ev_time_train = self.event_times[idx_train]
        ev_time_val = self.event_times[idx_val]
        ev_time_test = self.event_times[idx_test]
        dt_train = self.dt[idx_train]
        dt_val = self.dt[idx_val]
        dt_test = self.dt[idx_test]

        return (
            idx_train.cpu(), idx_val.cpu(), idx_test.cpu(),
            ev_time_train, ev_time_val, ev_time_test,
            dt_train, dt_val, dt_test
        )

    # =========================
    # Seen / OOD 标记（核心口径）
    # =========================
    def get_seen_ood_flags(
        self,
        idx_train: torch.Tensor,
        idx_val: torch.Tensor,
        idx_test: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        ✅ Seen/OOD 判定口径（论文用）：

        基于 (src, dst, coarse_type) 三元组，生成 Train/Val/Test 的 Seen/OOD 标记。

        - Seen: test 三元组 (i, j, c) 在训练集出现过
        - OOD : test 三元组 (i, j, c) 在训练集从未出现过
        """
        if self._seen_ood_flags is not None:
            return self._seen_ood_flags

        device = self.device

        # 1) 收集 train 三元组 (src, dst, coarse_type)
        train_triplets = set()
        for i in idx_train:
            i_int = int(i.item())
            trip = (
                int(self.src[i_int].item()),
                int(self.dst[i_int].item()),
                int(self.coarse_types[i_int].item()),   # ✅ 用 coarse_types
            )
            train_triplets.add(trip)

        # helper：给定索引集合，生成 seen/ood 标记
        def _build_flags_for_split(indices):
            seen_flags = []
            ood_flags = []
            for i in indices:
                i_int = int(i.item())
                trip = (
                    int(self.src[i_int].item()),
                    int(self.dst[i_int].item()),
                    int(self.coarse_types[i_int].item()),  # ✅ 用 coarse_types
                )
                if trip in train_triplets:
                    seen_flags.append(1)
                    ood_flags.append(0)
                else:
                    seen_flags.append(0)
                    ood_flags.append(1)
            seen_tensor = torch.tensor(seen_flags, dtype=torch.bool, device=device)
            ood_tensor = torch.tensor(ood_flags, dtype=torch.bool, device=device)
            return seen_tensor, ood_tensor

        # 2) 为三个划分分别生成标记
        seen_train, ood_train = _build_flags_for_split(idx_train)
        seen_val,   ood_val   = _build_flags_for_split(idx_val)
        seen_test,  ood_test  = _build_flags_for_split(idx_test)

        self._seen_ood_flags = {
            "seen_train": seen_train,
            "seen_val":   seen_val,
            "seen_test":  seen_test,
            "ood_train":  ood_train,
            "ood_val":    ood_val,
            "ood_test":   ood_test,
        }
        return self._seen_ood_flags

    def get_triplets_split(
        self,
        idx_train: torch.Tensor,
        idx_val: torch.Tensor,
        idx_test: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        提供标准化的三元组切分接口，仅用于 debug / typed-proxy。

        - type_*：fine-grained CAMEO code（ev_type）
        - coarse_*：coarse type id（coarse_types）
        """
        try:
            src_train = self.src[idx_train].cpu().long()
            dst_train = self.dst[idx_train].cpu().long()
            type_train = self.ev_type[idx_train].cpu().long()
            coarse_train = self.coarse_types[idx_train].cpu().long()

            src_val = self.src[idx_val].cpu().long()
            dst_val = self.dst[idx_val].cpu().long()
            type_val = self.ev_type[idx_val].cpu().long()
            coarse_val = self.coarse_types[idx_val].cpu().long()

            src_test = self.src[idx_test].cpu().long()
            dst_test = self.dst[idx_test].cpu().long()
            type_test = self.ev_type[idx_test].cpu().long()
            coarse_test = self.coarse_types[idx_test].cpu().long()

            return {
                "src_train": src_train, "dst_train": dst_train, "type_train": type_train, "coarse_train": coarse_train,
                "src_val":   src_val,   "dst_val":   dst_val,   "type_val":   type_val,   "coarse_val":   coarse_val,
                "src_test":  src_test,  "dst_test":  dst_test,  "type_test":  type_test,  "coarse_test":  coarse_test,
                "reason": "ok",
            }
        except Exception as e:
            return {
                "src_train": None, "dst_train": None, "type_train": None, "coarse_train": None,
                "src_val":   None, "dst_val":   None, "type_val":   None, "coarse_val":   None,
                "src_test":  None, "dst_test":  None, "type_test":  None, "coarse_test":  None,
                "reason": f"error: {e}",
            }

    # =========================
    # coarse 类型支持
    # =========================
    def _build_coarse_types(self) -> torch.Tensor:
        """基于 fine 类型构建 coarse 类型。

        说明：
        - 旧 ICEWS CSV 路线里 fine type 是 CAMEO 整数 code，可以映射 coarse。
        - icews0515 的 fine type 是 relation 的重编号（0..R-1），无法直接用 CAMEO 映射。
          为了让现有 pipeline 不炸，这里把所有事件都归到 coarse=0（Other）。
          后续如果你有 relation->CAMEO 或 relation->coarse 的映射表，再改这里即可。
        """

        if self.mode == "icews0515":
            return torch.zeros_like(self.ev_type, dtype=torch.long, device=self.device)

        coarse_list = [
            map_event_type_to_coarse(int(t.item()))
            for t in self.ev_type
        ]
        return torch.tensor(coarse_list, dtype=torch.long, device=self.device)

    def get_event_coarse_types(self) -> torch.Tensor:
        """
        返回每条事件对应的 coarse 类型 ID，长度与 event_times / dt 一致。
        """
        return self.coarse_types

    # =========================
    # 事件处理部分（src, dst, ev_type）
    # =========================
    def _build_or_load_triplets(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_path, dst_path, type_path = self._triplets_pt_paths()

        # New: icews0515 从 txt 直接加载 (src, dst, ev_type)
        if self.mode == "icews0515":
            data = load_icews0515_txt_dir(
                dir_path=os.path.join(self.events_dir, "icews0515"),
                device=self.device,
                split="facts",
                time_mode="index",
            )
            return data.src, data.dst, data.ev_type

        # Top-K 模式：直接从对应 CSV 重建，不走 .pt 缓存路径
        if self.mode == "icews_real_topk500":
            csv_path = os.path.join(self.events_dir, "icews_real_topk500.csv")
            return self._load_triplets_from_icews_csv(csv_path, time_col="time")

        if self.mode == "icews_real_topk500_K100":
            csv_path = os.path.join(self.events_dir, "icews_real_topk500_K100.csv")
            return self._load_triplets_from_icews_csv(csv_path, time_col="time")

        if self.mode == "icews_real_topk500_K500":
            csv_path = os.path.join(self.events_dir, "icews_real_topk500_K500.csv")
            return self._load_triplets_from_icews_csv(csv_path, time_col="time")

        if self.mode == "icews_real_topk500_K1000":
            csv_path = os.path.join(self.events_dir, "icews_real_topk500_K1000.csv")
            return self._load_triplets_from_icews_csv(csv_path, time_col="time")

        # 其他模式：优先尝试从缓存 .pt 加载
        if os.path.exists(src_path) and os.path.exists(dst_path) and os.path.exists(type_path):
            src = torch.load(src_path, map_location=self.device)
            dst = torch.load(dst_path, map_location=self.device)
            ev_type = torch.load(type_path, map_location=self.device)
            return src.to(self.device), dst.to(self.device), ev_type.to(self.device)

        # 否则根据 mode 从 CSV 构建
        if self.mode == "toy":
            src = torch.zeros(self.event_times.numel(), dtype=torch.long, device=self.device)
            dst = torch.zeros(self.event_times.numel(), dtype=torch.long, device=self.device)
            ev_type = torch.zeros(self.event_times.numel(), dtype=torch.long, device=self.device)
        elif self.mode in ["icews_toy", "icews_real"]:
            # 注意：你仓库里目前是 icews_events_icews_real.csv / icews_events_icews_toy.csv 这一套命名
            csv_path = os.path.join(self.events_dir, f"icews_events_{self.mode}.csv")
            src, dst, ev_type = self._load_triplets_from_icews_csv(csv_path, time_col="time")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.save_to_disk:
            torch.save(src, src_path)
            torch.save(dst, dst_path)
            torch.save(ev_type, type_path)

        return src, dst, ev_type

    def _load_triplets_from_icews_csv(self, csv_path: str, time_col: str = "time"):
        """
        从 ICEWS 类 CSV 中加载 (src, dst, event_type) 三元组。

        关键点：
        - src/dst 仍然重编号为 [0, num_entities)
        - event_type 直接使用 CSV 中的 CAMEO 整数 code（不再重编号）
        """
        df = pd.read_csv(csv_path)
        df = df.sort_values(time_col)

        # 实体转为字符串以便重编号
        df["src"] = df["src"].astype(str)
        df["dst"] = df["dst"].astype(str)

        # 实体重编号
        all_entities = pd.unique(pd.concat([df["src"], df["dst"]], ignore_index=True))
        ent2id = {e: i for i, e in enumerate(all_entities.tolist())}
        src_ids = df["src"].map(ent2id).astype(int).to_numpy()
        dst_ids = df["dst"].map(ent2id).astype(int).to_numpy()

        # 事件类型：直接使用 CAMEO 整数 code
        df["event_type"] = df["event_type"].astype(int)
        type_ids = df["event_type"].to_numpy()

        src = torch.tensor(src_ids, dtype=torch.long, device=self.device)
        dst = torch.tensor(dst_ids, dtype=torch.long, device=self.device)
        ev_type = torch.tensor(type_ids, dtype=torch.long, device=self.device)
        return src, dst, ev_type

    def _load_triplets_from_topk_csv(self, csv_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        （可选）从 Top-K 风格的 CSV 加载三元组。
        这里假定 event_type 也是 CAMEO 整数 code。
        """
        df = pd.read_csv(csv_path)
        df = df.sort_values("time")
        df["src"] = df["src"].astype(str)
        df["dst"] = df["dst"].astype(str)

        all_entities = pd.unique(pd.concat([df["src"], df["dst"]], ignore_index=True))
        ent2id = {e: i for i, e in enumerate(all_entities.tolist())}
        src_ids = df["src"].map(ent2id).astype(int).to_numpy()
        dst_ids = df["dst"].map(ent2id).astype(int).to_numpy()

        df["event_type"] = df["event_type"].astype(int)
        type_ids = df["event_type"].to_numpy()

        src = torch.tensor(src_ids, dtype=torch.long, device=self.device)
        dst = torch.tensor(dst_ids, dtype=torch.long, device=self.device)
        ev_type = torch.tensor(type_ids, dtype=torch.long, device=self.device)
        return src, dst, ev_type

    def _triplets_pt_paths(self):
        src_path = os.path.join(self.events_dir, f"src_{self.mode}.pt")
        dst_path = os.path.join(self.events_dir, f"dst_{self.mode}.pt")
        type_path = os.path.join(self.events_dir, f"ev_type_{self.mode}.pt")
        return src_path, dst_path, type_path

    def _snapshots_pt_paths(self):
        x_path = os.path.join(self.snapshots_dir, "X_list.pt")
        e_path = os.path.join(self.snapshots_dir, "edge_index.pt")
        return x_path, e_path

    def _events_pt_paths(self):
        ev_path = os.path.join(self.events_dir, f"event_times_{self.mode}.pt")
        dt_path = os.path.join(self.events_dir, f"dt_{self.mode}.pt")
        return ev_path, dt_path

    # ===========================
    # 图快照部分
    # ===========================
    def _build_or_load_snapshots(self):
        x_path, e_path = self._snapshots_pt_paths()
        if os.path.exists(x_path) and os.path.exists(e_path):
            X_list = torch.load(x_path, map_location=self.device)
            edge_index = torch.load(e_path, map_location=self.device)
            print("[GC_TPP_Dataset] Loaded X_list & edge_index from disk.")
            return X_list, edge_index

        print("[GC_TPP_Dataset] Building new snapshots & edge_index ...")
        edge_index = torch.tensor(
            [
                [i for i in range(self.N - 1)],
                [i + 1 for i in range(self.N - 1)],
            ],
            dtype=torch.long,
            device=self.device,
        )
        X_list = []
        for _ in range(self.T):
            x_t = torch.randn(self.N, self.F_in, device=self.device)
            X_list.append(x_t)

        if self.save_to_disk:
            torch.save(X_list, x_path)
            torch.save(edge_index, e_path)
            print("[GC_TPP_Dataset] Saved X_list & edge_index to disk.")

        return X_list, edge_index

    # ===========================
    # 事件时间部分（event_times / dt）
    # ===========================
    def _build_or_load_events(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """构建或加载事件时间序列 event_times 和 dt。"""

        ev_path, dt_path = self._events_pt_paths()

        # 优先尝试从缓存加载
        if os.path.exists(ev_path) and os.path.exists(dt_path):
            event_times = torch.load(ev_path, map_location=self.device)
            dt = torch.load(dt_path, map_location=self.device)
            print(f"[GC_TPP_Dataset] Loaded event_times & dt for mode={self.mode} from disk.")
            return event_times.to(self.device), dt.to(self.device)

        # 否则从 CSV 构建
        if self.mode == "toy":
            csv_path = os.path.join(self.events_dir, "toy_events.csv")
            print(f"[GC_TPP_Dataset] Loading events (mode=toy) from CSV: {csv_path}")
            event_times, dt = self._load_from_toy_csv(csv_path)
        elif self.mode == "icews_toy":
            csv_path = os.path.join(self.events_dir, "icews_events_toy.csv")
            print(f"[GC_TPP_Dataset] Loading events (mode=icews_toy) from CSV: {csv_path}")
            event_times, dt = self._load_from_icews_toy_csv(csv_path)
        elif self.mode == "icews_real":
            csv_path = os.path.join(self.events_dir, "icews_real.csv")
            print(f"[GC_TPP_Dataset] NOTE: icews_real mode, may truncate to first {self.truncate_icews_real_to} events.")
            print(f"[GC_TPP_Dataset] Loading events (mode=icews_real) from CSV: {csv_path}")
            event_times, dt = self._load_from_icews_real_csv(csv_path)
        elif self.mode == "icews_real_topk500":
            csv_path = os.path.join(self.events_dir, "icews_real_topk500.csv")
            print(f"[GC_TPP_Dataset] Loading events (mode=icews_real_topk500) from CSV: {csv_path}")
            event_times, dt = self._load_from_icews_real_topk500_csv(csv_path)
        elif self.mode == "icews_real_topk500_K100":
            csv_path = os.path.join(self.events_dir, "icews_real_topk500_K100.csv")
            print(f"[GC_TPP_Dataset] Loading events (mode=icews_real_topk500_K100) from CSV: {csv_path}")
            event_times, dt = self._load_from_icews_real_topk500_csv(csv_path)
        elif self.mode == "icews_real_topk500_K500":
            csv_path = os.path.join(self.events_dir, "icews_real_topk500_K500.csv")
            print(f"[GC_TPP_Dataset] Loading events (mode=icews_real_topk500_K500) from CSV: {csv_path}")
            event_times, dt = self._load_from_icews_real_topk500_csv(csv_path)
        elif self.mode == "icews_real_topk500_K1000":
            csv_path = os.path.join(self.events_dir, "icews_real_topk500_K1000.csv")
            print(f"[GC_TPP_Dataset] Loading events (mode=icews_real_topk500_K1000) from CSV: {csv_path}")
            event_times, dt = self._load_from_icews_real_topk500_csv(csv_path)
        elif self.mode == "icews0515":
            data = load_icews0515_txt_dir(
                dir_path=os.path.join(self.events_dir, "icews0515"),
                device=self.device,
                split="facts",
                time_mode="index",
            )
            return data.event_times, data.dt
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.save_to_disk:
            torch.save(event_times, ev_path)
            torch.save(dt, dt_path)
            print(f"[GC_TPP_Dataset] Saved event_times & dt for mode={self.mode} to disk.")

        return event_times, dt

    def _load_from_toy_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)
        # toy_events.csv 的时间列名是 "timestamp"，统一改名为 "time"
        if "time" not in df.columns and "timestamp" in df.columns:
            df = df.rename(columns={"timestamp": "time"})
        df = df.sort_values("time")
        times = torch.tensor(df["time"].values, dtype=torch.float32, device=self.device)
        dt = self._compute_dt(times)
        return times, dt

    def _load_from_icews_toy_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df = df.sort_values("time")
        times = torch.tensor(df["time"].values, dtype=torch.float32, device=self.device)
        dt = self._compute_dt(times)
        return times, dt

    def _load_from_icews_real_csv(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df = df.sort_values("time")

        # 论文写作期：可选截断，以加速训练
        if self.truncate_icews_real_to is not None and self.truncate_icews_real_to > 0:
            if len(df) > self.truncate_icews_real_to:
                df = df.iloc[: self.truncate_icews_real_to].copy()
                print(f"[GC_TPP_Dataset] DEBUG truncate icews_real to first {self.truncate_icews_real_to} events.")

        times = torch.tensor(df["time"].values, dtype=torch.float32, device=self.device)
        dt = self._compute_dt(times)
        return times, dt

    def _load_from_icews_real_topk500_csv(self, csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        从 icews_real_topk500 及其 Top-K 子集 CSV 中加载事件时间。
        """
        df = pd.read_csv(csv_path)
        df = df.sort_values("time")

        basename = os.path.basename(csv_path)

        if "K1000" in basename:
            max_events_debug = 8000
        elif "K500" in basename:
            max_events_debug = 5000
        elif "K100" in basename:
            max_events_debug = 2000
        else:
            max_events_debug = 8000

        if max_events_debug is not None and max_events_debug > 0 and len(df) > max_events_debug:
            before = len(df)
            df = df.iloc[: max_events_debug].copy()
            print(
                f"[GC_TPP_Dataset] DEBUG truncate {basename} "
                f"to first {max_events_debug} events (original={before})."
            )

        times = torch.tensor(df["time"].values, dtype=torch.float32, device=self.device)
        dt = self._compute_dt(times)
        return times, dt

    def _compute_dt(self, event_times: torch.Tensor) -> torch.Tensor:
        """
        将绝对时间序列 event_times 转为间隔序列 dt。
        """
        dt = torch.zeros_like(event_times)
        if event_times.numel() > 0:
            dt[0] = event_times[0]
        if event_times.numel() > 1:
            dt[1:] = event_times[1:] - event_times[:-1]
        return dt


# =============================
# Optional convenience patch:
# 使 GC_TPP_Dataset 支持 len(ds) / ds[i]，便于快速 smoke test。
# 不影响 flags / splits / training 逻辑。
# =============================
def _gctpp__len__(self):
    return int(getattr(self, "event_times").shape[0])

def _gctpp__getitem__(self, idx):
    return self.event_times[idx], self.dt[idx]

if "__len__" not in GC_TPP_Dataset.__dict__:
    GC_TPP_Dataset.__len__ = _gctpp__len__
if "__getitem__" not in GC_TPP_Dataset.__dict__:
    GC_TPP_Dataset.__getitem__ = _gctpp__getitem__
