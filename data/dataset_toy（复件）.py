import os
from typing import Tuple

import torch
import pandas as pd


class GC_TPP_Dataset:
    """
    统一管理 GC-TPP 用到的 toy / ICEWS 事件数据集。

    支持三种 mode:
      - "toy"        : data/events/toy_events.csv
      - "icews_toy"  : data/events/icews_events_toy.csv
      - "icews_real" : data/events/icews_real.csv  （由 2010_2011_AllProtests 清洗而来）

    统一输出:
      - X_list: list[Tensor(N,F)]
      - edge_index: (2,E)
      - event_times: (T,)
      - dt: (T,)

    并提供:
      - get_all()
      - get_train_val_test_split()
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
        self.truncate_icews_real_to = truncate_icews_real_to

        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.events_dir, exist_ok=True)

        # 1) 构建或加载图快照
        self.X_list, self.edge_index = self._build_or_load_snapshots()

        # 2) 构建或加载事件时间序列
        self.event_times, self.dt = self._build_or_load_events()

    # =========================
    # 1. 图快照 (X_list, edge_index)
    # =========================
    def _snapshots_pt_paths(self):
        x_path = os.path.join(self.snapshots_dir, "X_list.pt")
        e_path = os.path.join(self.snapshots_dir, "edge_index.pt")
        return x_path, e_path

    def _build_or_load_snapshots(self):
        x_path, e_path = self._snapshots_pt_paths()

        if os.path.exists(x_path) and os.path.exists(e_path):
            X_list = torch.load(x_path, map_location=self.device)
            edge_index = torch.load(e_path, map_location=self.device)
            print("[GC_TPP_Dataset] Loaded X_list & edge_index from disk.")
            return X_list, edge_index

        print("[GC_TPP_Dataset] Building new snapshots & edge_index ...")

        # 简单链式图：0-1-2-...-(N-1)
        edge_index = torch.tensor(
            [
                [i for i in range(self.N - 1)],
                [i + 1 for i in range(self.N - 1)],
            ],
            dtype=torch.long,
            device=self.device,
        )

        # 随机特征快照
        X_list = []
        for _ in range(self.T):
            x_t = torch.randn(self.N, self.F_in, device=self.device)
            X_list.append(x_t)

        if self.save_to_disk:
            torch.save(X_list, x_path)
            torch.save(edge_index, e_path)
            print("[GC_TPP_Dataset] Saved X_list & edge_index to disk.")

        return X_list, edge_index

    # =========================
    # 2. 事件时间序列 (event_times, dt)
    # =========================
    def _events_pt_paths(self):
        # 为不同 mode 分别缓存 .pt
        ev_path = os.path.join(self.events_dir, f"event_times_{self.mode}.pt")
        dt_path = os.path.join(self.events_dir, f"dt_{self.mode}.pt")
        return ev_path, dt_path

    def _build_or_load_events(self):
        ev_path, dt_path = self._events_pt_paths()

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

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.save_to_disk:
            torch.save(event_times, ev_path)
            torch.save(dt, dt_path)
            print(f"[GC_TPP_Dataset] Saved event_times & dt for mode={self.mode} to disk.")

        return event_times, dt

    def _compute_dt(self, event_times: torch.Tensor) -> torch.Tensor:
        """
        根据 event_times 计算 Δt 序列：
          dt[0] = event_times[0]
          dt[i] = event_times[i] - event_times[i-1]
        """
        dt = torch.zeros_like(event_times)
        if event_times.numel() > 0:
            dt[0] = event_times[0]
        if event_times.numel() > 1:
            dt[1:] = event_times[1:] - event_times[:-1]
        return dt

    def _load_from_toy_csv(self, csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        df = pd.read_csv(csv_path)
        # toy_events.csv: event_id,timestamp,label
        # 以 timestamp 升序排序
        df = df.sort_values("timestamp")
        times = torch.tensor(df["timestamp"].values, dtype=torch.float32, device=self.device)
        dt = self._compute_dt(times)
        print(f"[GC_TPP_Dataset] Loaded {len(times)} events (mode=toy).")
        return times, dt

    def _load_from_icews_toy_csv(self, csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        df = pd.read_csv(csv_path)
        # icews_events_toy.csv: event_id,time,src,dst,country,event_type,label
        df = df.sort_values("time")
        times = torch.tensor(df["time"].values, dtype=torch.float32, device=self.device)
        dt = self._compute_dt(times)
        print(f"[GC_TPP_Dataset] Loaded {len(times)} events (mode=icews_toy).")
        return times, dt

    def _load_from_icews_real_csv(self, csv_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        df = pd.read_csv(csv_path)
        # 由 create_icews_real_from_allprotests.py 生成：
        # columns: event_id,time,src,dst,country,event_type,label
        df = df.sort_values("time")

        # 如有需要，截断前 truncate_icews_real_to 个事件
        if self.truncate_icews_real_to is not None and len(df) > self.truncate_icews_real_to:
            print(
                f"[GC_TPP_Dataset] Truncate ICEWS real to first {self.truncate_icews_real_to} events (after sort)."
            )
            df = df.head(self.truncate_icews_real_to)

        times = torch.tensor(df["time"].values, dtype=torch.float32, device=self.device)
        dt = self._compute_dt(times)
        print(f"[GC_TPP_Dataset] Loaded {len(times)} events (mode=icews_real).")
        return times, dt

    # =========================
    # 3. 对外接口
    # =========================
    def get_all(self):
        """
        返回 (X_list, edge_index, event_times, dt)
        """
        return self.X_list, self.edge_index, self.event_times, self.dt

    def get_train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ):
        """
        按时间顺序将事件划分为 Train / Val / Test 三段。

        - Train: 前 train_ratio
        - Val  : 之后 val_ratio
        - Test : 剩余

        统一返回 8 个值：
          X_list,
          edge_index,
          train_event_times, train_dt,
          val_event_times,   val_dt,
          test_event_times,  test_dt
        """
        event_times = self.event_times
        dt = self.dt
        T = event_times.shape[0]

        if T < 3:
            # 事件太少，就全部当成 train，其余留空
            print(
                f"[GC_TPP_Dataset] WARNING: only {T} events, not enough for Train/Val/Test split. "
                f"Will use all events as Train."
            )
            return (
                self.X_list,
                self.edge_index,
                event_times,
                dt,
                event_times.new_empty(0),
                dt.new_empty(0),
                event_times.new_empty(0),
                dt.new_empty(0),
            )

        # 先按比例取整
        n_train = int(T * train_ratio)
        n_val = int(T * val_ratio)
        n_test = T - n_train - n_val

        # 保证三者都 >= 1
        if n_train < 1:
            n_train = 1
        if n_val < 1:
            n_val = 1
        if n_train + n_val >= T:
            # 至少留 1 个给 test
            n_val = max(1, T - n_train - 1)
        n_test = T - n_train - n_val
        if n_test < 1:
            # 再次兜底
            n_test = 1
            n_train = max(1, T - n_val - n_test)

        assert n_train + n_val + n_test == T, "Split sizes must sum to total T"

        train_event_times = event_times[:n_train]
        train_dt = dt[:n_train]

        val_event_times = event_times[n_train : n_train + n_val]
        val_dt = dt[n_train : n_train + n_val]

        test_event_times = event_times[n_train + n_val :]
        test_dt = dt[n_train + n_val :]

        print(
            f"[INFO] Total events = {T}, "
            f"train_events = {train_event_times.shape[0]}, "
            f"val_events = {val_event_times.shape[0]}, "
            f"test_events = {test_event_times.shape[0]}"
        )

        return (
            self.X_list,
            self.edge_index,
            train_event_times,
            train_dt,
            val_event_times,
            val_dt,
            test_event_times,
            test_dt,
        )

