"""data/icews0515_loader.py

将 data/events/icews0515/ 下的 (src, relation, dst, date) 文本数据
转换为本项目 GC_TPP_Dataset 所需要的张量：
- event_times: [E] float/long (按时间排序后用 0..E-1 作为事件序)
- dt:         [E] float (相邻 event_times 差；这里默认 dt=1)
- src/dst:    [E] long (实体重编号)
- ev_type:    [E] long (relation 重编号)

注意：这套数据是 TKG 风格（日期粒度）。在当前 GC-TPP 代码里，核心回归目标是 log(dt)。
如果你希望 dt 更有意义，可以改成“按日期差(天)”或“同一天多事件 dt=0+eps”。
目前实现选择最稳定、最不容易炸的默认：事件序 dt=1。

接口尽量保持纯函数，便于 dataset_toy.py 调用。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import torch


@dataclass
class Icews0515Data:
    event_times: torch.Tensor  # [E]
    dt: torch.Tensor  # [E]
    src: torch.Tensor  # [E]
    dst: torch.Tensor  # [E]
    ev_type: torch.Tensor  # [E]
    ent2id: Dict[str, int]
    rel2id: Dict[str, int]


def _parse_line_4cols(line: str) -> Tuple[str, str, str, str]:
    parts = line.rstrip("\n").split("\t")
    if len(parts) == 1:
        # 数据里看起来是用若干空格做分隔
        parts = line.rstrip("\n").split()
    if len(parts) < 4:
        raise ValueError(f"Expected 4 columns (src rel dst date), got {len(parts)}: {line[:200]}")
    src, rel, dst, date = parts[0], parts[1], parts[2], parts[3]
    return src, rel, dst, date


def _date_to_int(date_str: str) -> int:
    # 形如 2005-01-01
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())


def load_icews0515_txt_dir(
    dir_path: str,
    device: torch.device,
    split: str = "train",
    time_mode: str = "index",
) -> Icews0515Data:
    """加载 icews0515 的某个 split。

    Args:
        dir_path: data/events/icews0515
        split: train/valid/test/facts (文件名为 {split}.txt)
        time_mode:
          - "index": event_times = 0..E-1, dt=1（最稳定，适配 log(dt)）
          - "date":  event_times = unix_timestamp(date), dt = diff in seconds (clamp>=1)

    Returns:
        Icews0515Data
    """

    txt_path = os.path.join(dir_path, f"{split}.txt")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"icews0515 split file not found: {txt_path}")

    ent2id: Dict[str, int] = {}
    rel2id: Dict[str, int] = {}

    src_ids: List[int] = []
    dst_ids: List[int] = []
    rel_ids: List[int] = []
    times: List[int] = []

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s, r, d, t = _parse_line_4cols(line)

            if s not in ent2id:
                ent2id[s] = len(ent2id)
            if d not in ent2id:
                ent2id[d] = len(ent2id)
            if r not in rel2id:
                rel2id[r] = len(rel2id)

            src_ids.append(ent2id[s])
            dst_ids.append(ent2id[d])
            rel_ids.append(rel2id[r])
            times.append(_date_to_int(t))

    # sort by time (and keep stable within same time)
    order = sorted(range(len(times)), key=lambda i: (times[i], i))
    src = torch.tensor([src_ids[i] for i in order], dtype=torch.long, device=device)
    dst = torch.tensor([dst_ids[i] for i in order], dtype=torch.long, device=device)
    ev_type = torch.tensor([rel_ids[i] for i in order], dtype=torch.long, device=device)
    times_sorted = [times[i] for i in order]

    if time_mode == "date":
        event_times = torch.tensor(times_sorted, dtype=torch.float32, device=device)
        if event_times.numel() == 0:
            dt = torch.zeros(0, dtype=torch.float32, device=device)
        else:
            dt = torch.zeros_like(event_times)
            dt[0] = 1.0
            diff = event_times[1:] - event_times[:-1]
            dt[1:] = torch.clamp(diff, min=1.0)
    elif time_mode == "index":
        E = len(times_sorted)
        event_times = torch.arange(E, dtype=torch.float32, device=device)
        dt = torch.ones(E, dtype=torch.float32, device=device)
        if E > 0:
            dt[0] = 1.0
    else:
        raise ValueError(f"Unknown time_mode: {time_mode}")

    return Icews0515Data(
        event_times=event_times,
        dt=dt,
        src=src,
        dst=dst,
        ev_type=ev_type,
        ent2id=ent2id,
        rel2id=rel2id,
    )

