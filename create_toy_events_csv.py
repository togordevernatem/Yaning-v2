import os
import csv
import random

# 确保 data/events 目录存在
os.makedirs("./data/events", exist_ok=True)
csv_path = "./data/events/toy_events.csv"

# 我们造一批“更真实一点”的 toy 事件：
# 整个时间区间 [0, 20)
random.seed(42)

events = []
current_t = 0.0
event_id = 0

# 前半段 [0, 10)：事件比较稀疏
current_t = 0.0
while current_t < 10.0:
    # 间隔在 [0.8, 3.0] 之间随机
    dt = random.uniform(0.8, 3.0)
    current_t += dt
    if current_t >= 20.0:
        break
    # 大多数是标签 0，少数 1
    label = 0 if random.random() < 0.7 else 1
    events.append((event_id, current_t, label))
    event_id += 1

# 后半段 [10, 20)：事件更密集（模拟“高风险”区间）
current_t = 10.0
while current_t < 20.0:
    # 间隔在 [0.3, 1.0] 之间随机
    dt = random.uniform(0.3, 1.0)
    current_t += dt
    if current_t >= 20.0:
        break
    # 大多数是标签 1，少数 0
    label = 1 if random.random() < 0.7 else 0
    events.append((event_id, current_t, label))
    event_id += 1

# 按时间排序，保证 timestamp 单调递增
events.sort(key=lambda x: x[1])

print(f"[INFO] Will write {len(events)} toy events to {csv_path}")

# 写入 CSV 文件
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["event_id", "timestamp", "label"])
    writer.writerows(events)

print("[INFO] Done.")
