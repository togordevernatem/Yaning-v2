import os
import csv
import random

if __name__ == "__main__":
    # 目标路径
    events_dir = "./data/events"
    os.makedirs(events_dir, exist_ok=True)
    csv_path = os.path.join(events_dir, "icews_events_toy.csv")

    # 超参数：最长时间和事件个数
    T_max = 20.0
    n_events = 40

    random.seed(2025)

    countries = ["CHN", "USA", "RUS", "JPN"]
    event_types = ["material", "verbal", "conflict", "cooperate"]

    rows = []
    current_t = 0.5
    for eid in range(n_events):
        # 每个事件的时间间隔在 [0.2, 1.5] 之间
        dt = random.uniform(0.2, 1.5)
        current_t += dt
        if current_t >= T_max:
            break

        src = f"actor_{random.randint(0, 4)}"
        dst = f"actor_{random.randint(0, 4)}"
        country = random.choice(countries)
        etype = random.choice(event_types)
        label = random.randint(0, 1)  # 先放一个 0/1 标签位

        rows.append({
            "event_id": eid,
            "time": current_t,
            "src": src,
            "dst": dst,
            "country": country,
            "event_type": etype,
            "label": label,
        })

    print(f"[INFO] Will write {len(rows)} ICEWS-style toy events to {csv_path}")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["event_id", "time", "src", "dst", "country", "event_type", "label"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("[INFO] Done.")
