from typing import Dict

"""
Coarse 类型归一化映射（Stage-10）

当前版本是一个占位实现：
- 假设 fine_id 是 ICEWS/CAMEO 的整数 code（例如 141、192 等）；
- 我们根据 code 的前两位或关键词，将细粒度类型聚合到少数 coarse bucket；
- 你后续可以用更精确的 CAMEO -> coarse 映射替换 FINE_TO_COARSE。
"""

# 例子：根据经验把常见 code 聚到 4~5 个 coarse
FINE_TO_COARSE: Dict[int, int] = {
    # Protests（例如 14x）
    141: 1, 142: 1, 143: 1,
    # Violence / Military conflict（例如 19x, 20x）
    190: 2, 192: 2, 200: 2,
    # Diplomacy / Talks（例如 03x–05x）
    30: 3, 31: 3, 40: 3,
    # Economic / Aid / Sanctions（例如 07x–10x）
    70: 4, 71: 4, 82: 4,
    # ... 后续你可以根据实际 code 表逐步补充
}

COARSE_LABELS: Dict[int, str] = {
    0: "Other",
    1: "Protest",
    2: "Violence",
    3: "Diplomacy",
    4: "Economic",
}


def map_event_type_to_coarse(fine_id: int) -> int:
    """
    将细粒度事件类型 ID 映射到 coarse 类型 ID。

    fine_id: 通常是 ICEWS/CAMEO 的整数 code。
    返回值: [0..C-1] 的 coarse ID，0 作为 Other/Unknown。
    """
    return FINE_TO_COARSE.get(fine_id, 0)
