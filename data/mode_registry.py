# data/mode_registry.py
"""
Central registry for supported data_mode strings.

This file is used by both main.py and tools to avoid
'choices mismatch' between argparse and dataset implementation.
"""

DATA_MODES = [
    "toy",
    "icews_real",
    "icews_real_topk500",
    "icews_real_topk500_K100",
    "icews_real_topk500_K500",
    "icews_real_topk500_K1000",
]
