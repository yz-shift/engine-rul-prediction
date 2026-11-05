"""把实验结果写成 JSON。"""

import json
from pathlib import Path

# 随机森林多线程汇总预测时，浮点数相加的顺序不固定，第 13 位小数之后会有抖动。
# 保留 4 位小数已远超指标的实际精度，也让结果文件每次运行都完全一致。
DECIMALS = 4


# 这些键下面是模型配置，之后会被读回来训练，绝不能四舍五入，否则结果对应的就不是这组配置。
EXACT_KEYS = frozenset({"model_params", "chosen_per_fold"})


def rounded(value):
    """递归把结果里的指标四舍五入，模型配置保持原样。"""
    if isinstance(value, float):
        return round(value, DECIMALS)
    if isinstance(value, dict):
        return {k: v if k in EXACT_KEYS else rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    return value


def write_json(path: Path, result: dict) -> None:
    path.write_text(json.dumps(rounded(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
