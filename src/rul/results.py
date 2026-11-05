"""把实验结果写成 JSON。"""

import json
from pathlib import Path

# 随机森林多线程汇总预测时，浮点数相加的顺序不固定，第 13 位小数之后会有抖动。
# 保留 4 位小数已远超指标的实际精度，也让结果文件每次运行都完全一致。
DECIMALS = 4


def rounded(value):
    """递归把结果里的浮点数四舍五入。"""
    if isinstance(value, float):
        return round(value, DECIMALS)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    return value


def write_json(path: Path, result: dict) -> None:
    path.write_text(json.dumps(rounded(result), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
