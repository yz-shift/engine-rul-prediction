"""读取 C-MAPSS 原始文本文件。

原始文件没有表头，以空格分隔，行尾还带多余空格。这里统一转成带列名的
DataFrame，并在读入时校验结构：列数不对或有缺失时立刻报错，不把问题留到
建模阶段才以奇怪的指标表现出来。
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# src/rul/data.py -> 项目根目录/data/raw
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SPLITS = ("train", "test")
# 官方 readme：FD001、FD003 只有一种工况（海平面），FD002、FD004 有六种。
N_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}

ID_COLUMNS = ["unit", "cycle"]
SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS  # 官方说明：共 26 列


def _check_subset(subset: str) -> None:
    if subset not in SUBSETS:
        raise ValueError(f"未知子集 {subset!r}，可选：{SUBSETS}")


def load_trajectories(
    subset: str = "FD001",
    split: str = "train",
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """读取训练或测试轨迹，每行是一台发动机的一个运行周期。"""
    _check_subset(subset)
    if split not in SPLITS:
        raise ValueError(f"未知划分 {split!r}，可选：{SPLITS}")

    path = Path(data_dir) / f"{split}_{subset}.txt"
    # 不直接传 names=COLUMNS：列数不符时 pandas 可能静默错位或补 NaN，
    # 所以先按实际列数读入，校验后再命名。
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(COLUMNS):
        raise ValueError(f"{path.name} 应有 {len(COLUMNS)} 列，实际 {df.shape[1]} 列")
    if df.isna().any(axis=None):
        raise ValueError(f"{path.name} 存在缺失值")

    df.columns = COLUMNS
    df[ID_COLUMNS] = df[ID_COLUMNS].astype(int)
    return df


def load_rul(
    subset: str = "FD001",
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> pd.Series:
    """读取测试集答案：每台发动机在轨迹截断时的真实 RUL，索引是发动机编号。

    官方文件只有一列，没有编号，第 N 行对应第 N 台发动机。
    """
    _check_subset(subset)
    path = Path(data_dir) / f"RUL_{subset}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != 1:
        raise ValueError(f"{path.name} 应有 1 列，实际 {df.shape[1]} 列")

    rul = df[0].astype(int).rename("rul")
    rul.index = pd.RangeIndex(1, len(rul) + 1, name="unit")
    return rul


def add_rul(df: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    """返回带 rul 列的副本：RUL = 该发动机最后一个周期 - 当前周期。

    只适用于训练集：训练轨迹一直运行到故障，最后一行才是 RUL = 0。
    测试轨迹在故障前就截断了，不能这样算。

    cap 是分段线性标签的上限，比如 125：早期健康阶段的信号很弱（见 01_eda
    第 7 节），都当作同一个 RUL，不要求模型区分。
    """
    if cap is not None and cap <= 0:
        raise ValueError(f"cap 必须为正数，实际 {cap}")

    result = df.copy()
    result["rul"] = result.groupby("unit")["cycle"].transform("max") - result["cycle"]
    if cap is not None:
        result["rul"] = result["rul"].clip(upper=cap)
    return result


def engine_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """按发动机划分交叉验证折，逐折给出 (训练行位置, 验证行位置)。

    同一台发动机的相邻周期几乎一样。如果按行随机划分，模型等于见过验证
    发动机，分数会虚高。按发动机划分才能模拟“预测一台没见过的发动机”。
    """
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from splitter.split(df, groups=df["unit"])
