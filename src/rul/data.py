"""读取 C-MAPSS 原始文本文件。

原始文件没有表头，以空格分隔，行尾还带多余空格。这里统一转成带列名的
DataFrame，并在读入时校验结构：列数不对或有缺失时立刻报错，不把问题留到
建模阶段才以奇怪的指标表现出来。
"""

from pathlib import Path

import pandas as pd

# src/rul/data.py -> 项目根目录/data/raw
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SPLITS = ("train", "test")

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
