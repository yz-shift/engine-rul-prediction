"""实验定义：模型、特征和截顶的组合。

训练脚本、notebook 和 Streamlit 都从这里取配置，保证三处用的是同一套逻辑。
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.pipeline import Pipeline

from rul.data import SENSOR_COLUMNS, add_rul
from rul.models import SEED, build_model

# sensor_6 只有两种取值（见 01_eda 第 3 节）；settings 在 FD001 里只是抖动。
RAW_FEATURES = [c for c in SENSOR_COLUMNS if c != "sensor_6"]


def _raw_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    return df, RAW_FEATURES


FEATURE_SETS = {
    "raw": _raw_table,
}


@dataclass(frozen=True)
class Experiment:
    model: str
    features: str
    cap: int | None

    @property
    def name(self) -> str:
        return f"{self.model}__{self.features}__cap_{self.cap if self.cap else 'none'}"


EXPERIMENTS = [Experiment("linear_regression", "raw", cap) for cap in (125, None)]


def prepare_table(df: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    """把原始轨迹转成特征表，返回 (特征表, 特征列名)。"""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"未知特征集 {feature_set!r}，可选：{sorted(FEATURE_SETS)}")
    return FEATURE_SETS[feature_set](df)


def fit_final(experiment: Experiment, train: pd.DataFrame) -> Pipeline:
    """用全部训练发动机拟合最终模型。train 是原始轨迹。"""
    table, columns = prepare_table(train, experiment.features)
    labels = add_rul(table[["unit", "cycle"]], cap=experiment.cap)["rul"]
    return build_model(experiment.model, SEED).fit(table[columns], labels)
