"""实验定义：模型、特征和截顶的组合。

训练脚本、notebook 和 Streamlit 都从这里取配置，保证三处用的是同一套逻辑。
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from rul.data import SENSOR_COLUMNS, SETTING_COLUMNS, add_rul
from rul.features import add_window_features
from rul.models import SEED, build_model

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

# sensor_6 只有两种取值（见 01_eda 第 3 节）；settings 在 FD001 里只是抖动。
RAW_FEATURES = [c for c in SENSOR_COLUMNS if c != "sensor_6"]
# 对照组：Nc、NRc 的变化方向因发动机而异（见 01_eda 第 6 节）。
NO_CORE_SPEED_FEATURES = [c for c in RAW_FEATURES if c not in ("sensor_9", "sensor_14")]
# 对照组：不做人工筛选的全部列，恒定列由 Pipeline 的方差筛选去掉。
ALL_FEATURES = SETTING_COLUMNS + SENSOR_COLUMNS


def _window_table(df: pd.DataFrame, base: list[str]) -> tuple[pd.DataFrame, list[str]]:
    table, window_columns = add_window_features(df, base)
    return table, base + window_columns


FEATURE_SETS = {
    "raw": lambda df: (df, RAW_FEATURES),
    "window": lambda df: _window_table(df, RAW_FEATURES),
    "window_no_core_speed": lambda df: _window_table(df, NO_CORE_SPEED_FEATURES),
    "window_all_columns": lambda df: _window_table(df, ALL_FEATURES),
}


@dataclass(frozen=True)
class Experiment:
    model: str
    features: str
    cap: int | None

    @property
    def name(self) -> str:
        return f"{self.model}__{self.features}__cap_{self.cap if self.cap else 'none'}"


EXPERIMENTS = [
    Experiment(model, features, cap)
    for model in ("linear_regression", "random_forest", "lightgbm")
    for features in ("raw", "window")
    for cap in (125, None)
] + [
    # 特征对照只在 LightGBM + 截顶上跑，回答 01_eda 留下的两个问题
    Experiment("lightgbm", "window_no_core_speed", 125),
    Experiment("lightgbm", "window_all_columns", 125),
]


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


def best_experiment(results_dir: Path = RESULTS_DIR) -> Experiment:
    """按交叉验证的临近故障 RMSE 选最优实验。不看官方测试集结果。"""
    summary = pd.read_csv(results_dir / "summary.csv", keep_default_na=False)
    best = summary.loc[summary["cv_rmse_near_failure"].idxmin()]
    cap = None if best["cap"] == "none" else int(best["cap"])
    return Experiment(best["model"], best["features"], cap)
