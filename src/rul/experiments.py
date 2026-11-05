"""实验定义：模型、特征和截顶的组合。

训练脚本、notebook 和 Streamlit 都从这里取配置，保证三处用的是同一套逻辑。
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from rul.data import N_CONDITIONS, SENSOR_COLUMNS, SETTING_COLUMNS, add_rul
from rul.features import WINDOW, ConditionNormalizer, add_window_features
from rul.models import SEED, build_model

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
FINAL_FILE = "final.json"

# sensor_6 在 FD001 里只有两种取值（见 01_eda 第 3 节）；settings 在 FD001 里只是抖动。
RAW_FEATURES = [c for c in SENSOR_COLUMNS if c != "sensor_6"]
# 对照组：Nc、NRc 的变化方向因发动机而异（见 01_eda 第 6 节）。
NO_CORE_SPEED_FEATURES = [c for c in RAW_FEATURES if c not in ("sensor_9", "sensor_14")]
# 对照组：不做人工筛选的全部列，恒定列由 Pipeline 的方差筛选去掉。
ALL_FEATURES = SETTING_COLUMNS + SENSOR_COLUMNS


class Featurizer:
    """把原始轨迹转成特征表。先 fit 训练数据，再 transform 任意数据。

    工况归一化要从数据拟合，所以特征必须在每一折内部计算，不能切分前算好。
    """

    def __init__(
        self,
        base_columns: list[str],
        window: int | None = None,
        n_conditions: int = 0,
    ):
        self.base_columns = base_columns
        self.window = window
        self.n_conditions = n_conditions  # 0 表示不做工况归一化

    def fit(self, train: pd.DataFrame) -> "Featurizer":
        if self.n_conditions:
            self.normalizer_ = ConditionNormalizer(self.base_columns, self.n_conditions, SEED).fit(train)
        return self

    def transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """返回 (特征表, 特征列名)。"""
        table = self.normalizer_.transform(df) if self.n_conditions else df
        if self.window is None:
            return table, self.base_columns
        table, window_columns = add_window_features(table, self.base_columns, self.window)
        return table, self.base_columns + window_columns


def build_featurizer(feature_set: str, subset: str = "FD001", window: int = WINDOW) -> Featurizer:
    """按名称创建特征器。condition_* 按该子集的工况数归一化。"""
    conditions = N_CONDITIONS[subset]
    options = {
        "raw": (RAW_FEATURES, None, 0),
        "window": (RAW_FEATURES, window, 0),
        "window_no_core_speed": (NO_CORE_SPEED_FEATURES, window, 0),
        "window_all_columns": (ALL_FEATURES, window, 0),
        "condition_raw": (SENSOR_COLUMNS, None, conditions),
        "condition_window": (SENSOR_COLUMNS, window, conditions),
    }
    if feature_set not in options:
        raise ValueError(f"未知特征集 {feature_set!r}，可选：{sorted(options)}")
    return Featurizer(*options[feature_set])


@dataclass(frozen=True)
class Experiment:
    model: str
    features: str
    cap: int | None
    subset: str = "FD001"
    window: int = WINDOW
    model_params: tuple = ()  # (参数名, 值) 的元组，覆盖模型默认参数；用元组才能保持 frozen
    tag: str = ""  # 比如 tuned，用于区分同一组合的不同参数

    @property
    def name(self) -> str:
        base = f"{self.model}__{self.features}__cap_{self.cap if self.cap else 'none'}"
        return f"{base}__{self.tag}" if self.tag else base

    def make_featurizer(self) -> Featurizer:
        return build_featurizer(self.features, self.subset, self.window)

    def make_model(self) -> Pipeline:
        return build_model(self.model, SEED, dict(self.model_params))


_MODELS = ("linear_regression", "random_forest", "lightgbm")

_FD001_EXPERIMENTS = [
    Experiment(model, features, cap, "FD001")
    for model in _MODELS
    for features in ("raw", "window")
    for cap in (125, None)
] + [
    # 特征对照只在 LightGBM + 截顶上跑，回答 01_eda 留下的两个问题
    Experiment("lightgbm", "window_no_core_speed", 125, "FD001"),
    Experiment("lightgbm", "window_all_columns", 125, "FD001"),
]


def _other_subset_experiments(subset: str) -> list[Experiment]:
    # window_all_columns 带工况设置，是“让模型自己学工况”的对照组。
    return [
        Experiment(model, features, 125, subset)
        for model in _MODELS
        for features in ("raw", "window_all_columns", "condition_raw", "condition_window")
    ] + [Experiment("lightgbm", "condition_window", None, subset)]


EXPERIMENTS = {
    "FD001": _FD001_EXPERIMENTS,
    **{subset: _other_subset_experiments(subset) for subset in ("FD002", "FD003", "FD004")},
}


@dataclass
class FittedModel:
    """训练好的特征器和模型。预测时必须用同一个特征器。"""

    experiment: Experiment
    featurizer: Featurizer
    model: Pipeline
    columns: list[str]


def fit_final(experiment: Experiment, train: pd.DataFrame) -> FittedModel:
    """用全部训练发动机拟合最终模型。train 是原始轨迹。"""
    featurizer = experiment.make_featurizer().fit(train)
    table, columns = featurizer.transform(train)
    labels = add_rul(train[["unit", "cycle"]], cap=experiment.cap)["rul"]
    model = experiment.make_model().fit(table[columns], labels)
    return FittedModel(experiment, featurizer, model, columns)


def best_experiment(subset: str = "FD001", results_dir: Path = RESULTS_DIR) -> Experiment:
    """返回该子集的最终模型：优先读 final.json，还没有时用默认参数里 CV 最优的实验。"""
    path = results_dir / subset / FINAL_FILE
    if not path.exists():
        return best_cv_experiment(subset, results_dir)
    final = json.loads(path.read_text(encoding="utf-8"))
    return Experiment(
        final["model"],
        final["features"],
        final["cap"],
        subset,
        window=final["window"],
        model_params=tuple(final["model_params"].items()),
        tag=final["tag"],
    )


def best_cv_experiment(subset: str = "FD001", results_dir: Path = RESULTS_DIR) -> Experiment:
    """默认参数的实验里，按交叉验证的临近故障 RMSE 选最优。不看官方测试集结果。"""
    summary = pd.read_csv(results_dir / subset / "summary.csv", keep_default_na=False)
    best = summary.loc[summary["cv_rmse_near_failure"].idxmin()]
    cap = None if best["cap"] == "none" else int(best["cap"])
    return Experiment(best["model"], best["features"], cap, subset)
