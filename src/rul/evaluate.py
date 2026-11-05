"""交叉验证和官方测试集评估。

交叉验证的主要指标只统计“临近故障区间”（真实 RUL <= 125）：

- 这是模型有信号、也真正影响维修决策的区间（见 01_eda 第 7 节）。
- FD001 训练发动机寿命都不少于 128 周期，每台在这个区间的行数相同，权重一样。
- 不抽样，结果没有随机性。

曾经试过按官方测试集的轨迹长度随机截断验证发动机，但模拟出的 RUL 最高到 313，
一个样本就能产生几十万的罚分，分数完全被极端点主导，所以放弃了。

所有指标都和未截顶的真实 RUL 比较，这样截顶和不截顶的模型才有可比性。
截顶只影响训练标签。
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

from rul.data import add_rul, engine_folds
from rul.metrics import nasa_score, rmse

NEAR_FAILURE_RUL = 125


def predict_rul(model: RegressorMixin, features: pd.DataFrame) -> np.ndarray:
    """预测 RUL。线性模型可能给出负数，而剩余寿命不可能小于 0，所以截断到 0。"""
    return np.clip(model.predict(features), 0, None)


def _fold_metrics(truth: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    near = truth <= NEAR_FAILURE_RUL
    return {
        "rmse_near_failure": rmse(truth[near], predictions[near]),
        "rmse_all_cycles": rmse(truth, predictions),
        # 官方分数是求和，和样本数有关。换算成每 100 个预测才能和测试集（100 台）比较。
        "score_per_100": nasa_score(truth[near], predictions[near]) / near.sum() * 100,
    }


def cross_validate(
    build: Callable[[], RegressorMixin],
    table: pd.DataFrame,
    feature_columns: list[str],
    cap: int | None,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """按发动机做交叉验证，每折一行指标。table 需要包含 unit、cycle 和特征列。"""
    truth = add_rul(table[["unit", "cycle"]])["rul"].to_numpy()
    labels = add_rul(table[["unit", "cycle"]], cap=cap)["rul"].to_numpy()

    rows = []
    for fold, (train_idx, val_idx) in enumerate(engine_folds(table, n_splits, seed)):
        model = build()
        model.fit(table[feature_columns].iloc[train_idx], labels[train_idx])
        predictions = predict_rul(model, table[feature_columns].iloc[val_idx])
        rows.append({"fold": fold, **_fold_metrics(truth[val_idx], predictions)})
    return pd.DataFrame(rows).set_index("fold")


def evaluate_on_test(
    model: RegressorMixin,
    test_table: pd.DataFrame,
    feature_columns: list[str],
    true_rul: pd.Series,
) -> dict[str, float]:
    """官方测试集评估：每台发动机只用最后一个周期预测。"""
    last_rows = test_table.sort_values("cycle").groupby("unit").tail(1).sort_values("unit")
    predictions = predict_rul(model, last_rows[feature_columns])
    truth = true_rul.loc[last_rows["unit"]].to_numpy()  # 按编号对齐，不依赖行顺序
    return {"rmse": rmse(truth, predictions), "score": nasa_score(truth, predictions)}
