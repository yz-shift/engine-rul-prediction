"""交叉验证和官方测试集评估。

交叉验证的主要指标只统计“临近故障区间”（真实 RUL <= 125）：

- 这是模型有信号、也真正影响维修决策的区间（见 01_eda 第 7 节）。
- 四个子集的训练发动机寿命都不少于 128 周期，每台在这个区间的行数相同，权重一样。
- 不抽样，结果没有随机性。

曾经试过按官方测试集的轨迹长度随机截断验证发动机，但模拟出的 RUL 最高到 313，
一个样本就能产生几十万的罚分，分数完全被极端点主导，所以放弃了。

所有指标都和未截顶的真实 RUL 比较，这样截顶和不截顶的模型才有可比性。
截顶只影响训练标签。

这个指标的局限：它看不到 RUL > 125 的误差，也看不到健康阶段的误报，所以还要配合
rul.alerts 的预警评估一起看。这个方案是看过测试结果后调整的，测试集因此已不算完全独立。

特征器在每一折内部只用训练发动机拟合，防止工况归一化把验证折的信息带进来。
"""

from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin

from rul.data import add_rul, engine_folds
from rul.experiments import Experiment, Featurizer
from rul.metrics import nasa_score, rmse

NEAR_FAILURE_RUL = 125


def predict_rul(model: RegressorMixin, features: pd.DataFrame) -> np.ndarray:
    """预测 RUL。线性模型可能给出负数，而剩余寿命不可能小于 0，所以截断到 0。"""
    return np.clip(model.predict(features), 0, None)


def fold_metrics(truth: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    near = truth <= NEAR_FAILURE_RUL
    return {
        "rmse_near_failure": rmse(truth[near], predictions[near]),
        "rmse_all_cycles": rmse(truth, predictions),
        # 官方分数是求和，和样本数有关。换算成每 100 个预测才能和测试集比较。
        "score_per_100": nasa_score(truth[near], predictions[near]) / near.sum() * 100,
    }


def _fold_predictions(
    build_model: Callable[[], RegressorMixin],
    build_featurizer: Callable[[], Featurizer],
    trajectories: pd.DataFrame,
    cap: int | None,
    n_splits: int,
    seed: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """逐折给出 (验证行位置, 验证行预测)。特征器和模型都只用训练发动机拟合。"""
    labels = add_rul(trajectories[["unit", "cycle"]], cap=cap)["rul"].to_numpy()
    for train_idx, val_idx in engine_folds(trajectories, n_splits, seed):
        train, validation = trajectories.iloc[train_idx], trajectories.iloc[val_idx]
        featurizer = build_featurizer().fit(train)
        train_table, columns = featurizer.transform(train)
        val_table, _ = featurizer.transform(validation)

        model = build_model()
        model.fit(train_table[columns], labels[train_idx])
        yield val_idx, predict_rul(model, val_table[columns])


def cross_validate(
    build_model: Callable[[], RegressorMixin],
    build_featurizer: Callable[[], Featurizer],
    trajectories: pd.DataFrame,
    cap: int | None,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """按发动机做交叉验证，每折一行指标。trajectories 是原始轨迹。"""
    truth = add_rul(trajectories[["unit", "cycle"]])["rul"].to_numpy()
    rows = [
        {"fold": fold, **fold_metrics(truth[val_idx], predictions)}
        for fold, (val_idx, predictions) in enumerate(
            _fold_predictions(build_model, build_featurizer, trajectories, cap, n_splits, seed)
        )
    ]
    return pd.DataFrame(rows).set_index("fold")


def cross_validate_experiment(
    experiment: Experiment,
    trajectories: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """按实验配置做交叉验证。窗口、模型参数和最终训练用的是同一份配置。"""
    return cross_validate(
        experiment.make_model, experiment.make_featurizer, trajectories, experiment.cap, n_splits, seed
    )


def out_of_fold_predictions(
    experiment: Experiment,
    trajectories: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """每台发动机的逐周期预测都来自没见过它的模型，返回 unit、cycle、rul、predicted。"""
    result = add_rul(trajectories[["unit", "cycle"]])
    result["predicted"] = np.nan
    column = result.columns.get_loc("predicted")
    for val_idx, predictions in _fold_predictions(
        experiment.make_model, experiment.make_featurizer, trajectories, experiment.cap, n_splits, seed
    ):
        result.iloc[val_idx, column] = predictions
    return result


def evaluate_on_test(
    model: RegressorMixin,
    test_table: pd.DataFrame,
    feature_columns: list[str],
    true_rul: pd.Series,
) -> dict[str, float]:
    """官方测试集评估：每台发动机只用最后一个周期预测。

    rmse 和 score 是官方口径（全部发动机）。rmse_near_failure 只算真实 RUL <= 125
    的发动机，和交叉验证的主要指标口径一致，两者才能直接比较。
    """
    last_rows = test_table.sort_values("cycle").groupby("unit").tail(1).sort_values("unit")
    predictions = predict_rul(model, last_rows[feature_columns])
    truth = true_rul.loc[last_rows["unit"]].to_numpy()  # 按编号对齐，不依赖行顺序
    near = truth <= NEAR_FAILURE_RUL
    return {
        "rmse": rmse(truth, predictions),
        "score": nasa_score(truth, predictions),
        "rmse_near_failure": rmse(truth[near], predictions[near]),
        "n_engines": int(len(truth)),
        "n_above_near_failure": int((~near).sum()),
    }
