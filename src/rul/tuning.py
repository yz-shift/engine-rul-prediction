"""选择最终模型的完整流程，以及评估这个流程的嵌套交叉验证。

select_pipeline 把所有选择放在一起：特征集、窗口长度、LightGBM 参数、不对称损失权重。
嵌套 CV 在外层每折的训练发动机上调用同一个函数，外层留出的发动机完全不参与任何选择，
所以它评估的是“整个选择流程”，而不是某一个已经选好的配置。最终模型则是在全部训练
发动机上再调用一次这个函数。

普通 CV 的最优值是偏乐观的：试的组合越多，越容易碰上恰好在这几折表现好的配置。
默认参数也是候选之一，所以搜索没有优势时会自然退回默认参数。

截顶值不参与搜索，固定为 125。选参指标只统计真实 RUL <= 125 的周期，看不到 125 以上的
误差，把截顶调低的代价不会计入指标。第一版搜索就掉进了这个盲区：所有子集都选中截顶 100，
CV 变好，官方测试 RMSE 却变差。这是查看测试结果后才发现的。不同截顶的取舍单独报告。
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rul.asymmetric import choose_late_weight
from rul.data import N_CONDITIONS, add_rul, engine_folds
from rul.evaluate import cross_validate_experiment, fold_metrics, predict_rul
from rul.experiments import Experiment, build_featurizer, fit_final
from rul.features import WINDOW
from rul.models import SEED, build_model

N_TRIALS = 20
OUTER_SPLITS = 5
INNER_SPLITS = 3
SEARCH_CAP = 125
OUTER_SEED = SEED  # 和默认参数的 CV 相同的折，才能逐折配对比较
INNER_SEED = SEED + 1
LATE_WEIGHTS = (1, 1.5, 2, 3, 5)
ASYMMETRIC_MODEL = "lightgbm_asymmetric"


@dataclass(frozen=True)
class Configuration:
    """一组候选配置：窗口影响特征，截顶影响标签，其余是 LightGBM 参数。"""

    window: int
    cap: int
    model_params: tuple[tuple[str, float | int], ...]

    def params(self) -> dict:
        return dict(self.model_params)

    def as_dict(self) -> dict:
        return {"window": self.window, "cap": self.cap, **self.params()}


@dataclass
class Selection:
    """一次完整选择的结果。"""

    experiment: Experiment  # 最终选中的配置
    symmetric: Experiment  # 不用不对称损失时的最优配置，用来比较取舍
    late_weight: float
    configurations: list[dict]  # 每个候选配置的 CV 指标
    weights: list[dict]  # 每个权重的 CV 指标


def default_configuration() -> Configuration:
    """models.py 里的默认参数，作为搜索的基准候选。"""
    return Configuration(WINDOW, SEARCH_CAP, ())


def sample_configurations(n_trials: int = N_TRIALS, seed: int = SEED) -> list[Configuration]:
    """从搜索空间随机抽样。随机搜索比网格搜索更省：多数参数其实影响不大。"""
    rng = np.random.default_rng(seed)
    return [
        Configuration(
            window=int(rng.choice([10, 20, 30, 50])),
            cap=SEARCH_CAP,
            model_params=(
                ("n_estimators", int(rng.choice([300, 600]))),
                ("learning_rate", float(10 ** rng.uniform(-2, -1))),
                ("num_leaves", int(rng.choice([7, 15, 31, 63]))),
                ("min_child_samples", int(rng.choice([10, 20, 50, 100]))),
                ("subsample", float(rng.uniform(0.6, 1.0))),
                ("colsample_bytree", float(rng.uniform(0.4, 1.0))),
                ("reg_lambda", float(10 ** rng.uniform(-2, 1))),
            ),
        )
        for _ in range(n_trials)
    ]


def feature_candidates(subset: str) -> tuple[str, str]:
    """候选特征集：人工筛选或工况归一化，对比不做筛选的全部列。"""
    primary = "condition_window" if subset != "FD001" else "window"
    return primary, "window_all_columns"


def evaluate_configurations(
    configurations: Sequence[Configuration],
    trajectories: pd.DataFrame,
    feature_set: str,
    subset: str,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    """在同一组折上评估每个配置，返回每个配置的平均指标（行顺序和输入相同）。

    特征按窗口长度缓存，每折只算一次；窗口特征计算比较耗时，而配置之间常常共用窗口。
    """
    truth = add_rul(trajectories[["unit", "cycle"]])["rul"].to_numpy()
    totals = [[] for _ in configurations]

    for train_idx, val_idx in engine_folds(trajectories, n_splits, seed):
        train, validation = trajectories.iloc[train_idx], trajectories.iloc[val_idx]
        tables = {}
        for index, configuration in enumerate(configurations):
            if configuration.window not in tables:
                featurizer = build_featurizer(feature_set, subset, configuration.window).fit(train)
                train_table, columns = featurizer.transform(train)
                val_table, _ = featurizer.transform(validation)
                tables[configuration.window] = (train_table, val_table, columns)
            train_table, val_table, columns = tables[configuration.window]

            labels = add_rul(train[["unit", "cycle"]], cap=configuration.cap)["rul"].to_numpy()
            model = build_model("lightgbm", SEED, configuration.params())
            model.fit(train_table[columns], labels)
            predictions = predict_rul(model, val_table[columns])
            totals[index].append(fold_metrics(truth[val_idx], predictions))

    return pd.DataFrame([pd.DataFrame(rows).mean() for rows in totals])


def select_pipeline(
    train: pd.DataFrame,
    subset: str,
    n_splits: int,
    seed: int,
    n_trials: int = N_TRIALS,
) -> Selection:
    """只用 train 里的发动机，依次选特征集和参数、不对称权重。"""
    configurations = [default_configuration(), *sample_configurations(n_trials)]

    records = []
    best_rmse, best_features, best_configuration = np.inf, None, None
    for features in feature_candidates(subset):
        metrics = evaluate_configurations(configurations, train, features, subset, n_splits, seed)
        for configuration, (_, row) in zip(configurations, metrics.iterrows()):
            records.append({"features": features, **configuration.as_dict(), **row.to_dict()})
            if row["rmse_near_failure"] < best_rmse:
                best_rmse, best_features, best_configuration = (
                    row["rmse_near_failure"],
                    features,
                    configuration,
                )

    symmetric = Experiment(
        "lightgbm",
        best_features,
        best_configuration.cap,
        subset,
        window=best_configuration.window,
        model_params=best_configuration.model_params,
        tag="selected",
    )

    weights = []
    for late_weight in LATE_WEIGHTS:
        candidate = _asymmetric_experiment(symmetric, late_weight)
        folds = cross_validate_experiment(candidate, train, n_splits, seed)
        weights.append({"late_weight": late_weight, **folds.mean().to_dict()})
    late_weight = choose_late_weight(weights)

    experiment = symmetric if late_weight == 1 else _asymmetric_experiment(symmetric, late_weight)
    return Selection(experiment, symmetric, late_weight, records, weights)


def _asymmetric_experiment(symmetric: Experiment, late_weight: float) -> Experiment:
    return Experiment(
        ASYMMETRIC_MODEL,
        symmetric.features,
        symmetric.cap,
        symmetric.subset,
        window=symmetric.window,
        model_params=(*symmetric.model_params, ("late_weight", late_weight)),
        tag="selected",
    )


def nested_cross_validate(
    trajectories: pd.DataFrame,
    subset: str,
    n_trials: int = N_TRIALS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """嵌套 CV。返回 (每个外层折的指标和选中配置, 外层留出发动机的逐周期预测)。"""
    truth = add_rul(trajectories[["unit", "cycle"]])["rul"].to_numpy()
    rows, predictions = [], []
    for fold, (train_idx, val_idx) in enumerate(engine_folds(trajectories, OUTER_SPLITS, OUTER_SEED)):
        train, validation = trajectories.iloc[train_idx], trajectories.iloc[val_idx]

        # 所有选择只看外层训练发动机；外层验证发动机到这里才第一次被用到
        selection = select_pipeline(train, subset, INNER_SPLITS, INNER_SEED, n_trials)
        fitted = fit_final(selection.experiment, train)
        val_table, _ = fitted.featurizer.transform(validation)
        fold_predictions = predict_rul(fitted.model, val_table[fitted.columns])

        rows.append(
            {
                "fold": fold,
                **fold_metrics(truth[val_idx], fold_predictions),
                "chosen": {
                    "model": selection.experiment.model,
                    "features": selection.experiment.features,
                    "window": selection.experiment.window,
                    "late_weight": selection.late_weight,
                    **dict(selection.symmetric.model_params),
                },
            }
        )
        predictions.append(
            validation[["unit", "cycle"]].assign(rul=truth[val_idx], predicted=fold_predictions)
        )
    return pd.DataFrame(rows).set_index("fold"), pd.concat(predictions).sort_index()
