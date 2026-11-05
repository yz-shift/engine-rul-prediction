"""LightGBM 超参数的随机搜索和嵌套交叉验证。

普通 CV 的最优值是偏乐观的：试的组合越多，越容易碰上“恰好在这几折表现好”的
参数。所以这里用嵌套 CV 估计真实水平：外层每折留出的发动机完全不参与选参数，
只在外层训练发动机上再做内层 CV 选参数。

决策规则：只有嵌套 CV 的 RMSE 低于默认参数，才采用搜索结果。

截顶值不参与搜索，固定为 125。选参指标只统计真实 RUL <= 125 的周期，看不到 125 以上
的误差，而把截顶调低能让模型集中拟合这一段，所以指标一定偏向低截顶。第一版搜索就掉进
了这个盲区：所有子集都选中截顶 100，CV 明显变好，官方测试 RMSE 却变差。不同截顶的
取舍单独报告（cap_sensitivity），不用这个指标自动决定。
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from rul.data import add_rul, engine_folds
from rul.evaluate import fold_metrics, predict_rul
from rul.experiments import build_featurizer
from rul.models import SEED, build_model

N_TRIALS = 20
OUTER_SPLITS = 5
INNER_SPLITS = 3
SEARCH_CAP = 125
OUTER_SEED = SEED  # 和默认参数的 CV 相同的折，才能逐折配对比较
INNER_SEED = SEED + 1


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


def select_best(configurations: Sequence[Configuration], metrics: pd.DataFrame) -> Configuration:
    """按临近故障 RMSE 选最优配置。"""
    return configurations[int(metrics["rmse_near_failure"].to_numpy().argmin())]


def nested_cross_validate(
    configurations: Sequence[Configuration],
    trajectories: pd.DataFrame,
    feature_set: str,
    subset: str,
) -> pd.DataFrame:
    """嵌套 CV，每个外层折一行：外层指标加上该折选中的配置。"""
    truth = add_rul(trajectories[["unit", "cycle"]])["rul"].to_numpy()
    rows = []
    for fold, (train_idx, val_idx) in enumerate(engine_folds(trajectories, OUTER_SPLITS, OUTER_SEED)):
        train, validation = trajectories.iloc[train_idx], trajectories.iloc[val_idx]

        # 内层只看外层训练发动机；外层验证发动机不参与选参数
        inner = evaluate_configurations(configurations, train, feature_set, subset, INNER_SPLITS, INNER_SEED)
        chosen = select_best(configurations, inner)

        featurizer = build_featurizer(feature_set, subset, chosen.window).fit(train)
        train_table, columns = featurizer.transform(train)
        val_table, _ = featurizer.transform(validation)
        labels = add_rul(train[["unit", "cycle"]], cap=chosen.cap)["rul"].to_numpy()
        model = build_model("lightgbm", SEED, chosen.params()).fit(train_table[columns], labels)
        predictions = predict_rul(model, val_table[columns])

        rows.append(
            {
                "fold": fold,
                **fold_metrics(truth[val_idx], predictions),
                "inner_rmse": float(inner["rmse_near_failure"].min()),
                "chosen": chosen.as_dict(),
            }
        )
    return pd.DataFrame(rows).set_index("fold")
