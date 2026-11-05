"""实验配置测试。"""

import pandas as pd
import pytest

from rul.experiments import (
    EXPERIMENTS,
    Experiment,
    NO_CORE_SPEED_FEATURES,
    best_experiment,
    fit_final,
    prepare_table,
)


@pytest.fixture
def trajectories():
    rows = [
        {"unit": unit, "cycle": cycle, **{f"sensor_{i}": cycle * i * 0.1 for i in range(1, 22)},
         "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0}
        for unit in (1, 2)
        for cycle in range(1, 21)
    ]
    return pd.DataFrame(rows)


def test_experiment_names_are_unique():
    names = [e.name for e in EXPERIMENTS]
    assert len(names) == len(set(names))


def test_core_speed_control_removes_only_nc_and_nrc():
    assert "sensor_9" not in NO_CORE_SPEED_FEATURES
    assert "sensor_14" not in NO_CORE_SPEED_FEATURES
    assert len(NO_CORE_SPEED_FEATURES) == 18  # 21 - sensor_6 - 2


@pytest.mark.parametrize(
    ("feature_set", "count"),
    [("raw", 20), ("window", 80), ("window_no_core_speed", 72), ("window_all_columns", 96)],
)
def test_feature_set_sizes(trajectories, feature_set, count):
    table, columns = prepare_table(trajectories, feature_set)
    assert len(columns) == count
    assert set(columns) <= set(table.columns)


def test_prepare_table_rejects_unknown_feature_set(trajectories):
    with pytest.raises(ValueError, match="未知特征集"):
        prepare_table(trajectories, "nope")


def test_fit_final_respects_cap(trajectories):
    # 随机森林的预测是标签的平均值，标签被截顶后预测也不会超过上限
    experiment = Experiment("random_forest", "raw", cap=5)
    model = fit_final(experiment, trajectories)
    table, columns = prepare_table(trajectories, "raw")
    assert model.predict(table[columns]).max() <= 5


def test_best_experiment_uses_cross_validation_not_test(tmp_path):
    pd.DataFrame(
        {
            "model": ["lightgbm", "random_forest", "linear_regression"],
            "features": ["window", "raw", "raw"],
            "cap": ["125", "none", "125"],
            "cv_rmse_near_failure": [20.0, 10.0, 30.0],
            "test_rmse": [1.0, 50.0, 2.0],  # 测试集最好的不能被选中
        }
    ).to_csv(tmp_path / "summary.csv", index=False)

    assert best_experiment(tmp_path) == Experiment("random_forest", "raw", None)
