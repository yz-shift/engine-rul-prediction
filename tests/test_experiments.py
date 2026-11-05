"""实验配置测试。"""

import json

import pandas as pd
import pytest

from rul.data import SUBSETS
from rul.experiments import (
    EXPERIMENTS,
    Experiment,
    NO_CORE_SPEED_FEATURES,
    best_experiment,
    build_featurizer,
    fit_final,
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


@pytest.mark.parametrize("subset", SUBSETS)
def test_experiment_names_are_unique_within_subset(subset):
    names = [e.name for e in EXPERIMENTS[subset]]
    assert len(names) == len(set(names))
    assert all(e.subset == subset for e in EXPERIMENTS[subset])


def test_core_speed_control_removes_only_nc_and_nrc():
    assert "sensor_9" not in NO_CORE_SPEED_FEATURES
    assert "sensor_14" not in NO_CORE_SPEED_FEATURES
    assert len(NO_CORE_SPEED_FEATURES) == 18  # 21 - sensor_6 - 2


@pytest.mark.parametrize(
    ("feature_set", "count"),
    [
        ("raw", 20),
        ("window", 80),
        ("window_no_core_speed", 72),
        ("window_all_columns", 96),
        ("condition_raw", 21),
        ("condition_window", 84),
    ],
)
def test_feature_set_sizes(trajectories, feature_set, count):
    table, columns = build_featurizer(feature_set).fit(trajectories).transform(trajectories)
    assert len(columns) == count
    assert set(columns) <= set(table.columns)


def test_build_featurizer_rejects_unknown_feature_set():
    with pytest.raises(ValueError, match="未知特征集"):
        build_featurizer("nope")


def test_condition_featurizer_uses_the_subsets_number_of_conditions():
    assert build_featurizer("condition_window", "FD004").n_conditions == 6
    assert build_featurizer("condition_window", "FD003").n_conditions == 1
    assert build_featurizer("window", "FD004").n_conditions == 0


def test_window_length_is_configurable(trajectories):
    short, _ = build_featurizer("window", window=2).fit(trajectories).transform(trajectories)
    # 长度 2 的窗口：第 5 周期的均值是第 4、5 周期的平均
    expected = trajectories["sensor_2"].iloc[3:5].mean()
    assert short["sensor_2_mean"].iloc[4] == pytest.approx(expected)


def test_fit_final_respects_cap(trajectories):
    # 随机森林的预测是标签的平均值，标签被截顶后预测也不会超过上限
    fitted = fit_final(Experiment("random_forest", "raw", cap=5), trajectories)
    table, columns = fitted.featurizer.transform(trajectories)
    assert fitted.model.predict(table[columns]).max() <= 5


def test_best_experiment_prefers_the_recorded_final_model(tmp_path):
    (tmp_path / "FD003").mkdir()
    (tmp_path / "FD003" / "final.json").write_text(
        json.dumps(
            {
                "model": "lightgbm_asymmetric",
                "features": "condition_window",
                "cap": 125,
                "window": 50,
                "model_params": {"late_weight": 2, "num_leaves": 31},
                "tag": "final",
            }
        ),
        encoding="utf-8",
    )

    experiment = best_experiment("FD003", tmp_path)

    assert experiment.window == 50
    assert dict(experiment.model_params) == {"late_weight": 2, "num_leaves": 31}
    assert experiment.name == "lightgbm_asymmetric__condition_window__cap_125__final"


def test_best_experiment_uses_cross_validation_not_test(tmp_path):
    (tmp_path / "FD002").mkdir()
    pd.DataFrame(
        {
            "model": ["lightgbm", "random_forest", "linear_regression"],
            "features": ["window", "raw", "raw"],
            "cap": ["125", "none", "125"],
            "cv_rmse_near_failure": [20.0, 10.0, 30.0],
            "test_rmse": [1.0, 50.0, 2.0],  # 测试集最好的不能被选中
        }
    ).to_csv(tmp_path / "FD002" / "summary.csv", index=False)

    assert best_experiment("FD002", tmp_path) == Experiment("random_forest", "raw", None, "FD002")
