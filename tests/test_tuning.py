"""模型选择流程测试：搜索空间、可复现、嵌套 CV 不泄漏。"""

import numpy as np
import pandas as pd
import pytest

from rul import tuning
from rul.evaluate import cross_validate_experiment
from rul.models import build_model
from rul.tuning import (
    Configuration,
    default_configuration,
    evaluate_configurations,
    feature_candidates,
    nested_cross_validate,
    sample_configurations,
    select_pipeline,
)


@pytest.fixture
def trajectories():
    # 15 台发动机，读数随退化线性变化，模型能学到规律
    rng = np.random.default_rng(0)
    rows = []
    for unit in range(1, 16):
        life = 30 + unit
        for cycle in range(1, life + 1):
            rul = life - cycle
            row = {"unit": unit, "cycle": cycle, "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0}
            row.update({f"sensor_{i}": -rul * i + rng.normal() for i in range(1, 22)})
            rows.append(row)
    return pd.DataFrame(rows)


def _config(window=5, cap=125, **params):
    defaults = {"n_estimators": 10, "num_leaves": 7, "min_child_samples": 5}
    return Configuration(window, cap, tuple({**defaults, **params}.items()))


@pytest.fixture
def small_search(monkeypatch):
    """把搜索缩小到几棵树，让流程测试足够快。"""
    monkeypatch.setattr(tuning, "default_configuration", lambda: _config(window=5))
    monkeypatch.setattr(tuning, "sample_configurations", lambda n_trials: [_config(window=10)])
    monkeypatch.setattr(tuning, "LATE_WEIGHTS", (1, 2))
    monkeypatch.setattr(
        tuning,
        "_asymmetric_experiment",
        _small_asymmetric(tuning._asymmetric_experiment),
    )


def _small_asymmetric(original):
    return lambda symmetric, late_weight: original(symmetric, late_weight)


def test_sampling_is_reproducible_and_within_the_search_space():
    first = sample_configurations(30, seed=1)
    assert first == sample_configurations(30, seed=1)
    assert first != sample_configurations(30, seed=2)
    for configuration in first:
        params = configuration.params()
        assert configuration.window in (10, 20, 30, 50)
        assert configuration.cap == 125  # 选参指标看不到 RUL > 125，截顶不能参与搜索
        assert 0.01 <= params["learning_rate"] <= 0.1
        assert 0.6 <= params["subsample"] <= 1.0


def test_default_parameters_are_a_candidate():
    assert default_configuration().params() == {}
    assert default_configuration().window == 30


def test_feature_candidates_compare_against_all_columns():
    assert feature_candidates("FD001") == ("window", "window_all_columns")
    assert feature_candidates("FD004") == ("condition_window", "window_all_columns")


def test_model_params_are_applied_to_lightgbm():
    model = build_model("lightgbm", params={"num_leaves": 7, "learning_rate": 0.05})
    assert model.named_steps["model"].num_leaves == 7
    assert model.named_steps["model"].learning_rate == 0.05


def test_evaluate_configurations_returns_one_row_per_configuration(trajectories):
    configurations = [_config(window=5), _config(window=10, cap=20), _config(window=5, n_estimators=20)]
    metrics = evaluate_configurations(configurations, trajectories, "window", "FD001", 3, 42)

    assert len(metrics) == 3
    assert (metrics["rmse_near_failure"] > 0).all()
    # 缓存特征后，顺序不能错位：和单独评估同一配置的结果比较
    alone = evaluate_configurations([configurations[1]], trajectories, "window", "FD001", 3, 42)
    assert metrics["rmse_near_failure"].iloc[1] == pytest.approx(alone["rmse_near_failure"].iloc[0])


def test_select_pipeline_records_every_candidate_and_picks_the_lowest(trajectories, small_search):
    selection = select_pipeline(trajectories, "FD001", 3, 42)

    records = pd.DataFrame(selection.configurations)
    assert len(records) == 4  # 2 个配置 × 2 个特征集
    best = records.loc[records["rmse_near_failure"].idxmin()]
    assert selection.symmetric.features == best["features"]
    assert selection.symmetric.window == best["window"]
    assert [row["late_weight"] for row in selection.weights] == [1, 2]


def test_selected_weight_is_part_of_the_final_experiment(trajectories, small_search, monkeypatch):
    monkeypatch.setattr(tuning, "choose_late_weight", lambda table: 2)
    selection = select_pipeline(trajectories, "FD001", 3, 42)
    assert selection.experiment.model == "lightgbm_asymmetric"
    assert dict(selection.experiment.model_params)["late_weight"] == 2

    monkeypatch.setattr(tuning, "choose_late_weight", lambda table: 1.0)
    selection = select_pipeline(trajectories, "FD001", 3, 42)
    assert selection.experiment == selection.symmetric
    assert selection.experiment.model == "lightgbm"
    # 记录的是真正返回的普通 LightGBM 的成绩，不是权重表里包装器的成绩
    own = cross_validate_experiment(selection.symmetric, trajectories, 3, 42).mean()
    assert selection.selected_cv["rmse_near_failure"] == pytest.approx(own["rmse_near_failure"])


def test_selected_cv_is_the_chosen_weight_row(trajectories, small_search, monkeypatch):
    monkeypatch.setattr(tuning, "choose_late_weight", lambda table: 2)
    selection = select_pipeline(trajectories, "FD001", 3, 42)
    row = next(r for r in selection.weights if r["late_weight"] == 2)
    assert selection.selected_cv == {k: v for k, v in row.items() if k != "late_weight"}


def test_nested_cv_final_fit_never_sees_held_out_engines(trajectories, small_search, monkeypatch):
    fitted_on = []
    original = tuning.fit_final

    def spy(experiment, train):
        fitted_on.append(set(train["unit"]))
        return original(experiment, train)

    monkeypatch.setattr(tuning, "fit_final", spy)
    monkeypatch.setattr(tuning, "OUTER_SPLITS", 3)

    _, predictions = nested_cross_validate(trajectories, "FD001")

    assert len(fitted_on) == 3
    held_out = [set(range(1, 16)) - units for units in fitted_on]
    assert all(len(units) == 5 for units in held_out)  # 外层最终模型只用外层训练发动机
    assert set().union(*held_out) == set(predictions["unit"])


def test_nested_cv_never_selects_anything_on_held_out_engines(trajectories, small_search, monkeypatch):
    seen = []
    original = select_pipeline

    def spy(train, *args, **kwargs):
        seen.append(set(train["unit"]))
        return original(train, *args, **kwargs)

    monkeypatch.setattr(tuning, "select_pipeline", spy)
    monkeypatch.setattr(tuning, "OUTER_SPLITS", 3)

    folds, predictions = nested_cross_validate(trajectories, "FD001")

    assert len(folds) == 3
    held_out = [set(range(1, 16)) - units for units in seen]
    assert all(len(units) == 5 for units in held_out)  # 15 台分 3 折，每折留 5 台
    assert set().union(*held_out) == set(range(1, 16))  # 每台都恰好被留出一次
    # 每行都得到一次留出预测，真实 RUL 没有错位
    assert len(predictions) == len(trajectories)
    assert predictions["predicted"].notna().all()
    assert (predictions.groupby("unit")["rul"].min() == 0).all()
    assert "late_weight" in folds["chosen"].iloc[0]
