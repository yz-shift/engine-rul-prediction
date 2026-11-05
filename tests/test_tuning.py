"""超参数搜索测试：搜索空间、可复现、嵌套 CV 不泄漏。"""

import numpy as np
import pandas as pd
import pytest

from rul import tuning
from rul.models import build_model
from rul.tuning import (
    Configuration,
    evaluate_configurations,
    nested_cross_validate,
    sample_configurations,
    select_best,
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


def test_model_params_are_applied_to_lightgbm():
    model = build_model("lightgbm", params={"num_leaves": 7, "learning_rate": 0.05})
    assert model.named_steps["model"].num_leaves == 7
    assert model.named_steps["model"].learning_rate == 0.05


def test_select_best_picks_lowest_rmse():
    configurations = [_config(window=5), _config(window=10), _config(window=20)]
    metrics = pd.DataFrame({"rmse_near_failure": [3.0, 1.0, 2.0]})
    assert select_best(configurations, metrics).window == 10


def test_evaluate_configurations_returns_one_row_per_configuration(trajectories):
    configurations = [_config(window=5), _config(window=10, cap=20), _config(window=5, n_estimators=20)]
    metrics = evaluate_configurations(configurations, trajectories, "window", "FD001", 3, 42)

    assert len(metrics) == 3
    assert (metrics["rmse_near_failure"] > 0).all()
    # 配置不同，结果也应该不同；顺序错位时这条检测不出，所以再比较单独评估的结果
    alone = evaluate_configurations([configurations[1]], trajectories, "window", "FD001", 3, 42)
    assert metrics["rmse_near_failure"].iloc[1] == pytest.approx(alone["rmse_near_failure"].iloc[0])


def test_nested_cv_never_selects_parameters_on_held_out_engines(trajectories, monkeypatch):
    seen = []
    original = evaluate_configurations

    def spy(configurations, data, *args):
        seen.append(set(data["unit"]))
        return original(configurations, data, *args)

    monkeypatch.setattr(tuning, "evaluate_configurations", spy)
    monkeypatch.setattr(tuning, "OUTER_SPLITS", 3)

    result = nested_cross_validate([_config(), _config(window=10)], trajectories, "window", "FD001")

    assert len(result) == 3
    held_out = [set(range(1, 16)) - units for units in seen]
    assert all(len(units) == 5 for units in held_out)  # 15 台分 3 折，每折留 5 台
    assert set().union(*held_out) == set(range(1, 16))  # 每台都恰好被留出一次
    assert result["chosen"].iloc[0]["window"] in (5, 10)
