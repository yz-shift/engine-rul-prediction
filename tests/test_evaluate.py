"""评估流程测试：重点是不泄漏、标签处理正确。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

from rul.evaluate import NEAR_FAILURE_RUL, cross_validate, evaluate_on_test, predict_rul
from rul.metrics import nasa_score
from rul.models import build_model


class ConstantSpy(RegressorMixin, BaseEstimator):
    """记录训练时看到的数据，并总是预测同一个值。"""

    fits: list = []

    def __init__(self, value=10.0):
        self.value = value

    def fit(self, X, y):
        ConstantSpy.fits.append({"units": set(X["marker"]), "max_label": y.max()})
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


@pytest.fixture
def table():
    # 10 台发动机，寿命 40、42、…、58；marker 让模型“看到”发动机编号
    rows = [
        {"unit": unit, "cycle": cycle, "marker": unit}
        for unit in range(1, 11)
        for cycle in range(1, 39 + 2 * unit)
    ]
    return pd.DataFrame(rows)


def test_cross_validation_never_trains_on_validation_engines(table):
    ConstantSpy.fits = []
    folds = cross_validate(ConstantSpy, table, ["marker"], cap=None, n_splits=5)

    assert len(folds) == 5
    # 5 折、10 台：每折恰好留出 2 台，每台恰好被留出一次
    held_out = [set(range(1, 11)) - fit["units"] for fit in ConstantSpy.fits]
    assert all(len(units) == 2 for units in held_out)
    assert set().union(*held_out) == set(range(1, 11))


def test_cap_applies_to_training_labels_but_not_to_evaluation_truth(table):
    ConstantSpy.fits = []
    capped = cross_validate(ConstantSpy, table, ["marker"], cap=12)
    uncapped = cross_validate(ConstantSpy, table, ["marker"], cap=None)

    assert all(fit["max_label"] == 12 for fit in ConstantSpy.fits[:5])
    assert all(fit["max_label"] > 12 for fit in ConstantSpy.fits[5:])
    # 模型预测恒定，所以如果真值也不受 cap 影响，两次指标应完全一样
    pd.testing.assert_frame_equal(capped, uncapped)


def test_near_failure_metrics_ignore_healthy_cycles():
    # 2 台发动机，寿命都是 200，都预测 0，所以误差 = 真实 RUL
    table = pd.DataFrame(
        {
            "unit": np.repeat([1, 2], 200),
            "cycle": np.tile(np.arange(1, 201), 2),
            "marker": np.repeat([1, 2], 200),
        }
    )
    folds = cross_validate(lambda: ConstantSpy(0.0), table, ["marker"], cap=None, n_splits=2)

    near = np.arange(NEAR_FAILURE_RUL + 1, dtype=float)
    everything = np.arange(200, dtype=float)
    assert folds["rmse_near_failure"].iloc[0] == pytest.approx(np.sqrt(np.mean(near**2)))
    assert folds["rmse_all_cycles"].iloc[0] == pytest.approx(np.sqrt(np.mean(everything**2)))
    expected_score = nasa_score(near, np.zeros_like(near)) / len(near) * 100
    assert folds["score_per_100"].iloc[0] == pytest.approx(expected_score)


def test_predict_rul_clips_negative_values():
    class Negative:
        def predict(self, X):
            return np.array([-5.0, 3.0])

    assert predict_rul(Negative(), None).tolist() == [0.0, 3.0]


def test_evaluate_on_test_uses_last_cycle_and_aligns_by_unit():
    class LastCycleModel:
        def predict(self, X):
            return X["cycle"].to_numpy(dtype=float)

    # 行顺序故意打乱；最后周期：1 号 = 20，2 号 = 30
    test_table = pd.DataFrame({"unit": [2, 1, 2, 1], "cycle": [30, 10, 5, 20]})
    true_rul = pd.Series({1: 20, 2: 40})

    metrics = evaluate_on_test(LastCycleModel(), test_table, ["cycle"], true_rul)

    # 误差 0、-10 → rmse = sqrt(50)，score = e^(10/13) - 1
    assert metrics["rmse"] == pytest.approx(np.sqrt(50))
    assert metrics["score"] == pytest.approx(np.exp(10 / 13) - 1)


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="未知模型"):
        build_model("nope")


def test_pipeline_ignores_constant_columns():
    X = pd.DataFrame({"constant": [1.0] * 6, "signal": [1.0, 2, 3, 4, 5, 6]})
    y = np.array([2.0, 4, 6, 8, 10, 12])

    model = build_model("linear_regression").fit(X, y)

    assert model.predict(X) == pytest.approx(y)
    assert model.named_steps["drop_constant"].get_support().tolist() == [False, True]
