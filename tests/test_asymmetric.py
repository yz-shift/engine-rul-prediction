"""不对称损失测试。"""

import numpy as np
import pytest
from lightgbm import LGBMRegressor

from rul.asymmetric import LateWeightedLightGBM, choose_late_weight, late_weighted_objective


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 3))
    y = 80 + 25 * X[:, 0] + rng.normal(scale=15, size=1500)
    return X, y


def test_objective_hand_calculated():
    gradient, hessian = late_weighted_objective(3.0)(np.array([10.0, 10.0]), np.array([12.0, 7.0]))
    # 报晚 e = 2：3 * 2 = 6；报早 e = -3：1 * -3 = -3
    assert gradient.tolist() == [6.0, -3.0]
    assert hessian.tolist() == [3.0, 1.0]


def test_objective_gradient_matches_finite_differences():
    def loss(y_true, y_pred, weight):
        error = y_pred - y_true
        return np.where(error > 0, weight, 1.0) * error**2 / 2

    y_true = np.array([10.0, 10.0, 10.0])
    y_pred = np.array([14.0, 3.0, 10.5])
    gradient, _ = late_weighted_objective(2.5)(y_true, y_pred)

    step = 1e-6
    numeric = (loss(y_true, y_pred + step, 2.5) - loss(y_true, y_pred - step, 2.5)) / (2 * step)
    assert gradient == pytest.approx(numeric, rel=1e-5)


def test_weight_one_matches_ordinary_squared_error(regression_data):
    X, y = regression_data
    params = {"n_estimators": 60, "learning_rate": 0.1, "num_leaves": 7, "subsample_freq": 0, "subsample": 1.0}
    ours = LateWeightedLightGBM(late_weight=1.0, colsample_bytree=1.0, **params).fit(X, y).predict(X)
    standard = (
        LGBMRegressor(colsample_bytree=1.0, random_state=42, deterministic=True, force_row_wise=True, verbose=-1, **params)
        .fit(X, y)
        .predict(X)
    )
    assert ours == pytest.approx(standard, abs=1e-6)


def test_larger_weight_predicts_lower_and_reduces_late_errors(regression_data):
    X, y = regression_data
    symmetric = LateWeightedLightGBM(late_weight=1.0, n_estimators=100).fit(X, y).predict(X)
    weighted = LateWeightedLightGBM(late_weight=5.0, n_estimators=100).fit(X, y).predict(X)

    assert weighted.mean() < symmetric.mean() - 3
    assert (weighted > y).mean() < (symmetric > y).mean()


def test_training_is_reproducible(regression_data):
    X, y = regression_data
    first = LateWeightedLightGBM(n_estimators=30).fit(X, y).predict(X)
    second = LateWeightedLightGBM(n_estimators=30).fit(X, y).predict(X)
    assert first == pytest.approx(second)


def _row(weight, score, rmse):
    return {"late_weight": weight, "score_per_100": score, "rmse_near_failure": rmse}


def test_adopts_weight_when_score_improves_enough_and_rmse_barely_moves():
    table = [_row(1, 1000, 20.0), _row(2, 850, 20.5), _row(3, 880, 20.2)]
    assert choose_late_weight(table) == 2


def test_keeps_symmetric_loss_when_score_gain_is_small():
    assert choose_late_weight([_row(1, 1000, 20.0), _row(2, 950, 20.0)]) == 1.0


def test_keeps_symmetric_loss_when_rmse_gets_too_much_worse():
    assert choose_late_weight([_row(1, 1000, 20.0), _row(2, 500, 22.0)]) == 1.0
