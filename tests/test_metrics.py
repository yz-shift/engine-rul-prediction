"""评估指标测试，期望值都是手算的。"""

import math

import pytest

from rul.metrics import nasa_score, rmse


def test_rmse_hand_calculated():
    # 误差 3、-4 → sqrt((9 + 16) / 2)
    assert rmse([10, 20], [13, 16]) == pytest.approx(math.sqrt(12.5))


def test_score_is_zero_for_perfect_prediction():
    assert nasa_score([50, 80], [50, 80]) == 0.0


def test_score_hand_calculated_early_and_late():
    # d = -13 → e^1 - 1；d = +10 → e^1 - 1；d = 0 → 0
    expected = 2 * (math.e - 1)
    assert nasa_score([100, 100, 100], [87, 110, 100]) == pytest.approx(expected)


def test_score_penalises_late_more_than_early():
    late = nasa_score([100], [120])  # e^(20/10) - 1 = e^2 - 1
    early = nasa_score([100], [80])  # e^(20/13) - 1
    assert late == pytest.approx(math.exp(2) - 1)
    assert early == pytest.approx(math.exp(20 / 13) - 1)
    assert late > early


def test_score_sums_over_samples():
    assert nasa_score([100, 100], [110, 110]) == pytest.approx(2 * nasa_score([100], [110]))


@pytest.mark.parametrize("metric", [rmse, nasa_score])
def test_metrics_reject_misaligned_inputs(metric):
    # 长度 1 和 3 能被 numpy 广播，这里必须报错
    with pytest.raises(ValueError, match="形状不一致"):
        metric([1, 2, 3], [1])
