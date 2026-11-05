"""预测逻辑和界面的测试。"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from rul.data import DEFAULT_DATA_DIR
from rul.inference import engine_index, first_alert_cycle, needs_maintenance, predict_engine

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


class CycleCountdown:
    """假模型：预测 100 - cycle，可能为负。"""

    def predict(self, X):
        return 100.0 - X["cycle"].to_numpy()


@pytest.fixture
def table():
    return pd.DataFrame({"unit": [1, 1, 2, 2, 2], "cycle": [1, 2, 3, 1, 2]})  # 2 号发动机未排序


def test_predict_engine_selects_one_engine_in_cycle_order(table):
    result = predict_engine(CycleCountdown(), table, ["cycle"], unit=2, final_rul=10)
    assert result["cycle"].tolist() == [1, 2, 3]  # 夹具里的行顺序是打乱的


def test_true_rul_is_reconstructed_from_final_value():
    frame = pd.DataFrame({"unit": [5, 5, 5], "cycle": [1, 2, 3]})
    result = predict_engine(CycleCountdown(), frame, ["cycle"], unit=5, final_rul=10)
    assert result["true"].tolist() == [12, 11, 10]
    assert result["predicted"].tolist() == [99.0, 98.0, 97.0]


def test_predictions_are_not_negative():
    frame = pd.DataFrame({"unit": [1], "cycle": [150]})
    result = predict_engine(CycleCountdown(), frame, ["cycle"], unit=1, final_rul=0)
    assert result["predicted"].tolist() == [0.0]


def test_predict_engine_rejects_unknown_unit(table):
    with pytest.raises(ValueError, match="没有发动机"):
        predict_engine(CycleCountdown(), table, ["cycle"], unit=99, final_rul=0)


@pytest.mark.parametrize(
    ("requested", "expected"),
    [("34", 2), ("1", 0), (None, 0), ("999", 0), ("abc", 0), ("-1", 0), ("", 0)],
)
def test_engine_index_falls_back_to_first_engine(requested, expected):
    assert engine_index([1, 2, 34], requested) == expected


def test_first_alert_cycle():
    trajectory = pd.DataFrame({"cycle": [1, 2, 3, 4], "predicted": [90.0, 50.0, 40.0, 60.0]})
    assert first_alert_cycle(trajectory, 40) == 3  # 等于阈值也算
    assert first_alert_cycle(trajectory, 10) is None


@pytest.mark.parametrize(
    ("predicted", "threshold", "expected"),
    [(39.0, 40, True), (40.0, 40, True), (41.0, 40, False), (np.float64(10.0), 40, True)],
)
def test_needs_maintenance(predicted, threshold, expected):
    assert needs_maintenance(predicted, threshold) is expected


real_data = pytest.mark.skipif(
    not (DEFAULT_DATA_DIR / "train_FD001.txt").exists(),
    reason="未下载数据：运行 python scripts/download_data.py",
)


@pytest.mark.real_data
@real_data
def test_app_runs_and_reacts_to_inputs():
    app = AppTest.from_file(APP, default_timeout=120).run()
    assert not app.exception
    assert app.title[0].value == "发动机剩余寿命（RUL）预警"

    # 1 号测试发动机：真实 RUL 官方答案 112，健康，不该预警
    assert app.metric[2].value == "112"
    assert app.success and not app.error

    # 把预警线调到比预测值更高，必须转为维修提示
    predicted = float(app.metric[1].value)
    app.slider[0].set_value(80).run()
    assert not app.exception
    assert bool(app.error) == (predicted <= 80)

    # 切到另一台发动机，指标要随之变化
    app.selectbox[0].set_value(34).run()
    assert not app.exception
    assert app.metric[2].value != "112"

    # 不选传感器也不能报错
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert app.info
