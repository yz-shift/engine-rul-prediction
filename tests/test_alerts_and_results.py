"""预警评估和结果保存的测试。"""

import json

import pandas as pd
import pytest

from rul.alerts import alert_table
from rul.evaluate import cross_validate_experiment, out_of_fold_predictions
from rul.experiments import Experiment, best_experiment, build_featurizer, fit_final
from rul.results import rounded, write_json


@pytest.fixture
def predictions():
    # 1 号：预测准确；2 号：直到故障都高估，从不预警
    one = pd.DataFrame({"unit": 1, "rul": [150, 60, 35, 10, 0], "predicted": [150.0, 60, 35, 10, 0]})
    two = pd.DataFrame({"unit": 2, "rul": [150, 60, 35, 10, 0], "predicted": [150.0, 120, 110, 100, 90]})
    return pd.concat([one, two], ignore_index=True)


def test_alert_table_hand_calculated(predictions):
    row = alert_table(predictions, thresholds=(40,))[0]
    assert row["engines_without_alert"] == 1  # 2 号
    assert row["median_rul_at_first_alert"] == 35
    assert row["lowest_rul_at_first_alert"] == 35
    assert row["first_alert_rul_below_10"] == 0
    assert row["healthy_cycle_alert_rate"] == 0


def test_alert_table_counts_early_and_late_alerts():
    data = pd.DataFrame(
        {
            "unit": [1, 1, 2, 2, 3, 3],
            "rul": [130, 5, 110, 5, 50, 5],
            "predicted": [30.0, 30, 30, 30, 90, 30],
        }
    )
    row = alert_table(data, thresholds=(40,))[0]
    assert row["first_alert_rul_above_100"] == 2  # 第一次预警在 RUL 130、110
    assert row["first_alert_rul_below_10"] == 1  # 3 号到 RUL 5 才预警
    assert row["healthy_cycle_alert_rate"] == 1.0  # 唯一的健康周期（RUL 130）被误报


def test_alert_table_handles_threshold_with_no_alerts(predictions):
    row = alert_table(predictions.assign(predicted=500.0), thresholds=(40,))[0]
    assert row["engines_without_alert"] == 2
    assert row["median_rul_at_first_alert"] is None


def test_metrics_are_rounded_but_configurations_stay_exact():
    result = rounded(
        {
            "rmse": 1.234567,
            "model_params": {"learning_rate": 0.017123456789},
            "nested": {"chosen_per_fold": [{"subsample": 0.712612345}], "score": 9.87654321},
        }
    )
    assert result["rmse"] == 1.2346
    assert result["nested"]["score"] == 9.8765
    assert result["model_params"]["learning_rate"] == 0.017123456789
    assert result["nested"]["chosen_per_fold"][0]["subsample"] == 0.712612345


def test_saved_configuration_round_trips_exactly(tmp_path):
    params = {"learning_rate": 0.017123456789012345, "num_leaves": 63, "late_weight": 1.5}
    (tmp_path / "FD001").mkdir()
    write_json(
        tmp_path / "FD001" / "final.json",
        {
            "model": "lightgbm_asymmetric",
            "features": "window",
            "cap": 125,
            "window": 50,
            "model_params": params,
            "tag": "selected",
            "metric": 1.23456789,
        },
    )

    assert dict(best_experiment("FD001", tmp_path).model_params) == params
    saved = json.loads((tmp_path / "FD001" / "final.json").read_text(encoding="utf-8"))
    assert saved["metric"] == 1.2346


@pytest.fixture
def fleet():
    rows = [
        {"unit": unit, "cycle": cycle, "setting_1": 0.0, "setting_2": 0.0, "setting_3": 100.0,
         **{f"sensor_{i}": (cycle - 25 - unit) * i * 0.1 for i in range(1, 22)}}
        for unit in range(1, 11)
        for cycle in range(1, 26 + unit)
    ]
    return pd.DataFrame(rows)


def test_custom_window_and_parameters_reach_cv_and_final_training(fleet):
    """非默认配置必须同时用在交叉验证和最终训练上。"""
    params = (("n_estimators", 5), ("num_leaves", 3), ("min_child_samples", 2))
    experiment = Experiment("lightgbm", "window", 125, window=7, model_params=params)

    assert experiment.make_model().named_steps["model"].num_leaves == 3
    assert experiment.make_featurizer().window == 7

    fitted = fit_final(experiment, fleet)
    assert fitted.model.named_steps["model"].n_estimators == 5
    assert fitted.featurizer.window == 7

    custom = cross_validate_experiment(experiment, fleet, n_splits=2)
    default = cross_validate_experiment(
        Experiment("lightgbm", "window", 125), fleet, n_splits=2
    )
    assert not custom.equals(default)  # 如果 CV 忽略配置，两者会完全一样


def test_out_of_fold_predictions_cover_every_row_with_true_rul(fleet):
    experiment = Experiment("lightgbm", "raw", None, model_params=(("n_estimators", 5), ("min_child_samples", 2)))
    predictions = out_of_fold_predictions(experiment, fleet, n_splits=2)

    assert len(predictions) == len(fleet)
    assert predictions["predicted"].notna().all()
    assert (predictions.groupby("unit")["rul"].min() == 0).all()
    assert build_featurizer("raw").window is None
