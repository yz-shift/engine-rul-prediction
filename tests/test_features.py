"""滑动窗口特征测试：重点是不看未来、不跨发动机。"""

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_selection import VarianceThreshold

from rul.data import DEFAULT_DATA_DIR, load_trajectories
from rul.experiments import build_featurizer
from rul.features import ConditionNormalizer, add_window_features


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "unit": [1] * 8 + [2] * 6,
            "cycle": list(range(1, 9)) + list(range(1, 7)),
            "x": rng.normal(size=14).cumsum(),
        }
    )


def test_mean_and_std_match_manual_window(df):
    table, _ = add_window_features(df, ["x"], window=3)
    values = df["x"].iloc[3:6].to_numpy()  # 1 号发动机第 4–6 周期

    assert table["x_mean"].iloc[5] == pytest.approx(values.mean())
    assert table["x_std"].iloc[5] == pytest.approx(values.std(ddof=1))


def test_slope_matches_polyfit(df):
    table, _ = add_window_features(df, ["x"], window=4)
    for position in [3, 5, 7, 11]:
        window = df.iloc[position - 3 : position + 1]
        expected = np.polyfit(window["cycle"], window["x"], 1)[0]
        assert table["x_slope"].iloc[position] == pytest.approx(expected)


def test_slope_of_linear_signal_is_exact():
    df = pd.DataFrame({"unit": [1] * 6, "cycle": range(1, 7), "x": [3.0 * c + 9000 for c in range(1, 7)]})
    table, _ = add_window_features(df, ["x"], window=3)
    assert table["x_slope"].iloc[1:].tolist() == pytest.approx([3.0] * 5)


def test_constant_signal_has_exactly_zero_slope_and_std():
    # 518.67 这种值如果不先平移，浮点误差会产生 1e-14 级的假斜率，骗过方差筛选
    df = pd.DataFrame({"unit": [1] * 300, "cycle": range(1, 301), "x": [518.67] * 300})
    table, _ = add_window_features(df, ["x"])
    assert (table["x_slope"] == 0).all()
    assert (table["x_std"] == 0).all()


@pytest.mark.real_data
@pytest.mark.skipif(
    not (DEFAULT_DATA_DIR / "train_FD001.txt").exists(),
    reason="未下载数据：运行 python scripts/download_data.py",
)
def test_constant_sensors_produce_no_noise_features_on_real_data():
    # 14 个有变化的传感器，各有原值、均值、标准差、斜率，共 56 列
    train = load_trajectories("FD001", "train")
    table, columns = build_featurizer("window").fit(train).transform(train)
    assert VarianceThreshold().fit(table[columns]).get_support().sum() == 56


def test_features_do_not_use_future_cycles(df):
    # 改掉第 5 周期以后的读数，前 5 个周期的特征不能变
    baseline, columns = add_window_features(df, ["x"], window=3)
    changed = df.copy()
    changed.loc[5:7, "x"] += 1000
    result, _ = add_window_features(changed, ["x"], window=3)

    pd.testing.assert_frame_equal(baseline[columns].iloc[:5], result[columns].iloc[:5])
    assert not baseline["x_mean"].iloc[5:8].equals(result["x_mean"].iloc[5:8])


def test_windows_do_not_cross_engines(df):
    # 改掉 1 号发动机，2 号的特征不能变
    baseline, columns = add_window_features(df, ["x"], window=3)
    changed = df.copy()
    changed.loc[changed["unit"] == 1, "x"] += 1000
    result, _ = add_window_features(changed, ["x"], window=3)

    second = df["unit"] == 2
    pd.testing.assert_frame_equal(baseline.loc[second, columns], result.loc[second, columns])


def test_first_cycle_has_no_missing_values(df):
    table, columns = add_window_features(df, ["x"], window=5)
    assert not table[columns].isna().any(axis=None)
    first = table[table["cycle"] == 1]
    assert (first["x_std"] == 0).all()
    assert (first["x_slope"] == 0).all()
    assert (first["x_mean"] == first["x"]).all()


def test_short_trajectory_uses_available_cycles():
    df = pd.DataFrame({"unit": [1, 1], "cycle": [1, 2], "x": [1.0, 3.0]})
    table, _ = add_window_features(df, ["x"], window=30)
    assert table["x_mean"].iloc[1] == pytest.approx(2.0)
    assert table["x_slope"].iloc[1] == pytest.approx(2.0)


def test_rejects_unsorted_input(df):
    with pytest.raises(ValueError, match="升序"):
        add_window_features(df.iloc[::-1], ["x"])


def test_returns_three_features_per_column(df):
    table, columns = add_window_features(df.assign(y=1.0), ["x", "y"], window=3)
    assert columns == ["x_mean", "x_std", "x_slope", "y_mean", "y_std", "y_slope"]
    assert set(columns) <= set(table.columns)
    assert len(table) == len(df)


def _conditions_frame():
    # 两种工况：读数基准差很多，同一工况内才有可比性
    return pd.DataFrame(
        {
            "setting_1": [0.0] * 4 + [40.0] * 4,
            "setting_2": [0.0] * 4 + [0.8] * 4,
            "setting_3": [100.0] * 4 + [60.0] * 4,
            "x": [10.0, 11.0, 12.0, 13.0, 500.0, 502.0, 504.0, 506.0],
            "constant_in_condition": [7.0] * 4 + [9.0] * 4,
        }
    )


def test_condition_normalizer_uses_each_conditions_own_statistics():
    df = _conditions_frame()
    result = ConditionNormalizer(["x", "constant_in_condition"], n_conditions=2).fit(df).transform(df)

    for start in (0, 4):
        values = result["x"].iloc[start : start + 4]
        assert values.mean() == pytest.approx(0.0)
        assert values.std() == pytest.approx(1.0)
    # 不同工况的读数差 490，归一化后同一相对位置的值应该一样
    assert result["x"].iloc[0] == pytest.approx(result["x"].iloc[4])
    assert (result["constant_in_condition"] == 0).all()


def test_condition_normalizer_only_learns_from_fit_data():
    train = _conditions_frame()
    normalizer = ConditionNormalizer(["x"], n_conditions=2).fit(train)
    expected = normalizer.transform(train)["x"].iloc[0]

    # 新数据的读数再极端，也不能改变已经学到的统计量
    new = train.iloc[[0, 4]].assign(x=[10.0, 99999.0])
    assert normalizer.transform(new)["x"].iloc[0] == pytest.approx(expected)
    assert normalizer.transform(new)["x"].iloc[1] > 1000


def test_condition_normalizer_does_not_modify_input():
    df = _conditions_frame()
    before = df.copy()
    ConditionNormalizer(["x"], n_conditions=2).fit(df).transform(df)
    pd.testing.assert_frame_equal(df, before)


def test_single_condition_does_not_need_settings():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    result = ConditionNormalizer(["x"], n_conditions=1).fit(df).transform(df)
    assert result["x"].tolist() == pytest.approx([-1.0, 0.0, 1.0])
