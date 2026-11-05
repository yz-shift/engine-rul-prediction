"""RUL 标签和按发动机切分的测试。"""

import pandas as pd
import pytest

from rul.data import add_rul, engine_folds


@pytest.fixture
def trajectories():
    # 3 台发动机，寿命分别为 3、2、4 个周期
    return pd.DataFrame(
        {
            "unit": [1, 1, 1, 2, 2, 3, 3, 3, 3],
            "cycle": [1, 2, 3, 1, 2, 1, 2, 3, 4],
        }
    )


@pytest.fixture
def fleet():
    # 20 台发动机，每台 5 个周期
    return pd.DataFrame(
        {"unit": [u for u in range(1, 21) for _ in range(5)], "cycle": list(range(1, 6)) * 20}
    )


def test_add_rul_counts_down_to_zero_per_engine(trajectories):
    result = add_rul(trajectories)
    assert result["rul"].tolist() == [2, 1, 0, 1, 0, 3, 2, 1, 0]


def test_add_rul_cap_limits_only_large_values(trajectories):
    result = add_rul(trajectories, cap=2)
    assert result["rul"].tolist() == [2, 1, 0, 1, 0, 2, 2, 1, 0]


def test_add_rul_does_not_modify_input(trajectories):
    add_rul(trajectories)
    assert "rul" not in trajectories.columns


@pytest.mark.parametrize("cap", [0, -5])
def test_add_rul_rejects_non_positive_cap(trajectories, cap):
    with pytest.raises(ValueError, match="cap"):
        add_rul(trajectories, cap=cap)


def test_folds_have_no_engine_in_both_train_and_validation(fleet):
    for train_idx, val_idx in engine_folds(fleet):
        train_units = set(fleet["unit"].iloc[train_idx])
        val_units = set(fleet["unit"].iloc[val_idx])
        assert train_units.isdisjoint(val_units)


def test_every_engine_is_validated_exactly_once(fleet):
    validated = []
    for _, val_idx in engine_folds(fleet, n_splits=5):
        validated.extend(fleet["unit"].iloc[val_idx].unique())
    assert sorted(validated) == list(range(1, 21))


def test_folds_keep_all_rows_of_an_engine_together(fleet):
    for train_idx, val_idx in engine_folds(fleet):
        assert len(train_idx) + len(val_idx) == len(fleet)
        assert fleet["unit"].iloc[val_idx].value_counts().eq(5).all()


def _validation_units(df, seed):
    return [tuple(sorted(df["unit"].iloc[val].unique())) for _, val in engine_folds(df, seed=seed)]


def test_folds_are_reproducible_with_fixed_seed(fleet):
    assert _validation_units(fleet, seed=42) == _validation_units(fleet, seed=42)
    assert _validation_units(fleet, seed=42) != _validation_units(fleet, seed=7)
