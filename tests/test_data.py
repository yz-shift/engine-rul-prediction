"""读取模块测试。

合成数据测试不依赖真实文件，任何机器都能跑；real_data 测试校验官方数据的
行列数，数据不存在时跳过。
"""

import pandas as pd
import pytest

from rul.data import (
    COLUMNS,
    DEFAULT_DATA_DIR,
    SENSOR_COLUMNS,
    SETTING_COLUMNS,
    load_rul,
    load_trajectories,
)


def _row(unit: int, cycle: int, n_values: int = 24) -> str:
    # 和官方文件一样，行尾带两个空格
    values = " ".join(f"{unit + cycle + i * 0.5:.4f}" for i in range(n_values))
    return f"{unit} {cycle} {values}  \n"


def test_columns_match_official_layout():
    assert len(COLUMNS) == 26
    assert len(SETTING_COLUMNS) == 3
    assert len(SENSOR_COLUMNS) == 21


def test_load_trajectories_parses_trailing_spaces_and_names_columns(tmp_path):
    (tmp_path / "train_FD001.txt").write_text(
        _row(1, 1) + _row(1, 2) + _row(2, 1), encoding="utf-8"
    )

    df = load_trajectories("FD001", "train", tmp_path)

    assert df.shape == (3, 26)
    assert list(df.columns) == COLUMNS
    assert df["unit"].tolist() == [1, 1, 2]
    assert df["cycle"].tolist() == [1, 2, 1]
    assert pd.api.types.is_integer_dtype(df["unit"])
    assert pd.api.types.is_integer_dtype(df["cycle"])
    assert df["setting_1"].iloc[0] == pytest.approx(2.0)  # unit + cycle + 0 * 0.5


def test_load_trajectories_rejects_wrong_column_count(tmp_path):
    (tmp_path / "train_FD001.txt").write_text(_row(1, 1, n_values=23), encoding="utf-8")

    with pytest.raises(ValueError, match="应有 26 列"):
        load_trajectories("FD001", "train", tmp_path)


def test_load_trajectories_rejects_missing_values(tmp_path):
    bad_row = "1 2 NaN " + " ".join(["1.0"] * 23) + "\n"
    (tmp_path / "train_FD001.txt").write_text(_row(1, 1) + bad_row, encoding="utf-8")

    with pytest.raises(ValueError, match="缺失值"):
        load_trajectories("FD001", "train", tmp_path)


@pytest.mark.parametrize(
    "kwargs", [{"subset": "FD005"}, {"subset": "FD001", "split": "valid"}]
)
def test_load_trajectories_rejects_unknown_arguments(kwargs, tmp_path):
    with pytest.raises(ValueError, match="未知"):
        load_trajectories(data_dir=tmp_path, **kwargs)


def test_load_rul_indexes_by_unit_starting_from_one(tmp_path):
    (tmp_path / "RUL_FD001.txt").write_text("112 \n98 \n69 \n", encoding="utf-8")

    rul = load_rul("FD001", tmp_path)

    assert rul.to_dict() == {1: 112, 2: 98, 3: 69}
    assert rul.index.name == "unit"


def test_load_rul_rejects_extra_columns(tmp_path):
    (tmp_path / "RUL_FD001.txt").write_text("112 1\n98 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="应有 1 列"):
        load_rul("FD001", tmp_path)


real_data = pytest.mark.skipif(
    not (DEFAULT_DATA_DIR / "train_FD001.txt").exists(),
    reason="未下载数据：运行 python scripts/download_data.py",
)


# 官方 readme 给出 26 列和 100/100 台发动机，但没写行数。
# 这里的行数 20631 / 13096 来自对原始文件的 wc -l，用来防止读取时丢行或多行。
@pytest.mark.real_data
@real_data
def test_real_fd001_train_matches_official_description():
    df = load_trajectories("FD001", "train")
    assert df.shape == (20631, 26)
    assert df["unit"].nunique() == 100


@pytest.mark.real_data
@real_data
def test_real_fd001_test_and_rul_match_official_description():
    df = load_trajectories("FD001", "test")
    rul = load_rul("FD001")
    assert df.shape == (13096, 26)
    assert df["unit"].nunique() == 100
    assert len(rul) == 100
    assert set(df["unit"]) == set(rul.index)  # 每台测试发动机都有答案
