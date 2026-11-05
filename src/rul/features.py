"""滑动窗口特征：最近 N 个周期的均值、标准差和斜率。

单个周期的读数噪声大（见 01_eda 第 7 节），退化趋势比单点读数更有信息。

窗口只看当前和过去的周期，部署时也能算出来；如果用到未来的周期，就是时间上的
数据泄漏。窗口按发动机分别计算，不跨发动机。
"""

import pandas as pd

# 官方测试集最短轨迹是 31 周期，窗口取 30，每台测试发动机在预测时都能凑满窗口。
WINDOW = 30


def _rolling_slope(cycle: pd.Series, values: pd.Series, units: pd.Series, window: int) -> pd.Series:
    """最小二乘斜率的闭式解：cov(t, y) / var(t)，每个窗口一个值，比逐窗 polyfit 快得多。"""

    def rolling_mean(series: pd.Series) -> pd.Series:
        return series.groupby(units).rolling(window, min_periods=1).mean().droplevel(0)

    # 先减去每台发动机的首个读数：斜率不受平移影响，而恒定信号会变成精确的 0。
    # 否则浮点误差会让恒定传感器的斜率变成 1e-14 级的噪声，骗过方差筛选。
    values = values - values.groupby(units).transform("first")
    covariance = rolling_mean(cycle * values) - rolling_mean(cycle) * rolling_mean(values)
    variance = rolling_mean(cycle**2) - rolling_mean(cycle) ** 2
    # 只有 1 个点时方差为 0，斜率没有定义。这里写 0：还没有观察到趋势。
    # 用 0.25 判断：两个相邻周期的方差正好是 0.25，再小的就只可能是浮点误差。
    return (covariance / variance).where(variance >= 0.25, 0.0)


def add_window_features(
    df: pd.DataFrame,
    columns: list[str],
    window: int = WINDOW,
) -> tuple[pd.DataFrame, list[str]]:
    """返回 (带窗口特征的副本, 新增列名)。df 中的周期必须按发动机升序排列。

    轨迹开头不足一个窗口时，用已有的周期计算。
    """
    ordered = df.sort_values(["unit", "cycle"])
    if not ordered.index.equals(df.index):
        raise ValueError("df 必须按 unit、cycle 升序排列")

    new_columns = {}
    for column in columns:
        rolling = df.groupby("unit")[column].rolling(window, min_periods=1)
        new_columns[f"{column}_mean"] = rolling.mean().droplevel(0)
        # 只有 1 个点时标准差是 NaN，写 0：还没有观察到波动。
        new_columns[f"{column}_std"] = rolling.std().droplevel(0).fillna(0.0)
        new_columns[f"{column}_slope"] = _rolling_slope(
            df["cycle"].astype(float), df[column], df["unit"], window
        )

    features = pd.DataFrame(new_columns)
    return pd.concat([df, features], axis=1), list(new_columns)
