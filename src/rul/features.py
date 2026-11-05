"""滑动窗口特征：最近 N 个周期的均值、标准差和斜率。

单个周期的读数噪声大（见 01_eda 第 7 节），退化趋势比单点读数更有信息。

窗口只看当前和过去的周期，部署时也能算出来；如果用到未来的周期，就是时间上的
数据泄漏。窗口按发动机分别计算，不跨发动机。
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from rul.data import SETTING_COLUMNS

# FD001 官方测试集最短轨迹是 31 周期，窗口取 30，预测时每台都能凑满窗口。
# FD002、FD004 最短只有 21、19 周期，凑不满时用已有的周期计算。
WINDOW = 30
# 标准差低于这个值视为恒定：恒定读数的标准差可能因浮点误差略大于 0。
_MIN_STD = 1e-9


class ConditionNormalizer:
    """按工况把传感器读数转成 z 分数。

    FD002、FD004 有六种工况，工况切换造成的读数跳变远大于退化信号。先按工况设置
    聚类，再用每种工况自己的均值和标准差归一化，留下的主要是相对该工况的偏离。

    聚类和统计量都要从数据拟合，所以必须只在训练数据上 fit，否则就是泄漏。
    """

    def __init__(self, columns: list[str], n_conditions: int, seed: int = 42):
        self.columns = columns
        self.n_conditions = n_conditions
        self.seed = seed

    def _labels(self, df: pd.DataFrame) -> np.ndarray:
        if self.n_conditions == 1:
            return np.zeros(len(df), dtype=int)
        return self.kmeans_.predict(self.scaler_.transform(df[SETTING_COLUMNS]))

    def fit(self, df: pd.DataFrame) -> "ConditionNormalizer":
        if self.n_conditions > 1:
            # 三个设置量纲差很多（0–42、0–0.84、60–100），不缩放聚类会被最大的主导
            self.scaler_ = StandardScaler().fit(df[SETTING_COLUMNS])
            self.kmeans_ = KMeans(n_clusters=self.n_conditions, n_init=10, random_state=self.seed)
            self.kmeans_.fit(self.scaler_.transform(df[SETTING_COLUMNS]))
        grouped = df[self.columns].groupby(self._labels(df))
        self.means_ = grouped.mean()
        self.stds_ = grouped.std()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        labels = self._labels(df)
        means = self.means_.reindex(labels).to_numpy()
        stds = self.stds_.reindex(labels).to_numpy()
        values = df[self.columns].to_numpy(dtype=float)
        # 该工况下恒定的传感器没有信息，写 0
        z_scores = np.divide(values - means, stds, out=np.zeros_like(values), where=stds > _MIN_STD)
        return df.assign(**dict(zip(self.columns, z_scores.T)))


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
