"""模型定义。所有模型都包在 Pipeline 里，让筛选和缩放只在训练折上拟合。"""

from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def _linear_regression(seed: int) -> Pipeline:
    # 线性回归对量纲和共线性敏感：传感器量纲差很多，所以要标准化。seed 用不上。
    return Pipeline(
        [
            ("drop_constant", VarianceThreshold()),
            ("scale", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )


def _random_forest(seed: int) -> Pipeline:
    # 树模型按阈值切分，不受量纲影响，所以不需要标准化。
    # min_samples_leaf 防止记住单个周期的噪声；max_features 让各棵树差异更大。
    return Pipeline(
        [
            ("drop_constant", VarianceThreshold()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    min_samples_leaf=5,
                    max_features=0.33,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )


def _lightgbm(seed: int) -> Pipeline:
    # 小学习率配较多棵树；num_leaves 较小，防止对 80 台发动机过拟合。
    return Pipeline(
        [
            ("drop_constant", VarianceThreshold()),
            (
                "model",
                LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.03,
                    num_leaves=15,
                    min_child_samples=20,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.8,
                    random_state=seed,
                    deterministic=True,
                    force_row_wise=True,
                    verbose=-1,
                ),
            ),
        ]
    )


MODEL_BUILDERS = {
    "linear_regression": _linear_regression,
    "random_forest": _random_forest,
    "lightgbm": _lightgbm,
}


def build_model(name: str, seed: int = SEED) -> Pipeline:
    """按名称创建未训练的模型。"""
    if name not in MODEL_BUILDERS:
        raise ValueError(f"未知模型 {name!r}，可选：{sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](seed)
