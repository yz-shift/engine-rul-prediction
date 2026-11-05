"""模型定义。所有模型都包在 Pipeline 里，让筛选和缩放只在训练折上拟合。"""

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


MODEL_BUILDERS = {
    "linear_regression": _linear_regression,
}


def build_model(name: str, seed: int = SEED) -> Pipeline:
    """按名称创建未训练的模型。"""
    if name not in MODEL_BUILDERS:
        raise ValueError(f"未知模型 {name!r}，可选：{sorted(MODEL_BUILDERS)}")
    return MODEL_BUILDERS[name](seed)
