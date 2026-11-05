"""报晚罚得更重的 LightGBM。

官方评分对报晚（高估寿命）罚得更重，而默认的平方损失对两边一视同仁。这里用加权
平方损失：误差 e = 预测 - 真实，损失 = w * e^2 / 2，报晚时 w = late_weight，报早时 w = 1。

为什么不直接优化官方评分：它是指数函数，大误差处梯度爆炸，训练很不稳定，而且分数会
被个别样本主导。加权平方损失更稳定，只有一个参数，也容易解释。

决策规则：候选必须满足分数比 late_weight = 1 低至少 10%，且临近故障 RMSE 上升不超过 5%；
在满足条件的候选中选 CV 官方分数最低的，没有候选满足时不用不对称损失。

规则的局限：两个指标都只统计 RUL <= 125，看不到健康阶段被低估的代价。
"""

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin

MIN_SCORE_IMPROVEMENT = 0.10
MAX_RMSE_INCREASE = 0.05


def late_weighted_objective(late_weight: float):
    """返回 LightGBM 自定义目标函数，给出损失对预测值的一阶、二阶导数。"""

    def objective(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        error = y_pred - y_true
        weight = np.where(error > 0, late_weight, 1.0)
        return weight * error, weight

    return objective


class LateWeightedLightGBM(RegressorMixin, BaseEstimator):
    """用 late_weighted_objective 训练的 LightGBM。

    LightGBM 使用自定义目标时不会从标签均值开始提升，而是从 0 开始。这里先减去标签
    均值，预测时再加回来，让 late_weight = 1 时和普通平方损失等价。

    开启列抽样时，自定义目标下的随机流和普通 LightGBM 不同，预测逐点不相等，只是误差水平
    相同。所以比较权重时要以这个类的 late_weight = 1 为基线。
    """

    def __init__(
        self,
        late_weight: float = 2.0,
        n_estimators: int = 500,
        learning_rate: float = 0.03,
        num_leaves: int = 15,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        subsample_freq: int = 1,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 0.0,
        random_state: int = 42,
    ):
        self.late_weight = late_weight
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state

    def fit(self, X, y):
        self.offset_ = float(np.mean(y))
        self.model_ = LGBMRegressor(
            objective=late_weighted_objective(self.late_weight),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            colsample_bytree=self.colsample_bytree,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
        self.model_.fit(X, np.asarray(y, dtype=float) - self.offset_)
        return self

    def predict(self, X):
        return self.model_.predict(X) + self.offset_


def choose_late_weight(table: list[dict]) -> float:
    """按文件开头的决策规则选权重。table 的每行有 late_weight、score_per_100、rmse_near_failure。"""
    baseline = next(row for row in table if row["late_weight"] == 1)
    max_score = baseline["score_per_100"] * (1 - MIN_SCORE_IMPROVEMENT)
    max_rmse = baseline["rmse_near_failure"] * (1 + MAX_RMSE_INCREASE)
    acceptable = [
        row
        for row in table
        if row["late_weight"] > 1 and row["score_per_100"] <= max_score and row["rmse_near_failure"] <= max_rmse
    ]
    return min(acceptable, key=lambda row: row["score_per_100"])["late_weight"] if acceptable else 1.0
