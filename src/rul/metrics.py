"""RUL 预测的评估指标。"""

import numpy as np
from numpy.typing import ArrayLike

# 官方评分函数（Saxena et al., 2008）的时间常数：预测偏晚时惩罚增长更快。
EARLY_SCALE = 13.0
LATE_SCALE = 10.0


def _errors(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """返回 预测 - 真实。形状不一致时报错，防止 numpy 广播掩盖对齐错误。"""
    true, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(f"形状不一致：真实 {true.shape}，预测 {pred.shape}")
    return pred - true


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """均方根误差，单位是周期。早报和晚报同等对待。"""
    return float(np.sqrt(np.mean(_errors(y_true, y_pred) ** 2)))


def nasa_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """官方评分（越小越好），按样本求和。

    d = 预测 - 真实。d < 0（报早了）罚 e^(-d/13) - 1，d >= 0（报晚了）罚
    e^(d/10) - 1。报晚意味着发动机可能在计划维修前就坏了，所以罚得更重。

    因为是指数增长又是求和，分数主要由少数大误差决定，并且随样本数变大。
    比较时样本数必须相同，否则要先除以样本数。
    """
    d = _errors(y_true, y_pred)
    penalty = np.where(d < 0, np.exp(-d / EARLY_SCALE), np.exp(d / LATE_SCALE)) - 1
    return float(np.sum(penalty))
