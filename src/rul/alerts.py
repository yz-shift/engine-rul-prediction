"""评估“预测 RUL 低于阈值就预警”这条维修规则。

RMSE 好不等于维修决策好。实际要问的是：发动机能提前多久收到预警？会不会有发动机直到
故障都没有预警？健康阶段会不会频繁误报？
"""

import pandas as pd

THRESHOLDS = (20, 30, 40, 50, 60)
# 真实 RUL 高于这个值时报警，算作健康阶段的误报
HEALTHY_RUL = 125
LATE_ALERT_RUL = 10


def alert_table(predictions: pd.DataFrame, thresholds: tuple[int, ...] = THRESHOLDS) -> list[dict]:
    """predictions 是发动机完整轨迹的逐周期预测，需要 unit、rul、predicted 三列。

    必须用留出发动机的预测，否则结果没有意义。
    """
    n_engines = predictions["unit"].nunique()
    healthy = predictions["rul"] > HEALTHY_RUL
    rows = []
    for threshold in thresholds:
        alerts = predictions["predicted"] <= threshold
        # 第一次预警时的真实 RUL。RUL 随周期递减，所以每台的最大值就是第一次
        first_alert = predictions.loc[alerts].groupby("unit")["rul"].max()
        rows.append(
            {
                "threshold": threshold,
                "engines_without_alert": n_engines - len(first_alert),
                "median_rul_at_first_alert": float(first_alert.median()) if len(first_alert) else None,
                "lowest_rul_at_first_alert": int(first_alert.min()) if len(first_alert) else None,
                "first_alert_rul_below_10": int((first_alert < LATE_ALERT_RUL).sum()),
                "first_alert_rul_above_100": int((first_alert > 100).sum()),
                # 健康阶段里被判成需要维修的周期比例
                "healthy_cycle_alert_rate": float(alerts[healthy].mean()) if healthy.any() else 0.0,
            }
        )
    return rows
