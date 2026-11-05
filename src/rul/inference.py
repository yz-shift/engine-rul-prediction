"""界面用到的预测逻辑。放在这里而不是 Streamlit 脚本里，方便单元测试。"""

import pandas as pd
from sklearn.base import RegressorMixin

from rul.evaluate import predict_rul


def predict_engine(
    model: RegressorMixin,
    table: pd.DataFrame,
    columns: list[str],
    unit: int,
    final_rul: int,
) -> pd.DataFrame:
    """逐周期预测一台测试发动机，返回 cycle、predicted、true 三列。

    官方只给出轨迹最后一个周期的真实 RUL。RUL 每过一个周期减 1，所以可以
    倒推出更早周期的真实值：true = final_rul + (最后周期 - 当前周期)。
    """
    engine = table[table["unit"] == unit].sort_values("cycle")
    if engine.empty:
        raise ValueError(f"没有发动机 {unit}")
    cycles = engine["cycle"].to_numpy()
    return pd.DataFrame(
        {
            "cycle": cycles,
            "predicted": predict_rul(model, engine[columns]),
            "true": final_rul + cycles.max() - cycles,
        }
    )


def engine_index(units: list[int], requested: str | None) -> int:
    """把 URL 参数里的发动机编号转成下拉框位置；无效时退回第一台，不报错。"""
    if requested is not None and requested.isdecimal() and int(requested) in units:
        return units.index(int(requested))
    return 0


def first_alert_cycle(trajectory: pd.DataFrame, threshold: float) -> int | None:
    """预测 RUL 第一次低于等于阈值的周期；还没触发时返回 None。"""
    alerts = trajectory.loc[trajectory["predicted"] <= threshold, "cycle"]
    return int(alerts.iloc[0]) if not alerts.empty else None


def needs_maintenance(predicted_rul: float, threshold: float) -> bool:
    """当前预测 RUL 低于等于预警线时，应该安排维修。"""
    return bool(predicted_rul <= threshold)
