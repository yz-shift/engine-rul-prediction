"""发动机剩余寿命预警界面。

启动：python -m streamlit run app/streamlit_app.py

只展示官方测试集的发动机：模型只用训练集的 100 台拟合，这些发动机它没见过。
"""

import json

import altair as alt
import pandas as pd
import streamlit as st

from rul.data import DEFAULT_DATA_DIR, N_CONDITIONS, SENSOR_COLUMNS, SUBSETS, load_rul, load_trajectories
from rul.experiments import FINAL_FILE, RESULTS_DIR, best_experiment, fit_final
from rul.inference import engine_index, first_alert_cycle, needs_maintenance, predict_engine

# 默认阈值来自 02_modeling 第 4 节：训练集交叉验证外预测中，阈值 40 时第一次预警最晚还剩 26 周期。
DEFAULT_THRESHOLD = 40

SENSOR_NAMES = {
    "sensor_2": "T24 · LPC outlet temperature (°R)",
    "sensor_3": "T30 · HPC outlet temperature (°R)",
    "sensor_4": "T50 · LPT outlet temperature (°R)",
    "sensor_7": "P30 · HPC outlet pressure (psia)",
    "sensor_8": "Nf · fan speed (rpm)",
    "sensor_9": "Nc · core speed (rpm)",
    "sensor_11": "Ps30 · HPC outlet static pressure (psia)",
    "sensor_12": "phi · fuel flow / Ps30 (pps/psi)",
    "sensor_13": "NRf · corrected fan speed (rpm)",
    "sensor_14": "NRc · corrected core speed (rpm)",
    "sensor_15": "BPR · bypass ratio",
    "sensor_17": "htBleed · bleed enthalpy",
    "sensor_20": "W31 · HPT coolant bleed (lbm/s)",
    "sensor_21": "W32 · LPT coolant bleed (lbm/s)",
}
DEFAULT_SENSORS = ["sensor_3", "sensor_4", "sensor_11", "sensor_12"]

PREDICTED_COLOR = "#2a78d6"
TRUE_COLOR = "#eb6834"
ALERT_COLOR = "#8f8e88"


@st.cache_resource(show_spinner="正在用训练发动机拟合模型……")
def load_model(subset: str):
    """拟合最终模型并算好测试集特征。缓存后切换发动机不用重新训练。"""
    fitted = fit_final(best_experiment(subset), load_trajectories(subset, "train"))
    table, _ = fitted.featurizer.transform(load_trajectories(subset, "test"))
    return fitted, table, load_rul(subset)


@st.cache_data
def load_final_metrics(subset: str) -> dict:
    return json.loads((RESULTS_DIR / subset / FINAL_FILE).read_text(encoding="utf-8"))


def rul_chart(trajectory: pd.DataFrame, threshold: float) -> alt.LayerChart:
    long = trajectory.melt("cycle", ["predicted", "true"], var_name="series", value_name="RUL")
    long["series"] = long["series"].map({"predicted": "Predicted RUL", "true": "True RUL"})
    x = alt.X("cycle:Q", title="Cycle", scale=alt.Scale(nice=False))
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=2)
        .encode(
            x=x,
            y=alt.Y("RUL:Q", title="RUL (cycles)", scale=alt.Scale(domainMin=0)),
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(
                    domain=["Predicted RUL", "True RUL"],
                    range=[PREDICTED_COLOR, TRUE_COLOR],
                ),
                legend=alt.Legend(orient="top"),
            ),
            strokeDash=alt.StrokeDash(
                "series:N",
                scale=alt.Scale(
                    domain=["Predicted RUL", "True RUL"],
                    range=[[1, 0], [5, 3]],
                ),
                legend=None,
            ),
        )
    )
    hover = (
        alt.Chart(trajectory)
        .mark_rule(opacity=0)
        .encode(
            x="cycle:Q",
            tooltip=[
                alt.Tooltip("cycle:Q", title="Cycle"),
                alt.Tooltip("predicted:Q", title="Predicted RUL", format=".0f"),
                alt.Tooltip("true:Q", title="True RUL"),
            ],
        )
    )
    alert_line = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(
        color=ALERT_COLOR, strokeDash=[2, 2], strokeWidth=1.5
    ).encode(y="y:Q")
    alert_label = alt.Chart(
        pd.DataFrame({"y": [threshold], "text": [f"Alert line = {threshold}"]})
    ).mark_text(align="left", baseline="bottom", dx=4, dy=-3, color=ALERT_COLOR).encode(
        x=alt.value(0), y="y:Q", text="text:N"
    )
    return alt.layer(lines, alert_line, alert_label, hover).properties(height=320)


def sensor_chart(engine: pd.DataFrame, sensor: str) -> alt.Chart:
    return (
        alt.Chart(engine, title=alt.Title(SENSOR_NAMES[sensor], fontSize=13, anchor="start"))
        .mark_line(strokeWidth=1.5, color=PREDICTED_COLOR)
        .encode(
            x=alt.X("cycle:Q", title="Cycle", scale=alt.Scale(nice=False)),
            y=alt.Y(f"{sensor}:Q", title=None, scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("cycle:Q", title="Cycle"), alt.Tooltip(f"{sensor}:Q", title="Reading")],
        )
        .properties(height=200)
    )


def show_status(predicted: float, threshold: float, alert_cycle: int | None) -> None:
    if needs_maintenance(predicted, threshold):
        since = f"从第 {alert_cycle} 周期起首次触发。" if alert_cycle is not None else ""
        st.error(f"**该安排维修了**：预测剩余 {predicted:.0f} 周期，已低于预警线 {threshold}。{since}", icon="🔧")
    elif alert_cycle is not None:
        st.warning(
            f"**需要留意**：当前预测 {predicted:.0f} 周期，高于预警线 {threshold}，但第 {alert_cycle} 周期曾触发过预警。",
            icon="⚠️",
        )
    else:
        st.success(f"**运行正常**：预测剩余 {predicted:.0f} 周期，高于预警线 {threshold}。", icon="✅")


def main() -> None:
    st.set_page_config(page_title="发动机剩余寿命预警", page_icon="🛠️", layout="wide")
    st.title("发动机剩余寿命（RUL）预警")

    if not (DEFAULT_DATA_DIR / "train_FD001.txt").exists():
        st.error("没有找到数据。请先运行：python scripts/download_data.py")
        st.stop()

    with st.sidebar:
        st.header("设置")
        requested_subset = st.query_params.get("subset")
        subset = st.selectbox(
            "数据子集",
            SUBSETS,
            index=SUBSETS.index(requested_subset) if requested_subset in SUBSETS else 0,
            help="FD001、FD003 只有一种工况，FD002、FD004 有六种；FD003、FD004 有两种故障模式。",
        )

    fitted, table, true_rul = load_model(subset)
    experiment, model, columns = fitted.experiment, fitted.model, fitted.columns
    final = load_final_metrics(subset)

    with st.sidebar:
        units = sorted(table["unit"].unique())
        # 支持 ?engine=34 这样的链接，可以直接打开并分享某一台发动机
        unit = st.selectbox(
            "测试发动机编号", units, index=engine_index(units, st.query_params.get("engine"))
        )
        threshold = st.slider(
            "预警线（预测 RUL，周期）",
            min_value=10,
            max_value=80,
            value=DEFAULT_THRESHOLD,
            step=5,
            help="预测 RUL 低于等于这条线时提示维修。默认 40 来自训练集的交叉验证外预测，见 02_modeling 第 4 节。",
        )
        sensors = st.multiselect(
            "传感器",
            [s for s in SENSOR_COLUMNS if s in SENSOR_NAMES],
            default=DEFAULT_SENSORS,
            format_func=lambda s: SENSOR_NAMES[s],
        )
        st.divider()
        st.caption(
            f"模型：`{experiment.name}`，只用训练发动机拟合。"
            f"RUL ≤ 125 时，交叉验证 RMSE {final['cv']['rmse_near_failure']:.1f} 周期，"
            f"官方测试集 RMSE {final['test']['rmse_near_failure']:.1f} 周期；"
            f"测试集全部发动机 RMSE {final['test']['rmse']:.1f} 周期。"
        )

    trajectory = predict_engine(model, table, columns, unit, int(true_rul.loc[unit]))
    current = trajectory.iloc[-1]
    alert_cycle = first_alert_cycle(trajectory, threshold)

    show_status(current["predicted"], threshold, alert_cycle)

    observed, predicted, actual, error = st.columns(4)
    observed.metric("已观测周期", int(current["cycle"]))
    predicted.metric("预测 RUL", f"{current['predicted']:.0f}")
    actual.metric("真实 RUL（官方答案）", int(current["true"]))
    error.metric("预测 - 真实", f"{current['predicted'] - current['true']:+.0f}", help="正数表示报晚（高估寿命），风险更大。")

    st.subheader("剩余寿命预测")
    st.altair_chart(rul_chart(trajectory, threshold), width="stretch")
    st.caption(
        "模型训练时把 RUL 截顶，所以健康阶段的预测会停在上限附近，这不代表真实寿命只剩这么多。"
        "真实 RUL 曲线是由官方答案倒推出来的，模型看不到。"
    )

    st.subheader("传感器曲线")
    if experiment.features.startswith("condition") and N_CONDITIONS[subset] > 1:
        st.caption("这个子集有多种工况，这里显示的是按工况归一化后的读数（z 分数），原始读数会被工况切换主导。")
    if not sensors:
        st.info("在左侧至少选择一个传感器。")
    engine = table[table["unit"] == unit]
    for start in range(0, len(sensors), 2):
        for column, sensor in zip(st.columns(2), sensors[start : start + 2]):
            column.altair_chart(sensor_chart(engine, sensor), width="stretch")

    st.caption(f"数据：NASA C-MAPSS {subset} 仿真数据。结果不能直接外推到真实设备，只用于演示预测性维护的方法。")


if __name__ == "__main__":
    main()
