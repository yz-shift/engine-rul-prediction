"""运行完整的模型选择流程，确定每个子集的最终模型。

用法：python scripts/tune_models.py [--subsets FD001 ...]

输出 results/<子集>/tuning.json（完整记录）和 final.json（最终模型，界面读取它）。
需要先运行 train_models.py，它提供默认参数的基准。

只用训练发动机做选择。官方测试集只评估选中的模型，不参与这里的任何决策。
"""

import argparse
import json

import pandas as pd

from rul.alerts import alert_table
from rul.data import SUBSETS, load_rul, load_trajectories
from rul.evaluate import evaluate_on_test, out_of_fold_predictions
from rul.experiments import FINAL_FILE, RESULTS_DIR, Experiment, best_cv_experiment, fit_final
from rul.models import SEED
from rul.results import write_json
from rul.tuning import (
    INNER_SPLITS,
    N_TRIALS,
    OUTER_SEED,
    OUTER_SPLITS,
    Configuration,
    evaluate_configurations,
    nested_cross_validate,
    select_pipeline,
)

CAPS_TO_REPORT = (100, 125, 150)


def cap_sensitivity(experiment: Experiment, train: pd.DataFrame) -> list[dict]:
    """不同截顶的取舍：只报告，不自动决定（原因见 rul.tuning 开头）。"""
    variants = [Configuration(experiment.window, cap, experiment.model_params) for cap in CAPS_TO_REPORT]
    metrics = evaluate_configurations(
        variants, train, experiment.features, experiment.subset, OUTER_SPLITS, OUTER_SEED
    )
    return [{"cap": cap, **row.to_dict()} for cap, (_, row) in zip(CAPS_TO_REPORT, metrics.iterrows())]


def test_metrics(experiment: Experiment, train: pd.DataFrame, test: pd.DataFrame, true_rul: pd.Series) -> dict:
    fitted = fit_final(experiment, train)
    table, _ = fitted.featurizer.transform(test)
    return evaluate_on_test(fitted.model, table, fitted.columns, true_rul)


def run_subset(subset: str) -> None:
    train = load_trajectories(subset, "train")
    test = load_trajectories(subset, "test")
    true_rul = load_rul(subset)

    default = best_cv_experiment(subset)
    default_result = json.loads((RESULTS_DIR / subset / f"{default.name}.json").read_text(encoding="utf-8"))
    default_folds = pd.DataFrame(default_result["cv_folds"]).set_index("fold")["rmse_near_failure"]

    # 嵌套 CV 评估整个选择流程，是对新发动机表现最可信的估计
    nested, nested_predictions = nested_cross_validate(train, subset)
    print(f"{subset} 嵌套 CV 完成", flush=True)
    differences = nested["rmse_near_failure"] - default_folds

    # 在全部训练发动机上再跑一次同样的流程，得到最终模型
    selection = select_pipeline(train, subset, OUTER_SPLITS, OUTER_SEED)
    chosen_row = next(r for r in selection.weights if r["late_weight"] == selection.late_weight)
    print(f"{subset} 最终选择完成：{selection.experiment.name}", flush=True)

    nested_summary = {
        "rmse_near_failure": nested["rmse_near_failure"].mean(),
        "rmse_near_failure_std": nested["rmse_near_failure"].std(),
        "rmse_all_cycles": nested["rmse_all_cycles"].mean(),
        "score_per_100": nested["score_per_100"].mean(),
    }
    test_final = test_metrics(selection.experiment, train, test, true_rul)
    final = {
        "subset": subset,
        "seed": SEED,
        "model": selection.experiment.model,
        "features": selection.experiment.features,
        "cap": selection.experiment.cap,
        "window": selection.experiment.window,
        "model_params": dict(selection.experiment.model_params),
        "tag": selection.experiment.tag,
        "late_weight": selection.late_weight,
        "nested_cv": nested_summary,
        "selection_cv_optimistic": {k: v for k, v in chosen_row.items() if k != "late_weight"},
        "test": test_final,
        "alerts_nested": alert_table(nested_predictions),
    }
    write_json(RESULTS_DIR / subset / FINAL_FILE, final)

    tuning = {
        "subset": subset,
        "seed": SEED,
        "n_trials": N_TRIALS,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "default": {
            "experiment": default.name,
            "cv_rmse_near_failure": default_result["cv_mean"]["rmse_near_failure"],
            "cv_score_per_100": default_result["cv_mean"]["score_per_100"],
            "test": default_result["test"],
        },
        "nested_cv": {
            **nested_summary,
            "fold_rmse": nested["rmse_near_failure"].tolist(),
            "chosen_per_fold": nested["chosen"].tolist(),
        },
        "paired_vs_default": {
            "mean_rmse_change": differences.mean(),
            "folds_better": int((differences < 0).sum()),
        },
        "symmetric": {
            "experiment": selection.symmetric.name,
            "model_params": dict(selection.symmetric.model_params),
            "window": selection.symmetric.window,
            "features": selection.symmetric.features,
            "test": test_metrics(selection.symmetric, train, test, true_rul),
            # 下面两张表用的是选择时的 CV，比嵌套结果略乐观，但两者口径相同，可以互相比较
            "alerts_cv": alert_table(out_of_fold_predictions(selection.symmetric, train)),
        },
        "selected_alerts_cv": alert_table(out_of_fold_predictions(selection.experiment, train)),
        "late_weights_cv": selection.weights,
        "cap_sensitivity_cv": cap_sensitivity(selection.symmetric, train),
        "configurations_cv": selection.configurations,
    }
    write_json(RESULTS_DIR / subset / "tuning.json", tuning)

    print(
        f"{subset}: 默认 CV {default_result['cv_mean']['rmse_near_failure']:.2f} | "
        f"嵌套 CV {nested_summary['rmse_near_failure']:.2f}（{(differences < 0).sum()}/5 折更好）| "
        f"权重 {selection.late_weight} | 测试 RMSE {test_final['rmse']:.2f}，"
        f"RUL<=125 {test_final['rmse_near_failure']:.2f}，分数 {test_final['score']:.1f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="运行完整的模型选择流程")
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    for subset in parser.parse_args().subsets:
        run_subset(subset)


if __name__ == "__main__":
    main()
