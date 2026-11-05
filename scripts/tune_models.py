"""LightGBM 超参数搜索，结果写入 results/<子集>/tuning.json。

用法：python scripts/tune_models.py [--subsets FD001 ...]

只用训练发动机选参数。官方测试集只在最后评估默认参数和选中的参数各一次。
已经有结果的子集会跳过，所以运行中断后可以直接重跑。加 --force 重新计算。
"""

import argparse
import json

import pandas as pd

from rul.data import SUBSETS, load_rul, load_trajectories
from rul.evaluate import evaluate_on_test
from rul.experiments import RESULTS_DIR, Experiment, best_cv_experiment, fit_final
from rul.models import SEED
from rul.results import write_json
from rul.tuning import (
    INNER_SPLITS,
    Configuration,
    N_TRIALS,
    OUTER_SEED,
    OUTER_SPLITS,
    evaluate_configurations,
    nested_cross_validate,
    sample_configurations,
    select_best,
)


CAPS_TO_REPORT = (100, 125, 150)


def tune_subset(subset: str) -> dict:
    train = load_trajectories(subset, "train")
    test = load_trajectories(subset, "test")
    true_rul = load_rul(subset)

    # 特征集和默认参数的基准，都来自 train_models.py 的交叉验证
    default = best_cv_experiment(subset)
    default_result = json.loads(
        (RESULTS_DIR / subset / f"{default.name}.json").read_text(encoding="utf-8")
    )
    configurations = sample_configurations()

    nested = nested_cross_validate(configurations, train, default.features, subset)
    print(f"{subset} 嵌套 CV 完成", flush=True)

    # 全部训练发动机上的普通 CV，用来选最终配置。这个最优值偏乐观，只作参考。
    search = evaluate_configurations(
        configurations, train, default.features, subset, OUTER_SPLITS, OUTER_SEED
    )
    chosen = select_best(configurations, search)

    # 不同截顶的取舍：只报告，不自动决定（原因见 rul.tuning 开头）
    cap_variants = [Configuration(chosen.window, cap, chosen.model_params) for cap in CAPS_TO_REPORT]
    cap_metrics = evaluate_configurations(cap_variants, train, default.features, subset, OUTER_SPLITS, OUTER_SEED)
    cap_sensitivity = [
        {"cap": cap, **row.to_dict()} for cap, (_, row) in zip(CAPS_TO_REPORT, cap_metrics.iterrows())
    ]

    tuned = Experiment(
        "lightgbm",
        default.features,
        chosen.cap,
        subset,
        window=chosen.window,
        model_params=chosen.model_params,
        tag="tuned",
    )
    fitted = fit_final(tuned, train)
    test_table, _ = fitted.featurizer.transform(test)

    default_folds = pd.DataFrame(default_result["cv_folds"]).set_index("fold")["rmse_near_failure"]
    differences = nested["rmse_near_failure"] - default_folds
    return {
        "subset": subset,
        "seed": SEED,
        "n_trials": N_TRIALS,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "features": default.features,
        "default": {
            "experiment": default.name,
            "cv_rmse_near_failure": default_result["cv_mean"]["rmse_near_failure"],
            "cv_score_per_100": default_result["cv_mean"]["score_per_100"],
            "test": default_result["test"],
        },
        "nested_cv": {
            "rmse_near_failure": nested["rmse_near_failure"].mean(),
            "rmse_near_failure_std": nested["rmse_near_failure"].std(),
            "score_per_100": nested["score_per_100"].mean(),
            "fold_rmse": nested["rmse_near_failure"].tolist(),
            "chosen_per_fold": nested["chosen"].tolist(),
        },
        "paired_vs_default": {
            "mean_rmse_change": differences.mean(),
            "folds_better": int((differences < 0).sum()),
        },
        "search_best_cv_optimistic": {
            "rmse_near_failure": float(search["rmse_near_failure"].min()),
            "score_per_100": float(search.loc[search["rmse_near_failure"].idxmin(), "score_per_100"]),
        },
        "cap_sensitivity_cv": cap_sensitivity,
        "chosen": chosen.as_dict(),
        "tuned_test": evaluate_on_test(fitted.model, test_table, fitted.columns, true_rul),
        # 决策规则见 rul.tuning：嵌套 CV 赢过默认参数才采用
        "adopt_tuned": bool(differences.mean() < 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM 超参数搜索")
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for subset in args.subsets:
        path = RESULTS_DIR / subset / "tuning.json"
        if path.exists() and not args.force:
            print(f"跳过 {subset}：已有结果", flush=True)
            continue
        result = tune_subset(subset)
        write_json(path, result)
        print(
            f"{subset}: 默认 {result['default']['cv_rmse_near_failure']} | "
            f"嵌套 {result['nested_cv']['rmse_near_failure']} | "
            f"搜索最优（偏乐观）{result['search_best_cv_optimistic']['rmse_near_failure']} | "
            f"测试 {result['default']['test']['rmse']} -> {result['tuned_test']['rmse']} | "
            f"采用 {result['adopt_tuned']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
