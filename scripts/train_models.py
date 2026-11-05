"""运行全部实验，把每个模型的指标写进 results/。

用法：python scripts/train_models.py [--subsets FD001 FD002 ...]

流程：先在训练集上做交叉验证，再用全部训练发动机拟合，在官方测试集上评估。
模型参数是事先定好的，不根据测试集结果调整；选模型只看交叉验证。
"""

import argparse
from dataclasses import asdict

import pandas as pd

from rul.data import SUBSETS, load_rul, load_trajectories
from rul.evaluate import cross_validate_experiment, evaluate_on_test
from rul.experiments import EXPERIMENTS, RESULTS_DIR, Experiment, fit_final
from rul.models import SEED
from rul.results import write_json


def run(experiment: Experiment, train: pd.DataFrame, test: pd.DataFrame, true_rul: pd.Series) -> dict:
    folds = cross_validate_experiment(experiment, train, seed=SEED)

    final = fit_final(experiment, train)
    test_table, _ = final.featurizer.transform(test)

    return {
        **asdict(experiment),
        "seed": SEED,
        "feature_columns": final.columns,
        "cv_mean": folds.mean().to_dict(),
        "cv_std": folds.std().to_dict(),
        "cv_folds": folds.reset_index().to_dict(orient="records"),
        "test": evaluate_on_test(final.model, test_table, final.columns, true_rul),
    }


def summarise(results: list[dict]) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "model": r["model"],
            "features": r["features"],
            "cap": r["cap"] if r["cap"] else "none",
            "cv_rmse_near_failure": r["cv_mean"]["rmse_near_failure"],
            "cv_rmse_near_failure_std": r["cv_std"]["rmse_near_failure"],
            "cv_rmse_all_cycles": r["cv_mean"]["rmse_all_cycles"],
            "cv_score_per_100": r["cv_mean"]["score_per_100"],
            "test_rmse": r["test"]["rmse"],
            "test_rmse_near_failure": r["test"]["rmse_near_failure"],
            "test_score": r["test"]["score"],
        }
        for r in results
    )
    return summary.sort_values("cv_rmse_near_failure").round(2)


def run_subset(subset: str) -> None:
    train = load_trajectories(subset, "train")
    test = load_trajectories(subset, "test")
    true_rul = load_rul(subset)

    output_dir = RESULTS_DIR / subset
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for experiment in EXPERIMENTS[subset]:
        result = run(experiment, train, test, true_rul)
        path = output_dir / f"{experiment.name}.json"
        write_json(path, result)
        results.append(result)
        print(f"完成 {subset} {experiment.name}", flush=True)

    summary = summarise(results)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行全部实验")
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    for subset in parser.parse_args().subsets:
        run_subset(subset)


if __name__ == "__main__":
    main()
