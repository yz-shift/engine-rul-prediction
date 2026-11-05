"""运行全部实验，把每个模型的指标写进 results/。

用法：python scripts/train_models.py

流程：先在训练集上做交叉验证，再用全部训练发动机拟合，在官方测试集上评估。
模型参数是事先定好的，不根据测试集结果调整；选模型只看交叉验证。
"""

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from rul.data import load_rul, load_trajectories
from rul.evaluate import cross_validate, evaluate_on_test
from rul.experiments import EXPERIMENTS, Experiment, fit_final, prepare_table
from rul.models import SEED, build_model

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
SUBSET = "FD001"


def run(experiment: Experiment, train: pd.DataFrame, test: pd.DataFrame, true_rul: pd.Series) -> dict:
    table, columns = prepare_table(train, experiment.features)
    folds = cross_validate(
        lambda: build_model(experiment.model, SEED), table, columns, experiment.cap, seed=SEED
    )

    final = fit_final(experiment, train)
    test_table, _ = prepare_table(test, experiment.features)

    return {
        **asdict(experiment),
        "subset": SUBSET,
        "seed": SEED,
        "feature_columns": columns,
        "cv_mean": folds.mean().to_dict(),
        "cv_std": folds.std().to_dict(),
        "cv_folds": folds.reset_index().to_dict(orient="records"),
        "test": evaluate_on_test(final, test_table, columns, true_rul),
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
            "test_score": r["test"]["score"],
        }
        for r in results
    )
    return summary.sort_values("cv_rmse_near_failure").round(2)


def main() -> None:
    train = load_trajectories(SUBSET, "train")
    test = load_trajectories(SUBSET, "test")
    true_rul = load_rul(SUBSET)

    RESULTS_DIR.mkdir(exist_ok=True)
    results = []
    for experiment in EXPERIMENTS:
        result = run(experiment, train, test, true_rul)
        path = RESULTS_DIR / f"{experiment.name}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        results.append(result)
        print(f"完成 {experiment.name}", flush=True)

    summary = summarise(results)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
