"""比较不对称损失的权重，并确定每个子集的最终模型，写入 results/<子集>/final.json。

用法：python scripts/tune_asymmetric.py [--subsets FD001 ...]

先读 tuning.json：搜索结果被采用就用搜索到的参数，否则用默认参数。然后用同样的 5 折
比较不同的 late_weight。选权重只看交叉验证；官方测试集只评估最终模型。
"""

import argparse
import json

from rul.asymmetric import choose_late_weight
from rul.data import SUBSETS, load_rul, load_trajectories
from rul.evaluate import cross_validate, evaluate_on_test
from rul.experiments import (
    FINAL_FILE,
    RESULTS_DIR,
    Experiment,
    best_cv_experiment,
    build_featurizer,
    fit_final,
)
from rul.features import WINDOW
from rul.models import SEED, build_model
from rul.results import write_json

LATE_WEIGHTS = (1, 1.5, 2, 3, 5)
ASYMMETRIC_MODEL = "lightgbm_asymmetric"


def base_configuration(subset: str) -> tuple[str, int | None, int, dict, str]:
    """返回 (特征集, 截顶, 窗口, LightGBM 参数, 来源)。"""
    tuning = json.loads((RESULTS_DIR / subset / "tuning.json").read_text(encoding="utf-8"))
    if tuning["adopt_tuned"]:
        chosen = dict(tuning["chosen"])
        return tuning["features"], chosen.pop("cap"), chosen.pop("window"), chosen, "tuned"
    default = best_cv_experiment(subset)
    return default.features, default.cap, WINDOW, {}, "default"


def compare_weights(subset: str, train) -> dict:
    features, cap, window, params, source = base_configuration(subset)
    table = []
    for late_weight in LATE_WEIGHTS:
        folds = cross_validate(
            lambda: build_model(ASYMMETRIC_MODEL, SEED, {**params, "late_weight": late_weight}),
            lambda: build_featurizer(features, subset, window),
            train,
            cap,
            seed=SEED,
        )
        table.append({"late_weight": late_weight, **folds.mean().to_dict()})
        print(f"{subset} late_weight={late_weight} {table[-1]}", flush=True)

    late_weight = choose_late_weight(table)
    if late_weight == 1:
        # 不采用不对称损失时，最终模型就是普通 LightGBM
        model, final_params, tag = "lightgbm", params, source if source == "tuned" else ""
    else:
        model, final_params, tag = ASYMMETRIC_MODEL, {**params, "late_weight": late_weight}, "final"
    experiment = Experiment(
        model, features, cap, subset, window=window, model_params=tuple(final_params.items()), tag=tag
    )
    return {"experiment": experiment, "params_source": source, "late_weight": late_weight, "weights": table}


def main() -> None:
    parser = argparse.ArgumentParser(description="比较不对称损失并确定最终模型")
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    for subset in parser.parse_args().subsets:
        train = load_trajectories(subset, "train")
        comparison = compare_weights(subset, train)
        experiment = comparison.pop("experiment")

        chosen_row = next(r for r in comparison["weights"] if r["late_weight"] == comparison["late_weight"])
        fitted = fit_final(experiment, train)
        test_table, _ = fitted.featurizer.transform(load_trajectories(subset, "test"))
        result = {
            "subset": subset,
            "seed": SEED,
            "model": experiment.model,
            "features": experiment.features,
            "cap": experiment.cap,
            "window": experiment.window,
            "model_params": dict(experiment.model_params),
            "tag": experiment.tag,
            **comparison,
            "cv": {k: v for k, v in chosen_row.items() if k != "late_weight"},
            "test": evaluate_on_test(fitted.model, test_table, fitted.columns, load_rul(subset)),
        }
        path = RESULTS_DIR / subset / FINAL_FILE
        write_json(path, result)
        print(f"{subset} 最终：{experiment.name} {result['cv']} 测试 {result['test']}", flush=True)


if __name__ == "__main__":
    main()
