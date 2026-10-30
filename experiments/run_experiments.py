import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import data_summary, load_data
from src.evaluate import mcnemar_bowker, save_fold_results, save_json, summarize_predictions
from src.hpo_hyperopt import search_hyperopt
from src.hpo_optuna import search_optuna
from src.pipelines import svm_pipeline


def get_estimator(model, variant, params=None):
    params = params or {}
    if model == "svm":
        return svm_pipeline(scaled=(variant == "scaled"), **params)
    if model == "xgb":
        return XGBClassifier(
            objective="multi:softprob", num_class=4, eval_metric="mlogloss",
            random_state=42, n_jobs=-1, **params
        )
    if model == "lgbm":
        return LGBMClassifier(
            objective="multiclass", num_class=4, random_state=42,
            n_jobs=-1, verbose=-1, **params
        )
    raise ValueError(model)


def run_cv(args, seed):
    X, y = load_data(args.data)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    predictions = {"svm_hyp": [], "svm_opt": []}
    truths = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        models = [
            ("baseline", "raw", "svm", {}),
        ]

        if args.mode in {"all", "robust", "pca"}:
            models.append(("baseline", "scaled", "svm", {}))

        if args.mode in {"all", "boosting"}:
            models.extend([
                ("baseline", "raw", "xgb", {}),
                ("baseline", "raw", "lgbm", {}),
            ])

        if args.mode in {"all", "pca"}:
            for components in [5, 10, 15, 20, 0.95]:
                estimator = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=components)),
                    ("svc", SVC(random_state=seed)),
                ])
                started = time.perf_counter()
                estimator.fit(X_train, y_train)
                pred = estimator.predict(X_test)
                elapsed = time.perf_counter() - started
                metrics = summarize_predictions(y_test, pred)
                rows.append({
                    "seed": seed, "fold": fold, "model": "svm_pca",
                    "variant": f"pca_{components}", "seconds": elapsed, **metrics,
                })

        for model_name, variant, model_type, params in models:
            estimator = get_estimator(model_type, variant, params)
            started = time.perf_counter()
            estimator.fit(X_train, y_train)
            pred = estimator.predict(X_test)
            elapsed = time.perf_counter() - started
            metrics = summarize_predictions(y_test, pred)
            rows.append({
                "seed": seed, "fold": fold, "model": model_type,
                "variant": variant, "seconds": elapsed, **metrics,
            })

        if args.mode in {"all", "reproduce", "robust"}:
            paper_space = args.mode == "reproduce"
            scaled = args.mode == "robust"

            hyp_started = time.perf_counter()
            hyp_result = search_hyperopt(
                X_train, y_train, seed=seed + fold,
                n_trials=args.trials, scaled=scaled,
                paper_space=paper_space,
            )
            hyp_elapsed = time.perf_counter() - hyp_started
            hyp_estimator = get_estimator("svm", "scaled" if scaled else "raw", hyp_result["params"])
            hyp_estimator.fit(X_train, y_train)
            hyp_pred = hyp_estimator.predict(X_test)
            hyp_metrics = summarize_predictions(y_test, hyp_pred)
            rows.append({
                "seed": seed, "fold": fold, "model": "svm_hyp",
                "variant": "robust_scaled" if scaled else "paper_raw",
                "seconds": hyp_elapsed, "best_inner_accuracy": hyp_result["best_inner_accuracy"],
                "best_params": hyp_result["params"], **hyp_metrics,
            })
            hyp_result["trials_dataframe"].to_csv(
                args.output / f"hyperopt_fold_{fold}_trials.csv", index=False
            )

            opt_started = time.perf_counter()
            opt_result = search_optuna(
                X_train, y_train, seed=seed + fold,
                n_trials=args.trials, scaled=scaled,
                paper_space=paper_space,
            )
            opt_elapsed = time.perf_counter() - opt_started
            opt_estimator = get_estimator("svm", "scaled" if scaled else "raw", opt_result["params"])
            opt_estimator.fit(X_train, y_train)
            opt_pred = opt_estimator.predict(X_test)
            opt_metrics = summarize_predictions(y_test, opt_pred)
            rows.append({
                "seed": seed, "fold": fold, "model": "svm_opt",
                "variant": "robust_scaled" if scaled else "paper_raw",
                "seconds": opt_elapsed, "best_inner_accuracy": opt_result["best_inner_accuracy"],
                "best_params": opt_result["params"], **opt_metrics,
            })
            opt_result["trials_dataframe"].to_csv(
                args.output / f"optuna_fold_{fold}_trials.csv", index=False
            )

            truths.extend(y_test.tolist())
            predictions["svm_hyp"].extend(hyp_pred.tolist())
            predictions["svm_opt"].extend(opt_pred.tolist())

    frame = save_fold_results(rows, args.output / "fold_metrics.csv")
    if predictions["svm_hyp"]:
        bowker = mcnemar_bowker(
            np.array(truths),
            np.array(predictions["svm_hyp"]),
            np.array(predictions["svm_opt"]),
        )
        save_json(bowker, args.output / "mcnemar_bowker.json")

    return frame


def run_single_splits(args):
    X, y = load_data(args.data)
    rows = []
    for ratio_name, test_size in [("70-30", 0.30), ("80-20", 0.20), ("85-15", 0.15)]:
        for seed in range(42, 52):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=y
            )
            baseline = SVC()
            baseline.fit(X_train, y_train)
            pred = baseline.predict(X_test)
            metrics = summarize_predictions(y_test, pred)
            rows.append({"split": ratio_name, "seed": seed, "model": "svm", **metrics})
    return save_fold_results(rows, args.output / "single_split_results.csv")


def parse_args():
    parser = argparse.ArgumentParser(description="SVM mobile price classification experiments")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--mode", choices=["baseline", "reproduce", "robust", "all", "boosting", "pca", "splits"], default="baseline")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    X, y = load_data(args.data)
    summary = data_summary(X, y)
    save_json(summary, args.output / "data_summary.json")
    print(summary)

    if args.mode == "splits":
        frame = run_single_splits(args)
    else:
        frame = run_cv(args, args.seed)

    print("\nSaved results to", args.output)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
