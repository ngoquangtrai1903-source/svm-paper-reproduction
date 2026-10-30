from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from statsmodels.stats.contingency_tables import SquareTable
from sklearn.svm import SVC

METRIC_NAMES = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]


def classification_metrics(y_true, y_pred) -> dict:
    report = classification_report(
        y_true, y_pred, labels=[0, 1, 2, 3], output_dict=True, zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": report["macro avg"]["precision"],
        "recall_macro": report["macro avg"]["recall"],
        "f1_macro": report["macro avg"]["f1-score"],
    }


def summarize_predictions(y_true, y_pred) -> dict:
    result = classification_metrics(y_true, y_pred)
    result["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=[0, 1, 2, 3]
    ).tolist()
    return result


def save_fold_results(rows: list[dict], output_path: Path) -> pd.DataFrame:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    return frame


def print_summary(frame: pd.DataFrame) -> None:
    numeric = [c for c in METRIC_NAMES if c in frame.columns]
    print("\nCross-validation summary:")
    print(frame.groupby(["model", "variant"])[numeric].agg(["mean", "std"]).round(6))


def mcnemar_bowker(y_true, pred_a, pred_b) -> dict:
    table = np.zeros((4, 4), dtype=int)
    for a, b in zip(pred_a, pred_b):
        table[int(a), int(b)] += 1
    if np.array_equal(table, np.diag(np.diag(table))):
        return {"statistic": 0.0, "p_value": 1.0, "table": table.tolist()}
    result = SquareTable(table).symmetry(method="bowker")
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "table": table.tolist(),
    }


def save_json(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def inner_cv_score(params: dict, X_inner, y_inner, scaled: bool, seed: int) -> float:
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold

    estimator = SVC(random_state=seed, max_iter=-1, **params)
    if scaled:
        estimator = Pipeline([("scaler", StandardScaler()), ("svc", estimator)])

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scores = cross_val_score(estimator, X_inner, y_inner, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(scores.mean())
