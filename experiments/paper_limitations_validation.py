"""Reproducible, additive experiments. Run from repo experiments/ directory.

The deliberate test-selection diagnostic is NOT a reproduction of the paper.
All production HPO uses only outer training data, through the original repo functions.
"""

import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import binomtest, chi2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_data, data_summary
from src.evaluate import classification_metrics


def dump(path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def basic(out, X, y):
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    models = {
        "svm_raw": SVC(),
        "svm_scaled": make_pipeline(StandardScaler(), SVC()),
        "xgb_default": XGBClassifier(
            objective="multi:softprob",
            num_class=4,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=2,
        ),
        "lgbm_default": LGBMClassifier(random_state=42, n_jobs=2, verbose=-1),
    }
    for n in [5, 10, 15, 20, 0.95]:
        models[f"pca_{n}"] = make_pipeline(
            StandardScaler(), PCA(n_components=n, svd_solver="full"), SVC()
        )
    rows, preds = [], []
    for fold, (tr, te) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), 1
    ):
        for name, estimator in models.items():
            start = time.perf_counter()
            m = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
            p = m.predict(X.iloc[te])
            rows.append(
                dict(
                    model=name,
                    fold=fold,
                    seconds=time.perf_counter() - start,
                    **classification_metrics(y.iloc[te], p),
                )
            )
            preds.extend(
                dict(
                    model=name, fold=fold, index=int(i), truth=int(t), prediction=int(v)
                )
                for i, t, v in zip(te, y.iloc[te], p)
            )
        pd.DataFrame(rows).to_csv(out / "basic_metrics.csv", index=False)
        print("basic fold", fold, flush=True)
    pd.DataFrame(preds).to_csv(out / "basic_predictions.csv", index=False)
    rows = []
    for size in [0.3, 0.2, 0.15]:
        for seed in range(42, 52):
            tr, te = train_test_split(
                np.arange(len(y)), test_size=size, stratify=y, random_state=seed
            )
            for name, m in [
                ("raw_default", SVC()),
                (
                    "scaled_linear_C1",
                    make_pipeline(StandardScaler(), SVC(kernel="linear", C=1)),
                ),
            ]:
                p = m.fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
                correct = int((p == y.iloc[te]).sum())
                ci = binomtest(correct, len(te)).proportion_ci()
                rows.append(
                    dict(
                        model=name,
                        test_size=size,
                        seed=seed,
                        n_test=len(te),
                        errors=len(te) - correct,
                        accuracy=correct / len(te),
                        ci_low=ci.low,
                        ci_high=ci.high,
                    )
                )
    pd.DataFrame(rows).to_csv(out / "split_sensitivity.csv", index=False)
    # Demonstrate exact output-shape and inactive-gamma behavior on matched data.
    tr, te = next(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    rows = []
    for kernel in ["linear", "poly", "rbf", "sigmoid"]:
        a = make_pipeline(
            StandardScaler(), SVC(kernel=kernel, decision_function_shape="ovo")
        )
        b = make_pipeline(
            StandardScaler(), SVC(kernel=kernel, decision_function_shape="ovr")
        )
        pa = a.fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        pb = b.fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        rows.append(
            dict(
                kernel=kernel,
                different_predictions=int(np.sum(pa != pb)),
                n_test=len(te),
                support_equal=np.array_equal(a[-1].support_, b[-1].support_),
            )
        )
    pd.DataFrame(rows).to_csv(out / "decision_shape.csv", index=False)
    z = StandardScaler().fit_transform(X.iloc[tr])
    pc = PCA(svd_solver="full").fit(z)
    j = X.columns.get_loc("ram")
    rows = []
    for n in [5, 10, 15, 20]:
        total = np.sum(pc.explained_variance_ * pc.components_[:, j] ** 2)
        kept = np.sum(pc.explained_variance_[:n] * pc.components_[:n, j] ** 2)
        rows.append(
            dict(
                components=n,
                total_variance_retained=pc.explained_variance_ratio_[:n].sum(),
                ram_variance_retained=kept / total,
            )
        )
    pd.DataFrame(rows).to_csv(out / "pca_ram_diagnostic.csv", index=False)


def hpo(out, X, y, protocol, trials, only_fold=None):
    from src.hpo_optuna import search_optuna
    from src.hpo_hyperopt import search_hyperopt

    scaled = protocol == "robust"
    for fold, (tr, te) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), 1
    ):
        if only_fold is not None and fold != only_fold:
            continue
        assert not np.intersect1d(tr, te).size
        for name, search in [("hyperopt", search_hyperopt), ("optuna", search_optuna)]:
            dest = out / f"{protocol}_{name}_{fold}.json"
            if dest.exists():
                saved = json.loads(dest.read_text(encoding="utf-8"))
                if saved["trials"] != trials:
                    raise ValueError(
                        "Use a separate output directory for a changed budget"
                    )
                continue
            print("START", protocol, name, fold, trials, flush=True)
            start = time.perf_counter()
            result = search(
                X.iloc[tr],
                y.iloc[tr],
                seed=42 + fold,
                n_trials=trials,
                scaled=scaled,
                paper_space=not scaled,
            )
            m = SVC(**result["params"])
            m = make_pipeline(StandardScaler(), m) if scaled else m
            p = m.fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
            result["trials_dataframe"].to_csv(
                out / f"{protocol}_{name}_{fold}_trials.csv", index=False
            )
            row = dict(
                protocol=protocol,
                model=name,
                fold=fold,
                trials=trials,
                seed=42,
                inner_seed=42 + fold,
                seconds=time.perf_counter() - start,
                best_params=result["params"],
                best_inner_accuracy=result["best_inner_accuracy"],
                train_indices=tr.tolist(),
                test_indices=te.tolist(),
                truth=y.iloc[te].tolist(),
                predictions=p.tolist(),
                **classification_metrics(y.iloc[te], p),
            )
            dump(dest, row)
            print(
                "DONE",
                protocol,
                name,
                fold,
                row["accuracy"],
                row["seconds"],
                flush=True,
            )


def diagnostics(out, X, y):
    # Paired factorial ablation: SAME C/gamma candidate quantiles and outer splits.
    # RBF-only deliberately isolates gamma from kernel selection; 20 fixed candidates.
    rng = np.random.default_rng(713)
    u = rng.uniform(size=(20, 2))
    rows = []
    for fold, (tr, te) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), 1
    ):
        inner = StratifiedKFold(3, shuffle=True, random_state=42 + fold)
        for scaled in [False, True]:
            for wide in [False, True]:
                candidates = []
                scores = []
                for uc, ug in u:
                    params = dict(
                        kernel="rbf",
                        C=10 ** (-2 + 4 * uc),
                        gamma=10 ** (-9 + 10 * ug) if wide else 4.701e-7 + 2e-7 * ug,
                    )
                    m = SVC(**params)
                    m = make_pipeline(StandardScaler(), m) if scaled else m
                    scores.append(
                        cross_val_score(
                            m, X.iloc[tr], y.iloc[tr], cv=inner, n_jobs=1
                        ).mean()
                    )
                    candidates.append(m)
                best = int(np.argmax(scores))
                p = candidates[best].fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
                rows.append(
                    dict(
                        fold=fold,
                        scaled=scaled,
                        wide_gamma=wide,
                        inner_accuracy=scores[best],
                        accuracy=accuracy_score(y.iloc[te], p),
                    )
                )
        pd.DataFrame(rows).to_csv(out / "gamma_ablation.csv", index=False)
        print("gamma ablation", fold, flush=True)
    # Degree/coef0: matched 24 candidates, same C/gamma, baseline fixed defaults.
    rng = np.random.default_rng(714)
    pars = []
    for i in range(24):
        pars.append(
            dict(
                C=float(10 ** rng.uniform(-2, 2)),
                gamma=float(10 ** rng.uniform(-4, 0)),
                degree=int(rng.integers(2, 6)),
                coef0=float(rng.choice([0, 0.5, 1, 2])),
            )
        )
    rows = []
    for fold, (tr, te) in enumerate(
        StratifiedKFold(5, shuffle=True, random_state=42).split(X, y), 1
    ):
        cv = StratifiedKFold(3, shuffle=True, random_state=42 + fold)
        for expanded in [False, True]:
            candidates = []
            scores = []
            for par in pars:
                par = par.copy()
                if not expanded:
                    par.update(degree=3, coef0=0)
                m = make_pipeline(StandardScaler(), SVC(kernel="poly", **par))
                candidates.append(m)
                scores.append(
                    cross_val_score(m, X.iloc[tr], y.iloc[tr], cv=cv, n_jobs=1).mean()
                )
            best = int(np.argmax(scores))
            p = candidates[best].fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
            rows.append(
                dict(
                    fold=fold,
                    expanded=expanded,
                    inner_accuracy=scores[best],
                    accuracy=accuracy_score(y.iloc[te], p),
                    params=str(candidates[best][-1].get_params()),
                )
            )
        pd.DataFrame(rows).to_csv(out / "kernel_ablation.csv", index=False)
        print("kernel ablation", fold, flush=True)
    # Deliberate selection on a pseudo-test versus untouched audit set, across 10 seeds.
    # All 30 candidates fixed a priori. This is a diagnostic, not TPE reproduction.
    rows = []
    rng = np.random.default_rng(715)
    params = [
        dict(
            kernel="rbf",
            C=float(10 ** rng.uniform(-2, 2)),
            gamma=float(10 ** rng.uniform(-4, 1)),
        )
        for _ in range(30)
    ]
    for seed in range(42, 52):
        train, rest = train_test_split(
            np.arange(len(y)), test_size=0.4, stratify=y, random_state=seed
        )
        val, audit = train_test_split(
            rest, test_size=0.5, stratify=y.iloc[rest], random_state=seed
        )
        assert (
            not set(train) & set(val)
            and not set(train) & set(audit)
            and not set(val) & set(audit)
        )
        v = []
        a = []
        inn = []
        for par in params:
            m = make_pipeline(StandardScaler(), SVC(**par))
            inn.append(
                cross_val_score(
                    m,
                    X.iloc[train],
                    y.iloc[train],
                    cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                    n_jobs=1,
                ).mean()
            )
            m.fit(X.iloc[train], y.iloc[train])
            v.append(m.score(X.iloc[val], y.iloc[val]))
            a.append(m.score(X.iloc[audit], y.iloc[audit]))
        for budget in [1, 5, 10, 30]:
            contaminated = int(np.argmax(v[:budget]))
            clean = int(np.argmax(inn[:budget]))
            rows.append(
                dict(
                    seed=seed,
                    budget=budget,
                    selected_trial=contaminated,
                    reported_pseudotest=v[contaminated],
                    audit_accuracy=a[contaminated],
                    gap=v[contaminated] - a[contaminated],
                    clean_selected_trial=clean,
                    clean_test_accuracy=v[clean],
                    clean_audit_accuracy=a[clean],
                )
            )
        pd.DataFrame(rows).to_csv(out / "selection_bias_diagnostic.csv", index=False)
        print("selection diagnostic", seed, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["basic", "hpo", "diagnostics"], required=True)
    p.add_argument("--protocol", choices=["paper", "robust"], default="paper")
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--fold", type=int, choices=range(1, 6))
    p.add_argument(
        "--output", type=Path, default=ROOT / "reports" / "paper_limitations" / "data"
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    X, y = load_data(ROOT / "data" / "train.csv")
    dump(
        a.output / f"manifest_{a.stage}_{a.protocol}_{a.fold}.json",
        dict(
            python=platform.python_version(),
            packages={
                s: md.version(s)
                for s in [
                    "numpy",
                    "pandas",
                    "scikit-learn",
                    "optuna",
                    "hyperopt",
                    "xgboost",
                    "lightgbm",
                    "scipy",
                    "statsmodels",
                ]
            },
            data_sha256=hashlib.sha256(
                (ROOT / "data" / "train.csv").read_bytes()
            ).hexdigest(),
            data_summary=data_summary(X, y),
            arguments=vars(a),
        ),
    )
    if a.stage == "basic":
        basic(a.output, X, y)
    elif a.stage == "hpo":
        hpo(a.output, X, y, a.protocol, a.trials, a.fold)
    else:
        diagnostics(a.output, X, y)
    print("COMPLETE", a.stage, a.protocol, flush=True)


if __name__ == "__main__":
    main()
