import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def search_hyperopt(X_inner, y_inner, seed=42, n_trials=100, scaled=False,
                    paper_space=True) -> dict:
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    gamma_space = (
        hp.uniform("gamma", 4.701e-7, 6.701e-7)
        if paper_space
        else hp.loguniform("gamma", np.log(1e-4), np.log(10))
    )
    space = {
        "kernel": hp.choice("kernel", ["linear", "poly", "rbf", "sigmoid"]),
        "C": hp.loguniform("C", np.log(0.01), np.log(100)),
        "gamma": gamma_space,
        "decision_function_shape": hp.choice("decision_function_shape", ["ovo", "ovr"]),
    }

    def objective(params):
        svc = SVC(random_state=seed, **params)
        estimator = Pipeline([("scaler", StandardScaler()), ("svc", svc)]) if scaled else svc
        score = float(cross_val_score(
            estimator, X_inner, y_inner, cv=inner_cv,
            scoring="accuracy", n_jobs=-1
        ).mean())
        return {"loss": -score, "status": STATUS_OK}

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=n_trials,
        trials=trials,
        rstate=np.random.default_rng(seed),
        show_progressbar=False,
    )
    resolved_params = space_eval(space, best)

    trial_rows = []
    for trial in trials.trials:
        vals = trial["misc"]["vals"]
        trial_rows.append({
            "number": trial["tid"],
            "accuracy": -float(trial["result"]["loss"]),
            "kernel": vals["kernel"][0] if vals["kernel"] else None,
            "C": vals["C"][0] if vals["C"] else None,
            "gamma": vals["gamma"][0] if vals["gamma"] else None,
            "decision_function_shape": vals["decision_function_shape"][0]
            if vals["decision_function_shape"] else None,
        })

    import pandas as pd
    return {
        "params": resolved_params,
        "best_inner_accuracy": -float(trials.best_trial["result"]["loss"]),
        "n_trials": len(trials.trials),
        "trials_dataframe": pd.DataFrame(trial_rows),
    }
