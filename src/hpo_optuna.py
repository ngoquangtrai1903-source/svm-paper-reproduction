import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

optuna.logging.set_verbosity(optuna.logging.WARNING)


def search_optuna(X_inner, y_inner, seed=42, n_trials=100, scaled=False,
                  paper_space=True) -> dict:
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
        C = trial.suggest_float("C", 0.01, 100, log=True)
        decision_shape = trial.suggest_categorical(
            "decision_function_shape", ["ovo", "ovr"]
        )
        gamma = (
            trial.suggest_float("gamma", 4.701e-7, 6.701e-7)
            if paper_space
            else trial.suggest_float("gamma", 1e-4, 10, log=True)
        )

        svc = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            decision_function_shape=decision_shape,
            random_state=seed,
        )
        estimator = Pipeline([("scaler", StandardScaler()), ("svc", svc)]) if scaled else svc

        return float(cross_val_score(
            estimator, X_inner, y_inner, cv=inner_cv,
            scoring="accuracy", n_jobs=-1
        ).mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {
        "params": study.best_params,
        "best_inner_accuracy": study.best_value,
        "n_trials": len(study.trials),
        "trials_dataframe": study.trials_dataframe(attrs=("number", "value", "params", "state")),
    }
