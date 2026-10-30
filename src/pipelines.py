from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def svm_pipeline(scaled: bool, **svc_params) -> Pipeline | SVC:
    estimator = SVC(random_state=42, **svc_params)
    if scaled:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svc", estimator),
        ])
    return estimator
