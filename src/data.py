from pathlib import Path
import pandas as pd

REQUIRED_TARGET = "price_range"


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Missing dataset: {csv_path}. Place Kaggle train.csv at data/train.csv."
        )

    frame = pd.read_csv(csv_path)
    if REQUIRED_TARGET not in frame.columns:
        raise ValueError(f"Expected target column '{REQUIRED_TARGET}'. Found: {list(frame.columns)}")

    y = frame[REQUIRED_TARGET].astype(int)
    X = frame.drop(columns=[REQUIRED_TARGET])

    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values; paper dataset should contain none.")
    if X.shape[1] != 20:
        raise ValueError(f"Expected 20 features, found {X.shape[1]}.")
    if sorted(y.unique().tolist()) != [0, 1, 2, 3]:
        raise ValueError(f"Expected four classes 0-3, found {sorted(y.unique().tolist())}.")

    return X, y


def data_summary(X: pd.DataFrame, y: pd.Series) -> dict:
    return {
        "n_samples": len(X),
        "n_features": X.shape[1],
        "missing_values": int(X.isna().sum().sum()),
        "duplicate_rows_after_features_target": int(pd.concat([X, y], axis=1).duplicated().sum()),
        "class_counts": y.value_counts().sort_index().to_dict(),
        "feature_types": X.dtypes.astype(str).value_counts().to_dict(),
    }
