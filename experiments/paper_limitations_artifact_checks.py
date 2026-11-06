"""Validate real experiment artifacts, including all four metrics and trial selection."""

import hashlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluate import classification_metrics

D = ROOT / "reports" / "paper_limitations" / "data"


class ArtifactChecks(unittest.TestCase):
    def test_hpo_integrity(self):
        source = pd.read_csv(ROOT / "data" / "train.csv")
        for protocol in ["paper", "robust"]:
            for method in ["hyperopt", "optuna"]:
                seen = []
                for fold in range(1, 6):
                    with self.subTest(protocol=protocol, method=method, fold=fold):
                        r = json.loads(
                            (D / f"{protocol}_{method}_{fold}.json").read_text(
                                encoding="utf-8"
                            )
                        )
                        self.assertEqual(len(r["train_indices"]), 1600)
                        self.assertEqual(len(r["test_indices"]), 400)
                        self.assertFalse(
                            set(r["train_indices"]) & set(r["test_indices"])
                        )
                        self.assertEqual(
                            source.iloc[r["test_indices"]].price_range.tolist(),
                            r["truth"],
                        )
                        for k, v in classification_metrics(
                            r["truth"], r["predictions"]
                        ).items():
                            self.assertAlmostEqual(v, r[k], places=12)
                        trials = pd.read_csv(
                            D / f"{protocol}_{method}_{fold}_trials.csv"
                        )
                        self.assertEqual(len(trials), 100)
                        col = "accuracy" if method == "hyperopt" else "value"
                        self.assertAlmostEqual(
                            trials[col].max(), r["best_inner_accuracy"], places=12
                        )
                        seen.extend(r["test_indices"])
                self.assertEqual(sorted(seen), list(range(2000)))

    def test_pca_full_rotation_preserves_predictions(self):
        p = pd.read_csv(D / "basic_predictions.csv")
        a = p[p.model == "svm_scaled"].sort_values("index")
        b = p[p.model == "pca_20"].sort_values("index")
        self.assertEqual(a.prediction.tolist(), b.prediction.tolist())

    def test_data_provenance(self):
        digest = hashlib.sha256((ROOT / "data" / "train.csv").read_bytes()).hexdigest()
        for path in D.glob("manifest*.json"):
            r = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(r["data_sha256"], digest)
            self.assertEqual(r["packages"]["scikit-learn"], "1.4.2")


if __name__ == "__main__":
    unittest.main(verbosity=2)
