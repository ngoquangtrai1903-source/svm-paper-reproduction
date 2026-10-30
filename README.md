# Mobile Price SVM / HPO Reproducibility Project

This project implements the experiments described in the paper *Support Vector Machines with Hyperparameter Optimization Frameworks for Classifying Mobile Phone Prices in Multi-Class*.

## Data

Download the Kaggle dataset `Mobile Price Classification` and place `train.csv` here:

```text
data/train.csv
```

Do not rename the target column. It must be named `price_range`.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Commands

Baseline reproduction:

```powershell
python experiments/run_experiments.py --mode baseline
```

Full SVM paper-style reproduction, including Optuna and Hyperopt:

```powershell
python experiments/run_experiments.py --mode reproduce --trials 100
```

Robust nested-CV experiment with StandardScaler and wider gamma:

```powershell
python experiments/run_experiments.py --mode robust --trials 100
```

Boosting comparison:

```powershell
python experiments/run_experiments.py --mode boosting
```

Literature split comparison:

```powershell
python experiments/run_experiments.py --mode splits
```

## Important design distinction

`reproduce` follows the paper's raw-feature gamma range and 100-trial setup. The paper's single test-fold HPO objective is not observable from the text, so this implementation uses inner 3-fold cross-validation inside each outer 5-fold training partition to prevent outer-test leakage.

`robust` adds StandardScaler and searches gamma from `1e-4` to `10` on log scale. Treat it as a separate research extension, not as a direct reproduction of the paper.

## Outputs

All result files are written to `results/`, including fold metrics, data summary, trial histories, and McNemar-Bowker disagreement analysis.
