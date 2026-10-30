# Phân tích paper: SVM + Hyperparameter Optimization cho bài toán Mobile Phone Price Classification

*Nguồn: Chang, Lin, Pai — Electronics 2025, 14, 2173 (MDPI). Tài liệu phân tích và kế hoạch tái hiện (reproduce) thực nghiệm.*

## 1. Tổng quan paper

**Bài toán:** Dự đoán khoảng giá (price\_range) của điện thoại từ 20 thông số kỹ thuật. Đây là bài toán **multi-class classification** với 4 lớp: 0 = Low, 1 = Medium, 2 = High, 3 = Very High.

**Ý tưởng chính:** Dùng **SVM** làm mô hình phân loại, rồi tự động tìm hyperparameter bằng hai framework **HPO (Hyperparameter Optimization)** là **Hyperopt (HYP)** và **Optuna (OPT)**, tạo ra hai mô hình **SVM\_HYP** và **SVM\_OPT**. Đánh giá bằng **5-fold cross-validation**.

**Đóng góp tuyên bố:**

- Áp dụng HPO frameworks (TPE-based) cho multi-class SVM trên dataset giá điện thoại, vượt các kết quả trong 12 paper trước đó.
- Dùng 5-fold CV để chứng minh tính generalizability và robustness (các paper trước chủ yếu dùng single split).
- So sánh SVM với XGBoost, LightGBM; thử PCA; phân tích kernel nào hiệu quả.

**Kết quả chính (average 5-fold accuracy):**

| Mô hình | Accuracy | MA\_Precision | MA\_Recall | MA\_F1 |
| --- | --- | --- | --- | --- |
| SVM (W/O HPO, default) | 94.95% | 94.97% | 94.95% | 94.93% |
| SVM\_HYP (Hyperopt) | 97.40% | 97.44% | 97.40% | 97.40% |
| SVM\_OPT (Optuna) | 97.80% | 97.81% | 97.80% | 97.80% |
| SVM\_OPT + PCA | 97.90% | - | - | - |
| XGBoost / LGBM (W/O HPO) | 91.15% / 90.70% | - | - | - |
| XGBoost\_OPT / LGBM\_OPT | 93.55% / 92.95% | - | - | - |

McNemar–Bowker test giữa SVM\_OPT và SVM\_HYP: p-value = 0.995 (> 0.05), nghĩa là **khác biệt không có ý nghĩa thống kê**.

## 2. Dataset

- Nguồn: Kaggle "Mobile Price Classification" (train.csv), **2000 samples**, **20 features** + 1 target, không có missing values, các lớp **cân bằng** (khoảng 500 mẫu/lớp).
- Features: battery\_power, blue, clock\_speed, dual\_sim, fc, four\_g, int\_memory, m\_dep, mobile\_wt, n\_cores, pc, px\_height, px\_width, ram, sc\_h, sc\_w, talk\_time, three\_g, touch\_screen, wifi.
- Kiểu dữ liệu: int64 (hầu hết), float64 (clock\_speed, m\_dep); các biến binary (blue, dual\_sim, four\_g, three\_g, touch\_screen, wifi).
- Chia dữ liệu: **5-fold CV**, mỗi fold 1600 train / 400 test (80/20). Không có validation set riêng.

## 3. Kiến thức nền tảng được sử dụng

### 3.1 Support Vector Machine (SVM)

- Tìm **hyperplane** w·x + b = 0 với **maximum margin**. Ràng buộc: y\_i(w·x\_i + b) ≥ 1; hàm mục tiêu: minimize ½‖w‖².
- Dữ liệu không tách tuyến tính được thì dùng **Kernel Trick** để ánh xạ lên không gian chiều cao.
- Các kernel: **linear**, **polynomial** K = (γ·xi·xj + r)^d, **RBF** K = exp(−γ‖xi−xj‖²), **sigmoid** K = tanh(γ·xi·xj + r).
- Hyperparameters được tune: **kernel**, **C** (trade-off margin vs lỗi; C lớn dễ overfitting), **gamma**, **decision\_function\_shape (DF\_S)**.
- Gamma mặc định `scale` = 1 / (n\_features × X.var()).
- **Multi-class strategy:** One-vs-One (OvO) cần k(k−1)/2 classifier (4 lớp → 6); One-vs-Rest (OvR) cần k classifier (4 lớp → 4).

### 3.2 HPO frameworks

| Tiêu chí | Hyperopt | Optuna |
| --- | --- | --- |
| Hướng tối ưu | Chỉ minimize (trả về −Accuracy) | Minimize hoặc maximize (trả về +Accuracy) |
| Search space | `hp.uniform`, `hp.choice`, ... | `trial.suggest_float`, `suggest_int`, `suggest_categorical` |
| Thuật toán | Random, TPE, Adaptive TPE | TPE (mặc định), Random, Grid, CMA-ES, NSGA-II, QMC, GP, BoTorch |
| Số vòng lặp | `max_evals` | `n_trials` |
| Chạy | `fmin()` | `study.optimize()` |

Cả hai đều dùng **TPE (Tree-structured Parzen Estimator)**, một dạng Bayesian optimization.

### 3.3 Evaluation metrics

Accuracy, **Macro-averaged Precision/Recall/F1** (tính từ Confusion Matrix, Eq. 10–13).

## 4. Pipeline phương pháp (Algorithm 1)

1. Load dataset, kiểm tra missing values.
2. Tạo 5-fold CV (mỗi fold 80% train / 20% test).
3. Với mỗi fold: định nghĩa objective (−Acc cho Hyperopt, +Acc cho Optuna) → định nghĩa search space → chạy **100 trials** → lấy best hyperparameters → fit SVM cuối → dự đoán trên test fold.
4. Tính Accuracy, MA\_Precision, MA\_Recall, MA\_F1 cho từng fold, rồi **average** qua 5 fold.

**Search space (Table 3):**

| Hyperparameter | Default | Search range |
| --- | --- | --- |
| kernel | rbf | linear, poly, rbf, sigmoid |
| decision\_function\_shape | ovr | ovr, ovo |
| C | 1 | 1e-2 → 1e2 |
| gamma | scale | **4.701e-7 → 6.701e-7** |

Tác giả thử thu hẹp gamma dần: 1e-7\~1 → 1e-7\~1e-1 → 1e-7\~1e-5 → cuối cùng 4.701e-7\~6.701e-7 vì khoảng rộng làm training chậm và không ổn định.

**Môi trường:** Python 3.9.15, scikit-learn 1.4.2, Optuna 4.0.0, Hyperopt 0.2.7; Intel i5-12500, RTX 4070 Ti Super, 16 GB RAM.

## 5. Kết quả thực nghiệm chi tiết

### 5.1 Best hyperparameters theo fold

**Hyperopt (Table 5):**

| Fold | Best trial | Kernel | C | Gamma | DF\_S | Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 73 | poly | 24.0911 | 5.7263e-7 | ovr | 181 |
| 2 | 2 | linear | 8.0367 | 6.6103e-7 | ovo | 530 |
| 3 | 34 | linear | 0.0339 | 5.6089e-7 | ovo | 436 |
| 4 | 48 | poly | 5.0618 | 6.0715e-7 | ovo | 66 |
| 5 | 15 | linear | 7.5976 | 6.2982e-7 | ovr | 751 |

**Optuna (Table 6):**

| Fold | Best trial | Kernel | C | Gamma | DF\_S | Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 6 | poly | 28.5306 | 5.3762e-7 | ovr | 159 |
| 2 | 93 | linear | 0.0361 | 5.2013e-7 | ovo | 1101 |
| 3 | 0 | poly | 61.8569 | 5.8244e-7 | ovr | 370 |
| 4 | 87 | poly | 86.5157 | 6.5313e-7 | ovr | 63 |
| 5 | 9 | linear | 86.8067 | 5.4853e-7 | ovr | 1505 |

### 5.2 Accuracy theo fold

| Fold | SVM | SVM\_HYP | SVM\_OPT |
| --- | --- | --- | --- |
| 1 | 95.75 | 97.75 | 97.75 |
| 2 | 95.75 | 98.00 | 98.50 |
| 3 | 94.00 | 98.25 | 98.00 |
| 4 | 94.25 | 96.00 | 96.50 |
| 5 | 95.00 | 97.00 | 98.25 |
| **Avg** | **94.95** | **97.40** | **97.80** |

### 5.3 Các thí nghiệm bổ sung

- **Kernel analysis (Fig. 11):** linear và poly cho accuracy cao nhất; sigmoid luôn thấp nhất.
- **PCA (Table 10):** SVM 94.95 → 95.15; SVM\_HYP 97.40 → 97.45; SVM\_OPT 97.80 → 97.90 (cải thiện rất nhỏ). Số principal components không được nêu trong paper.
- **XGBoost / LGBM (Table 11):** thấp hơn SVM rõ rệt, kể cả sau HPO.
- **So sánh với literature (Tables 12–14):**

| Split | SVM\_HYP | SVM\_OPT | Literature tốt nhất |
| --- | --- | --- | --- |
| 70/30 | 97.50 | 97.67 | 96.31 (Nasser) |
| 80/20 | 98.25 | 98.50 | 96.77 (Pramanik) |
| 85/15 | 99.00 | 99.67 | 95.50 (Zhang) |

## 6. Nhận xét phê bình (quan trọng khi tái hiện)

Các điểm sau là **phân tích của tôi**, không phải kết luận của tác giả:

1. **Nguy cơ data leakage / optimistic bias:** Algorithm 1 không nói rõ objective của HPO được tính trên tập nào. Nếu accuracy trong objective được đo trên test fold (và bảng báo cáo cũng là test fold), kết quả bị lạc quan. Cách đúng: **nested CV** hoặc tách validation từ training set.
2. **Gamma range cực hẹp (4.7e-7 \~ 6.7e-7):** được chọn quanh giá trị default `scale`, tức là "data-dependent" và gần như hard-code. Giá trị gamma \~1e-7 cho thấy feature **chưa được scale** (ram \~ hàng nghìn MB, px \~ hàng nghìn). Paper không đề cập **StandardScaler/MinMaxScaler**; đây là chi tiết cần kiểm chứng khi tái hiện (X.var() lớn → gamma default nhỏ).
3. **Kernel linear/poly thắng:** gợi ý dữ liệu gần tách tuyến tính (đặc biệt feature `ram` rất mạnh). Với gamma nhỏ như vậy, RBF gần như suy biến thành linear.
4. **Không tune `degree`, `coef0`:** poly kernel dùng default (degree=3).
5. **Chênh lệch SVM\_OPT vs SVM\_HYP không có ý nghĩa thống kê** (p = 0.995); lợi thế "Optuna tốt hơn" chỉ là 0.4%.
6. **Thiếu chi tiết:** random seed, stratified hay không, shuffle hay không, cách chia 70/30, 80/20, 85/15 (single split nào), search space của XGBoost/LGBM, số PCA components, cross-validation bên trong HPO.
7. **Kết quả 85/15 = 99.67%** trên 300 mẫu test chỉ tương đương sai 1 mẫu, rất nhạy với random split.
8. Thời gian chạy dao động lớn (63s đến 1505s) do linear kernel + C lớn hội tụ chậm.

## 7. Kế hoạch triển khai thực nghiệm

### Giai đoạn 0 — Chuẩn bị môi trường

- Python 3.9+ (paper dùng 3.9.15), `scikit-learn==1.4.2`, `optuna==4.0.0`, `hyperopt==0.2.7`, `xgboost`, `lightgbm`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `statsmodels`/`scipy`.
- Tải `train.csv` từ Kaggle, đặt vào `data/`.
- Cố định `SEED = 42`.

```bash
pip install scikit-learn==1.4.2 optuna==4.0.0 hyperopt==0.2.7 xgboost lightgbm pandas numpy matplotlib seaborn statsmodels
```

### Giai đoạn 1 — Data preparation & EDA

- Kiểm tra missing, kiểu dữ liệu, phân phối lớp (kỳ vọng 500/lớp).
- Vẽ correlation heatmap; kỳ vọng `ram` tương quan mạnh nhất với `price_range`.
- Tạo `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

### Giai đoạn 2 — Baseline SVM (W/O HPO)

```python
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

skf = StratifiedKFold(5, shuffle=True, random_state=42)
for tr, te in skf.split(X, y):
    m = SVC()                      # kernel=rbf, C=1, gamma=scale, ovr
    m.fit(X.iloc[tr], y.iloc[tr])
    p = m.predict(X.iloc[te])
    acc = accuracy_score(y.iloc[te], p)
    pr, rc, f1, _ = precision_recall_fscore_support(y.iloc[te], p, average='macro')
```

Mục tiêu: đạt \~94.95% average accuracy (Table 7). Chạy hai biến thể (raw features và có scaling) để kiểm chứng giả thuyết ở mục 6.

### Giai đoạn 3 — SVM\_OPT (Optuna)

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    kernel = trial.suggest_categorical('kernel', ['linear','poly','rbf','sigmoid'])
    C      = trial.suggest_float('C', 1e-2, 1e2, log=True)
    gamma  = trial.suggest_float('gamma', 4.701e-7, 6.701e-7)
    dfs    = trial.suggest_categorical('decision_function_shape', ['ovr','ovo'])
    clf = SVC(kernel=kernel, C=C, gamma=gamma, decision_function_shape=dfs)
    # Phiên bản "paper": đánh giá trên fold; phiên bản "chuẩn": inner CV trên train
    return cross_val_score(clf, X_tr, y_tr, cv=3).mean()

study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100)
```

### Giai đoạn 4 — SVM\_HYP (Hyperopt)

```python
from hyperopt import fmin, tpe, hp, Trials
import numpy as np

space = {
  'kernel': hp.choice('kernel', ['linear','poly','rbf','sigmoid']),
  'C': hp.loguniform('C', np.log(1e-2), np.log(1e2)),
  'gamma': hp.uniform('gamma', 4.701e-7, 6.701e-7),
  'dfs': hp.choice('dfs', ['ovr','ovo']),
}
def f(p):
    clf = SVC(kernel=p['kernel'], C=p['C'], gamma=p['gamma'], decision_function_shape=p['dfs'])
    return -cross_val_score(clf, X_tr, y_tr, cv=3).mean()   # minimize -Accuracy

trials = Trials()
best = fmin(f, space, algo=tpe.suggest, max_evals=100, trials=trials,
            rstate=np.random.default_rng(42))
```

Lưu ý: `hp.choice` trả về **index**, cần dùng `space_eval` để chuyển về giá trị thật.

### Giai đoạn 5 — Đánh giá & thống kê

- Với mỗi fold: Accuracy, MA\_Precision, MA\_Recall, MA\_F1, confusion matrix; sau đó tính trung bình.
- Ghi lại best hyperparameters, best trial number, thời gian chạy để so với Table 5–6.
- **McNemar–Bowker test** (`statsmodels.stats.contingency_tables.SquareTable.symmetry`) giữa dự đoán của SVM\_OPT và SVM\_HYP trên fold tốt nhất; kỳ vọng p lớn (không khác biệt).
- Vẽ: bar chart accuracy theo fold (Fig. 10), đường optimization history (Fig. 8), histogram accuracy theo kernel (Fig. 11).

### Giai đoạn 6 — Các thí nghiệm mở rộng

| Thí nghiệm | Nội dung | Ghi chú |
| --- | --- | --- |
| PCA | `PCA` trước SVM, so sánh with/without | Paper không nêu số components; thử `n_components` ∈ {5,10,15,20} hoặc 95% variance |
| XGBoost / LGBM | Baseline + HPO (Optuna & Hyperopt, 100 trials) | Tự định nghĩa search space: n\_estimators, max\_depth, learning\_rate, subsample... |
| So sánh literature | Single split 70/30, 80/20, 85/15 | Dùng nhiều seed (ví dụ 10 seed) và báo cáo mean ± std thay vì một số duy nhất |
| Kernel analysis | Lấy toàn bộ trials, vẽ accuracy theo kernel | `study.trials_dataframe()` / `trials.trials` |

### Giai đoạn 7 — Mở rộng cải thiện độ tin cậy (khuyến nghị)

- **Nested CV** (outer 5-fold, inner 3-fold cho HPO) để loại bỏ leakage.
- Thêm `StandardScaler` trong `Pipeline` và mở rộng gamma search space (log-uniform 1e-4 \~ 1e1) để xem HPO có giúp hơn khi dữ liệu đã chuẩn hóa.
- Thêm tune `degree`, `coef0` cho poly/sigmoid.
- Chạy lặp nhiều random seed, báo cáo mean ± std, kiểm định thống kê.
- Thử thêm Random Search/Grid Search làm baseline HPO.

### Cấu trúc project đề xuất

```
project/
├── data/train.csv
├── notebooks/01_eda.ipynb
├── src/
│   ├── data.py            # load, CV splits
│   ├── models.py          # SVM, XGB, LGBM builders
│   ├── hpo_optuna.py
│   ├── hpo_hyperopt.py
│   ├── evaluate.py        # metrics, McNemar-Bowker
│   └── plots.py
├── experiments/run_all.py
└── results/ (tables, figures, best_params.json)
```

### Checklist kiểm chứng kết quả tái hiện

| Hạng mục | Giá trị paper | Sai số chấp nhận |
| --- | --- | --- |
| SVM baseline avg acc | 94.95% | ±1% |
| SVM\_HYP avg acc | 97.40% | ±1% |
| SVM\_OPT avg acc | 97.80% | ±1% |
| Kernel chọn nhiều nhất | poly / linear | cùng xu hướng |
| Gamma tối ưu | \~5e-7 \~ 6.6e-7 | trong search range |
| McNemar–Bowker p | 0.995 | > 0.05 |

## 8. Workflow diagrams

*(Các sơ đồ workflow được bổ sung ngay bên dưới.)*

### Sơ đồ 1 — Pipeline tổng thể (Algorithm 1)

&#91;embedded content: pipeline tổng thể · Algorithm 1\]

Dữ liệu qua 5-fold CV; trong mỗi fold, HPO chọn bộ hyperparameter rồi SVM cuối được chấm trên 400 mẫu test, sau đó lấy trung bình.

### Sơ đồ 2 — HPO workflow: Hyperopt vs Optuna

&#91;embedded content: HPO workflow · Hyperopt vs Optuna\]

Hai khung đi qua cùng 5 bước; điểm khác chính là objective của Hyperopt phải trả về −Accuracy để minimize, còn Optuna trả trực tiếp +Accuracy.

### Sơ đồ 3 — Lộ trình triển khai thực nghiệm

&#91;embedded content: roadmap triển khai · 7 phase\]

Phase 3 và 4 chạy song song rồi gộp ở Phase 5; Phase 7 là phần tôi đề xuất thêm, không có trong paper.
