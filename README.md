# Mobile Price SVM / HPO Reproducibility Project

Project triển khai thí nghiệm phân loại giá điện thoại đa lớp trong paper bằng SVM kết hợp Hyperopt và Optuna.

## 1. Môi trường và dữ liệu đầu vào

### Cài đặt

```powershell
python -m pip install -r requirements.txt
```

### Dữ liệu

Download Kaggle dataset `Mobile Price Classification`, sau đó đặt file gốc tại:

```text
data/train.csv
```

Chương trình đọc dữ liệu từ `data/train.csv` bằng [src/data.py](src/data.py).

Đầu vào bắt buộc:

- 2000 dòng dữ liệu.
- 20 cột đặc trưng.
- Một cột nhãn tên chính xác là `price_range`.
- Giá trị nhãn thuộc tập `{0, 1, 2, 3}`.
- Không có giá trị thiếu.

20 đặc trưng đầu vào gồm thông số phần cứng như pin, RAM, camera, độ phân giải màn hình, số lõi, hỗ trợ 4G/3G, Bluetooth và Wi-Fi. Nhãn đầu ra là bốn mức giá:

| `price_range` | Ý nghĩa |
|---:|---|
| 0 | Low |
| 1 | Medium |
| 2 | High |
| 3 | Very High |

Project kiểm tra số lượng dòng, số đặc trưng, nhãn, missing value và dòng trùng lặp trước khi huấn luyện.

---

## 2. Quy trình chung được các chế độ 5-fold sử dụng

Với các lệnh dùng cross-validation:

```text
Outer 5-fold Stratified Cross-validation
├── Outer training data
│   ├── Fit model baseline hoặc final model
│   └── Với HPO:
│       └── Inner 3-fold Stratified Cross-validation
│           └── Optuna hoặc Hyperopt chạy tối đa N trial
├── Outer validation/test data
└── Dự đoán và tính metric
```

Phân phối lớp trong split được giữ cân bằng bằng `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)`.

Các metric được lưu sau mỗi fold:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Confusion matrix
- Thời gian chạy
- Best inner accuracy
- Best hyperparameters

Các metric này được tính tại [src/evaluate.py](src/evaluate.py).

---

## 3. Giải thích từng lệnh

### A. Baseline SVM

```powershell
python experiments/run_experiments.py --mode baseline
```

Lệnh này chỉ huấn luyện SVM mặc định trên dữ liệu thô.

#### Model

```text
SVC(random_state=42)
```

Vì không truyền hyperparameter tùy chỉnh, scikit-learn sử dụng thiết lập mặc định, gồm:

- Kernel: `rbf`
- `C = 1`
- `gamma = "scale"`
- Multi-class strategy: One-vs-Rest mặc định

#### Kỹ thuật được sử dụng

- Stratified 5-fold cross-validation.
- Dữ liệu không được chuẩn hóa.
- Không dùng HPO.
- Không dùng PCA.
- Không dùng XGBoost hoặc LightGBM.
- Không có inner CV.
- Không dùng split 70/30 hoặc 85/15.

#### Kết quả và nơi lưu

- Trung bình mỗi metric được ghi trong `results/fold_metrics.csv`.
- Model chạy trên cả 20 đặc trưng.
- Mỗi dòng kết quả tương ứng một outer fold.
- Mục tiêu là so sánh với bảng `SVM W/O HPO` của paper.

---

### B. Tái hiện paper bằng SVM, Hyperopt và Optuna

```powershell
python experiments/run_experiments.py --mode reproduce --trials 100
```

Lệnh này chạy mô hình baseline và hai mô hình tối ưu SVM.

#### Model được chạy

1. `svm`: baseline raw SVM.
2. `svm_hyp`: SVM với Hyperopt.
3. `svm_opt`: SVM với Optuna.

#### Hyperparameter được tìm kiếm

| Hyperparameter | Search space |
|---|---|
| kernel | `linear`, `poly`, `rbf`, `sigmoid` |
| `C` | log scale từ `0.01` đến `100` |
| gamma | uniform từ `4.701e-7` đến `6.701e-7` |
| decision_function_shape | `ovo`, `ovr` |

Gamma trong chế độ này là hẹp và chính xác theo search range mô tả trong paper.

#### Kỹ thuật được sử dụng

- Outer 5-fold Stratified Cross-validation.
- Inner 3-fold Stratified Cross-validation để chọn hyperparameter.
- Hyperopt dùng TPE và 100 trial mặc định.
- Optuna dùng TPE và 100 trial mặc định.
- Tối ưu Accuracy trên inner training data.
- Fit lại mô hình cuối trên toàn bộ outer training data.
- Dự đoán trên outer test fold.
- Tính toàn bộ metric sau mỗi fold.
- Không chuẩn hóa dữ liệu bằng `StandardScaler`.
- Không dùng PCA.
- Không tune `degree`, `coef0` hoặc tham số poly/sigmoid khác.
- Không dùng XGBoost hoặc LightGBM.

#### HPO objective

Hyperopt:

```text
minimize(-accuracy)
```

Optuna:

```text
maximize(accuracy)
```

#### Trial history

Các file:

- `results/optuna_fold_1_trials.csv` đến `optuna_fold_5_trials.csv`
- `results/hyperopt_fold_1_trials.csv` đến `hyperopt_fold_5_trials.csv`

#### So sánh Hyperopt và Optuna

Sau khi chạy, project tạo `results/mcnemar_bowker.json`.

Không chạy McNemar-Bowker nếu HPO không được kích hoạt.

---

### C. Phiên bản robust có scaling và gamma rộng

```powershell
python experiments/run_experiments.py --mode robust --trials 100
```

Lệnh này là thí nghiệm cải tiến, không phải tái hiện nguyên bản paper.

#### Model được chạy

1. `svm` baseline raw.
2. `svm` baseline với `StandardScaler`.
3. `svm_hyp` trong pipeline `StandardScaler + SVC`.
4. `svm_opt` trong pipeline `StandardScaler + SVC`.

#### Pipeline của model HPO

```text
StandardScaler
    ↓
SVC
    ↓
Price range prediction
```

Scaler được fit theo inner/outer training data. Outer test data không tham gia fit scaler.

#### Kỹ thuật khác với `reproduce`

- Dùng `StandardScaler` cho toàn bộ SVM trong mô hình robust.
- Gamma được tìm bằng log scale trong khoảng `1e-4` đến `10`.
- Inner 3-fold được dùng để chọn hyperparameter.
- Outer 5-fold được dùng để báo kết quả cuối.

Các hyperparameter khác vẫn giữ:

- Kernel: `linear`, `poly`, `rbf`, `sigmoid`
- `C`: log scale từ `0.01` đến `100`
- `decision_function_shape`: `ovo`, `ovr`

#### Không sử dụng trong mode này

- Không dùng gamma hẹp của paper.
- Không dùng PCA.
- Không tune polynomial `degree`, `coef0`.
- Không dùng XGBoost hoặc LightGBM.
- Không dùng search algorithm ngoài Hyperopt TPE và Optuna TPE.

Lệnh này dùng để kiểm tra tác động của scaling và gamma rộng, vì dữ liệu gốc có các đặc trưng khác đơn vị và thang giá trị.

---

### D. So sánh SVM, XGBoost và LightGBM

```powershell
python experiments/run_experiments.py --mode boosting
```

#### Model được chạy

1. SVM baseline raw.
2. XGBoost baseline raw.
3. LightGBM baseline raw.

#### Cấu hình XGBoost

- Objective: `multi:softprob`
- `num_class = 4`
- Evaluation metric: `mlogloss`
- Không HPO.
- Không scaling.
- Không PCA.
- Không dùng tham số tùy chỉnh ngoài tham số mặc định.

#### Cấu hình LightGBM

- Objective: `multiclass`
- `num_class = 4`
- Không HPO.
- Không scaling.
- Không PCA.
- Không dùng tăng cường tham số tùy chỉnh ngoài tham số mặc định.

#### Kỹ thuật được sử dụng

- Outer stratified 5-fold cross-validation.
- SVM raw baseline.
- XGBoost và LightGBM ở chế độ mặc định.
- Accuracy, Macro Precision, Macro Recall, Macro F1.

#### Kỹ thuật không được sử dụng

- Không chạy Hyperopt cho XGBoost hoặc LightGBM.
- Không chạy Optuna cho XGBoost hoặc LightGBM.
- Không tune learning rate, depth, estimators, subsample hoặc tham số boosting khác.
- Không dùng inner CV cho boosting.
- Không dùng PCA.

---

### E. Thí nghiệm PCA

```powershell
python experiments/run_experiments.py --mode pca
```

#### Model được chạy

1. SVM baseline raw.
2. SVM baseline với `StandardScaler`.
3. Năm biến thể `StandardScaler + PCA + SVC`:

| Variant | Số thành phần PCA |
|---|---:|
| `pca_5` | 5 |
| `pca_10` | 10 |
| `pca_15` | 15 |
| `pca_20` | 20 |
| `pca_0.95` | Giữ 95% phương sai được chọn bởi PCA |

#### Pipeline PCA

```text
StandardScaler
    ↓
PCA
    ↓
SVC mặc định
    ↓
Price range prediction
```

#### Kỹ thuật được sử dụng

- Stratified 5-fold cross-validation.
- PCA được fit độc lập trong từng outer training fold.
- Outer test fold không tham gia scaler hoặc PCA.
- Thử fixed component count và variance threshold.

#### Kỹ thuật không được sử dụng

- Không dùng HPO.
- Không tune `C`, gamma hoặc kernel.
- Không compare Optuna/Hyperopt.
- Không dùng XGBoost hoặc LightGBM.
- Không tune PCA cùng SVM bằng nested CV.

Lệnh này chạy PCA như một thí nghiệm tiền xử lý cố định, không phải tối ưu hóa đồng thời PCA và SVM.

---

### F. So sánh theo các split của literature

```powershell
python experiments/run_experiments.py --mode splits
```

Lệnh này không dùng cross-validation.

#### Split được thực hiện

| Tên split | Training | Testing |
|---|---:|---:|
| `70-30` | 70% | 30% |
| `80-20` | 80% | 20% |
| `85-15` | 85% | 15% |

#### Chiến lược chạy

- Với mỗi split, chạy 10 seed từ 42 đến 51.
- Mỗi seed dùng `train_test_split(..., stratify=y)`.
- Mỗi run dùng SVM mặc định `SVC()`.
- Không sử dụng Hyperopt hoặc Optuna.
- Không scaling.
- Không PCA.
- Không XGBoost hoặc LightGBM.
- Không dùng 5-fold CV trong mode này.

#### Dữ liệu đầu ra

Kết quả được lưu tại:

```text
results/single_split_results.csv
```

Mỗi dòng tương ứng một cặp `split + seed`.

Mục đích là đối chiếu xu hướng với các con số 70/30, 80/20 và 85/15 trong phần literature của paper, không dùng kết quả này thay cho 5-fold CV.

---

## 4. Các file kết quả

Sau khi chạy thí nghiệm, tất cả file nằm trong thư mục `results/`.

| File | Nội dung |
|---|---|
| `data_summary.json` | Số mẫu, số feature, missing value, số dòng trùng, phân phối lớp |
| `fold_metrics.csv` | Accuracy và metric từng fold, confusion matrix, thời gian, hyperparameter |
| `optuna_fold_N_trials.csv` | Trial history của Optuna theo từng outer fold |
| `hyperopt_fold_N_trials.csv` | Trial history của Hyperopt theo từng outer fold |
| `mcnemar_bowker.json` | Phân tích sai biệt giữa dự đoán Optuna và Hyperopt |
| `single_split_results.csv` | Kết quả các split 70/30, 80/20 và 85/15 |

Cột quan trọng trong `fold_metrics.csv`:

| Cột | Ý nghĩa |
|---|---|
| `seed` | Random seed outer CV |
| `fold` | Outer fold 1–5 |
| `model` | Tên model: `svm`, `svm_hyp`, `svm_opt`, `svm_pca`, `xgb`, `lgbm` |
| `variant` | Biến thể dữ liệu hoặc preprocessing |
| `accuracy` | Accuracy của outer fold |
| `precision_macro` | Macro precision |
| `recall_macro` | Macro recall |
| `f1_macro` | Macro F1-score |
| `confusion_matrix` | Ma trận nhầm lẫn |
| `best_inner_accuracy` | Accuracy tốt nhất tìm được trong inner CV |
| `best_params` | Hyperparameter cuối cùng của mô hình HPO |
| `seconds` | Thời gian fit hoặc HPO |

---

## 5. Các kỹ thuật chưa được triển khai

README mô tả hiện trạng code, không phải danh sách ý tưởng trong paper. Các kỹ thuật sau chưa hoàn thành trong implementation:

- HPO cho XGBoost hoặc LightGBM.
- PCA kết hợp HPO.
- Tune polynomial `degree`, `coef0`.
- Random Search hoặc Grid Search làm baseline HPO.
- Multi-seed outer-CV mặc định cho tất cả mode.
- Kiểm tra confidence interval mặc định trên 5 fold.
- Trực tiếp tải dữ liệu Kaggle trong script; người dùng phải tự đặt `data/train.csv`.

Các mode vẫn độc lập. Ví dụ lệnh `robust` không chạy PCA; lệnh `pca` không chạy HPO; lệnh `boosting` không chạy Optimizer cho boosting.
