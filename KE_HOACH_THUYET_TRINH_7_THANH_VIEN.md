# Kế hoạch thuyết trình: Reproduce paper SVM + Hyperparameter Optimization

## 1. Thông tin chung

- **Đề tài:** *Support Vector Machines with Hyperparameter Optimization Frameworks for Classifying Mobile Phone Prices in Multi-Class*
- **Dữ liệu:** Mobile Price Classification, 2000 mẫu, 20 đặc trưng, 4 lớp giá.
- **Mục tiêu trình bày:** giải thích nội dung bài báo, tái hiện thí nghiệm, so sánh kết quả và thảo luận các phần mở rộng.
- **Số thành viên:** 7 người.
- **Thời lượng đề xuất:** 25-30 phút, gồm khoảng 24 phút thuyết trình và 5 phút demo/Q&A.
- **Ngôn ngữ slide đề xuất:** tiếng Việt; tên mô hình, thuật ngữ kỹ thuật giữ tiếng Anh.

## 2. Luồng trình bày tổng thể

| Phần | Thời lượng | Người trình bày |
|---|---:|---|
| 1. Mở đầu và động lực nghiên cứu | 2 phút | Thành viên 1 |
| 2. Dataset và bài toán phân loại | 3 phút | Thành viên 2 |
| 3. SVM và kiến thức nền tảng | 4 phút | Thành viên 3 |
| 4. Thiết kế reproduce và baseline | 3 phút | Thành viên 4 |
| 5. Hyperopt trong thí nghiệm | 4 phút | Thành viên 5 |
| 6. Optuna, đánh giá và thống kê sai khác | 4 phút | Thành viên 6 |
| 7. Kết quả, mở rộng, kết luận và demo | 4 phút | Thành viên 7 |
| **Q&A và demo chi tiết** | 5 phút | Toàn nhóm |

---

## 3. Phân công chi tiết từng thành viên

### Thành viên 1 - Chủ trình bày, tổng quan paper và giả thuyết nghiên cứu

**Trách nhiệm chính**
- Giới thiệu tên bài báo, tác giả, năm xuất bản và nguồn bài báo.
- Nêu bối cảnh smartphone, lý do cần dự đoán khoảng giá.
- Trình bày câu hỏi nghiên cứu: HPO có giúp SVM cải thiện accuracy trên bài toán 4 lớp hay không?
- Giới thiệu ba nội dung chính: SVM, Hyperopt và Optuna.
- Chuyển mạch sang Thành viên 2 bằng câu hỏi: Dữ liệu được sử dụng ra sao?

**Slide đề xuất**
- Slide 1: tiêu đề, danh sách nhóm.
- Slide 2: động lực và câu hỏi nghiên cứu.
- Slide 3: đóng góp chính của bài báo và sơ đồ pipeline tổng quát.

**Phần code/artifact cần chuẩn bị**
- Không phụ trách triển khai riêng.
- Chuẩn bị số liệu tổng quan từ `README.md` và kết quả dataset từ `results/data_summary.json`.

**Thời gian:** 2 phút.

---

### Thành viên 2 - Dataset và tiền xử lý dữ liệu

**Trách nhiệm chính**
- Giải thích Mobile Price Classification gồm 2000 mẫu và 20 đặc trưng.
- Liệt kê các đặc trưng tiêu biểu, đặc biệt là `ram`, `battery_power`, `px_width`, `px_height`.
- Trình bày 4 lớp: Low, Medium, High và Very High.
- Xác nhận dữ liệu cân bằng, không thiếu dữ liệu và không có duplicate sau khi ghép feature-target.
- Giải thích tại sao cần chia dữ liệu Stratified 5-fold và mỗi fold có khoảng 1600 mẫu train/400 mẫu test.
- Trình bày phân biệt `train.csv` và `test.csv`; nếu `test.csv` không có nhãn thì chỉ dùng để kiểm tra cấu trúc dữ liệu, không dùng làm metric đánh giá.

**Slide đề xuất**
- Slide 4: mô tả dataset và phân bố 4 lớp.
- Slide 5: kiểm tra chất lượng dữ liệu, danh sách feature và chiến lược chia dữ liệu.

**Phần code/artifact cần chuẩn bị**
- Giải thích logic trong `src/data.py`.
- Trình bày bảng tổng hợp từ `results/data_summary.json`.
- Dùng biểu đồ phân bố lớp và biểu đồ tương quan/boxplot của các feature quan trọng.

**Thời gian:** 3 phút.

---

### Thành viên 3 - Lý thuyết SVM, multi-class và hyperparameters

**Trách nhiệm chính**
- Trình bày maximum margin, support vectors và ảnh hưởng của tham số `C`.
- Giải thích kernel trick và các kernel: linear, polynomial, RBF, sigmoid.
- Giải thích `gamma` và vì sao dữ liệu chưa chuẩn hóa có thể làm `gamma` rất nhỏ.
- Trình bày chiến lược multi-class `ovo` và `ovr`.
- Trình bày các hyperparameter được tune: kernel, C, gamma và decision_function_shape.
- Liên hệ lý thuyết với paper: paper tìm gamma quanh khoảng rất hẹp, còn bản mở rộng của nhóm dùng StandardScaler và gamma rộng hơn.

**Slide đề xuất**
- Slide 6: maximum margin và hard/soft margin.
- Slide 7: kernel và multi-class strategy.
- Slide 8: hyperparameter và search space.

**Phần code/artifact cần chuẩn bị**
- Đọc và giải thích `src/pipelines.py`.
- Đưa ra ví dụ khác biệt pipeline raw SVM và scaled SVM.
- Chuẩn bị tranh minh họa phù hợp với dữ liệu, không lấy hình dán thô từ nguồn chưa kiểm tra.

**Thời gian:** 4 phút.

---

### Thành viên 4 - Thiết kế reproduce, baseline và nested cross-validation

**Trách nhiệm chính**
- Trình bày 5-fold outer CV.
- Giải thích trong mỗi fold, phần train được dùng tiếp để chạy inner 3-fold CV cho HPO.
- Phân biệt baseline SVM và two tuned models `SVM_HYP`, `SVM_OPT`.
- Giải thích cách tránh test-fold leakage: 5-fold outer test không tham gia objective của HPO.
- Trình bày baseline kết quả hiện có và cách chạy lệnh reproduce.
- Phân biệt rõ:
  - `reproduce`: giữ khoảng gamma giống paper và dùng dữ liệu raw.
  - `robust`: StandardScaler + gamma rộng log-scale.

**Slide đề xuất**
- Slide 9: nested CV và luồng pipeline.
- Slide 10: baseline và mapping giữa paper với code.
- Slide 11: mapping file/mô hình trong repo.

**Phần code/artifact cần chuẩn bị**
- Giải thích `experiments/run_experiments.py`.
- Đảm bảo môi trường chạy được lệnh:
  ```powershell
  python experiments/run_experiments.py --mode baseline
  python experiments/run_experiments.py --mode reproduce --trials 100
  ```
- Chạy trước và lưu kết quả để tránh lỗi slide khi thuyết trình.

**Thời gian:** 3 phút.

---

### Thành viên 5 - Hyperopt và phân tích trial history

**Trách nhiệm chính**
- Trình bày TPE có tên đầy đủ là Tree-structured Parzen Estimator.
- Giải thích Hyperopt biểu diễn không gian tham số bằng `hp.choice`, `hp.loguniform` và mục tiêu minimize negative accuracy.
- So sánh cách chạy:
  - `fmin(...)`
  - `Trials` lưu lịch sử trial.
- Giải thích `search_hyperopt()` trong `src/hpo_hyperopt.py`.
- Trình bày tối thiểu một best hyperparameter của nhóm nếu kết quả reproduction hoàn tất.
- Phân tích trial history không chọn trial theo outer test accuracy; chỉ dùng inner CV accuracy.

**Slide đề xuất**
- Slide 12: cơ chế Hyperopt/TPE.
- Slide 13: search space và trial history.

**Phần code/artifact cần chuẩn bị**
- Vẽ biểu đồ accuracy theo trial từ `results/hyperopt_fold_*_trials.csv`.
- Chuẩn bị tối thiểu 1 bảng so sánh hyperparameter giữa 2-3 fold.
- Chuẩn bị giải thích nếu paper và kết quả nội bộ khác nhau.

**Thời gian:** 4 phút.

---

### Thành viên 6 - Optuna, metric đánh giá và kiểm định thống kê

**Trách nhiệm chính**
- Giải thích Optuna TPE, API `suggest_float`, `suggest_categorical` và `study.optimize`.
- So sánh trực tiếp Hyperopt và Optuna về API, cách định hướng tối ưu và kết quả.
- Trình bày metric Accuracy, Macro Precision, Macro Recall và Macro F1.
- Giải thích Confusion Matrix theo 4 lớp.
- Trình bày McNemar-Bowker là kiểm định sai khác giữa hai hệ phân loại, hỗ trợ từ kết quả `mcnemar_bowker.json`.
- Không kết luận hai mô hình khác nhau nếu p-value lớn hơn 0.05.

**Slide đề xuất**
- Slide 14: cơ chế Optuna/TPE.
- Slide 15: metric, confusion matrix và McNemar-Bowker.
- Slide 16: bảng so sánh Hyperopt/Optuna.

**Phần code/artifact cần chuẩn bị**
- Giải thích `src/hpo_optuna.py` và `src/evaluate.py`.
- Tạo bảng mean ± std trên 5 fold, không chỉ báo mean accuracy.
- Chuẩn bị giải thích `results/fold_metrics.csv` và `results/mcnemar_bowker.json` nếu file đã được tạo.

**Thời gian:** 4 phút.

---

### Thành viên 7 - Tổng hợp kết quả, phần mở rộng, kết luận và điều phối Q&A

**Trách nhiệm chính**
- Tổng hợp kết quả baseline, reproduce và robust thành một bảng thống nhất.
- Trình bày các phần mở rộng thực nghiệm đã chọn:
  - StandardScaler + gamma rộng.
  - PCA 5, 10, 15, 20 và 95% variance.
  - XGBoost và LightGBM.
  - 70/30, 80/20, 85/15 split.
  - McNemar-Bowker.
- Trình bày mở rộng hiệu quả nhất, nếu không chạy đủ thí nghiệm thì phải nói rõ phần chưa hoàn tất, không phỏng đoán kết quả.
- Trình bày ưu điểm, hạn chế và các điểm cần cải thiện của paper:
  - search space gamma quá hẹp;
  - thiếu chi tiết preprocessing trong paper;
  - best parameter không ổn định giữa các fold;
  - thang đo reproducibility có thể bị ảnh hưởng nếu seed/code thiếu rõ ràng;
  - nested CV là cách kiểm tra và giảm nguy cơ data leakage tốt hơn reproduction đơn giản.
- Kết luận xem nhóm ủng hộ hay không ủng hộ claim của paper dựa trên kết quả thực nghiệm.
- Điều phối phần hỏi đáp và chuyển câu hỏi đến thành viên phụ trách sâu.

**Slide đề xuất**
- Slide 17: bảng tổng hợp kết quả.
- Slide 18: phần mở rộng và phân tích.
- Slide 19: kết luận, limitation và hướng phát triển.
- Slide 20: Q&A/demo.

**Phần code/artifact cần chuẩn bị**
- Tổng hợp hình ảnh/CSV từ thư mục `results/`.
- Chạy demo ngắn và chuẩn bị terminal trước kết quả.
- Chuẩn bị 5 câu hỏi dự kiến của giảng viên và câu trả lời ngắn.

**Thời gian:** 4 phút.

---

## 4. Phần mở rộng thực nghiệm bắt buộc nên hoàn tất

Nhóm nên ưu tiên các phần mở rộng theo thứ tự dưới đây để tránh thí nghiệm quá tải.

### Mức A - bắt buộc

1. **Baseline reproduction**
   - Chạy 5-fold raw SVM mặc định.
   - Ghi Accuracy, Macro Precision, Macro Recall và Macro F1.
2. **Nested-CV reproduction**
   - Chạy inner 3-fold HPO trong mỗi outer fold.
   - Chạy 100 trials cho Hyperopt và Optuna.
3. **So sánh kết quả 5 fold**
   - Tính mean và standard deviation.
   - Vẽ biểu đồ accuracy theo fold.
4. **Kiểm tra data leakage**
   - Xác nhận outer test không tham gia `cross_val_score`.
   - Giải thích rõ đây là design khác biệt với paper nếu paper không mô tả objective chi tiết.

### Mức B - mở rộng nghiên cứu

5. **Robust scaling**
   - StandardScaler trong pipeline.
   - Gamma log-scale từ `1e-4` đến `10`.
   - So sánh trực tiếp với kết quả paper-space raw.
6. **PCA**
   - Thử 5, 10, 15, 20 components và 95% variance.
   - Không kết luận PCA cải thiện nếu accuracy mean chỉ tăng rất nhỏ.
7. **Boosting baselines**
   - XGBoost và LightGBM mặc định.
   - So sánh với SVM để kiểm chứng claim bài báo.
8. **Literature-style splits**
   - Test 70/30, 80/20 và 85/15.
   - Cần lưu seed để kết quả có thể tái hiện.

### Mức C - nâng cao nếu còn thời gian

9. Phân tích tầm quan trọng feature bằng permutation importance.
10. So sánh Random Forest và Logistic Regression với cùng 5-fold protocol.
11. Chạy nhiều seed để đo độ ổn định thay vì 1 seed duy nhất.
12. Phân tích runtime và mức tăng chi phí khi tăng trials.

---

## 5. Ma trận phân công công việc

| Công việc | Người phụ trách chính | Người hỗ trợ |
|---|---|---|
| Tổng quan paper và slide mở đầu | Thành viên 1 | Thành viên 7 |
| Phân tích dataset | Thành viên 2 | Thành viên 4 |
| SVM/kernel/slide lý thuyết | Thành viên 3 | Thành viên 5 |
| Chạy baseline và reproduction | Thành viên 4 | Thành viên 5, 6 |
| Trial Hyperopt và biểu đồ | Thành viên 5 | Thành viên 4 |
| Trial Optuna và thống kê | Thành viên 6 | Thành viên 4, 7 |
| Robust/PCA/boosting/splits | Thành viên 7 | Thành viên 4 |
| Tổng hợp results và kiểm tra dữ liệu | Thành viên 7 | Thành viên 6 |
| Slide thống nhất | Thành viên 1 | Toàn nhóm |
| Rehearsal/Q&A | Thành viên 7 điều phối | Toàn nhóm |

## 6. Kế hoạch sản phẩm và mốc tiến độ

### Tuần 1 - Hiểu paper và thiết lập

- Thành viên 1: đọc Abstract/Introduction và tóm tắt claim.
- Thành viên 2: kiểm tra dữ liệu bằng `src/data.py` và `results/data_summary.json`.
- Thành viên 3: tóm tắt kernel, C, gamma, OvO và OvR.
- Thành viên 4: cài đặt, chạy baseline, kiểm tra môi trường.
- Thành viên 5: đọc `src/hpo_hyperopt.py`.
- Thành viên 6: đọc `src/hpo_optuna.py`, `src/evaluate.py`.
- Thành viên 7: lập list phần mở rộng và deadline.

### Tuần 2 - Triển khai và tạo dữ liệu

- Chạy baseline.
- Chạy reproduce với 100 trials nếu tài nguyên cho phép.
- Chạy robust.
- Phân công PCA/boosting/splits.
- Thu thập CSV/JSON vào thư mục `results/`.

### Tuần 3 - Hoàn thiện phân tích

- Vẽ chart và summary table.
- Viết phần so sánh với kết quả paper.
- Viết mỗi phần presentation theo đúng thời lượng.
- Kiểm tra thuật ngữ và câu chuyển slide.

### Tuần 4 - Rehearsal và hoàn tất

- Chạy rehearsal lần 1, đo thời gian từng thành viên.
- Chạy rehearsal lần 2 với Q&A.
- Dự phòng: screenshot kết quả, bản PDF slide và video ngắn demo.
- Trình bày lệnh code, không phụ thuộc terminal sống nếu máy phòng thuyết trình không ổn định.

## 7. File và nội dung slide đề xuất

| Slide | Nội dung | Phụ trách |
|---|---|---|
| 1 | Tiêu đề và thành viên | Thành viên 1 |
| 2 | Motivation và research question | Thành viên 1 |
| 3 | Tổng quan đóng góp và pipeline | Thành viên 1 |
| 4 | Dataset và các feature | Thành viên 2 |
| 5 | EDA và 5-fold stratified split | Thành viên 2 |
| 6 | SVM maximum margin | Thành viên 3 |
| 7 | Kernel và OvO/OvR | Thành viên 3 |
| 8 | Hyperparameters | Thành viên 3 |
| 9 | Reproduction strategy/Nested CV | Thành viên 4 |
| 10 | Baseline và mapping code | Thành viên 4 |
| 11 | Hyperopt/TPE | Thành viên 5 |
| 12 | Hyperopt search space/trials | Thành viên 5 |
| 13 | Optuna/TPE | Thành viên 6 |
| 14 | Metric và confusion matrix | Thành viên 6 |
| 15 | Hyperopt vs Optuna | Thành viên 6 |
| 16 | Tổng hợp kết quả | Thành viên 7 |
| 17 | PCA/Boosting/Splits/Robust | Thành viên 7 |
| 18 | Claim và limitation | Thành viên 7 |
| 19 | Conclusion | Thành viên 7 |
| 20 | Demo/Q&A | Toàn nhóm |

## 8. Kịch bản nối câu theo trình tự

- Thành viên 1: "Vậy bài toán thực tế là gì và dữ liệu dùng để giải quyết ra sao?"
- Thành viên 2: "Sau khi hiểu dữ liệu, mô hình cần giải quyết vấn đề multi-class như thế nào?"
- Thành viên 3: "Để model hoạt động tốt, cần tune những hyperparameter nào?"
- Thành viên 4: "Chúng tôi tái hiện paper bằng nested CV theo pipeline nào?"
- Thành viên 5: "Hyperopt tìm tham số ra sao?"
- Thành viên 6: "Optuna khác gì và kết quả đánh giá được đo bằng metric nào?"
- Thành viên 7: "Kết quả tổng hợp và phần nhóm mở rộng cho thấy điều gì?"

## 9. Câu hỏi có khả năng cao cần chuẩn bị

1. Vì sao dùng 5-fold thay vì train/test split?
2. `accuracy` và Macro F1 khác nhau thế nào?
3. StandardScaler có làm kết quả thay đổi không?
4. OvO và OvR khác nhau ra sao?
5. Gamma trong paper có hợp lý không?
6. Nested-CV khác reproduction trực tiếp như thế nào?
7. Vì sao Hyperopt và Optuna đều dựa trên TPE nhưng kết quả có thể khác?
8. Làm sao tránh data leakage trong HPO?
9. PCA có thực sự cải thiện accuracy không?
10. Kết quả reproduction có giống bảng kết quả paper không?
11. Vì sao baseline trên máy nhóm có thể khác với paper?
12. 100 trials có đảm bảo tối ưu toàn cục không?
13. Test.csv được dùng để làm gì?
14. McNemar-Bowker được diễn giải thế nào?
15. Nếu p-value > 0.05 thì có kết luận "Optuna tốt hơn Hyperopt" không?

## 10. Checklist trước buổi thuyết trình

- [ ] Mỗi thành viên đã đọc paper ở mức ít nhất phần được giao.
- [ ] Không phát biểu kết quả chưa chạy hoặc chỉ ghi là "ước lượng".
- [ ] Slide có đúng tên file, lệnh và kết quả thực tế.
- [ ] Mean ± std đã được tính trên 5 fold.
- [ ] 100 trials đã chạy xong hoặc có ghi rõ thí nghiệm dùng số trials khác.
- [ ] Runtime có thể quá dài nên đã lưu kết quả trước.
- [ ] Có tối thiểu 2 bản backup: PDF và PPTX.
- [ ] Screenshot các biểu đồ quan trọng đã lưu.
- [ ] Lệnh baseline chạy được trên máy dự phòng.
- [ ] Mỗi thành viên học thuộc phần chuyển slide.
- [ ] Thành viên 7 biết cách trả lời câu hỏi chuyển sang thành viên chuyên môn.
- [ ] Câu kết luận tách biệt: paper claim, nhóm reproduce và nhóm extension.

## 11. Tài liệu tham khảo trong repository

- `paper.md`: nội dung paper đầy đủ.
- `paper_keypoint.md`: tóm tắt phương pháp, kết quả và nhận xét.
- `README.md`: hướng dẫn chạy thí nghiệm.
- `src/data.py`: load/validate dữ liệu.
- `src/pipelines.py`: pipeline SVM và StandardScaler.
- `src/hpo_hyperopt.py`: triển khai Hyperopt.
- `src/hpo_optuna.py`: triển khai Optuna.
- `src/evaluate.py`: metric và McNemar-Bowker.
- `experiments/run_experiments.py`: điều phối baseline/reproduce/robust/open extensions.
- `results/data_summary.json`: thống kê dataset.
- `results/fold_metrics.csv`: kết quả baseline hiện có.
