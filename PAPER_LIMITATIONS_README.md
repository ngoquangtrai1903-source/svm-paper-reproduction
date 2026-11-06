# Thí nghiệm kiểm tra hạn chế cho hạn chế của bài báo

Đọc `reports/paper_limitations/PAPER_LIMITATIONS.md` và `reports/paper_limitations/comparison.csv` trước. Code bổ sung không thay đổi hàm HPO gốc.

## Chạy lại

Python 3.12; không cài vào Python hệ thống:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-paper_limitations.txt
.\.venv\Scripts\python.exe experiments/paper_limitations_validation.py --stage basic
.\.venv\Scripts\python.exe experiments/paper_limitations_validation.py --stage hpo --protocol paper --trials 100
.\.venv\Scripts\python.exe experiments/paper_limitations_validation.py --stage hpo --protocol robust --trials 100
.\.venv\Scripts\python.exe experiments/paper_limitations_validation.py --stage diagnostics
.\.venv\Scripts\python.exe experiments/paper_limitations_report.py
.\.venv\Scripts\python.exe experiments/paper_limitations_artifact_checks.py
```

HPO tự bỏ qua checkpoint đã xong cùng budget. Để chạy mới hoàn toàn, dùng `--output reports/paper_limitations/rerun` cho runner; report mặc định đọc `reports/paper_limitations/data`. Có thể chạy từng fold bằng `--fold 1` ... `--fold 5`, không chạy hai tiến trình vào cùng checkpoint. Không sửa dữ liệu/cấu hình rồi reuse output cũ. `environment_freeze.txt` ghi toàn bộ phiên bản đã chạy, `base_commit.txt` ghi commit gốc.

## Phạm vi đã thiết kế

- Baseline: raw/scaled SVC, default XGBoost/LightGBM, scaled PCA 5/10/15/20/95%.
- Nested HPO: 5 outer × 3 inner × 100 trials × 2 framework × 2 protocol.
- RBF factorial: raw/scaled × gamma narrow/wide; 20 ứng viên định trước, lựa chọn bằng inner CV.
- Polynomial: degree/coef0 mặc định so với mở rộng; 24 ứng viên, lựa chọn bằng inner CV.
- Độ nhạy split: 3 tỷ lệ × 10 seed × 2 mô hình cố định.
- Selection bias: 10 seed, 30 cấu hình, train/pseudo-test/audit=60/20/20; đây là thí nghiệm cố ý chọn theo pseudo-test, không phải cáo buộc paper.

## Các phát hiện khi đọc repo gốc

1. `search_optuna` và `search_hyperopt` nhận outer train và dùng inner CV; không tìm thấy outer-test leakage trong luồng reproduce/robust.
2. `--mode all` dùng `paper_space=False`, `scaled=False` nhưng ghi variant `paper_raw`: nhãn dễ gây hiểu nhầm. Runner hạn chế của bài báo gọi hai protocol tường minh để tránh trường hợp này.
3. Runner gốc ghi cùng `fold_metrics.csv` và tên trial không có seed/protocol: có nguy cơ ghi đè khi chạy nhiều mode. Runner mới lưu checkpoint riêng, không ghi vào kết quả gốc.
4. `SquareTable(table)` mặc định có điều chỉnh ô 0; thống kê bổ sung tính từ đếm nguyên bản và ghi rõ giới hạn bảng thưa/OOF phụ thuộc.
5. Python 3.12 venv thiếu pkg_resources; Hyperopt 0.2.7 cần setuptools tương thích, đã pin 80.9.0 riêng cho môi trường này.

## Đọc số liệu đúng

Mean ± SD theo fold không phải CI. Delta với paper không phải so sánh ghép cặp vì không biết split của tác giả. Không coi PCA scaled mặc định là tái hiện HPO+PCA của paper. Không coi p lớn là chứng minh tương đương. Không chọn seed/fold tốt nhất để báo cáo.

Các file nằm trong `reports/` để có thể được review/commit; `results/` của repo gốc bị gitignore. Nhánh review: paper-limitations.
