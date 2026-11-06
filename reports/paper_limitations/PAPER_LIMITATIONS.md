# Paper Limitations

## Mục đích và kết luận chính

Tài liệu này tổng hợp phân tích phản biện, thiết kế kiểm chứng, toàn bộ bảng kết quả đã chạy, cách diễn giải, kịch bản thuyết trình và hướng dẫn tái tạo thí nghiệm cho bài “Support Vector Machines with Hyperparameter Optimization Frameworks for Classifying Mobile Phone Prices in Multi-Class” (Chang, Lin và Pai, Electronics 2025, 14, 2173).

Kết quả nested CV của nhóm đạt 97,65% với Hyperopt và 97,70% với Optuna, gần các mức 97,40% và 97,80% của paper. Kết quả hỗ trợ ích lợi HPO trên bộ dữ liệu này; chưa chứng minh paper mắc data leakage hoặc một framework vượt trội ổn định. Các hạn chế phải được phân biệt giữa thiếu mô tả, nguy cơ có điều kiện và bằng chứng thực nghiệm đã quan sát.

## I. Sáu hạn chế của bài báo và mức độ ảnh hưởng

### 1. Objective của HPO chưa được mô tả đủ rõ

Algorithm 1, trang 9, ghi objective là ±Accuracy nhưng không nêu cụ thể tập dùng để tính điểm trong từng trial. Sơ đồ của paper thể hiện tách train/test và đặt HPO trong khối học mô hình, nên không thể chỉ dựa vào sơ đồ để kết luận có leakage. Điểm chắc chắn là thiếu thông tin để kiểm tra cách đánh giá trong HPO.

Nếu điểm outer test được trả cho HPO trong 100 trials, test đã ảnh hưởng vào việc chọn cấu hình dù không xuất hiện trong lệnh fit. HPO có thể chọn cả cấu hình tốt thật lẫn cấu hình may mắn phù hợp với tập test. Đây là thiên lệch lựa chọn; trung bình qua 5 fold hoặc fit lại mô hình sau khi chọn tham số không khôi phục tính độc lập của test. HPO dùng inner validation/CV trong outer train thì phù hợp. Chấm trên chính tập vừa fit là điểm train, không phải đánh giá khả năng tổng quát.

Đây là hạn chế cần ưu tiên kiểm chứng nhất vì có thể ảnh hưởng đồng thời accuracy cuối, mức cải thiện so với baseline và xếp hạng framework. Tuy nhiên, code repo của nhóm đã dùng inner 3-fold trên outer train. Nested CV vẫn đạt gần mức paper; không có cơ sở từ các thí nghiệm hiện tại để quy kết kết quả paper cao do leakage.

Khắc phục: công bố objective, giữ outer test ngoài quá trình chọn siêu tham số, fit scaler/PCA trong từng training split. Không đổi seed, search space hay tiền xử lý chỉ để đạt điểm outer test mong muốn. Đối chứng pseudo-test ở phần số liệu chỉ kiểm tra cơ chế, không tái hiện một lỗi đã được xác nhận của tác giả.

### 2. Khoảng gamma hẹp và phụ thuộc thang đo dữ liệu

Trang 10–11 mô tả thu hẹp gamma về [4.701e-7, 6.701e-7], cận trên chỉ khoảng 1,43 lần cận dưới. Phụ thuộc dữ liệu không tự nó là lỗi: khoảng phù hợp có thể tiết kiệm tính toán. Hạn chế là thiếu đánh giá độ nhạy và thiếu chi tiết về dữ liệu dùng để xác định khoảng, nên khó biết kết luận về kernel phụ thuộc bao nhiêu vào thiết kế tìm kiếm.

Với RBF, K(x,z)=exp(-gamma*||x-z||²). Gamma nhỏ không tự chứng minh dữ liệu chưa chuẩn hóa hoặc mô hình gần tuyến tính. Ví dụ một chiều chênh 1.000 đơn vị cho bình phương khoảng cách 1.000.000; gamma=5e-7 vẫn đóng góp 0,5 vào độ lớn số mũ. Paper chưa mô tả rõ scaler; “chưa chuẩn hóa” phải được giữ ở mức giả thuyết.

Đối chứng nhóm cho thấy RBF scaled+narrow chỉ đạt 69,95%, scaled+wide đạt 93,25%; raw+narrow lại đạt 96,35%, cao hơn raw+wide 95,60% với ngân sách 20 ứng viên. Vì thế không thể kết luận gamma càng rộng càng tốt hoặc scaling luôn cải thiện. 9/10 mô hình paper-space cuối của nhóm chọn linear, nơi gamma không ảnh hưởng hàm nhân; không nên diễn giải những giá trị gamma đó là tối ưu có ý nghĩa.

Khắc phục: chọn khoảng theo kernel/thang đo, ghi rõ căn cứ và đánh giá độ nhạy chỉ trên train/validation. Nếu dùng điểm test để thu hẹp khoảng, đó lại là tuning theo test. Không chuyển nguyên khoảng số sang dataset hoặc đơn vị đo khác.

### 3. Diễn giải mức hơn kém giữa Optuna và Hyperopt

Paper báo cáo 97,80% so với 97,40%, chênh 0,40 điểm phần trăm; kiểm định McNemar–Bowker có p=0,995. Có thể mô tả Optuna có mean cao hơn trong thí nghiệm, nhưng không suy ra ưu thế tổng quát. p lớn không chứng minh hai phương pháp tương đương; muốn kết luận tương đương cần ngưỡng chênh lệch có ý nghĩa thực tiễn và thiết kế kiểm định phù hợp.

Bowker kiểm tra tính đối xứng bảng phân loại ghép cặp, không trực tiếp chứng minh mọi khía cạnh hiệu năng bằng nhau. Trang 13 nói dùng trường hợp chính xác nhất của từng mô hình, trong khi best fold của Hyperopt và Optuna trong bảng không trùng nhau. Cần làm rõ tác giả có ghép đúng cùng quan sát hay không; chưa đủ bằng chứng khẳng định kiểm định đã được thực hiện sai.

Nhóm ghép đúng cùng index OOF và báo cáo thêm exact McNemar trên đúng/sai. Chênh paper-space chỉ 0,05 pp, tương ứng một dự đoán đúng nhiều hơn trên 2.000 mẫu. Bảng bất đồng thưa và các training fold chồng lấn khiến p-value OOF chỉ nên được dùng mô tả hỗ trợ. Không coi p lớn là bằng chứng tương đương, hoặc SD nhỏ qua 5 fold là chứng minh ổn định qua seed.

### 4. Thiếu chi tiết tái lập và đối chứng công bằng

Các chi tiết cần công bố gồm seed chia dữ liệu, shuffle/stratification, seed sampler, objective, scaler, số thành phần và vị trí fit PCA, search space của boosting. Thiếu chúng khiến khác biệt kết quả có thể đến từ triển khai thay vì bản chất phương pháp. Cùng 100 trials chưa đủ đảm bảo công bằng nếu không gian tìm kiếm của một mô hình thuận lợi hơn mô hình khác.

PCA là ví dụ cụ thể: nhóm thử scaled PCA 5/10/15/20/95%; accuracy thay đổi từ 33,45% đến 87,70%. Giữ phương sai đầu vào không đảm bảo giữ thông tin dự báo nhãn. Các cấu hình này khác HPO+PCA chưa được mô tả đầy đủ trong paper, nên không dùng chúng để phủ định trực tiếp Bảng 10. Tương tự, kết quả boosting mặc định chưa đủ để kết luận SVM tốt hơn boosting đã tuning.

Nhóm lưu phiên bản thư viện, hash dữ liệu, commit gốc, phân hoạch, dự đoán, tham số và lịch sử trials. Python thực chạy là 3.12, khác 3.9.15 của paper; các thư viện chính được pin theo repo. Các phát hiện về mode all, ghi đè kết quả và môi trường được giữ trong phần hướng dẫn cuối tài liệu.

### 5. Kết quả 99,67% nhạy với tập test nhỏ

Với 300 mẫu test, 299 đúng và 1 sai cho 99,67%; 2 sai cho 99,33%, 3 sai cho 99,00%. Mỗi mẫu đổi khoảng 0,333 điểm phần trăm. Điểm từ một split chưa mô tả hết biến động do các mẫu khó và trường hợp gần ranh giới lớp.

Tập test nhỏ làm tăng độ bất định, không tự gây thiên lệch lạc quan. Thiên lệch phát sinh khi chọn split/seed có điểm tốt nhất hoặc chỉnh mô hình theo test. Nhóm chạy 10 seed ở mỗi tỷ lệ với hai mô hình cố định để đo độ nhạy; đây không phải tái hiện single-split HPO của paper. Không dùng kết quả baseline thấp hơn 99,67% để tuyên bố không tái lập được mô hình HPO.

Khắc phục: lưu seed, báo số lỗi, mean/SD và khoảng bất định phù hợp; không chọn best seed. Các lần chia dùng chung dataset có chồng lấn nên không được coi là các nghiên cứu độc lập.

### 6. Không gian kernel chưa tận dụng đúng các tham số

Paper tune kernel/C/gamma/decision_function_shape nhưng giữ degree/coef0 cố định. Không bắt buộc tune mọi tham số, song cần giải thích và kiểm tra độ nhạy khi đưa ra kết luận về các kernel. Polynomial dùng (gamma*xᵀz+coef0)^degree; degree và coef0 tương tác với gamma và C. Sigmoid cũng chịu tác động của coef0.

Đối chứng polynomial scaled của nhóm, cùng 24 bộ C/gamma và cùng các fold, đạt 78,95% khi cố định degree=3, coef0=0 và 94,30% khi mở rộng cả hai. Đây là hiệu ứng của thiết kế tìm kiếm cụ thể, chưa tách riêng tác động từng tham số và không phải mức tăng dự đoán cho raw polynomial của paper.

Trong sklearn SVC, decision_function_shape='ovr' không đổi việc huấn luyện sang One-vs-Rest; SVC vẫn dùng One-vs-One. Kiểm tra cùng tham số trên 400 mẫu của fold 1 không thấy dự đoán thay đổi khi đổi shape ở cả bốn kernel, với break_ties mặc định. Gamma không tác động linear. Không gian tìm kiếm nên có điều kiện theo kernel để hạn chế thử các tham số không ảnh hưởng mô hình.

## Thứ tự ưu tiên và cách phát biểu phản biện

| Ưu tiên | Vấn đề | Kết luận được phép từ bằng chứng hiện có |
|---|---|---|
| 1 | Tính độc lập của đánh giá HPO | Paper thiếu mô tả; code nhóm dùng nested CV; chưa chứng minh paper leakage |
| 2 | Tái lập và công bằng đối chứng | Thiếu chi tiết khiến khó đối chiếu chính xác; cần code/config/splits |
| 3 | Scaling, gamma và kernel | Có ảnh hưởng thực đo nhưng phụ thuộc cấu hình và ngân sách |
| 4 | Hơn kém framework | Chỉ có chênh lệch quan sát; chưa chứng minh vượt trội hoặc tương đương |
| 5 | Test nhỏ và seed | Cần báo biến động, không dựa vào một split đẹp |

Nguồn: [bài báo gốc](https://doi.org/10.3390/electronics14112173), nội dung paper.md trong repo, kế hoạch thuyết trình đã cung cấp và các kết quả thực nghiệm bên dưới. Các nhận định mạnh hơn đã được sửa: gamma nhỏ không tự chứng minh chưa scale; p lớn không chứng minh tương đương; độ chính xác cao không tự chứng minh leakage.

## II. Phân tích thực nghiệm và nội dung trình bày

### Kết luận có bằng chứng

**HPO cải thiện SVM trong lần nested CV này.** Baseline raw đạt 95.15%, paper-space Hyperopt đạt 97.65%, paper-space Optuna đạt 97.70%. Paper lần lượt báo cáo 94,95%, 97,40%, 97,80%. Bảng comparison.csv có chênh lệch chính xác. Kết quả cùng xu hướng hỗ trợ ích lợi HPO trên dataset này, nhưng không chứng minh mọi dataset hoặc seed đều cải thiện. Không được dùng khác biệt số liệu để khẳng định paper có leakage vì chưa biết split và objective của tác giả.

**Không có bằng chứng outer-test leakage trong code HPO của repo đã đọc.** Search chỉ nhận outer train, cross_val_score dùng inner 3-fold; outer test chấm sau lựa chọn. Sơ đồ paper thiếu inner validation vẫn là hạn chế báo cáo. Thí nghiệm cố ý chọn theo pseudo-test của nhóm chỉ minh họa cơ chế, không đo trực tiếp leakage của paper.

**Chuẩn hóa đơn thuần không bảo đảm cải thiện.** Scaled SVC mặc định đạt 87.70%, thấp hơn raw 95.15%; scaled+HPO đạt 97.05% với Hyperopt và 97.15% với Optuna. Thang đo biến và gamma/kernel/C cần được xét chung. Không thể quy toàn bộ mức tăng cho riêng StandardScaler vì robust thay đổi cả không gian tìm kiếm.

**Gamma hẹp không thể chuyển nguyên xi sang dữ liệu đã scale.** Trong đối chứng RBF-only, raw+narrow đạt 96.35%, raw+wide 95.60%, scaled+narrow 69.95%, scaled+wide 93.25%. Khi giữ scaling cố định, đây là bằng chứng khoảng gamma ảnh hưởng đáng kể. Ngược lại, khoảng rộng không bảo đảm tốt hơn trên raw vì chỉ có 20 ứng viên; ngân sách hữu hạn và mật độ tìm kiếm cũng quan trọng.

**PCA cần công bố số thành phần và preprocessing.** Scaled PCA 5 đạt 33.45%, 10 đạt 47.55%, 15 đạt 82.35%, 20 đạt 87.70%, 95% variance đạt 87.45%. PCA 20 giữ đủ chiều và cho dự đoán giống scaled baseline trên toàn bộ OOF; phép quay trực giao giữ khoảng cách RBF. Trên training fold 1, PCA 5 chỉ giữ 22.84% phương sai RAM, PCA 15 giữ 99.60%. Điều này hỗ trợ cơ chế mất thông tin có ích khi giảm chiều; chưa phải phân tích nhân quả hoàn chỉnh. PCA tối ưu phương sai X, không tối ưu dự báo nhãn. Các kết quả này không phủ định Bảng 10 vì paper thiếu n_components/scaling và còn có HPO+PCA mà nhóm chưa chạy.

**Bỏ degree/coef0 có thể hạn chế polynomial.** Đối chứng polynomial scaled cho nhánh cố định 78.95% và nhánh mở rộng 94.30%. Hai nhánh cùng 24 bộ C/gamma, cùng split, inner CV; nhánh mở rộng thay cả degree và coef0. Đây là tác động của cả thiết kế không gian, chưa tách được riêng tham số nào gây cải thiện; không phải mức cải thiện dự đoán cho raw polynomial của paper.

**decision_function_shape không phải thử hai cách huấn luyện.** Cùng kernel và tham số, thay ovo/ovr không đổi dự đoán ở kiểm tra fold 1 của cả bốn kernel. SVC huấn luyện OvO; shape chỉ đổi dạng điểm quyết định với cấu hình mặc định. Đây là điểm nên sửa trong phần lý thuyết của nhóm để tránh bị hỏi khó.

**Boosting mặc định thấp hơn SVM raw trong lần chạy này.** XGBoost 91.30%, LightGBM 90.50%; paper tương ứng 91,15% và 90,70%. Hỗ trợ xu hướng trong thí nghiệm baseline, không kết luận SVM vượt mọi cấu hình boosting vì nhóm chưa tune boosting.

### Giải thích đối chứng selection bias

Ở ngân sách 30 ứng viên, điểm trung bình được báo cáo trên pseudo-test là 93.73%, audit độc lập là 93.67%, chênh lệch 0.05 điểm phần trăm. Nhánh chọn sạch bằng inner CV có pseudo-test 93.10% và audit 93.73%. Trong lần chạy này, gap pseudo-test so với audit là 0.050 pp, chưa cho thấy độ phồng lớn. Tuy nhiên, so với cách chọn sạch, cách chọn theo pseudo-test có chênh lệch điểm pseudo-test 0.625 pp và chênh lệch điểm audit -0.050 pp. Lợi thế trên tập dùng để chọn không chuyển thành lợi thế trên audit. Không diễn giải các chênh lệch nhỏ này như bằng chứng thống kê chắc chắn. Các giá trị ghi đúng dấu; không ép kết quả phải chứng minh giả thuyết. Chênh lệch giữa hai tập còn chịu dao động lấy mẫu. Khi đọc bảng theo budget, không mặc định gap phải tăng đơn điệu. Audit không được đưa vào lựa chọn cấu hình.

### Độ nhạy split và diễn giải thống kê

Đọc split_sensitivity.csv: 10 seed cho mỗi tỷ lệ 70/30, 80/20, 85/15. Hai mô hình là raw mặc định và scaled linear C=1 cố định, không phải HPO. Vì vậy không dùng các số này tuyên bố nhóm thất bại tái hiện 99,67% của paper. Ý nghĩa là một điểm số từ một split không mô tả hết biến động. 99,67% trên 300 mẫu là 299 đúng, 1 sai; chỉ thêm 1 lỗi đã còn 99,33%.

Đọc paired_statistics.json và mục 3 báo cáo số liệu. Kiểm định dùng cùng 2.000 mẫu OOF, không chọn best fold của từng phương pháp. p lớn không chứng minh hai framework tương đương. Bowker đánh giá tính đối xứng bảng dự đoán; exact McNemar bổ sung góc nhìn đúng/sai. Vì các mô hình CV dùng training overlap và bảng bất đồng có thể thưa, chỉ coi p-value là thông tin mô tả hỗ trợ. Nghiên cứu này chưa có repeated nested CV đủ nhiều seed để kết luận ưu thế bền vững của một framework.

### Kịch bản khoảng 4 phút

1. **0:00–0:50 — Bảng tổng hợp.** “Nhóm dùng cùng 5 outer fold và inner 3-fold để chọn siêu tham số. Baseline raw là 95.15%; Hyperopt paper-space 97.65%; Optuna paper-space 97.70%. Chúng tôi so sánh xu hướng với paper, không khẳng định trùng giao thức vì tác giả chưa công bố đầy đủ cách chia và objective.”
2. **0:50–1:40 — Scaling/gamma.** “Chỉ chuẩn hóa chưa chắc tốt: scaled mặc định là 87.70%. Khi tune lại, Hyperopt/Optuna robust lần lượt 97.05%/97.15%. Đối chứng RBF chứng minh gamma cần phù hợp thang đo.”
3. **1:40–2:30 — PCA/kernel.** “Giảm PCA quá mạnh làm mất thông tin dự báo; 5 thành phần đạt 33.45%. Mở rộng degree/coef0 trong đối chứng polynomial thay đổi kết quả từ 78.95% lên 94.30%. Đây là các mở rộng riêng, không phải kết quả nguyên bản paper.”
4. **2:30–3:20 — Độ tin cậy.** “Repo nhóm dùng nested CV nên outer test không tham gia HPO. Đối chứng cố ý chọn theo pseudo-test cho thấy cần đánh giá độc lập, nhưng không chứng minh tác giả đã dùng test. Chúng tôi báo cáo toàn bộ fold và seed, không chọn kết quả đẹp nhất.”
5. **3:20–4:00 — Kết luận và demo.** “Kết quả hỗ trợ lợi ích tuning SVM trên dữ liệu này. Các claim scaling/PCA luôn tăng accuracy hoặc một framework luôn tốt hơn chưa được chứng minh. Demo mở comparison.csv, một checkpoint JSON và trial CSV để đối chiếu.”

### Năm câu hỏi dự kiến

1. **Vì sao kết quả khác paper?** Seed/split/inner objective chưa được tác giả công bố đầy đủ; nhóm dùng nested CV rõ ràng và phiên bản môi trường đã lưu. Khác số liệu không tự chứng minh leakage.
2. **Chuẩn hóa sao lại kém?** Scaling thay đổi trọng số tương đối các biến trong khoảng cách và gamma mặc định; phải tune lại. Raw có thể vô tình đặt trọng số lớn lên biến hữu ích như RAM.
3. **PCA giữ 95% variance sao chưa tốt?** Giữ phương sai đầu vào không đồng nghĩa giữ thông tin dự báo nhãn; phải xét cùng scaling, số thành phần và mô hình.
4. **Optuna tốt hơn Hyperopt không?** Nêu mean/SD và chênh lệch thực đo, không chuyển một lần CV hoặc p lớn thành kết luận vượt trội/tương đương.
5. **Đã chứng minh paper leakage chưa?** Chưa. Đã xác minh luồng code nhóm tách outer test và thực hiện đối chứng minh họa selection bias; cần code/hàm objective của tác giả để kết luận paper.

### Phần chưa thực hiện

HPO boosting; HPO+PCA; repeated nested HPO nhiều seed; đánh giá dataset ngoài; tái hiện chính xác single-split HPO của paper. Không trình bày những mục này như kết quả đã có. Code và kết quả được lưu trên nhánh paper-limitations để nhóm review.

### Điểm quan trọng khi bảo vệ kết quả

Nested CV sạch vẫn đạt 97.70%, gần mức 97,80% paper. Thí nghiệm này không cho thấy một sự sụt giảm lớn đủ để nghi ngờ rằng toàn bộ hiệu quả HPO trong paper đến từ leakage. Không được dùng đối chứng cố ý sai để gán lỗi cho tác giả.

Nhánh mang tên robust không tự động tốt hơn paper-space: Hyperopt robust trừ raw là -0.60 pp, Optuna là -0.55 pp. Tên robust chỉ mô tả một thiết kế mở rộng; chưa chứng minh khả năng tổng quát tốt hơn.

Trong 10 mô hình HPO paper-space cuối cùng, 9/10 chọn linear. Với những mô hình đó gamma không tham gia hàm nhân, nên không diễn giải gamma được báo cáo là giá trị tối ưu có ý nghĩa. Các giá trị C/kernel giữa fold có thể khác mà dự đoán gần như nhau; biến động tham số chưa đủ chứng minh mô hình thiếu ổn định.

## III. Thiết kế, bảng số liệu và kiểm tra chất lượng
Mục tiêu: kiểm tra các hạn chế của paper bằng kết quả thực đo; không mặc định paper đã rò rỉ dữ liệu.

### 1. Thiết kế và mức độ so sánh

- Dữ liệu: 2.000 mẫu, 20 đặc trưng; chỉ train.csv có nhãn được dùng để đánh giá.
- Outer StratifiedKFold 5 fold, shuffle=True, seed=42; inner 3 fold, seed=42+fold.
- HPO: 100 trials cho mỗi framework trong từng fold; dùng nguyên hàm search của repo. Paper-space giữ raw và gamma hẹp; robust dùng StandardScaler và gamma 1e-4..10 log-scale.
- 5 fold chỉ là một lần CV; độ lệch chuẩn giữa fold không phải khoảng tin cậy và không đo hết bất định của HPO.
- Không tuyên bố tái lập chính xác: paper không công bố seed/phân hoạch/objective đầy đủ; Python chạy là 3.12 thay vì 3.9.15 của paper. Phiên bản thực tế lưu trong manifest và environment_freeze.txt.
- Basic PCA là scaled SVC mặc định với PCA fit trong train, không phải HPO+PCA của paper. Boosting chỉ chạy mặc định, không chạy HPO boosting.

### 2. Bảng so sánh kết quả

| model | accuracy_mean_pct | accuracy_std_pp | macro_f1_pct | paper_pct | delta_pp | total_seconds |
|---|---|---|---|---|---|---|
| lgbm_default | 90.500 | 1.723 | 90.501 | 90.700 | -0.200 | 1.039 |
| pca_0.95 | 87.450 | 2.341 | 87.513 | — | — | 1.007 |
| pca_10 | 47.550 | 4.009 | 47.174 | — | — | 0.577 |
| pca_15 | 82.350 | 5.273 | 82.422 | — | — | 0.927 |
| pca_20 | 87.700 | 2.161 | 87.767 | — | — | 0.834 |
| pca_5 | 33.450 | 4.159 | 32.206 | — | — | 0.800 |
| svm_raw | 95.150 | 1.318 | 95.124 | 94.950 | 0.200 | 0.188 |
| svm_scaled | 87.700 | 2.161 | 87.767 | — | — | 0.518 |
| xgb_default | 91.300 | 1.430 | 91.291 | 91.150 | 0.150 | 2.005 |
| paper_hyperopt | 97.650 | 0.652 | 97.651 | 97.400 | 0.250 | 851.448 |
| paper_optuna | 97.700 | 0.694 | 97.701 | 97.800 | -0.100 | 654.312 |
| robust_hyperopt | 97.050 | 0.694 | 97.046 | — | — | 77.551 |
| robust_optuna | 97.150 | 0.675 | 97.147 | — | — | 100.657 |

Đơn vị accuracy/F1: %. Delta là điểm phần trăm (pp). Dấu —: không có đối chứng trực tiếp phù hợp. Thời gian đo khi các tiến trình chạy đồng thời; không dùng để xếp hạng tốc độ framework.

### 3. So sánh framework và thống kê

| protocol | bowker_statistic | bowker_df_nominal | bowker_p_nominal | nonzero_pairs | hyperopt_only_correct | optuna_only_correct | mcnemar_exact_p |
|---|---|---|---|---|---|---|---|
| paper | 1.0 | 6 | 0.985612 | 3 | 3 | 4 | 1.0 |
| robust | 3.2 | 6 | 0.783358 | 3 | 4 | 6 | 0.753906 |

Dự đoán được ghép đúng cùng index ngoài mẫu. Bowker dùng bảng đếm nguyên bản, không cộng giả 0,5 vào ô 0; dùng df danh nghĩa 6 cho 4 lớp. Bảng thưa làm xấp xỉ chi-square kém tin cậy. McNemar exact kiểm tra bất đồng đúng/sai. Các p-value OOF chỉ mang tính mô tả vì mô hình ở các fold có training overlap; không coi 5 fold là 5 nghiên cứu độc lập. p lớn không chứng minh tương đương; không chọn fold tốt nhất để kiểm định.

### 4. Gamma và scaling: kiểm tra có kiểm soát

RBF-only, cùng 20 bộ quantile C/gamma định trước và cùng outer/inner splits. Wide gamma 1e-9..10 để bao cả thang raw/scaled; đây là đối chứng riêng, khác robust TPE. Các thông số được chọn bằng inner CV.

| scaled | wide_gamma | mean | std |
|---|---|---|---|
| False | False | 96.35 | 0.84 |
| False | True | 95.6 | 0.627 |
| True | False | 69.95 | 3.262 |
| True | True | 93.25 | 1.49 |

So sánh theo cùng hàng/cột để tách ảnh hưởng scaling và khoảng gamma. Không suy rộng kết quả RBF-only thành xếp hạng mọi kernel, và không coi scaling luôn cải thiện.

### 5. Degree và coef0

Polynomial-only, scaled, 24 cấu hình mỗi nhánh; cùng C/gamma. Nhánh cố định degree=3, coef0=0; nhánh mở rộng degree 2..5, coef0 thuộc {0,0.5,1,2}. Không gian lớn hơn với ngân sách hữu hạn không bảo đảm thắng.

| expanded | mean | std |
|---|---|---|
| False | 78.95 | 1.634 |
| True | 94.3 | 0.694 |

Kiểm tra decision_function_shape trên fold 1, cùng mọi tham số còn lại:

| kernel | different_predictions | n_test | support_equal |
|---|---|---|---|
| linear | 0 | 400 | True |
| poly | 0 | 400 | True |
| rbf | 0 | 400 | True |
| sigmoid | 0 | 400 | True |

SVC vẫn huấn luyện OvO. Thay shape ovo/ovr với break_ties mặc định không phải so sánh hai chiến lược huấn luyện khác nhau.

### 6. Đối chứng cố ý chọn theo pseudo-test

10 seed 42..51; chia 60% train / 20% pseudo-test / 20% audit. Cố định trước 30 cấu hình RBF scaled. Nhánh sai chọn cấu hình theo pseudo-test; audit không tham gia chọn. Nhánh sạch chọn theo inner CV trong train. Không refit thêm pseudo-test để giữ cùng kích thước train. Đây là chẩn đoán cơ chế selection bias bằng random candidates, KHÔNG phải tái hiện quy trình chưa được biết của paper.

| budget | reported_pseudotest | audit_accuracy | gap | clean_test_accuracy | clean_audit_accuracy |
|---|---|---|---|---|---|
| 1 | 68.35 | 68.9 | -0.55 | 68.35 | 68.9 |
| 5 | 92.7 | 93.525 | -0.825 | 92.7 | 93.525 |
| 10 | 93.725 | 93.675 | 0.05 | 93.1 | 93.725 |
| 30 | 93.725 | 93.675 | 0.05 | 93.1 | 93.725 |

Gap pseudo-test trừ audit bao gồm cả dao động lấy mẫu và hiệu ứng lựa chọn; không dùng nó làm ước lượng mức leakage của paper. Các split khác seed chồng lấn dữ liệu nên không coi là 10 mẫu nghiên cứu độc lập.

### 7. Độ nhạy tập test

| model | test_size | mean | std | min | max |
|---|---|---|---|---|---|
| raw_default | 0.15 | 95.133 | 0.878 | 94.0 | 96.667 |
| raw_default | 0.2 | 94.975 | 0.712 | 93.5 | 95.75 |
| raw_default | 0.3 | 94.7 | 0.565 | 93.833 | 96.0 |
| scaled_linear_C1 | 0.15 | 96.467 | 0.757 | 94.667 | 97.333 |
| scaled_linear_C1 | 0.2 | 96.4 | 0.868 | 95.25 | 98.0 |
| scaled_linear_C1 | 0.3 | 95.817 | 0.664 | 94.5 | 96.667 |

10 seed ở mỗi tỷ lệ, baseline raw mặc định và scaled linear C=1 cố định; không phải HPO của Bảng 12–14. 99,67% trên 300 mẫu tương ứng 1 lỗi; mỗi lỗi đổi 0,333 pp. CI nhị thức từng split có trong CSV và giả định mẫu độc lập; không xử lý dataset shift. Không chọn seed tốt nhất để báo cáo.

### 8. Kiểm tra chất lượng kết quả

- Đã xác nhận đủ 20 checkpoint HPO, mỗi checkpoint 100 trials.
- Train/test index không giao nhau; mỗi quan sát xuất hiện đúng một lần trong outer test cho mỗi mô hình.
- Accuracy tính lại từ dự đoán đã lưu khớp metric.
- Hai framework được ghép đúng index và nhãn khi tính thống kê.
- Scaler/PCA nằm trong pipeline; search chỉ nhận outer train.
- Không có kết quả HPO nhiều seed, HPO boosting hay HPO+PCA; không suy diễn những phần này.

## IV. Hướng dẫn tái chạy và kiểm tra code

Đọc `reports/paper_limitations/PAPER_LIMITATIONS.md` và `reports/paper_limitations/comparison.csv` trước. Code bổ sung không thay đổi hàm HPO gốc.

### Chạy lại

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

### Phạm vi đã thiết kế

- Baseline: raw/scaled SVC, default XGBoost/LightGBM, scaled PCA 5/10/15/20/95%.
- Nested HPO: 5 outer × 3 inner × 100 trials × 2 framework × 2 protocol.
- RBF factorial: raw/scaled × gamma narrow/wide; 20 ứng viên định trước, lựa chọn bằng inner CV.
- Polynomial: degree/coef0 mặc định so với mở rộng; 24 ứng viên, lựa chọn bằng inner CV.
- Độ nhạy split: 3 tỷ lệ × 10 seed × 2 mô hình cố định.
- Selection bias: 10 seed, 30 cấu hình, train/pseudo-test/audit=60/20/20; đây là thí nghiệm cố ý chọn theo pseudo-test, không phải cáo buộc paper.

### Các phát hiện khi đọc repo gốc

1. `search_optuna` và `search_hyperopt` nhận outer train và dùng inner CV; không tìm thấy outer-test leakage trong luồng reproduce/robust.
2. `--mode all` dùng `paper_space=False`, `scaled=False` nhưng ghi variant `paper_raw`: nhãn dễ gây hiểu nhầm. Runner hạn chế của bài báo gọi hai protocol tường minh để tránh trường hợp này.
3. Runner gốc ghi cùng `fold_metrics.csv` và tên trial không có seed/protocol: có nguy cơ ghi đè khi chạy nhiều mode. Runner mới lưu checkpoint riêng, không ghi vào kết quả gốc.
4. `SquareTable(table)` mặc định có điều chỉnh ô 0; thống kê bổ sung tính từ đếm nguyên bản và ghi rõ giới hạn bảng thưa/OOF phụ thuộc.
5. Python 3.12 venv thiếu pkg_resources; Hyperopt 0.2.7 cần setuptools tương thích, đã pin 80.9.0 riêng cho môi trường này.

### Đọc số liệu đúng

Mean ± SD theo fold không phải CI. Delta với paper không phải so sánh ghép cặp vì không biết split của tác giả. Không coi PCA scaled mặc định là tái hiện HPO+PCA của paper. Không coi p lớn là chứng minh tương đương. Không chọn seed/fold tốt nhất để báo cáo.

Các file nằm trong `reports/` để có thể được review/commit; `results/` của repo gốc bị gitignore. Nhánh review: paper-limitations.

Các đường dẫn output cũ trong manifest là lịch sử lần chạy, được giữ nguyên để bảo toàn nguồn gốc. Thư mục hiện tại là reports/paper_limitations.
