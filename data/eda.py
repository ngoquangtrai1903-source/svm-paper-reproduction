"""Vẽ ma trận tương quan (Pearson) cho train.csv.

Cài thư viện:  pip install pandas numpy seaborn matplotlib
Chạy:          python correlation_matrix.py
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CSV_PATH = "train.csv"                 # đổi thành đường dẫn file của bạn
OUT_PATH = "correlation_matrix_v2.png"
TARGET = "price_range"

# 1. Đọc dữ liệu và kiểm tra nhanh
df = pd.read_csv(CSV_PATH)
print("Kích thước:", df.shape)
print("Số ô thiếu:", df.isna().sum().sum())

# 2. Tính ma trận tương quan Pearson
corr = df.corr(method="pearson", numeric_only=True)

# 3. Chỉ giữ nửa tam giác dưới (giữ đường chéo),
#    bỏ hàng đầu và cột cuối vì chỉ chứa ô trống
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
corr_plot = corr.iloc[1:, :-1]
mask_plot = mask[1:, :-1]

# 4. Vẽ heatmap
plt.figure(figsize=(15, 12))
ax = sns.heatmap(
    corr_plot,
    mask=mask_plot,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    annot_kws={"size": 8},
    cbar_kws={"shrink": 0.8, "label": "Pearson r"},
)
ax.set_title("Ma tran tuong quan (Pearson) - train.csv", fontsize=15)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
plt.show()

# 5. In các tương quan nổi bật để đối chiếu
with_target = corr[TARGET].drop(TARGET)
print(f"\nTương quan với {TARGET}:")
print(with_target.reindex(with_target.abs().sort_values(ascending=False).index).round(3).head(5))

feat = corr.drop(index=TARGET, columns=TARGET)
pairs = feat.where(np.triu(np.ones_like(feat, dtype=bool), k=1)).stack()
print("\nCác cặp đặc trưng tương quan nhất:")
print(pairs.reindex(pairs.abs().sort_values(ascending=False).index).round(3).head(5))