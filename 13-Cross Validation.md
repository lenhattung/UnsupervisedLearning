# Bài 17: Cross-Validation (Kiểm định chéo)

> **Môn học:** Data Science với Python  
> **Dữ liệu sử dụng:** `breast_cancer`, `titanic` (seaborn), `wine`, `diabetes` (sklearn.datasets)

---

## 1. Động lực: Bias–Variance Tradeoff

Khi xây dựng mô hình học máy, chúng ta luôn đối mặt với câu hỏi: **mô hình có thực sự học được quy luật từ dữ liệu, hay chỉ đang "học thuộc lòng" (memorize) tập huấn luyện?**

Hiện tượng này được gọi là **Bias–Variance Tradeoff**:

- **Bias cao** (Underfitting): Mô hình quá đơn giản, không nắm bắt được quy luật thực.
- **Variance cao** (Overfitting): Mô hình quá phức tạp, khớp rất tốt với tập huấn luyện nhưng dự đoán kém trên dữ liệu mới.

**Mục tiêu:** Tìm mức độ phức tạp tối ưu để mô hình hoạt động tốt trên **dữ liệu mới** — không phải trên dữ liệu huấn luyện.

> 💡 **Câu hỏi đặt ra:** Làm thế nào để đánh giá hiệu quả "thực sự" của mô hình trên dữ liệu mà nó chưa từng thấy?  
> **Trả lời:** Sử dụng **Cross-Validation**.

---

## 2. Cross-Validation là gì?

**Cross-Validation (CV)** là kỹ thuật đánh giá mô hình bằng cách chia dữ liệu thành tập **huấn luyện (training)** và tập **kiểm định (validation)** theo nhiều cách khác nhau, sau đó lấy trung bình kết quả.

### Mục đích sử dụng CV:
1. **Đánh giá khách quan** hiệu năng thực sự của mô hình.
2. **Tinh chỉnh siêu tham số** (hyperparameters) như số cây trong Random Forest, độ sâu của cây, v.v.

---

## 3. Các phương pháp Cross-Validation

### 3.1. Holdout (Train/Test Split)

Chia dữ liệu một lần thành 2 phần:
- **Training set** (~70–80%): Dùng để huấn luyện mô hình.
- **Test set** (~20–30%): Dùng để đánh giá.

**Hạn chế:** Kết quả phụ thuộc nhiều vào lần chia ngẫu nhiên cụ thể — có thể biến động lớn giữa các lần chạy.

### 3.2. K-Fold Cross-Validation

Chia dữ liệu thành **K phần bằng nhau (fold)**. Lặp K lần:
- Mỗi lần, 1 fold làm tập validation, K−1 fold còn lại làm training.
- Kết quả cuối = **trung bình** của K lần đánh giá.

**K thường dùng:** 5 hoặc 10.

```
Fold 1: [TEST] [    ] [    ] [    ] [    ]
Fold 2: [    ] [TEST] [    ] [    ] [    ]
Fold 3: [    ] [    ] [TEST] [    ] [    ]
Fold 4: [    ] [    ] [    ] [TEST] [    ]
Fold 5: [    ] [    ] [    ] [    ] [TEST]
```

### 3.3. Repeated K-Fold Cross-Validation

Chạy K-Fold **nhiều lần** với các lần chia ngẫu nhiên khác nhau, rồi lấy trung bình tất cả kết quả. Giúp giảm phương sai (variance) của kết quả đánh giá.

### 3.4. Leave-One-Out Cross-Validation (LOO-CV)

Trường hợp đặc biệt: K = N (số quan sát). Mỗi lần, **1 quan sát** làm validation, N−1 còn lại làm training.

- **Ưu điểm:** Tận dụng tối đa dữ liệu cho huấn luyện, kết quả ổn định (không có yếu tố ngẫu nhiên).
- **Nhược điểm:** Tốn tính toán; các mô hình huấn luyện rất giống nhau nên ít đa dạng kịch bản test.

### 3.5. Bootstrap Cross-Validation

Lấy mẫu **có hoàn lại (with replacement)** N quan sát từ tập dữ liệu N quan sát làm training. Các quan sát **không được chọn** (~36.8%) làm validation. Lặp lại nhiều lần.

### So sánh tổng quan

| Phương pháp | Ưu điểm | Nhược điểm |
|---|---|---|
| Holdout | Nhanh, đơn giản | Kết quả biến động theo lần chia |
| K-Fold | Cân bằng, phổ biến nhất | Kết quả vẫn thay đổi giữa các lần chạy |
| Repeated K-Fold | Kết quả ổn định hơn | Tốn thời gian hơn |
| LOO-CV | Ổn định, không ngẫu nhiên | Rất chậm với tập lớn |
| Bootstrap | Linh hoạt | Một số quan sát có thể không bao giờ vào validation |

---

## 4. Những vấn đề cần lưu ý khi thiết lập CV

### 4.1. Stratification (Phân tầng)

Khi dữ liệu **mất cân bằng lớp** (imbalanced), một fold ngẫu nhiên có thể chứa rất ít mẫu của lớp thiểu số. Giải pháp: dùng **Stratified K-Fold** — đảm bảo mỗi fold giữ nguyên tỷ lệ lớp như dữ liệu gốc.

### 4.2. Dữ liệu chuỗi thời gian (Time Series)

Với time series, các quan sát **không độc lập** theo thời gian. Không thể chia ngẫu nhiên — phải chia theo **thứ tự thời gian**: quá khứ làm training, tương lai làm validation.

### 4.3. Rò rỉ thông tin (Information Leakage)

CV chỉ có giá trị nếu tập training và validation **độc lập** nhau. Các nguồn gây rò rỉ thường gặp:
- Nhiều quan sát từ cùng một người/đơn vị.
- Thực hiện chuẩn hóa dữ liệu (scaling) trên toàn bộ dữ liệu trước khi chia.

> ⚠️ **Lưu ý:** Rò rỉ thông tin dẫn đến đánh giá lạc quan quá mức — mô hình trông có vẻ tốt trong CV nhưng thất bại khi triển khai thực tế.

### 4.4. Train – Validate – Test

Nếu dùng CV để chọn siêu tham số, thông tin từ validation đã "rò" vào quá trình training. Giải pháp: giữ lại thêm một **tập test hoàn toàn độc lập** để đánh giá cuối cùng.

```
Dữ liệu gốc
    ├── Test set (giữ lại hoàn toàn, chỉ dùng ở cuối)
    └── Phần còn lại
            ├── Training fold 1..K
            └── Validation fold 1..K  (dùng cho CV & tuning)
```

---

## 5. Thực hành trong Python

### Cài đặt và nạp thư viện

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer, load_wine, load_diabetes
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    KFold, StratifiedKFold,
    RepeatedStratifiedKFold,
    LeaveOneOut,
    cross_val_score, cross_validate,
    GridSearchCV
)
from sklearn.metrics import roc_auc_score
import seaborn as sns
```

---

### 5.1. Nạp dữ liệu

Trong phần này chúng ta dùng **Breast Cancer Wisconsin** — dữ liệu chẩn đoán ung thư vú.  
Biến mục tiêu: `target` — khối u lành tính (1) hay ác tính (0).

```python
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("Kích thước dữ liệu:", X.shape)
print("\nPhân phối biến mục tiêu:")
print(y.value_counts())
print(y.value_counts(normalize=True).round(3))
```

> 📌 Lưu ý: Tỷ lệ lớp `1` (lành tính, ~63%) cho thấy dữ liệu mất cân bằng nhẹ.

---

### 5.2. Holdout — Train/Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Kích thước tập train:", X_train.shape[0])
print("Kích thước tập test: ", X_test.shape[0])

# Kiểm tra tỷ lệ lớp trong từng tập
print("\nTỷ lệ lớp train:", y_train.value_counts(normalize=True).round(3).to_dict())
print("Tỷ lệ lớp test: ", y_test.value_counts(normalize=True).round(3).to_dict())
```

**Huấn luyện Logistic Regression và đánh giá:**

```python
# Dùng Pipeline để tránh data leakage khi scale
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

pipe.fit(X_train, y_train)
pred_prob = pipe.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, pred_prob)
print(f"AUC (Holdout): {auc:.4f}")
```

**Quan sát sự biến động của AUC theo từng lần chia:**

```python
auc_scores = []

for i in range(20):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y  # không fix random_state
    )
    pipe_i = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    pipe_i.fit(X_tr, y_tr)
    prob_i = pipe_i.predict_proba(X_te)[:, 1]
    auc_scores.append(roc_auc_score(y_te, prob_i))

print(f"AUC min: {min(auc_scores):.4f}")
print(f"AUC max: {max(auc_scores):.4f}")
print(f"AUC sd:  {np.std(auc_scores):.4f}")

plt.hist(auc_scores, bins=8, color="steelblue", edgecolor="white")
plt.title("Phân phối AUC — Holdout (20 lần)")
plt.xlabel("AUC")
plt.tight_layout()
plt.show()
```

> 💡 **Nhận xét:** AUC biến động đáng kể giữa các lần chia — điều này chứng tỏ Holdout đơn lẻ không đủ tin cậy.

---

### 5.3. K-Fold Cross-Validation

#### Cách 1: Dùng `cross_val_score` (đơn giản, phổ biến)

```python
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

cv_10fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

scores = cross_val_score(pipe, X, y, cv=cv_10fold, scoring="roc_auc")

print(f"AUC từng fold: {scores.round(4)}")
print(f"AUC trung bình (10-Fold CV): {scores.mean():.4f}")
print(f"Độ lệch chuẩn:               {scores.std():.4f}")
```

#### Cách 2: Vòng lặp thủ công (hiểu rõ cơ chế)

```python
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=456)

auc_train_list = []
auc_val_list   = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    scaler = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)        # dùng thống kê từ train

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr_sc, y_tr)

    auc_tr  = roc_auc_score(y_tr,  clf.predict_proba(X_tr_sc)[:, 1])
    auc_val = roc_auc_score(y_val, clf.predict_proba(X_val_sc)[:, 1])

    auc_train_list.append(auc_tr)
    auc_val_list.append(auc_val)

    print(f"Fold {fold:2d} | Train AUC: {auc_tr:.4f} | Val AUC: {auc_val:.4f}")

print("\n--- Tổng kết ---")
print(f"Train AUC trung bình: {np.mean(auc_train_list):.4f}")
print(f"Val   AUC trung bình: {np.mean(auc_val_list):.4f}")
print(f"Chênh lệch (overfit?): {np.mean(auc_train_list) - np.mean(auc_val_list):.4f}")
```

> 💡 **Đọc kết quả:**  
> - Nếu Train AUC >> Val AUC → mô hình đang **overfitting**.  
> - Nếu cả hai đều thấp → mô hình đang **underfitting**.  
> - Hai giá trị gần nhau → mô hình **tổng quát hóa tốt**.

---

### 5.4. Repeated K-Fold Cross-Validation

```python
# 10-Fold CV lặp 5 lần (= 50 lần đánh giá tổng cộng)
repeated_cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=789)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

scores_rep = cross_val_score(pipe, X, y, cv=repeated_cv, scoring="roc_auc")

print(f"Số lần đánh giá: {len(scores_rep)}")
print(f"AUC trung bình (Repeated 10-Fold CV): {scores_rep.mean():.4f}")
print(f"Độ lệch chuẩn:                        {scores_rep.std():.4f}")
```

**So sánh sự ổn định giữa 5-Fold và 10-Fold (chạy nhiều lần):**

```python
results = []

for run in range(10):
    for k_val in [5, 10]:
        cv_k = StratifiedKFold(n_splits=k_val, shuffle=True)
        pipe_k = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ])
        scores_k = cross_val_score(pipe_k, X, y, cv=cv_k, scoring="roc_auc")
        results.append({"run": run + 1, "k": f"{k_val}-Fold", "auc": scores_k.mean()})

df_results = pd.DataFrame(results)
summary = df_results.groupby("k")["auc"].agg(["mean", "std"]).round(4)
print(summary)
```

> 💡 **Nhận xét:** 10-Fold thường cho variance thấp hơn 5-Fold vì mỗi validation fold lớn hơn, nhưng tốn thời gian gấp đôi.

---

### 5.5. Stratified K-Fold (Phân tầng)

Quan trọng khi dữ liệu mất cân bằng lớp. `StratifiedKFold` đảm bảo mỗi fold giữ nguyên tỷ lệ lớp — ta có thể kiểm tra:

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (_, val_idx) in enumerate(skf.split(X, y), start=1):
    y_fold = y.iloc[val_idx]
    pct_pos = (y_fold == 1).mean() * 100
    print(f"Fold {fold} — tỷ lệ lớp 1 (lành tính): {pct_pos:.1f}%")
```

**Ví dụ rõ hơn với dữ liệu Titanic (mất cân bằng hơn):**

```python
# Dùng dataset titanic từ seaborn
titanic = sns.load_dataset("titanic")

titanic_clean = (
    titanic[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare"]]
    .dropna()
    .copy()
)
titanic_clean["sex"] = (titanic_clean["sex"] == "male").astype(int)

X_t = titanic_clean.drop("survived", axis=1)
y_t = titanic_clean["survived"]

print("Tỷ lệ sống sót:")
print(y_t.value_counts(normalize=True).round(3))

# 5-Fold CV với stratification
pipe_t = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_t = cross_val_score(pipe_t, X_t, y_t, cv=cv_strat, scoring="roc_auc")

print(f"\nAUC Titanic (Stratified 5-Fold CV): {scores_t.mean():.4f} ± {scores_t.std():.4f}")
```

---

### 5.6. Leave-One-Out Cross-Validation (LOO-CV)

```python
# LOO-CV chỉ dùng cho tập dữ liệu nhỏ!
# Dùng tập wine (178 mẫu) để minh họa
wine = load_wine()
X_w = pd.DataFrame(wine.data, columns=wine.feature_names)
y_w = pd.Series((wine.target == 0).astype(int))  # class 0 vs rest

pipe_w = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

loo = LeaveOneOut()
scores_loo = cross_val_score(pipe_w, X_w, y_w, cv=loo, scoring="roc_auc")

print(f"Số lần đánh giá: {len(scores_loo)}")
print(f"AUC (LOO-CV, {len(X_w)} obs): {scores_loo.mean():.4f}")
```

> ⚠️ **Lưu ý:** LOO-CV chạy **N mô hình** — với tập lớn sẽ rất chậm. Chỉ dùng khi tập dữ liệu nhỏ (< vài trăm quan sát).

---

### 5.7. Bootstrap Cross-Validation

```python
# Bootstrap thủ công (sklearn không có sẵn bootstrap CV)
from sklearn.utils import resample

n_bootstrap = 50
auc_boot = []

X_arr = X.values
y_arr = y.values

np.random.seed(42)

for _ in range(n_bootstrap):
    # Lấy mẫu có hoàn lại
    X_boot, y_boot = resample(X_arr, y_arr)

    # Out-of-bag: các mẫu không được chọn
    all_idx  = set(range(len(X_arr)))
    boot_idx = set(np.where(np.isin(X_arr, X_boot).all(axis=1))[0])
    oob_idx  = list(all_idx - boot_idx)

    if len(oob_idx) == 0:
        continue

    X_oob = X_arr[oob_idx]
    y_oob = y_arr[oob_idx]

    scaler = StandardScaler()
    X_boot_sc = scaler.fit_transform(X_boot)
    X_oob_sc  = scaler.transform(X_oob)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_boot_sc, y_boot)

    prob_oob = clf.predict_proba(X_oob_sc)[:, 1]
    if len(np.unique(y_oob)) > 1:          # cần cả 2 lớp để tính AUC
        auc_boot.append(roc_auc_score(y_oob, prob_oob))

print(f"AUC trung bình (Bootstrap, {n_bootstrap} reps): {np.mean(auc_boot):.4f}")
print(f"Độ lệch chuẩn:                                  {np.std(auc_boot):.4f}")
```

---

### 5.8. Ứng dụng: CV để chọn siêu tham số

Ví dụ: Chọn số láng giềng `k` tối ưu cho KNN dùng tập **Wine**.

```python
wine = load_wine()
X_w = pd.DataFrame(wine.data, columns=wine.feature_names)
y_w = pd.Series(wine.target)

pipe_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier())
])

param_grid = {"clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21]}

cv_tune = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    pipe_knn,
    param_grid,
    cv=cv_tune,
    scoring="roc_ovr_weighted",   # multi-class AUC
    refit=True,
    n_jobs=-1
)

grid_search.fit(X_w, y_w)

# Kết quả theo từng giá trị k
cv_results = pd.DataFrame(grid_search.cv_results_)[
    ["param_clf__n_neighbors", "mean_test_score", "std_test_score"]
].rename(columns={
    "param_clf__n_neighbors": "k",
    "mean_test_score": "AUC_mean",
    "std_test_score": "AUC_std"
})
print(cv_results.to_string(index=False))

print(f"\nk tối ưu: {grid_search.best_params_['clf__n_neighbors']}")
print(f"AUC tối đa: {grid_search.best_score_:.4f}")

# Vẽ đồ thị
plt.figure(figsize=(7, 4))
plt.errorbar(
    cv_results["k"], cv_results["AUC_mean"],
    yerr=cv_results["AUC_std"], marker="o", capsize=4, color="steelblue"
)
plt.xlabel("k (số láng giềng)")
plt.ylabel("AUC (5-Fold CV)")
plt.title("Chọn k tối ưu cho KNN qua 5-Fold CV")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

---

### 5.9. Train – Validate – Test trong Python

```python
# Dùng lại dữ liệu Titanic
# Bước 1: Tách ra 20% làm test set (giữ lại hoàn toàn)
X_dev, X_test_final, y_dev, y_test_final = train_test_split(
    X_t, y_t, test_size=0.2, stratify=y_t, random_state=42
)

print(f"Train+Val set: {len(X_dev)} | Test set: {len(X_test_final)}")

# Bước 2: Dùng CV trên phần còn lại để đánh giá / chọn mô hình
pipe_final = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

cv_tvt = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_dev = cross_val_score(pipe_final, X_dev, y_dev, cv=cv_tvt, scoring="roc_auc")

print(f"AUC trong CV (validation): {scores_dev.mean():.4f} ± {scores_dev.std():.4f}")

# Bước 3: Huấn luyện lại trên toàn bộ dev set, đánh giá trên test set (chỉ 1 lần!)
pipe_final.fit(X_dev, y_dev)
prob_test = pipe_final.predict_proba(X_test_final)[:, 1]
auc_test  = roc_auc_score(y_test_final, prob_test)

print(f"AUC trên TEST SET (đánh giá cuối): {auc_test:.4f}")
```

> ⚠️ **Quy tắc vàng:** Test set chỉ được dùng **một lần duy nhất** — sau khi mọi quyết định về mô hình đã được hoàn tất. Nếu dùng nhiều lần để điều chỉnh mô hình, test set sẽ bị "ô nhiễm".

---

## 6. Tổng kết — Nên chọn phương pháp nào?

| Tình huống | Khuyến nghị |
|---|---|
| Tập dữ liệu nhỏ (< 500 obs) | LOO-CV hoặc Repeated K-Fold |
| Tập dữ liệu trung bình | 10-Fold CV |
| Tập dữ liệu lớn (> 10,000) | 5-Fold CV (hoặc Holdout) |
| Dữ liệu mất cân bằng | Stratified K-Fold |
| Cần kết quả ổn định nhất | Repeated K-Fold |
| Dữ liệu chuỗi thời gian | `TimeSeriesSplit` (không chia ngẫu nhiên) |
| Cần đánh giá cuối cùng tin cậy | Train – Validate – Test |

---

## 7. Bài tập

### Bài tập 1 — Breast Cancer (`load_breast_cancer`)

Dữ liệu chẩn đoán khối u vú (569 quan sát, 30 đặc trưng). Biến mục tiêu: `target`.

1. Chạy **5-Fold CV** với Logistic Regression. Ghi nhận AUC trung bình.
2. Chạy **Repeated 5-Fold CV** (5 lần lặp). So sánh độ ổn định với câu 1.
3. Thêm **stratification** — kết quả có thay đổi đáng kể không? Tại sao?
4. Kiểm tra: mô hình có overfitting không? (So sánh Train AUC vs Val AUC bằng vòng lặp thủ công)

### Bài tập 2 — Wine (`load_wine`)

Dữ liệu phân loại rượu vang theo 3 giống nho. Dự đoán nhị phân: class 0 vs. phần còn lại.

```
Đặc trưng: alcohol, malic_acid, ash, alcalinity_of_ash, magnesium,
           total_phenols, flavanoids, ... (13 đặc trưng hóa học)
→ Mục tiêu: class (0 / 1 / 2)
```

1. Huấn luyện **Logistic Regression** với 5-Fold CV. Tính AUC (one-vs-rest).
2. Huấn luyện **KNN** với 5-Fold CV — thử các giá trị k từ 3 đến 21. Giá trị k nào tốt nhất?
3. So sánh tác dụng của stratification giữa bài Titanic và bài Wine — tập nào được hưởng lợi nhiều hơn? Giải thích.
4. *(Nâng cao)* Thực hiện quy trình **Train–Validate–Test** hoàn chỉnh. Báo cáo AUC cuối cùng trên test set.

### Bài tập 3 — Diabetes (`load_diabetes` → phân loại)

Dữ liệu bệnh tiểu đường (442 quan sát). Chuyển thành bài toán phân loại: nhãn = 1 nếu chỉ số `target` > median, ngược lại = 0.

```python
from sklearn.datasets import load_diabetes
data = load_diabetes()
X_d = pd.DataFrame(data.data, columns=data.feature_names)
y_d = (pd.Series(data.target) > pd.Series(data.target).median()).astype(int)
```

1. Thực hiện **5-Fold CV** với Logistic Regression.
2. Thực hiện **LOO-CV** (tập chỉ có 442 obs — chấp nhận được).
3. So sánh kết quả AUC giữa K-Fold và LOO-CV. Giải thích sự khác biệt.

---

## 8. Tài liệu tham khảo

- **scikit-learn — Model selection:** <https://scikit-learn.org/stable/model_selection.html>
- **scikit-learn — cross_validate:** <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html>
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.

---

*Bài giảng được soạn cho môn Data Science với Python — Bài 17: Cross-Validation*
