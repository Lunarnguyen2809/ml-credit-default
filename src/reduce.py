"""Giảm chiều và trực quan hóa (mục 4.2). Mọi phương pháp fit trên train, validation chỉ transform.

Gồm PCA (#10) và FAMD (#11). Các hàm vẽ/đo dùng chung nhận ma trận tọa độ `Z` (n mẫu x k thành phần)
nên áp dụng được cho cả hai. Hàm vẽ lưu PNG (mặc định vào `results/figures/`), đóng figure sau khi lưu
và trả về đường dẫn; không đặt backend matplotlib. Hàm bảng trả về DataFrame.

Quy ước: `Z` và `y` cùng thứ tự dòng (ví dụ cùng là train, hoặc cùng là validation).
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prince
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from src.data import DERIVED_COLS, MONEY_COLS, PAY_AMT, PAY_STATUS, PROFILE_CAT
from src.eda import FIG_DIR, COLORS, save_fig
from src.preprocess import build_preprocessor, signed_log

# Cột đưa vào FAMD: category trước, rồi 24 cột số (20 gốc + 4 dẫn xuất).
FAMD_NUM = ["LIMIT_BAL", "AGE"] + PAY_STATUS + [f"BILL_AMT{i}" for i in range(1, 7)] + PAY_AMT + DERIVED_COLS
FAMD_COLS = PROFILE_CAT + FAMD_NUM

# Màu theo khối đặc trưng khi vẽ loadings
_BLOCK_COLORS = {"PAY": "#e45756", "BILL_AMT": "#4c78a8", "PAY_AMT": "#54a24b",
                 "Hồ sơ": "#b279a2", "Dẫn xuất": "#f58518"}


# ---------------------------------------------------------------- tiện ích dùng chung

def feature_block(name: str) -> str:
    """Khối của một tên đặc trưng: PAY (PAY_1..6), BILL_AMT, PAY_AMT, Dẫn xuất, còn lại là Hồ sơ.

    Hồ sơ gồm LIMIT_BAL, AGE và các cột/mức của SEX, EDUCATION, MARRIAGE (kể cả dạng one-hot `SEX_1`).
    """
    if re.fullmatch(r"PAY_\d", name):
        return "PAY"
    if name.startswith("BILL_AMT"):
        return "BILL_AMT"
    if name.startswith("PAY_AMT"):
        return "PAY_AMT"
    if name in DERIVED_COLS:
        return "Dẫn xuất"
    return "Hồ sơ"


def explained_ratio(model) -> np.ndarray:
    """Tỉ lệ phương sai (PCA) hoặc quán tính (FAMD) giải thích của từng thành phần, tổng <= 1."""
    if isinstance(model, PCA):
        return np.asarray(model.explained_variance_ratio_)
    return np.asarray(model.percentage_of_variance_) / 100.0  # prince trả về phần trăm


def n_components_for(ratio, threshold: float) -> int:
    """Số thành phần nhỏ nhất để tỉ lệ giải thích tích lũy đạt `threshold` (vd 0.8)."""
    cum = np.cumsum(ratio)
    if cum[-1] < threshold - 1e-12:
        raise ValueError(f"mô hình chỉ giữ {len(ratio)} thành phần, tích lũy {cum[-1]:.3f} < {threshold}")
    return int(np.searchsorted(cum, threshold - 1e-12) + 1)


def explained_summary(ratio, k: int = 11, thresholds=(0.8, 0.9)) -> dict:
    """Con số cho báo cáo/#12: số thành phần đạt từng ngưỡng và tích lũy tại `k` thành phần."""
    out = {f"n_for_{int(t * 100)}%": n_components_for(ratio, t) for t in thresholds}
    out[f"cum_at_{k}"] = float(np.sum(ratio[:k]))
    return out


def _sample_idx(n: int, size: int, seed: int) -> np.ndarray:
    """Chọn ngẫu nhiên <= size dòng (seed cố định), giữ nguyên thứ tự dòng."""
    if n <= size:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=size, replace=False))


# ---------------------------------------------------------------- PCA (#10)

def pca_input(df_train: pd.DataFrame, df_val: pd.DataFrame | None = None, use_derived: bool = True):
    """Đầu vào của PCA: `build_preprocessor` (one-hot + signed_log + StandardScaler) fit trên train.

    `df_*` là dữ liệu sau `add_derived_features` (có thể còn cột nhãn, bị bỏ qua). Trả về
    (pre, X_train, X_val) với X là DataFrame 33 cột (29 nếu `use_derived=False`) có tên đặc trưng
    từ `get_feature_names_out()`; X_val là None nếu không truyền `df_val`. Val chỉ transform.
    """
    pre = build_preprocessor(use_derived=use_derived)
    X_tr = pre.fit_transform(df_train)
    names = list(pre.named_steps["columns"].get_feature_names_out())
    X_tr = pd.DataFrame(X_tr, columns=names, index=df_train.index)
    X_va = None
    if df_val is not None:
        X_va = pd.DataFrame(pre.transform(df_val), columns=names, index=df_val.index)
    return pre, X_tr, X_va


def fit_pca(X_train, n_components=None, seed: int = 0) -> PCA:
    """Fit sklearn PCA trên `X_train` (đã chuẩn hóa, xem `pca_input`).

    `n_components=None` giữ đủ mọi thành phần (để vẽ scree toàn bộ). `random_state=seed` cố định
    (chỉ có tác dụng khi sklearn chọn solver ngẫu nhiên). Dùng `explained_variance_ratio_` để vẽ scree.
    """
    return PCA(n_components=n_components, random_state=seed).fit(X_train)


# ---------------------------------------------------------------- FAMD (#11)

def famd_input(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn bị dữ liệu hỗn hợp cho FAMD, KHÔNG one-hot, cột đúng thứ tự `FAMD_COLS`.

    - SEX, EDUCATION, MARRIAGE -> chuỗi (prince coi cột chuỗi là category; cột int sẽ bị coi nhầm là category).
    - 24 cột số (LIMIT_BAL, AGE, PAY_1..6, BILL_AMT1..6, PAY_AMT1..6 và 4 cột dẫn xuất) -> float.
    - signed_log cho `MONEY_COLS` như đầu vào PCA, để hai phương pháp so sánh công bằng.
    Cột nhãn (nếu có) bị bỏ. Không sửa df gốc.
    """
    out = df[FAMD_COLS].copy()
    out[MONEY_COLS] = signed_log(out[MONEY_COLS])
    out[FAMD_NUM] = out[FAMD_NUM].astype(float)
    for c in PROFILE_CAT:
        out[c] = out[c].astype(str)
    return out


def fit_famd(df_train: pd.DataFrame, n_components: int = 30, seed: int = 0) -> prince.FAMD:
    """Fit prince.FAMD trên dữ liệu hỗn hợp CHƯA one-hot (category SEX/EDUCATION/MARRIAGE + 24 cột số).

    `df_train` là dữ liệu sau `add_derived_features` (cột lạ như nhãn bị bỏ, xem `famd_input`).
    Lựa chọn để so sánh công bằng với PCA: cột tiền qua `signed_log` như đầu vào PCA, cột số được prince
    tự chuẩn hóa (trung bình 0, độ lệch chuẩn 1), category mã hóa kiểu MCA (one-hot nội bộ, trọng số theo
    tần suất) nên mức hiếm không bị thổi phồng. Mặc định `rescale_with_std=False` của prince (FAMD chuẩn):
    tổng quán tính = 24 + (9 - 3) = 30, nên tối đa 30 thành phần và tổng tỉ lệ giải thích = 1.
    `random_state=seed` cố định; fit chạy dưới 1 giây trên ~21 nghìn dòng.

    Ngoài ra lưu `model.p_train_` (Series: tần suất từng mức category trên TRAIN, index = `one_hot_columns_`)
    để `famd_transform` mã hóa dữ liệu mới bằng thống kê train. Dùng `famd_transform` cho train lẫn validation.
    """
    X = famd_input(df_train)
    model = prince.FAMD(n_components=n_components, random_state=seed).fit(X)
    oh = pd.get_dummies(X[PROFILE_CAT], dtype=float).reindex(columns=model.one_hot_columns_, fill_value=0.0)
    model.p_train_ = oh.mean(axis="rows")
    return model


def famd_transform(model: prince.FAMD, df: pd.DataFrame) -> pd.DataFrame:
    """Tọa độ dòng của `df` trên các thành phần FAMD đã fit, mã hóa HOÀN TOÀN bằng thống kê train.

    Không dùng `model.transform` của prince 0.23.1 vì nó tính lại tần suất category `p` từ chính dữ liệu
    đầu vào, nên mã hóa validation theo thống kê validation và hỏng với lô nhỏ/một dòng (p = 1 làm cột về 0).
    Ở đây, giống prince nhưng với thống kê train: cột số trừ trung bình/chia độ lệch chuẩn của train
    (`num_scaler_`), category one-hot rồi `(x - p_train) / sqrt(p_train)`, ghép [số, category], trừ trung
    bình cột của PCA nội bộ (`scaler_`, xấp xỉ 0 trên train), nhân `column_weight_`, chiếu lên `svd_.V`.
    Mức category chưa gặp ở train cho one-hot toàn 0 (chỉ còn phần `-p/sqrt(p)`); `df` có thể chỉ một dòng.
    Trên train kết quả trùng `model.transform`.
    """
    X = famd_input(df)
    num_cols = list(model.num_scaler_.feature_names_in_)
    X_num = pd.DataFrame(model.num_scaler_.transform(X[num_cols]), index=X.index, columns=num_cols)
    oh = pd.get_dummies(X[PROFILE_CAT], dtype=float).reindex(columns=model.one_hot_columns_, fill_value=0.0)
    p = model.p_train_
    X_cat = oh.sub(p, axis="columns").div(np.sqrt(p), axis="columns")
    Z = pd.concat([X_num, X_cat], axis=1).to_numpy(dtype=float)
    if hasattr(model, "scaler_"):
        Z = model.scaler_.transform(Z)
    return pd.DataFrame((Z * model.column_weight_) @ model.svd_.V.T, index=X.index).rename_axis(columns="component")


# ---------------------------------------------------------------- loadings

def get_loadings(model, feature_names=None) -> pd.DataFrame:
    """Bảng loadings (đặc trưng x thành phần), cùng nghĩa 'tương quan với thành phần' cho cả hai mô hình.

    - PCA: `components_.T * sqrt(explained_variance_)`; vì đầu vào đã chuẩn hóa nên đây là hệ số tương quan
      giữa đặc trưng và thành phần. Cần `feature_names` (33 tên).
    - FAMD: `column_coordinates_` của prince: với cột số là tương quan với thành phần; với category là tọa độ
      của từng mức (SEX_1, ...), thang đo khác nên chỉ so sánh tương đối.
    """
    if isinstance(model, PCA):
        load = model.components_.T * np.sqrt(model.explained_variance_)
        cols = [f"PC{i + 1}" for i in range(load.shape[1])]
        return pd.DataFrame(load, index=list(feature_names), columns=cols)
    load = model.column_coordinates_.copy()
    load.columns = [f"F{int(c) + 1}" for c in load.columns]
    load.index = load.index.astype(str)
    return load


def top_loadings(loadings: pd.DataFrame, n_components: int = 3, top: int = 6) -> pd.DataFrame:
    """Bảng `top` đặc trưng có |loading| lớn nhất ở mỗi thành phần (cột: thành phần, hạng, đặc trưng, khối, loading)."""
    rows = []
    for comp in loadings.columns[:n_components]:
        s = loadings[comp]
        for rank, name in enumerate(s.abs().sort_values(ascending=False).index[:top], 1):
            rows.append({"thành phần": comp, "hạng": rank, "đặc trưng": name,
                         "khối": feature_block(name), "loading": round(float(s[name]), 3)})
    return pd.DataFrame(rows)


def block_strength(loadings: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
    """Tỉ trọng bình phương loading theo khối (PAY, BILL_AMT, PAY_AMT, Hồ sơ, Dẫn xuất) ở từng thành phần.

    Mỗi cột cộng lại bằng 1, cho biết khối nào 'chi phối' thành phần đó.
    """
    sq = loadings.iloc[:, :n_components] ** 2
    blocks = pd.Series([feature_block(n) for n in sq.index], index=sq.index)
    tab = sq.groupby(blocks).sum()
    return (tab / tab.sum()).reindex([b for b in _BLOCK_COLORS if b in tab.index])


def plot_loadings(model, feature_names, n_components, path) -> Path:
    """Loadings của `n_components` thành phần đầu (2–3): cột theo đặc trưng, tô màu theo khối.

    `model` là PCA hoặc prince.FAMD; `feature_names` chỉ cần cho PCA (FAMD tự có tên cột).
    Các cột sắp theo khối để thấy khối PAY / BILL_AMT / PAY_AMT / hồ sơ gắn với thành phần nào.
    """
    load = get_loadings(model, feature_names)
    order = sorted(load.index, key=lambda n: (list(_BLOCK_COLORS).index(feature_block(n)), n))
    load = load.loc[order]
    fig, axes = plt.subplots(n_components, 1, figsize=(12, 3 * n_components), sharex=True)
    axes = np.atleast_1d(axes)
    colors = [_BLOCK_COLORS[feature_block(n)] for n in load.index]
    for ax, comp in zip(axes, load.columns[:n_components]):
        ax.bar(range(len(load)), load[comp], color=colors)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylabel(f"Loading {comp}")
    axes[-1].set_xticks(range(len(load)))
    axes[-1].set_xticklabels(load.index, rotation=75, ha="right", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in _BLOCK_COLORS.values()]
    lo, hi = axes[0].get_ylim()
    axes[0].set_ylim(lo, hi + 0.3 * (hi - lo))  # chừa chỗ cho chú giải, tránh đè cột
    axes[0].legend(handles, list(_BLOCK_COLORS), ncol=5, fontsize=8, loc="upper right")
    return save_fig(fig, path)


# ---------------------------------------------------------------- scree, scatter, quan hệ với nhãn

def plot_scree(ratio, path, title: str = "Scree plot", k: int = 11) -> Path:
    """Cột tỉ lệ giải thích từng thành phần + đường tích lũy; vạch ngang 80%, 90% và vạch đứng tại `k`."""
    ratio = np.asarray(ratio)
    x = np.arange(1, len(ratio) + 1)
    cum = np.cumsum(ratio)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x, ratio, color=COLORS[0], alpha=0.85, label="Từng thành phần")
    ax.plot(x, cum, marker="o", ms=3, color=COLORS[1], label="Tích lũy")
    for thr in (0.8, 0.9):
        ax.axhline(thr, color="gray", ls=":", lw=1)
        ax.text(x[-1], thr, f" {thr:.0%}", va="bottom", ha="right", fontsize=8)
    if k <= len(ratio):
        ax.axvline(k, color="black", ls="--", lw=1)
        ax.text(k + 0.3, 0.5, f"k={k}: {cum[k - 1]:.1%}", fontsize=9, bbox=dict(facecolor="white", alpha=0.8))
    ax.set_xlabel("Thành phần")
    ax.set_ylabel("Tỉ lệ giải thích")
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    ax.legend(loc="center right")
    return save_fig(fig, path)


def plot_pairwise(Z, y, n_components, path, max_points: int = 3000, seed: int = 0) -> Path:
    """Ma trận scatter từng cặp cho `n_components` thành phần đầu (4–6), tô màu theo nhãn.

    Để dễ đọc chỉ vẽ ngẫu nhiên `max_points` dòng (seed cố định, alpha nhỏ); đường chéo là histogram
    mật độ từng lớp vẽ trên TOÀN BỘ dòng. `Z` và `y` cùng thứ tự dòng.
    """
    Z = np.asarray(Z)[:, :n_components]
    y = np.asarray(y)
    idx = _sample_idx(len(Z), max_points, seed)
    fig, axes = plt.subplots(n_components, n_components, figsize=(2.6 * n_components, 2.6 * n_components))
    for i in range(n_components):
        for j in range(n_components):
            ax = axes[i, j]
            if i == j:
                lim = np.percentile(Z[:, i], [0.5, 99.5])
                for lab in (0, 1):
                    ax.hist(Z[y == lab, i], bins=40, range=tuple(lim), density=True,
                            histtype="stepfilled", alpha=0.45, color=COLORS[lab])
                ax.set_yticks([])
            else:
                for lab in (0, 1):
                    m = idx[y[idx] == lab]
                    ax.scatter(Z[m, j], Z[m, i], s=4, alpha=0.25, color=COLORS[lab], linewidths=0, rasterized=True)
                ax.set_xlim(*np.percentile(Z[:, j], [0.5, 99.5]))
                ax.set_ylim(*np.percentile(Z[:, i], [0.5, 99.5]))
            ax.tick_params(labelsize=6)
            if i == n_components - 1:
                ax.set_xlabel(f"Thành phần {j + 1}", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"Thành phần {i + 1}", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[lab], label=name)
               for lab, name in ((0, "Không vỡ nợ"), (1, "Vỡ nợ"))]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle(f"Scatter từng cặp ({len(idx)} / {len(Z)} điểm, màu = nhãn)")
    return save_fig(fig, path, dpi=90)  # 90 dpi để file nhỏ (~0,5 MB), vẫn đọc được


def component_vs_label(Z, y, n_components: int = 5) -> pd.DataFrame:
    """Số đo tách lớp của từng thành phần: trung bình theo lớp, Cohen's d và ROC-AUC đơn biến.

    `auc` là AUC của chính thành phần làm điểm số cho nhãn 1 (0.5 = không tách; dấu của thành phần là
    tùy ý nên <0.5 chỉ là bị lật dấu), `auc_abs` = max(auc, 1 - auc) là mức tách bất kể dấu.
    """
    Z = np.asarray(Z)[:, :n_components]
    y = np.asarray(y)
    rows = []
    for k in range(Z.shape[1]):
        z0, z1 = Z[y == 0, k], Z[y == 1, k]
        pooled = np.sqrt((z0.var(ddof=1) * (len(z0) - 1) + z1.var(ddof=1) * (len(z1) - 1)) / (len(z0) + len(z1) - 2))
        auc = roc_auc_score(y, Z[:, k])
        rows.append({"thành phần": k + 1, "mean_lớp0": z0.mean(), "mean_lớp1": z1.mean(),
                     "cohen_d": (z1.mean() - z0.mean()) / pooled, "auc": auc, "auc_abs": max(auc, 1 - auc)})
    return pd.DataFrame(rows).set_index("thành phần")


def default_rate_by_quantile(z, y, q: int = 10) -> pd.DataFrame:
    """Tỉ lệ vỡ nợ theo q khoảng phân vị của một thành phần (bin đều số mẫu; trùng mốc thì gộp)."""
    z = pd.Series(np.asarray(z))
    key = pd.qcut(z, q=q, duplicates="drop")
    g = pd.Series(np.asarray(y)).groupby(key, observed=True)
    out = pd.DataFrame({"count": g.size(), "default_rate": g.mean()})
    out.index = range(1, len(out) + 1)
    out.index.name = "khoảng"
    return out


def plot_component_boxplots(Z, y, n_components, path) -> Path:
    """Boxplot theo nhãn cho `n_components` thành phần đầu (một ô mỗi thành phần, không vẽ điểm ngoại lai)."""
    Z = np.asarray(Z)[:, :n_components]
    y = np.asarray(y)
    fig, axes = plt.subplots(1, n_components, figsize=(2.6 * n_components, 3.8))
    for k, ax in enumerate(np.atleast_1d(axes)):
        bp = ax.boxplot([Z[y == 0, k], Z[y == 1, k]], tick_labels=["0", "1"], patch_artist=True,
                        showfliers=False, widths=0.6)
        for patch, lab in zip(bp["boxes"], (0, 1)):
            patch.set_facecolor(COLORS[lab])
            patch.set_alpha(0.75)
        ax.set_title(f"Thành phần {k + 1}", fontsize=10)
        ax.set_xlabel("Nhãn", fontsize=8)
    fig.suptitle("Phân phối thành phần theo nhãn (0 = không vỡ nợ, 1 = vỡ nợ)")
    return save_fig(fig, path)


def plot_default_rate_by_component(Z, y, n_components, path, q: int = 10) -> Path:
    """Tỉ lệ vỡ nợ theo q khoảng phân vị của từng thành phần đầu; vạch đứt = tỉ lệ vỡ nợ chung."""
    Z = np.asarray(Z)[:, :n_components]
    y = np.asarray(y)
    fig, axes = plt.subplots(1, n_components, figsize=(2.8 * n_components, 3.6), sharey=True)
    for k, ax in enumerate(np.atleast_1d(axes)):
        tab = default_rate_by_quantile(Z[:, k], y, q=q)
        ax.plot(tab.index, tab["default_rate"], marker="o", color=COLORS[1])
        ax.axhline(y.mean(), color="gray", ls="--", lw=1)
        ax.set_title(f"Thành phần {k + 1}", fontsize=10)
        ax.set_xlabel(f"Phân vị (1 = thấp nhất, {len(tab)} = cao nhất)", fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("Tỉ lệ vỡ nợ")
    fig.suptitle(f"Tỉ lệ vỡ nợ theo {q} khoảng phân vị của thành phần")
    return save_fig(fig, path)


def plot_tsne(X, y, path, seed=0):
    """TODO(#12): t-SNE 2 chiều chỉ để trực quan hóa, không dùng làm đầu vào mô hình."""
    raise NotImplementedError
