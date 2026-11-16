"""Phân cụm (mục 4.3, issue #21), chạy trên 11 thành phần PCA của tập train, bỏ qua nhãn.

Nhãn chỉ dùng SAU khi phân cụm để đánh giá (`cluster_label_report`, tô hình dạng điểm). Mọi mô hình
(preprocessor, PCA, GMM, K-Means) fit trên train; validation chỉ transform / predict / predict_proba.
Hàm vẽ lưu PNG (mặc định vào `results/figures/`), đóng figure sau khi lưu, trả về đường dẫn; không đặt
backend matplotlib. Hàm bảng trả về DataFrame.

GMM đã fit được lưu bằng joblib vào `data/models/` (gitignore) cùng pipeline preprocessor + PCA, để tái lập
kết quả mục 4.3. Mô-đun 2 tầng (mục 5) KHÔNG dùng mô hình này: nó fit GMM riêng trên khối A (LIMIT_BAL, SEX,
EDUCATION, MARRIAGE, AGE), dùng lại `select_k_gmm` / `fit_gmm` / `save_models`.

Lưu ý về GMM trên PCA: các thành phần PCA có hướng gần như rời rạc (one-hot SEX/EDUCATION/MARRIAGE, PAY = 0), nên
GMM covariance đầy đủ dễ co một cụm về không gian con rời rạc (phương sai ~ 0 theo vài hướng, hội tụ rất nhạy với
khởi tạo). Vì vậy `reg_covar` mặc định 1e-2 (thay vì 1e-6 của sklearn), `n_init` giống nhau cho chọn k và fit cuối,
và k chọn theo độ ổn định giữa các seed (`k_stability`, `choose_k_stable`), kèm kiểm tra suy biến (`gmm_degeneracy`).
"""
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.mixture import GaussianMixture

from src.config import DATA_DIR
from src.data import BILL_AMT, PAY_AMT
from src.eda import COLORS, FIG_DIR, save_fig
from src.reduce import _sample_idx

MODEL_DIR = DATA_DIR / "models"
# reg_covar cộng vào đường chéo mọi covariance. 1e-6 (mặc định sklearn) quá nhỏ so với phương sai ~1 của thành phần
# PCA: một cụm có thể co về không gian con rời rạc (eigenvalue sát sàn). 1e-2 ~ 1% phương sai của một chiều chuẩn hóa.
REG_COVAR = 1e-2
N_INIT_GMM = 5      # dùng GIỐNG NHAU khi chọn k và fit cuối (cùng thiết lập thì BIC lúc chọn mới đúng cho mô hình cuối)
STABLE_ARI = 0.8    # ngưỡng ARI trung bình giữa các seed để coi k là ổn định
SIL_SAMPLE = 5000   # silhouette là O(n^2): tính trên mẫu con cố định (cùng seed) cho mọi k và hai thuật toán

# Màu cụm (tối đa 10 cụm), tách khỏi COLORS (màu của nhãn)
_CLUSTER_COLORS = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#9d755d", "#72b7b2", "#eeca3b", "#ff9da6",
                   "#bab0ac", "#59a14f"]
_LABEL_MARKERS = {0: "o", 1: "^"}
_LABEL_NAMES = {0: "Không vỡ nợ", 1: "Vỡ nợ"}


# ---------------------------------------------------------------- chọn k

def select_k_gmm(Z, k_range, seed: int = 0, covariance_type: str = "full", n_init: int = N_INIT_GMM,
                 reg_covar: float = REG_COVAR) -> pd.DataFrame:
    """BIC và AIC của GMM với k trong `k_range` (đóng hai đầu, vd (2, 10) hoặc danh sách).

    `covariance_type='full'`, mỗi k chạy `n_init` lần khởi tạo (lấy lần log-likelihood tốt nhất), seed cố định,
    `reg_covar` (xem `REG_COVAR`). Dùng cùng `n_init`/`reg_covar` với `fit_gmm`.
    Trả về DataFrame index = k, cột `bic`, `aic`, `converged`. BIC/AIC càng NHỎ càng tốt.
    """
    rows = []
    for k in _ks(k_range):
        gm = fit_gmm(Z, k, seed=seed, covariance_type=covariance_type, n_init=n_init, reg_covar=reg_covar)
        rows.append({"k": k, "bic": gm.bic(Z), "aic": gm.aic(Z), "converged": bool(gm.converged_)})
    return pd.DataFrame(rows).set_index("k")


def k_stability(Z, k_range, seeds, covariance_type: str = "full", n_init: int = N_INIT_GMM,
                reg_covar: float = REG_COVAR) -> dict:
    """Độ ổn định của GMM theo seed khởi tạo, cho từng k (cùng Z, chỉ đổi `random_state`).

    Trả về dict:
    - `summary`: DataFrame index = k, cột `bic_mean`, `bic_min`, `bic_max` (qua các seed), `ari_mean` (ARI trung bình
      giữa mọi cặp phân hoạch của hai seed khác nhau; 1 = mọi seed ra cùng cụm), `gain_%` (mức giảm `bic_mean` so với k-1).
    - `bic`, `aic`: DataFrame index = k, cột = seed.
    """
    Z = np.asarray(Z)
    seeds = list(seeds)
    ks = _ks(k_range)
    bic, aic, lab = {}, {}, {}
    for k in ks:
        for s in seeds:
            gm = fit_gmm(Z, k, seed=s, covariance_type=covariance_type, n_init=n_init, reg_covar=reg_covar)
            bic[(k, s)], aic[(k, s)], lab[(k, s)] = gm.bic(Z), gm.aic(Z), gm.predict(Z)
    bic_df = pd.DataFrame({s: [bic[(k, s)] for k in ks] for s in seeds}, index=pd.Index(ks, name="k"))
    aic_df = pd.DataFrame({s: [aic[(k, s)] for k in ks] for s in seeds}, index=pd.Index(ks, name="k"))
    ari = [np.mean([adjusted_rand_score(lab[(k, a)], lab[(k, b)]) for a, b in combinations(seeds, 2)])
           if len(seeds) > 1 else np.nan for k in ks]
    summary = pd.DataFrame({"bic_mean": bic_df.mean(axis=1), "bic_min": bic_df.min(axis=1),
                            "bic_max": bic_df.max(axis=1), "ari_mean": ari}, index=bic_df.index)
    summary["gain_%"] = -100 * summary["bic_mean"].diff() / summary["bic_mean"].shift()
    return {"summary": summary, "bic": bic_df, "aic": aic_df}


def choose_k_stable(summary: pd.DataFrame, min_ari: float = STABLE_ARI) -> int | None:
    """Chọn k của GMM từ `k_stability()['summary']`: trong các k ỔN ĐỊNH (`ari_mean` >= `min_ari`), lấy k có `bic_mean`
    nhỏ nhất, loại k nhỏ nhất của khoảng dò (k = 2 ổn định gần như tầm thường). Không k nào đạt -> trả về None.

    BIC min đơn thuần không đáng tin ở đây: BIC cực tiểu khác nhau rất nhiều giữa các seed (cực trị địa phương), nên một
    k chỉ được tin khi nhiều seed ra cùng một phân hoạch.
    """
    ok = summary[(summary["ari_mean"] >= min_ari) & (summary.index > summary.index.min())]
    return int(ok["bic_mean"].idxmin()) if len(ok) else None


def select_k_kmeans(Z, k_range, seed: int = 0, n_init: int = 10, sil_sample: int = SIL_SAMPLE) -> pd.DataFrame:
    """Inertia (elbow) và silhouette của K-Means với k trong `k_range`.

    Silhouette tính trên mẫu con cố định `sil_sample` dòng (cùng `seed`, giống nhau cho mọi k) vì độ phức tạp
    O(n^2); inertia tính trên toàn bộ. Trả về DataFrame index = k, cột `inertia`, `silhouette`.
    """
    Z = np.asarray(Z)
    idx = _sample_idx(len(Z), sil_sample, seed)
    rows = []
    for k in _ks(k_range):
        km = KMeans(k, n_init=n_init, random_state=seed).fit(Z)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Z[idx], km.labels_[idx])})
    return pd.DataFrame(rows).set_index("k")


def _ks(k_range) -> list:
    """(lo, hi) -> lo..hi gồm cả hai đầu (đúng dạng `clustering.k_range` trong config); list dài > 2 giữ nguyên."""
    k_range = list(k_range)
    return list(range(k_range[0], k_range[1] + 1)) if len(k_range) == 2 else k_range


def best_k_bic(table: pd.DataFrame) -> int:
    """k có BIC nhỏ nhất. Nếu là mốc cuối của khoảng dò, BIC chưa 'dừng giảm' và cần thêm lập luận (elbow ΔBIC)."""
    return int(table["bic"].idxmin())


def elbow_k(table: pd.DataFrame, col: str = "bic") -> int:
    """k tại 'khuỷu' của đường `col` (giảm dần): k có khoảng cách lớn nhất tới đường nối hai đầu mút.

    Hai trục chuẩn hóa về [0, 1] rồi tính khoảng cách vuông góc tới đoạn thẳng đầu-cuối (kiểu kneedle).
    Dùng khi BIC giảm đến cuối khoảng dò: chọn điểm sau đó mức giảm thêm không còn đáng kể.
    """
    k = table.index.to_numpy(dtype=float)
    v = table[col].to_numpy(dtype=float)
    x = (k - k[0]) / (k[-1] - k[0])
    y = (v - v.min()) / (v.max() - v.min())
    # đường nối (0, y[0]) -> (1, y[-1]); khoảng cách theo phương dọc đủ để so sánh vì hai trục cùng thang [0, 1]
    line = y[0] + (y[-1] - y[0]) * x
    return int(k[np.argmax(np.abs(line - y))])


def bic_gain_table(table: pd.DataFrame) -> pd.DataFrame:
    """ΔBIC từng bước (BIC(k) - BIC(k-1), âm = tốt hơn) và mức giảm tương đối so với BIC(k-1)."""
    d = table["bic"].diff()
    return pd.DataFrame({"bic": table["bic"], "delta_bic": d, "giảm_%": -100 * d / table["bic"].shift()})


# ---------------------------------------------------------------- fit và gán cụm

def fit_gmm(Z, k: int, seed: int = 0, covariance_type: str = "full", n_init: int = N_INIT_GMM,
            reg_covar: float = REG_COVAR) -> GaussianMixture:
    """Fit GMM với k đã chọn (`n_init` lần khởi tạo, seed cố định, `reg_covar`); dùng cùng giá trị với `select_k_gmm`."""
    return GaussianMixture(k, covariance_type=covariance_type, n_init=n_init, reg_covar=reg_covar,
                           random_state=seed).fit(Z)


def fit_kmeans(Z, k: int, seed: int = 0, n_init: int = 10) -> KMeans:
    """Fit K-Means cuối cùng với k đã chọn."""
    return KMeans(k, n_init=n_init, random_state=seed).fit(Z)


def relabel_by_size(clusters):
    """Đánh số lại cụm 0..k-1 theo kích thước giảm dần (cụm 0 lớn nhất) để so sánh các lần chạy dễ hơn.

    Trả về (clusters mới, bản đồ cũ -> mới). Nhãn cụm chỉ là quy ước, đổi số không đổi cấu trúc cụm.
    Dùng `apply_mapping` để áp bản đồ này cho tập khác (validation).
    """
    clusters = np.asarray(clusters)
    order = pd.Series(clusters).value_counts().index.to_numpy()
    mapping = {int(old): new for new, old in enumerate(order)}
    return np.vectorize(mapping.get)(clusters), mapping


def apply_mapping(clusters, mapping: dict) -> np.ndarray:
    """Áp bản đồ cũ -> mới của `relabel_by_size` cho cụm của tập khác. Cụm không có trong bản đồ (không có mẫu train) -> lỗi."""
    clusters = np.asarray(clusters)
    missing = set(np.unique(clusters).tolist()) - set(mapping)
    if missing:
        raise ValueError(f"cụm {sorted(missing)} không có mẫu train nên không có trong bản đồ đánh số")
    return np.array([mapping[int(c)] for c in clusters])


# ---------------------------------------------------------------- suy biến của GMM

def gmm_degeneracy(gmm: GaussianMixture, Z, labels_df: pd.DataFrame | None = None,
                   mapping: dict | None = None) -> pd.DataFrame:
    """Kiểm tra suy biến của GMM covariance đầy đủ, mỗi hàng một thành phần.

    Mặc định hàng được đánh số theo thành phần GỐC của `gmm` (chưa đánh số lại theo kích thước). Nếu truyền
    `mapping` (dict thành phần gốc -> nhãn mới) thì index đổi sang nhãn mới và các hàng sắp theo nhãn mới.

    Cột: `weight`, `min_eig`, `max_eig` (eigenvalue nhỏ/lớn nhất của covariance; `min_eig` sát `gmm.reg_covar` nghĩa là
    thành phần co về không gian con, gần suy biến), `logdet` (log định thức covariance, rất âm = hẹp), `size`.
    Nếu có `labels_df` (cùng thứ tự dòng với Z; cột SEX, EDUCATION, MARRIAGE và PAY_1..PAY_6 nếu có) thêm: `pure_SEX`,
    `pure_EDUCATION`, `pure_MARRIAGE` (tỉ lệ của giá trị phổ biến nhất trong cụm; 1 = cụm chỉ có một giá trị) và `pay_all0`
    (tỉ lệ dòng có mọi PAY_1..PAY_6 = 0). Cụm gán bằng `gmm.predict(Z)`.
    """
    lab = gmm.predict(np.asarray(Z))
    rows = []
    for c in range(gmm.n_components):
        ev = np.linalg.eigvalsh(gmm.covariances_[c])
        rows.append({"weight": gmm.weights_[c], "min_eig": ev.min(), "max_eig": ev.max(),
                     "logdet": float(np.log(ev).sum()), "size": int((lab == c).sum())})
    out = pd.DataFrame(rows)
    out.index.name = "thành phần"
    if labels_df is not None:
        d = labels_df.reset_index(drop=True)
        pay = [c for c in d.columns if c.startswith("PAY_") and c[4:].isdigit()]
        for col in ("SEX", "EDUCATION", "MARRIAGE"):
            if col in d:
                out[f"pure_{col}"] = [d.loc[lab == c, col].value_counts(normalize=True).max() if (lab == c).any()
                                      else np.nan for c in range(gmm.n_components)]
        if pay:
            all0 = (d[pay] == 0).all(axis=1).to_numpy()
            out["pay_all0"] = [all0[lab == c].mean() if (lab == c).any() else np.nan for c in range(gmm.n_components)]
    if mapping is not None:
        out = out.rename(index=mapping).sort_index()
    return out


# ---------------------------------------------------------------- chất lượng cụm (không dùng nhãn)

def quality_metrics(Z, clusters, seed: int = 0, sil_sample: int = SIL_SAMPLE) -> dict:
    """Silhouette (mẫu con cố định `sil_sample` dòng, cùng seed), Davies–Bouldin (nhỏ tốt), Calinski–Harabasz (lớn tốt)."""
    Z = np.asarray(Z)
    clusters = np.asarray(clusters)
    idx = _sample_idx(len(Z), sil_sample, seed)
    return {"silhouette": silhouette_score(Z[idx], clusters[idx]),
            "davies_bouldin": davies_bouldin_score(Z, clusters),
            "calinski_harabasz": calinski_harabasz_score(Z, clusters)}


# ---------------------------------------------------------------- quan hệ cụm - nhãn

def cluster_label_report(clusters, y) -> dict:
    """Quan hệ giữa cụm và nhãn (chỉ để đánh giá, phân cụm không dùng nhãn).

    Trả về dict:
    - `table`: DataFrame index = cụm; cột `size`, `share` (tỉ trọng cụm), `n_default`, `default_rate`.
    - `chi2`, `dof`, `p`: kiểm định chi-bình phương độc lập trên bảng chéo cụm x nhãn (`chi2_contingency`).
    - `cramers_v`: sqrt(chi2 / (n * min(r - 1, c - 1))) với r x c là kích thước bảng chéo; nhãn nhị phân (c = 2) nên = sqrt(chi2 / n); thang 0..1.
    - `ari`, `nmi`: Adjusted Rand Index và Normalized Mutual Information giữa cụm và nhãn.
    """
    clusters = np.asarray(clusters)
    y = np.asarray(y)
    g = pd.Series(y).groupby(clusters)
    table = pd.DataFrame({"size": g.size(), "share": g.size() / len(y), "n_default": g.sum(), "default_rate": g.mean()})
    table.index.name = "cụm"
    ct = pd.crosstab(clusters, y)
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum()
    v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
    return {"table": table, "chi2": float(chi2), "dof": int(dof), "p": float(p), "cramers_v": v,
            "ari": adjusted_rand_score(y, clusters), "nmi": normalized_mutual_info_score(y, clusters)}


def agreement_ari(a, b) -> float:
    """ARI giữa hai cách gán cụm (vd GMM vs K-Means; không phụ thuộc cách đánh số cụm)."""
    return adjusted_rand_score(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------- hồ sơ cụm

def cluster_profile(df: pd.DataFrame, clusters) -> pd.DataFrame:
    """Trung bình thuộc tính GỐC (đã làm sạch, trước mã hóa/scale/PCA) theo cụm; hàng cuối `Tất cả`.

    `df`: dữ liệu sau `clean` + `add_derived_features`, cùng thứ tự dòng với `clusters`. Cột:
    LIMIT_BAL, AGE, PAY_1, BILL_mean (trung bình BILL_AMT1..6), PAYAMT_mean (trung bình PAY_AMT1..6), N_LATE,
    UTIL_RATIO (trung bình), tỉ lệ nữ (`SEX=2`), tỉ lệ Sau ĐH (`EDUCATION=1`), Đại học (`=2`), Có gia đình
    (`MARRIAGE=1`) và `size`.
    """
    d = pd.DataFrame({
        "LIMIT_BAL": df["LIMIT_BAL"], "AGE": df["AGE"], "PAY_1": df["PAY_1"],
        "BILL_mean": df[BILL_AMT].mean(axis=1), "PAYAMT_mean": df[PAY_AMT].mean(axis=1),
        "N_LATE": df["N_LATE"], "UTIL_RATIO": df["UTIL_RATIO"],
        "nữ": (df["SEX"] == 2).astype(float), "sau_ĐH": (df["EDUCATION"] == 1).astype(float),
        "đại_học": (df["EDUCATION"] == 2).astype(float), "có_gia_đình": (df["MARRIAGE"] == 1).astype(float),
    })
    d = d.set_index(pd.Index(np.asarray(clusters), name="cụm"))
    out = d.groupby(level=0).mean()
    out["size"] = d.groupby(level=0).size()
    out.loc["Tất cả"] = list(d.mean()) + [len(d)]
    return out


# ---------------------------------------------------------------- hình

def plot_k_gmm(stab: dict, path, k_chosen: int | None = None, min_ari: float = STABLE_ARI) -> Path:
    """Trái: BIC trung bình theo k (đường) với dải min-max qua các seed; AIC trung bình (đường đứt). Phải: ARI trung
    bình giữa các seed theo k, vạch ngang = ngưỡng ổn định `min_ari`. `stab` = kết quả `k_stability`; vạch đứng = k chọn."""
    sm = stab["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.fill_between(sm.index, sm["bic_min"], sm["bic_max"], color=COLORS[0], alpha=0.2, label="BIC min–max qua seed")
    ax.plot(sm.index, sm["bic_mean"], marker="o", color=COLORS[0], label="BIC trung bình")
    ax.plot(sm.index, stab["aic"].mean(axis=1), marker="s", ls="--", color=COLORS[1], label="AIC trung bình")
    ax.set_ylabel("BIC / AIC (nhỏ hơn = tốt hơn)")
    ax.set_title("GMM – BIC, AIC theo k (trung bình, dải min–max)")
    ax.legend()
    axes[1].plot(sm.index, sm["ari_mean"], marker="o", color=_CLUSTER_COLORS[2])
    axes[1].axhline(min_ari, color="gray", ls=":", lw=1, label=f"ngưỡng ổn định = {min_ari}")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("ARI trung bình giữa các cặp seed")
    axes[1].set_title("GMM – độ ổn định theo seed")
    axes[1].legend()
    for a in axes:
        a.set_xlabel("Số cụm k")
        if k_chosen is not None:
            a.axvline(k_chosen, color="black", ls="--", lw=1)
    return save_fig(fig, path)


def plot_k_kmeans(table: pd.DataFrame, path, k_chosen: int | None = None) -> Path:
    """Elbow (inertia, trái) và silhouette (phải) theo k; vạch đứt = k đã chọn."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(table.index, table["inertia"], marker="o", color=COLORS[0])
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("K-Means – elbow")
    axes[1].plot(table.index, table["silhouette"], marker="o", color=COLORS[1])
    axes[1].set_ylabel(f"Silhouette (mẫu con {SIL_SAMPLE} dòng)")
    axes[1].set_title("K-Means – silhouette")
    for ax in axes:
        ax.set_xlabel("Số cụm k")
        if k_chosen is not None:
            ax.axvline(k_chosen, color="black", ls="--", lw=1)
    return save_fig(fig, path)


def plot_clusters(Z, clusters, y, path, max_points: int = 6000, seed: int = 0, title: str = "") -> Path:
    """Scatter PC1–PC2: MÀU = cụm, HÌNH DẠNG điểm = nhãn (tròn = không vỡ nợ, tam giác = vỡ nợ).

    Chỉ vẽ ngẫu nhiên `max_points` dòng (seed cố định, alpha nhỏ) cho dễ đọc; `Z`, `clusters`, `y` cùng thứ tự
    dòng. Có hai chú giải: màu cụm và hình dạng nhãn.
    """
    Z = np.asarray(Z)
    clusters = np.asarray(clusters)
    y = np.asarray(y)
    idx = _sample_idx(len(Z), max_points, seed)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ks = sorted(np.unique(clusters))
    for c in ks:
        for lab, mk in _LABEL_MARKERS.items():
            m = idx[(clusters[idx] == c) & (y[idx] == lab)]
            ax.scatter(Z[m, 0], Z[m, 1], s=14 if lab == 0 else 20, marker=mk, alpha=0.4,
                       color=_CLUSTER_COLORS[c % len(_CLUSTER_COLORS)], linewidths=0, rasterized=True)
    ax.set_xlim(*np.percentile(Z[:, 0], [0.3, 99.7]))
    ax.set_ylim(*np.percentile(Z[:, 1], [0.3, 99.7]))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title or f"PC1–PC2: màu = cụm, hình = nhãn ({len(idx)} / {len(Z)} điểm)")
    h_cluster = [plt.Line2D([], [], marker="o", ls="", color=_CLUSTER_COLORS[c % len(_CLUSTER_COLORS)],
                            label=f"Cụm {c}") for c in ks]
    h_label = [plt.Line2D([], [], marker=_LABEL_MARKERS[lab], ls="", color="gray", label=_LABEL_NAMES[lab])
               for lab in (0, 1)]
    leg1 = ax.legend(handles=h_cluster, title="Màu = cụm", loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=h_label, title="Hình = nhãn", loc="upper right", fontsize=8)
    return save_fig(fig, path)


def plot_cluster_default_rate(table: pd.DataFrame, overall: float, path, title: str = "") -> Path:
    """Cột tỉ lệ vỡ nợ từng cụm (từ `cluster_label_report()['table']`); vạch đứt = tỉ lệ chung."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(c) for c in table.index], table["default_rate"],
           color=[_CLUSTER_COLORS[int(c) % len(_CLUSTER_COLORS)] for c in table.index])
    ax.axhline(overall, color="gray", ls="--", lw=1, label=f"Chung = {overall:.3f}")
    for i, (r, s) in enumerate(zip(table["default_rate"], table["share"])):
        ax.text(i, r, f"{r:.3f}\n({s:.0%})", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, table["default_rate"].max() * 1.25)
    ax.set_xlabel("Cụm")
    ax.set_ylabel("Tỉ lệ vỡ nợ")
    ax.set_title(title or "Tỉ lệ vỡ nợ theo cụm (trong ngoặc: tỉ trọng cụm)")
    ax.legend()
    return save_fig(fig, path)


# ---------------------------------------------------------------- lưu / nạp (dùng lại cho mô-đun 2 tầng)

def save_models(path_prefix, pre, pca, gmm, kmeans=None) -> dict:
    """Lưu preprocessor, PCA, GMM (và K-Means nếu có) bằng joblib vào `data/models/<prefix>_*.joblib`.

    `path_prefix` là tên (vd 'cluster') hoặc đường dẫn không đuôi. Trả về dict tên -> Path đã lưu.
    """
    base = Path(path_prefix)
    if not base.is_absolute():
        base = MODEL_DIR / base
    base.parent.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, obj in (("pre", pre), ("pca", pca), ("gmm", gmm), ("kmeans", kmeans)):
        if obj is not None:
            p = base.parent / f"{base.name}_{name}.joblib"
            joblib.dump(obj, p)
            out[name] = p
    return out


def load_models(path_prefix) -> dict:
    """Nạp các mô hình đã `save_models`; trả về dict tên (`pre`, `pca`, `gmm`, `kmeans` nếu có) -> object."""
    base = Path(path_prefix)
    if not base.is_absolute():
        base = MODEL_DIR / base
    out = {}
    for name in ("pre", "pca", "gmm", "kmeans"):
        p = base.parent / f"{base.name}_{name}.joblib"
        if p.exists():
            out[name] = joblib.load(p)
    return out


def predict_proba_df(models: dict, df: pd.DataFrame, mapping: dict | None = None) -> pd.DataFrame:
    """P(cụm | x) của GMM đã lưu cho dữ liệu mới `df` (sau `add_derived_features`): preprocessor -> PCA -> `predict_proba`.

    Chỉ transform bằng mô hình fit trên train, không fit lại. Trả về DataFrame (index = index của `df`, cột = cụm).

    Mặc định cột là thứ tự thành phần GỐC của GMM (0..k-1), chưa đánh số lại theo kích thước. Truyền `mapping`
    (dict thành phần gốc -> nhãn mới, vd. nhãn sau khi đánh số lại) để đổi tên và sắp cột theo nhãn mới.
    """
    X = models["pre"].transform(df)
    names = getattr(models["pca"], "feature_names_in_", None)  # PCA fit trên DataFrame: giữ tên cột, tránh cảnh báo
    Z = models["pca"].transform(pd.DataFrame(X, columns=names, index=df.index) if names is not None else X)
    p = models["gmm"].predict_proba(Z)
    out = pd.DataFrame(p, index=df.index, columns=range(p.shape[1]))
    if mapping is not None:
        out = out.rename(columns=mapping).sort_index(axis=1)
    return out
