"""Thống kê và trực quan hóa dữ liệu (phần thống kê của mục 2.2, issue #8).

Mọi hàm chạy trên dữ liệu ĐÃ làm sạch (`clean(load_raw())`). Hàm bảng trả về DataFrame;
hàm vẽ lưu PNG vào `results/figures/` (mặc định), đóng figure sau khi lưu và trả về đường dẫn.
Không đặt backend matplotlib (hàm chỉ savefig, không show) nên import module không đổi cấu hình của notebook.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from src.config import ROOT
from src.data import BILL_AMT, MONEY_COLS, PAY_AMT, PAY_STATUS, PROFILE_NUM, TARGET
from src.preprocess import signed_log

FIG_DIR = ROOT / "results" / "figures"

# Cột số để thống kê / tương quan. Gồm PAY_1..6: đây là biến THỨ BẬC (-2..8) nhưng
# giữ dạng số trong mô hình (mục 4.1 bước 6) nên vẫn thống kê như cột số.
# Loại SEX, EDUCATION, MARRIAGE (mã category) và nhãn.
NUMERIC_COLS = PROFILE_NUM + PAY_STATUS + BILL_AMT + PAY_AMT

# BILL_AMT1/PAY_AMT1 là tháng gần nhất (9/2005), chỉ số 6 là tháng 4/2005
# -> thứ tự thời gian tăng dần là chỉ số 6, 5, ..., 1.
MONTH_LABELS = ["T4", "T5", "T6", "T7", "T8", "T9"]

AGE_BINS = [20, 30, 40, 50, 60, 80]
LIMIT_BINS = [0, 50_000, 100_000, 150_000, 200_000, 300_000, 500_000, 1_000_000]

SEX_NAMES = {1: "Nam", 2: "Nữ"}
EDU_NAMES = {1: "Sau ĐH", 2: "Đại học", 3: "THPT", 4: "Khác"}
MAR_NAMES = {1: "Có gia đình", 2: "Độc thân", 3: "Khác"}

_DPI = 120
_COLORS = {0: "#4c78a8", 1: "#e45756"}  # 0 = không vỡ nợ, 1 = vỡ nợ


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)
    return path


def describe_numeric(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """mean, median, std, min, max, skewness, kurtosis cho các cột số.

    Mặc định `cols = NUMERIC_COLS` (20 cột: LIMIT_BAL, AGE, PAY_1..6, BILL_AMT1..6,
    PAY_AMT1..6). PAY_x là biến thứ bậc nhưng vẫn được tính vì mô hình dùng chúng dạng
    số. Không gồm SEX/EDUCATION/MARRIAGE (mã category) và nhãn. Kurtosis là excess
    kurtosis của pandas (phân phối chuẩn = 0).
    """
    cols = list(NUMERIC_COLS if cols is None else cols)
    x = df[cols]
    return pd.DataFrame(
        {
            "mean": x.mean(),
            "median": x.median(),
            "std": x.std(),
            "min": x.min(),
            "max": x.max(),
            "skew": x.skew(),
            "kurtosis": x.kurt(),
        }
    )


def default_rate_by(df: pd.DataFrame, col: str, bins=None, target: str = TARGET) -> pd.DataFrame:
    """Bảng tần suất và tỉ lệ vỡ nợ theo từng mức (hoặc khoảng nếu có `bins`) của `col`.

    `bins`: danh sách mốc cho pd.cut (khoảng mở trái, đóng phải, `right=True`).
    Trả về index = mức/khoảng; cột `count`, `ratio` (tỉ trọng trong dữ liệu),
    `n_default`, `default_rate`.
    """
    key = pd.cut(df[col], bins=bins) if bins is not None else df[col]
    g = df.groupby(key, observed=True)[target]
    out = pd.DataFrame({"count": g.size(), "n_default": g.sum()})
    out["ratio"] = out["count"] / out["count"].sum()
    out["default_rate"] = out["n_default"] / out["count"]
    out.index.name = col
    return out[["count", "ratio", "n_default", "default_rate"]]


def plot_correlation_heatmap(df: pd.DataFrame, path=FIG_DIR / "correlation_heatmap.png") -> Path:
    """Heatmap tương quan Pearson giữa 20 cột số (xem `NUMERIC_COLS`).

    Khối BILL_AMT1..6 thường tương quan rất mạnh với nhau (nằm cạnh nhau trên hình).
    """
    corr = df[NUMERIC_COLS].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True,
                linewidths=0.3, cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Tương quan Pearson giữa các cột số")
    return _save(fig, path)


def _bar_rate(ax, tab: pd.DataFrame, labels, title, xlabel, overall: float):
    """Cột tỉ lệ vỡ nợ, ghi số mẫu (n) trên mỗi cột và vẽ đường tỉ lệ chung."""
    x = np.arange(len(tab))
    ax.bar(x, tab["default_rate"], color=_COLORS[1], alpha=0.85)
    for xi, r, n in zip(x, tab["default_rate"], tab["count"]):
        ax.text(xi, r, f"{r:.1%}\nn={n}", ha="center", va="bottom", fontsize=7, zorder=3,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.5))
    ax.axhline(overall, color="gray", ls="--", lw=1, label=f"Chung {overall:.1%}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Tỉ lệ vỡ nợ")
    ax.set_title(title)
    ax.set_ylim(0, min(1.0, tab["default_rate"].max() * 1.3 + 0.02))
    ax.legend(fontsize=8)


def plot_default_rate_pay1(df: pd.DataFrame, path=FIG_DIR / "default_rate_by_pay1.png") -> Path:
    """(1) Tỉ lệ vỡ nợ theo từng giá trị PAY_1: lịch sử trễ hạn gần nhất có quyết định không."""
    tab = default_rate_by(df, "PAY_1")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _bar_rate(ax, tab, tab.index.tolist(), "Tỉ lệ vỡ nợ theo PAY_1 (tháng 9/2005)",
              "PAY_1 (-2: không dùng, -1: trả đủ, 0: dư nợ quay vòng, k: trễ k tháng)",
              df[TARGET].mean())
    return _save(fig, path)


def plot_default_rate_limit(df: pd.DataFrame, path=FIG_DIR / "default_rate_by_limit_bal.png") -> Path:
    """(2) Tỉ lệ vỡ nợ theo khoảng LIMIT_BAL: hạn mức thấp có rủi ro cao hơn không."""
    tab = default_rate_by(df, "LIMIT_BAL", bins=LIMIT_BINS)
    labels = [f"{int(i.left / 1000)}–{int(i.right / 1000)}k" for i in tab.index]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    _bar_rate(ax, tab, labels, "Tỉ lệ vỡ nợ theo khoảng LIMIT_BAL",
              "LIMIT_BAL (NT$, khoảng mở trái đóng phải)", df[TARGET].mean())
    return _save(fig, path)


def plot_default_rate_demographics(df: pd.DataFrame, path=FIG_DIR / "default_rate_by_demographics.png") -> Path:
    """(3) Tỉ lệ vỡ nợ theo nhóm tuổi, học vấn, hôn nhân (và giới tính) trong 1 hình 2x2."""
    overall = df[TARGET].mean()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    tab = default_rate_by(df, "AGE", bins=AGE_BINS)
    labels = [f"{int(i.left) + 1}–{int(i.right)}" for i in tab.index]
    _bar_rate(axes[0, 0], tab, labels, "Theo nhóm tuổi", "AGE", overall)

    for ax, col, names, title in (
        (axes[0, 1], "EDUCATION", EDU_NAMES, "Theo học vấn"),
        (axes[1, 0], "MARRIAGE", MAR_NAMES, "Theo hôn nhân"),
        (axes[1, 1], "SEX", SEX_NAMES, "Theo giới tính"),
    ):
        tab = default_rate_by(df, col)
        _bar_rate(ax, tab, [names[i] for i in tab.index], title, col, overall)

    fig.suptitle("Tỉ lệ vỡ nợ theo hồ sơ nhân khẩu học")
    return _save(fig, path)


def monthly_means(df: pd.DataFrame, prefix: str, target: str = TARGET) -> pd.DataFrame:
    """Trung bình của `{prefix}1..6` theo nhãn; index theo THỜI GIAN (T4 -> T9/2005).

    Chỉ số 6 = tháng 4 (xa nhất), chỉ số 1 = tháng 9 (gần nhất) nên đảo thứ tự cột.
    """
    cols = [f"{prefix}{i}" for i in range(6, 0, -1)]
    m = df.groupby(target)[cols].mean().T
    m.index = MONTH_LABELS
    m.columns.name = target
    return m


def plot_monthly_trend(df: pd.DataFrame, path=FIG_DIR / "monthly_trend.png") -> Path:
    """(4) Trung bình BILL_AMT và PAY_AMT qua 6 tháng (T4 -> T9/2005), tách theo nhãn."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, prefix, title in ((axes[0], "BILL_AMT", "Sao kê (BILL_AMT)"),
                              (axes[1], "PAY_AMT", "Số tiền đã trả (PAY_AMT)")):
        m = monthly_means(df, prefix)
        for lab, name in ((0, "Không vỡ nợ"), (1, "Vỡ nợ")):
            ax.plot(m.index, m[lab], marker="o", color=_COLORS[lab], label=name)
        ax.set_title(f"Trung bình {title} theo tháng")
        ax.set_xlabel("Tháng (2005)")
        ax.set_ylabel("NT$")
        ax.legend()
    return _save(fig, path)


def plot_label_distribution(df: pd.DataFrame, path=FIG_DIR / "label_distribution.png") -> Path:
    """(5) Biểu đồ cột phân phối nhãn: mức độ mất cân bằng."""
    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["0: không vỡ nợ", "1: vỡ nợ"], counts.values,
           color=[_COLORS[0], _COLORS[1]])
    for i, c in enumerate(counts.values):
        ax.text(i, c, f"{c}\n({c / counts.sum():.1%})", ha="center", va="bottom")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.set_ylabel("Số mẫu")
    ax.set_title("Phân phối nhãn")
    return _save(fig, path)


def normalized_for_description(df: pd.DataFrame) -> pd.DataFrame:
    """Bản chuẩn hóa TẠM của LIMIT_BAL, BILL_AMT, PAY_AMT (signed_log) và AGE, rồi StandardScaler.

    Chỉ để MÔ TẢ (mục 4.1 bước 9): StandardScaler fit trên TOÀN BỘ dữ liệu đã làm sạch,
    nên KHÔNG dùng cho huấn luyện (sẽ rò rỉ). AGE chỉ scale, không log.
    TODO(#7): thay bằng pipeline `build_preprocessor` (fit trên train) khi #7 xong.
    """
    cols = MONEY_COLS + ["AGE"]
    x = df[cols].copy()
    x[MONEY_COLS] = signed_log(x[MONEY_COLS])
    scaled = StandardScaler().fit_transform(x)
    return pd.DataFrame(scaled, columns=cols, index=df.index)


def plot_normalization_histograms(df: pd.DataFrame, path=FIG_DIR / "hist_before_after_scaling.png") -> Path:
    """Histogram trước/sau chuẩn hóa cho cột tiền và AGE (mỗi biến 1 cặp ô: trước | sau).

    Chuẩn hóa tạm thời, xem `normalized_for_description` (sẽ thay bằng pipeline #7).
    """
    after = normalized_for_description(df)
    cols = list(after.columns)
    n_rows = 5  # 14 biến, 3 biến mỗi hàng -> mỗi hàng 6 ô
    fig, axes = plt.subplots(n_rows, 6, figsize=(18, 3 * n_rows))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for k, c in enumerate(cols):
        a_before, a_after = axes[2 * k], axes[2 * k + 1]
        a_before.axis("on")
        a_after.axis("on")
        a_before.hist(df[c], bins=40, color=_COLORS[0])
        a_before.set_title(f"{c} (trước)", fontsize=9)
        a_after.hist(after[c], bins=40, color="#54a24b")
        a_after.set_title(f"{c} (sau log+scale)" if c != "AGE" else f"{c} (sau scale)", fontsize=9)
        for a in (a_before, a_after):
            a.tick_params(labelsize=7)
    return _save(fig, path)


def make_all_figures(df: pd.DataFrame, out_dir=FIG_DIR) -> list:
    """Vẽ và lưu toàn bộ hình EDA vào `out_dir`; trả về danh sách đường dẫn."""
    out_dir = Path(out_dir)
    return [
        plot_correlation_heatmap(df, out_dir / "correlation_heatmap.png"),
        plot_default_rate_pay1(df, out_dir / "default_rate_by_pay1.png"),
        plot_default_rate_limit(df, out_dir / "default_rate_by_limit_bal.png"),
        plot_default_rate_demographics(df, out_dir / "default_rate_by_demographics.png"),
        plot_monthly_trend(df, out_dir / "monthly_trend.png"),
        plot_label_distribution(df, out_dir / "label_distribution.png"),
        plot_normalization_histograms(df, out_dir / "hist_before_after_scaling.png"),
    ]
