"""Thống kê và trực quan hóa dữ liệu gốc (phần thống kê của mục 2.2). TODO(#8)

Đầu ra lưu vào results/figures/, mỗi hình kèm 1–2 câu nhận xét trong notebook.
"""


def describe_numeric(df):
    """mean, median, std, min, max, skewness, kurtosis cho các cột số."""
    raise NotImplementedError


def default_rate_by(df, col, bins=None):
    """Tỉ lệ vỡ nợ theo từng mức (hoặc khoảng nếu có bins) của một cột."""
    raise NotImplementedError


def plot_correlation_heatmap(df, path):
    raise NotImplementedError


def plot_monthly_trend(df, path):
    """Đường trung bình BILL_AMT và PAY_AMT qua 6 tháng, tách theo nhãn."""
    raise NotImplementedError
