"""Phân cụm (mục 4.3), chạy trên 11 thành phần PCA của tập train, bỏ qua nhãn. TODO(#21)"""


def select_k_gmm(Z, k_range):
    """BIC và AIC với k trong k_range."""
    raise NotImplementedError


def select_k_kmeans(Z, k_range):
    """Elbow (inertia) và silhouette với k trong k_range."""
    raise NotImplementedError


def cluster_label_report(clusters, y):
    """Tỉ lệ vỡ nợ từng cụm, kiểm định chi-bình phương, ARI, NMI."""
    raise NotImplementedError


def plot_clusters(Z, clusters, y, path):
    """Scatter PC1–PC2: màu = cụm, hình dạng điểm = nhãn."""
    raise NotImplementedError
