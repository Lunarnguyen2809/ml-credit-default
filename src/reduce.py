"""Giảm chiều và trực quan hóa (mục 4.2). Mọi phương pháp fit trên train, validation chỉ transform."""


def fit_pca(X_train, n_components):
    """TODO(#10): trả về PCA đã fit; dùng explained_variance_ratio_ để vẽ scree và đường tích lũy."""
    raise NotImplementedError


def fit_famd(df_train, n_components):
    """TODO(#11): prince.FAMD trên dữ liệu hỗn hợp chưa one-hot; đo tỉ lệ quán tính giải thích."""
    raise NotImplementedError


def plot_pairwise(Z, y, n_components, path):
    """Scatter từng cặp cho n_components thành phần đầu (4–6), tô màu theo nhãn."""
    raise NotImplementedError


def plot_loadings(model, feature_names, n_components, path):
    """Loadings của 2–3 thành phần đầu."""
    raise NotImplementedError
