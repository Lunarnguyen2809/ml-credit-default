"""Ba mô hình phân loại (mục 4.4). Mỗi thành viên sở hữu một mô hình.

Dò siêu tham số: 5-fold CV có phân tầng trên train, tối ưu PR-AUC (configs/experiment.yaml).
"""


def build_naive_bayes(params=None):
    """TODO(#13) – Nhóm 1, mô hình sinh. GaussianNB, dò var_smoothing."""
    raise NotImplementedError


def build_logistic(params=None, class_weight=None):
    """TODO(#14) – Nhóm 2, phân biệt tuyến tính. Dò C và penalty L1/L2 (solver='saga' hoặc 'liblinear')."""
    raise NotImplementedError


def build_svm_rbf(params=None, class_weight=None):
    """TODO(#20) – Nhóm 3, phân biệt phi tuyến. Dò C, gamma trên mẫu con 10.000."""
    raise NotImplementedError


def tune(pipeline, grid, X_train, y_train, cfg):
    """GridSearchCV với StratifiedKFold(cfg['cv_folds']), scoring=cfg['cv_scoring']."""
    raise NotImplementedError


def choose_threshold(y_val, proba):
    """Chọn ngưỡng quyết định trên validation theo F1 lớp vỡ nợ."""
    raise NotImplementedError
