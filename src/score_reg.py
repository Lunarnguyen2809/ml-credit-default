"""Mục 2.4(c): chuyển sang hồi quy trên score của lớp "vỡ nợ". TODO(#23)

1. Huấn luyện Logistic và SVM chỉ trên train.
2. Lấy decision_function cho train và test làm đầu ra mới.
3. Hồi quy bằng Ridge và K-NN Regressor trên 33 chiều và 11 chiều, đủ 3 tỉ lệ chia.
4. Báo cáo R², RMSE, MAE trên test.
"""


def make_score_target(classifier, X_train, X_test):
    raise NotImplementedError


def fit_regressors(X_train, s_train, X_test, s_test, cfg):
    raise NotImplementedError
