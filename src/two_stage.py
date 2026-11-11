"""Mô-đun mở rộng 2 tầng (mục 5) – chỉ làm sau giữa kỳ. TODO(#26, #27)

Tầng 1: GMM trên khối A (hồ sơ) -> P(k | A).
Tầng 2: mỗi f_k huấn luyện trên toàn bộ train với sample_weight = P(k | A), dự đoán từ khối B.
y_hat = sum_k P(k | A) * f_k(B)
Bắt đầu bằng cổng kiểm tra likelihood ratio test; không qua thì viết kết luận âm tính và dừng.
"""


def interaction_gate(X_train, y_train, clusters):
    raise NotImplementedError


def soft_routing_predict(gmm, experts, X_A, X_B):
    raise NotImplementedError
