"""Test cho src/preprocess.py – chỉ dùng dữ liệu giả lập (data/raw không có trên CI)."""
import numpy as np
import pandas as pd
import pytest

from src.data import DERIVED_COLS
from src.preprocess import add_derived_features, load_splits, make_splits, save_splits

RATIOS = {"8:2": 0.2, "7:3": 0.3, "6:4": 0.4}
SEEDS = [0, 1]


def make_df(n=50, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"LIMIT_BAL": rng.integers(1, 10, n) * 10000.0})
    for i in range(1, 7):
        df[f"BILL_AMT{i}"] = rng.normal(20000, 30000, n)
        df[f"PAY_AMT{i}"] = rng.uniform(0, 30000, n)
        df[f"PAY_{i}"] = rng.integers(-2, 5, n)
    return df


def test_derived_them_dung_4_cot_va_khong_sua_input():
    df = make_df()
    goc = df.copy()
    out = add_derived_features(df)
    assert list(out.columns) == list(df.columns) + DERIVED_COLS
    pd.testing.assert_frame_equal(df, goc)  # df gốc không bị sửa


def test_util_ratio():
    df = make_df()
    out = add_derived_features(df)
    np.testing.assert_allclose(out["UTIL_RATIO"], df["BILL_AMT1"] / df["LIMIT_BAL"])
    # một dòng tính tay: 30000 / 120000 = 0.25
    row = make_df(1)
    row["LIMIT_BAL"] = 120000.0
    row["BILL_AMT1"] = 30000.0
    assert add_derived_features(row)["UTIL_RATIO"].iloc[0] == pytest.approx(0.25)


def test_pay_ratio_cat_va_gan_1_khi_bill_khong_duong():
    df = make_df(4)
    df["BILL_AMT2"] = [1000.0, 1000.0, 0.0, -500.0]
    df["PAY_AMT1"] = [500.0, 5000.0, 100.0, 100.0]
    out = add_derived_features(df)
    np.testing.assert_allclose(out["PAY_RATIO"], [0.5, 1.0, 1.0, 1.0])


def test_pay_ratio_am_bi_cat_ve_0():
    df = make_df(1)
    df["BILL_AMT2"] = 1000.0
    df["PAY_AMT1"] = -50.0
    assert add_derived_features(df)["PAY_RATIO"].iloc[0] == 0.0


def test_n_late_dem_pay_ge_1():
    df = make_df()
    out = add_derived_features(df)
    pay = df[[f"PAY_{i}" for i in range(1, 7)]].to_numpy()
    np.testing.assert_array_equal(out["N_LATE"], (pay >= 1).sum(axis=1))
    # một dòng tính tay: PAY = [-2, 0, 1, 1, 3, 0] -> 3 tháng trễ (>=1)
    row = make_df(1)
    for k, v in zip(range(1, 7), [-2, 0, 1, 1, 3, 0]):
        row[f"PAY_{k}"] = v
    assert add_derived_features(row)["N_LATE"].iloc[0] == 3


def test_late_trend_bang_polyfit_va_dau_dung():
    df = make_df()
    out = add_derived_features(df)
    t = np.arange(6)  # PAY_6 -> t=0 ... PAY_1 -> t=5
    for i in range(len(df)):
        y = [df.loc[i, f"PAY_{6 - k}"] for k in range(6)]
        assert out["LATE_TREND"].iloc[i] == pytest.approx(np.polyfit(t, y, 1)[0])
    # xấu đi dần (PAY_6 thấp, PAY_1 cao) -> dương
    row = make_df(1)
    for k, v in zip(range(6, 0, -1), [-1, 0, 1, 2, 3, 4]):
        row[f"PAY_{k}"] = v
    assert add_derived_features(row)["LATE_TREND"].iloc[0] > 0


def make_y(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = pd.Series((rng.random(n) < 0.22).astype(int))
    y.index = rng.permutation(n) + 1000  # index lệch, xáo trộn
    return y


def test_make_splits_khoa_va_ti_le():
    y = make_y()
    sp = make_splits(y, SEEDS, RATIOS)
    assert set(sp) == {(n, s) for n in RATIOS for s in SEEDS}
    for (name, _), (tr, va) in sp.items():
        assert len(va) / len(y) == pytest.approx(RATIOS[name], abs=0.01)


def test_make_splits_roi_nhau_phu_het_va_tra_nhan_index():
    y = make_y()
    for tr, va in make_splits(y, SEEDS, RATIOS).values():
        assert set(tr).isdisjoint(va)
        assert set(tr) | set(va) == set(y.index)
        assert len(tr) + len(va) == len(y)
        assert min(va) >= 1000  # nhãn index, không phải vị trí


def test_make_splits_phan_tang_giu_ti_le_lop():
    y = make_y(1000)
    for tr, va in make_splits(y, SEEDS, RATIOS).values():
        # số mẫu dương trong val phải đúng kỳ vọng (lệch tối đa 1 do làm tròn)
        assert abs(y.loc[va].sum() - y.sum() * len(va) / len(y)) <= 1
        assert abs(y.loc[tr].sum() - y.sum() * len(tr) / len(y)) <= 1


def test_make_splits_cung_seed_giong_nhau():
    y = make_y()
    a = make_splits(y, SEEDS, RATIOS)
    b = make_splits(y, SEEDS, RATIOS)
    for k in a:
        np.testing.assert_array_equal(a[k][0], b[k][0])
        np.testing.assert_array_equal(a[k][1], b[k][1])
    assert not np.array_equal(a[("7:3", 0)][1], a[("7:3", 1)][1])  # khác seed thì khác


def test_make_splits_index_trung_bao_loi():
    y = pd.Series([0, 1, 0, 1, 0, 1], index=[0, 0, 1, 2, 3, 4])
    with pytest.raises(ValueError):
        make_splits(y, [0], {"7:3": 0.3})


def test_make_splits_strata_xao_thu_tu_giong_strata_thang_hang():
    y = make_y(400)
    # nhóm độc lập với y và lệch (60/30/10) -> chỉ phân tầng theo y sẽ không giữ được cỡ nhóm
    nhom = pd.Series(
        np.random.default_rng(1).choice(["a", "b", "c"], len(y), p=[0.6, 0.3, 0.1]),
        index=y.index,
    )
    shuffled = nhom.sample(frac=1, random_state=3)
    a = make_splits(y, [0], {"7:3": 0.3}, strata=nhom)
    b = make_splits(y, [0], {"7:3": 0.3}, strata=shuffled)
    np.testing.assert_array_equal(a[("7:3", 0)][0], b[("7:3", 0)][0])
    np.testing.assert_array_equal(a[("7:3", 0)][1], b[("7:3", 0)][1])
    # mỗi nhóm trong val đúng tỉ lệ (lệch tối đa 1)
    va = a[("7:3", 0)][1]
    for g, n in nhom.value_counts().items():
        assert abs(nhom.loc[va].eq(g).sum() - n * 0.3) <= 1


def test_make_splits_strata_thieu_nhan_bao_loi():
    y = make_y()
    with pytest.raises(ValueError):
        make_splits(y, [0], {"7:3": 0.3}, strata=y.iloc[:-5])


def test_save_load_splits_round_trip(tmp_path):
    y = make_y()
    sp = make_splits(y, SEEDS, RATIOS)
    save_splits(sp, tmp_path / "splits")  # thư mục chưa tồn tại -> tự tạo
    assert (tmp_path / "splits" / "split_7-3_seed0.npz").exists()
    back = load_splits(SEEDS, RATIOS, tmp_path / "splits")
    assert set(back) == set(sp)
    for k in sp:
        np.testing.assert_array_equal(back[k][0], sp[k][0])
        np.testing.assert_array_equal(back[k][1], sp[k][1])
