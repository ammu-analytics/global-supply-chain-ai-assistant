import pandas as pd
import pytest

from supplychain import analytics as A
from supplychain.data import clean


def test_one_row_per_route_year(data):
    assert not data.routes.duplicated(["importer", "supplier", "year"]).any()
    assert len(data.routes) == 55  # 11 routes x 5 years


def test_germany_not_double_counted(data):
    # the total row in the raw file for Germany <- China 2023
    assert A.total_value(data.routes, "Germany", "China", 2023) == pytest.approx(1.75605e11)
    assert data.report["double_counted"] > 1e12


def test_product_lines_add_up_to_totals(data):
    pl = data.product_lines.groupby(["importer", "supplier", "year"])["value"].sum()
    for (imp, sup, yr), v in pl.items():
        assert v == pytest.approx(A.total_value(data.routes, imp, sup, yr), rel=0.01)


def test_group_without_total_row_is_summed():
    raw = pd.DataFrame({
        "importer": ["A", "A", "A"],
        "supplier": ["B", "B", "B"],
        "year": [2020, 2020, 2020],
        "trade_flow": ["Import"] * 3,
        "value": [10.0, 20.0, 5.0],
    })
    td = clean(raw)
    assert td.routes["value"].iloc[0] == 35.0
    assert td.report["groups_without_total_row"]


def test_bad_values_dropped():
    raw = pd.DataFrame({
        "importer": ["A", "A"], "supplier": ["B", "C"], "year": [2020, 2020],
        "trade_flow": ["Import"] * 2, "value": [None, 3.0],
    })
    td = clean(raw)
    assert len(td.routes) == 1


def test_dependency_shares_sum_to_100(data):
    table, hhi = A.dependency(data.routes, "Germany", 2023)
    assert table["share_pct"].sum() == pytest.approx(100, abs=0.2)
    assert 0 < hhi <= 1


def test_fmt_value():
    assert A.fmt_value(1.7562e11) == "175.62 B"
    assert A.fmt_value(1.2e12) == "1.20 T"
    assert A.fmt_value(950) == "950.00"
