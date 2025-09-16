import polars as pl

from splurge_lazyframe_compare.utils.formatting import (
    create_summary_table,
    format_column_list,
    format_dataframe_sample,
    format_large_number,
    format_number,
    format_percentage,
    format_validation_errors,
    truncate_string,
)


def test_number_and_percentage_formatting():
    assert format_number(3.14159, precision=3) == "3.142"
    assert format_percentage(0.1234) == "12.3%"
    assert format_percentage(0.5, include_symbol=False) == "50.0"


def test_large_number_and_truncation():
    assert format_large_number(1234567) == "1,234,567"
    long = "x" * 100
    t = truncate_string(long, max_length=10)
    assert len(t) == 10
    assert t.endswith("...")


def test_dataframe_sample_and_column_list():
    df = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    s = format_dataframe_sample(df, max_rows=2, max_cols=1)
    assert "shape" in s or "a" in s  # string representation

    cols = [f"c{i}" for i in range(12)]
    cl = format_column_list(cols, max_items=5)
    assert "`c0`" in cl and "(+7 more)" in cl


def test_validation_errors_formatting():
    assert format_validation_errors([]) == "No errors found"
    assert format_validation_errors(["only one"]) == "only one"
    multi = format_validation_errors(["e1", "e2"]) 
    assert "Multiple errors" in multi and "e1" in multi and "e2" in multi


def test_create_summary_table_simple():
    table = create_summary_table({"Left": 10, "Right": 12})
    assert isinstance(table, str)
    assert "Left" in table and "Right" in table


