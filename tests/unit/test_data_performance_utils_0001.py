import polars as pl

from splurge_lazyframe_compare.utils.data_helpers import (
    compare_dataframe_shapes,
    get_dataframe_info,
    get_null_summary,
    has_null_values,
    optimize_dataframe,
    safe_collect,
)


def test_get_dataframe_info_and_nulls():
    df = pl.LazyFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    info = get_dataframe_info(df)
    assert info["row_count"] == 3
    assert info["column_count"] == 2
    assert set(info["column_names"]) == {"a", "b"}

    assert has_null_values(df) is True
    assert has_null_values(df, columns=["b"]) is False

    nulls = get_null_summary(df)
    assert set(nulls.keys()) == {"a", "b"}
    assert nulls["a"]["has_nulls"] is True


def test_compare_shapes_and_optimize_and_safe_collect():
    left = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    right = pl.LazyFrame({"a": [1, 2, 3, 4], "c": [10, 20, 30, 40]})

    comparison = compare_dataframe_shapes(left, right)
    assert comparison["shape_comparison"]["same_row_count"] is False
    assert comparison["shape_comparison"]["row_difference"] == 1
    assert "a" in comparison["column_overlap"]["common_columns"]

    optimized = optimize_dataframe(left)
    assert isinstance(optimized, pl.LazyFrame)

    collected = safe_collect(left)
    assert collected is not None and collected.shape == (3, 2)


