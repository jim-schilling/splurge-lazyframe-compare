from datetime import datetime

import polars as pl

from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary


def test_summary_to_dict_contains_expected_fields() -> None:
    ts = datetime.now().isoformat()
    summary = ComparisonSummary(
        total_left_records=100,
        total_right_records=95,
        matching_records=80,
        value_differences_count=15,
        left_only_count=5,
        right_only_count=10,
        comparison_timestamp=ts,
    )

    d = summary.to_dict()
    assert d["total_left_records"] == 100
    assert d["comparison_timestamp"] == ts


def test_result_to_dict_includes_flags_and_summary() -> None:
    lf_non_empty = pl.LazyFrame({"a": [1]})
    lf_empty = pl.LazyFrame({"a": []})
    summary = ComparisonSummary(
        total_left_records=1,
        total_right_records=1,
        matching_records=1,
        value_differences_count=0,
        left_only_count=0,
        right_only_count=0,
        comparison_timestamp=datetime.now().isoformat(),
    )

    # value differences non-empty; others empty
    result = ComparisonResult(
        summary=summary,
        value_differences=lf_non_empty,
        left_only_records=lf_empty,
        right_only_records=lf_empty,
        config=None,  # type: ignore[arg-type]
    )

    d = result.to_dict()
    assert d["summary"]["total_left_records"] == 1
    assert d["has_value_differences"] is True
    assert d["has_left_only"] is False
    assert d["has_right_only"] is False
