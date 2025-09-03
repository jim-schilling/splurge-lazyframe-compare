from hypothesis import assume, given, strategies as st

from splurge_lazyframe_compare.models.comparison import ComparisonSummary


@given(
    total_left=st.integers(min_value=0, max_value=10_000),
    total_right=st.integers(min_value=0, max_value=10_000),
    left_only=st.integers(min_value=0, max_value=10_000),
    diffs=st.integers(min_value=0, max_value=10_000),
    right_only=st.integers(min_value=0, max_value=10_000),
)
def test_summary_invariants(total_left: int, total_right: int, left_only: int, diffs: int, right_only: int) -> None:
    # We assume counts are consistent with a feasible left dataset
    assume(left_only + diffs <= total_left)

    import polars as pl

    def make_len(n: int) -> pl.LazyFrame:
        return pl.LazyFrame({"x": list(range(max(n, 0)))})

    summary = ComparisonSummary.create(
        total_left_records=total_left,
        total_right_records=total_right,
        value_differences=make_len(diffs),
        left_only_records=make_len(left_only),
        right_only_records=make_len(right_only),
    )

    assert summary.matching_records >= 0
    assert summary.value_differences_count >= 0
    assert summary.left_only_count >= 0
    assert summary.right_only_count >= 0

