from splurge_lazyframe_compare.models.comparison import ComparisonSummary


def _make_len(n: int):
    """Create a small Polars LazyFrame with length n for test inputs."""
    import polars as pl

    return pl.LazyFrame({"x": list(range(max(n, 0)))})


def run_summary_and_assert(total_left: int, total_right: int, left_only: int, diff_count: int, right_only: int) -> None:
    """Helper that creates a ComparisonSummary and asserts basic non-negativity invariants.

    The original test used Hypothesis to draw random integers and assumed that
    left_only + diff_count <= total_left. Here we exercise several deterministic
    cases, including edge cases, to provide similar coverage.
    """

    # Skip infeasible combinations (mirrors the original assume())
    if left_only + diff_count > total_left:
        return

    summary = ComparisonSummary.create(
        total_left_records=total_left,
        total_right_records=total_right,
        value_differences=_make_len(diff_count),
        left_only_records=_make_len(left_only),
        right_only_records=_make_len(right_only),
    )

    assert summary.matching_records >= 0
    assert summary.value_differences_count >= 0
    assert summary.left_only_count >= 0
    assert summary.right_only_count >= 0


def test_summary_invariants_deterministic() -> None:
    """Deterministic test cases replacing Hypothesis-driven randomized checks."""

    # Typical small cases
    run_summary_and_assert(10, 10, 0, 0, 0)
    run_summary_and_assert(10, 10, 2, 3, 1)

    # Edge cases: zeros
    run_summary_and_assert(0, 0, 0, 0, 0)
    run_summary_and_assert(1, 0, 0, 1, 0)

    # Large-ish numbers (but deterministic)
    run_summary_and_assert(1000, 800, 10, 20, 5)

    # Cases where left_only + diff_count == total_left (boundary)
    run_summary_and_assert(5, 7, 2, 3, 1)

    # Infeasible case should be skipped (mirrors original assume semantics)
    run_summary_and_assert(3, 3, 2, 2, 0)


def test_summary_invariants_additional_cases() -> None:
    """Extra deterministic cases to broaden coverage without Hypothesis.

    These include additional boundaries, large values, and right-heavy scenarios.
    """

    cases = [
        # small boundaries
        (2, 0, 0, 2, 0),
        (2, 0, 1, 1, 0),
        # right-heavy (more right-only than left-related diffs)
        (10, 15, 1, 2, 5),
        # large but feasible
        (5000, 4000, 100, 200, 50),
        (10_000, 10_000, 0, 0, 0),
        # boundary where left_only + diff_count == total_left
        (7, 8, 3, 4, 1),
    ]

    for total_left, total_right, left_only, diff_count, right_only in cases:
        run_summary_and_assert(total_left, total_right, left_only, diff_count, right_only)


def test_matching_records_formula_for_feasible_inputs() -> None:
    """Assert matching_records equals total_left - left_only - diff_count for feasible inputs.

    This checks an expected invariant for a handful of deterministic inputs.
    """

    checks = [
        (10, 10, 2, 3, 0),
        (5, 5, 0, 0, 0),
        (100, 80, 10, 20, 5),
    ]

    for total_left, total_right, left_only, diff_count, right_only in checks:
        # Ensure we only assert when the combination is feasible
        if left_only + diff_count > total_left:
            continue

        summary = ComparisonSummary.create(
            total_left_records=total_left,
            total_right_records=total_right,
            value_differences=_make_len(diff_count),
            left_only_records=_make_len(left_only),
            right_only_records=_make_len(right_only),
        )

        expected_matching = total_left - left_only - diff_count
        assert summary.matching_records == expected_matching
