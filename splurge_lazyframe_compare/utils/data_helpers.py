"""Data manipulation helpers for the comparison framework.

Copyright (c) 2025 Jim Schilling.
Licensed under the MIT License. See the LICENSE file for details.
"""

from typing import Any

import polars as pl


class DataHelperConstants:
    """Constants for data manipulation operations."""

    # Data validation
    MIN_ROWS_THRESHOLD: int = 0
    MAX_ROWS_THRESHOLD: int = 1000000

    # Memory optimization
    BATCH_SIZE_DEFAULT: int = 10000
    MEMORY_LIMIT_MB: int = 1000

    # Memory estimation (bytes per value)
    BYTES_PER_INT64: int = 8
    BYTES_PER_INT32: int = 4
    BYTES_PER_FLOAT64: int = 8
    BYTES_PER_FLOAT32: int = 4
    BYTES_PER_BOOLEAN: int = 1
    BYTES_PER_STRING_AVG: int = 16  # Average string length in bytes
    BYTES_PER_DEFAULT: int = 8

    # Performance tuning
    OPTIMIZE_THRESHOLD: int = 100000
    PARALLEL_WORKERS: int = 4


def validate_dataframe(df: pl.LazyFrame, name: str = "DataFrame") -> None:
    """Validate basic DataFrame properties.

    Args:
        df: LazyFrame to validate.
        name: Name for error reporting.

    Raises:
        ValueError: If DataFrame is invalid.
    """
    if df is None:
        raise ValueError(f"{name} cannot be None")

    # Check if DataFrame has columns
    try:
        schema = df.collect_schema()
        if not schema.names():
            raise ValueError(f"{name} has no columns")
    except Exception as e:
        raise ValueError(f"Invalid {name}: {e}") from e


def get_dataframe_info(df: pl.LazyFrame) -> dict[str, Any]:
    """Get comprehensive information about a DataFrame.

    Args:
        df: LazyFrame to analyze.

    Returns:
        Dictionary with DataFrame information.
    """
    try:
        schema = df.collect_schema()
        row_count = df.select(pl.len()).collect().item()

        return {
            "row_count": row_count,
            "column_count": len(schema.names()),
            "column_names": schema.names(),
            "column_types": [str(dtype) for dtype in schema.dtypes()],
            "memory_estimate_mb": estimate_dataframe_memory(df),
            "has_nulls": has_null_values(df),
        }
    except Exception:
        return {
            "row_count": 0,
            "column_count": 0,
            "column_names": [],
            "column_types": [],
            "memory_estimate_mb": 0,
            "has_nulls": False,
        }


def estimate_dataframe_memory(df: pl.LazyFrame) -> float:
    """Estimate memory usage of a DataFrame in MB.

    Args:
        df: LazyFrame to analyze.

    Returns:
        Estimated memory usage in MB.
    """
    try:
        # Sample a subset to estimate
        sample_size = min(1000, df.select(pl.len()).collect().item())
        if sample_size == 0:
            return 0

        sample_df = df.limit(sample_size).collect()

        # Rough estimation based on Polars internals
        # This is approximate and may vary by system
        bytes_per_row = sum(
            DataHelperConstants.BYTES_PER_INT64 if dtype in [pl.Int64, pl.Float64, pl.Datetime] else
            DataHelperConstants.BYTES_PER_INT32 if dtype in [pl.Int32, pl.Float32] else
            DataHelperConstants.BYTES_PER_BOOLEAN if dtype == pl.Boolean else
            DataHelperConstants.BYTES_PER_STRING_AVG if dtype == pl.Utf8 else  # Average string length
            DataHelperConstants.BYTES_PER_DEFAULT  # Default for other types
            for dtype in sample_df.dtypes
        )

        total_rows = df.select(pl.len()).collect().item()
        total_bytes = bytes_per_row * total_rows

        return total_bytes / (1024 * 1024)  # Convert to MB

    except Exception:
        return 0


def has_null_values(df: pl.LazyFrame, columns: list[str] | None = None) -> bool:
    """Check if DataFrame has null values in specified columns.

    Args:
        df: LazyFrame to check.
        columns: Specific columns to check. If None, checks all columns.

    Returns:
        True if any null values found, False otherwise.
    """
    try:
        schema = df.collect_schema()
        all_columns = schema.names()
        check_columns = columns if columns else all_columns

        for col in check_columns:
            if col not in all_columns:
                continue

            null_count = df.select(pl.col(col).is_null().sum()).collect().item()
            if null_count > 0:
                return True

        return False

    except Exception:
        return False


def get_null_summary(df: pl.LazyFrame) -> dict[str, dict[str, int | float]]:
    """Get summary of null values in DataFrame.

    Args:
        df: LazyFrame to analyze.

    Returns:
        Dictionary with null statistics per column.
    """
    try:
        total_rows = df.select(pl.len()).collect().item()
        summary = {}
        schema = df.collect_schema()

        for col in schema.names():
            null_count = df.select(pl.col(col).is_null().sum()).collect().item()
            null_percentage = (null_count / total_rows * 100) if total_rows > 0 else 0

            summary[col] = {
                "null_count": null_count,
                "null_percentage": round(null_percentage, 2),
                "has_nulls": null_count > 0,
            }

        return summary

    except Exception:
        return {}


def optimize_dataframe(df: pl.LazyFrame) -> pl.LazyFrame:
    """Apply basic optimizations to DataFrame.

    Args:
        df: LazyFrame to optimize.

    Returns:
        Optimized LazyFrame.
    """
    try:
        # Apply basic optimizations
        optimized = df

        # Use lazy evaluation optimizations
        optimized = optimized.select(pl.all())

        return optimized

    except Exception:
        # Return original if optimization fails
        return df


def safe_collect(df: pl.LazyFrame) -> pl.DataFrame | None:
    """Safely collect a LazyFrame with error handling.

    Args:
        df: LazyFrame to collect.

    Returns:
        Collected DataFrame or None if collection fails.
    """
    try:
        return df.collect()
    except Exception:
        return None


def compare_dataframe_shapes(df1: pl.LazyFrame, df2: pl.LazyFrame) -> dict[str, Any]:
    """Compare shapes and basic properties of two DataFrames.

    Args:
        df1: First LazyFrame to compare.
        df2: Second LazyFrame to compare.

    Returns:
        Dictionary with comparison results.
    """
    try:
        info1 = get_dataframe_info(df1)
        info2 = get_dataframe_info(df2)

        return {
            "df1_info": info1,
            "df2_info": info2,
            "shape_comparison": {
                "same_row_count": info1["row_count"] == info2["row_count"],
                "same_column_count": info1["column_count"] == info2["column_count"],
                "row_difference": info2["row_count"] - info1["row_count"],
                "column_difference": info2["column_count"] - info1["column_count"],
            },
            "column_overlap": {
                "common_columns": list(set(info1["column_names"]) & set(info2["column_names"])),
                "df1_only_columns": list(set(info1["column_names"]) - set(info2["column_names"])),
                "df2_only_columns": list(set(info2["column_names"]) - set(info1["column_names"])),
            },
        }

    except Exception:
        return {
            "df1_info": {},
            "df2_info": {},
            "shape_comparison": {},
            "column_overlap": {},
        }


