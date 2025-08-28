"""Comparison result models for the comparison framework."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import polars as pl


@dataclass
class ComparisonConstants:
    """Constants for comparison operations."""

    # Default values
    DEFAULT_NULL_EQUALS_NULL: bool = True
    DEFAULT_IGNORE_CASE: bool = False
    DEFAULT_MAX_SAMPLES: int = 10

    # Column prefixes
    PRIMARY_KEY_PREFIX: str = "PK_"
    LEFT_PREFIX: str = "L_"
    RIGHT_PREFIX: str = "R_"
    LEFT_DF_NAME: str = "left DataFrame"
    RIGHT_DF_NAME: str = "right DataFrame"

    # Join types
    JOIN_INNER: str = "inner"
    JOIN_LEFT: str = "left"

    # Thresholds
    DUPLICATE_THRESHOLD: int = 1
    ZERO_THRESHOLD: int = 0

    # Error messages
    DUPLICATE_PK_MSG: str = "Duplicate primary keys found in {}: {} duplicates"


@dataclass
class ComparisonSummary:
    """Summary statistics for comparison results.

    Attributes:
        total_left_records: Total number of records in left DataFrame.
        total_right_records: Total number of records in right DataFrame.
        matching_records: Number of records that match exactly.
        value_differences_count: Number of records with value differences.
        left_only_count: Number of records only in left DataFrame.
        right_only_count: Number of records only in right DataFrame.
        comparison_timestamp: Timestamp when comparison was performed.
    """

    total_left_records: int
    total_right_records: int
    matching_records: int
    value_differences_count: int
    left_only_count: int
    right_only_count: int
    comparison_timestamp: str

    @classmethod
    def create(
        cls,
        *,
        total_left_records: int,
        total_right_records: int,
        value_differences: pl.LazyFrame,
        left_only_records: pl.LazyFrame,
        right_only_records: pl.LazyFrame,
    ) -> "ComparisonSummary":
        """Create a comparison summary from result DataFrames.

        Args:
            total_left_records: Total records in left DataFrame.
            total_right_records: Total records in right DataFrame.
            value_differences: LazyFrame containing value differences.
            left_only_records: LazyFrame containing left-only records.
            right_only_records: LazyFrame containing right-only records.

        Returns:
            ComparisonSummary with calculated statistics.
        """
        value_differences_count = value_differences.select(pl.len()).collect().item()
        left_only_count = left_only_records.select(pl.len()).collect().item()
        right_only_count = right_only_records.select(pl.len()).collect().item()

        # Calculate matching records
        matching_records = total_left_records - left_only_count - value_differences_count

        return cls(
            total_left_records=total_left_records,
            total_right_records=total_right_records,
            matching_records=matching_records,
            value_differences_count=value_differences_count,
            left_only_count=left_only_count,
            right_only_count=right_only_count,
            comparison_timestamp=datetime.now().isoformat(),
        )


@dataclass
class ValueDifference:
    """Details of a specific value difference.

    Attributes:
        primary_key_values: Dictionary of primary key column values.
        column_name: Name of the column with the difference.
        left_value: Value from left DataFrame.
        right_value: Value from right DataFrame.
        friendly_column_name: Human-readable column name.
    """

    primary_key_values: Dict[str, Any]
    column_name: str
    left_value: Any
    right_value: Any
    friendly_column_name: str


@dataclass
class ComparisonResult:
    """Container for all comparison results.

    Attributes:
        summary: Summary statistics for the comparison.
        value_differences: LazyFrame containing records with value differences.
        left_only_records: LazyFrame containing records only in left DataFrame.
        right_only_records: LazyFrame containing records only in right DataFrame.
        config: Configuration used for the comparison.
    """

    summary: ComparisonSummary
    value_differences: pl.LazyFrame
    left_only_records: pl.LazyFrame
    right_only_records: pl.LazyFrame
    config: "ComparisonConfig"  # Forward reference to avoid circular import
