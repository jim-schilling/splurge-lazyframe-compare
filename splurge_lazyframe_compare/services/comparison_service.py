"""Core comparison service for the comparison framework.

Copyright (c) 2025 Jim Schilling.
Licensed under the MIT License. See the LICENSE file for details.
"""

from typing import Literal, cast

import polars as pl

from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary
from splurge_lazyframe_compare.models.schema import ComparisonConfig
from splurge_lazyframe_compare.services.base_service import BaseService
from splurge_lazyframe_compare.services.preparation_service import DataPreparationService
from splurge_lazyframe_compare.services.validation_service import ValidationService
from splurge_lazyframe_compare.utils.constants import (
    JOIN_INNER,
    LEFT_PREFIX,
    PRIMARY_KEY_PREFIX,
    RIGHT_PREFIX,
    ZERO_THRESHOLD,
)
from splurge_lazyframe_compare.utils.logging_helpers import performance_monitor

DOMAINS: list[str] = ["services", "comparison", "processing"]


class ComparisonService(BaseService):
    """Core comparison service for LazyFrames.

    This service orchestrates the complete comparison process between
    two LazyFrames, including validation, preparation, comparison,
    and result generation.
    """

    def __init__(
        self,
        validation_service: ValidationService | None = None,
        preparation_service: DataPreparationService | None = None,
    ) -> None:
        """Initialize the comparison service.

        Args:
            validation_service: Optional validation service instance.
            preparation_service: Optional data preparation service instance.
        """
        super().__init__("ComparisonService")

        # Use provided services or create defaults
        self.validation_service = validation_service or ValidationService()
        self.preparation_service = preparation_service or DataPreparationService()

    def _validate_inputs(self, **kwargs) -> None:
        """Validate service inputs."""
        if "left" in kwargs and not isinstance(kwargs["left"], pl.LazyFrame):
            raise ValueError("left must be a polars LazyFrame")
        if "right" in kwargs and not isinstance(kwargs["right"], pl.LazyFrame):
            raise ValueError("right must be a polars LazyFrame")
        if "config" in kwargs and not isinstance(kwargs["config"], ComparisonConfig):
            raise ValueError("config must be a ComparisonConfig")

    def execute_comparison(
        self,
        *,
        left: pl.LazyFrame,
        right: pl.LazyFrame,
        config: ComparisonConfig,
    ) -> ComparisonResult:
        """Execute complete comparison between two LazyFrames.

        Args:
            left: Left LazyFrame to compare.
            right: Right LazyFrame to compare.
            config: Comparison configuration.

        Returns:
            ComparisonResult containing all comparison results.
        """
        try:
            self._validate_inputs(left=left, right=right, config=config)

            with performance_monitor(self.service_name, "execute_comparison") as ctx:
                # Validate DataFrames against schemas
                self.validation_service.validate_dataframe_schema(
                    df=left, schema=config.left_schema, df_name="Left DataFrame"
                )
                self.validation_service.validate_dataframe_schema(
                    df=right, schema=config.right_schema, df_name="Right DataFrame"
                )

                # Validate primary key uniqueness
                self.validation_service.validate_primary_key_uniqueness(
                    df=left, config=config, df_name="left DataFrame"
                )
                self.validation_service.validate_primary_key_uniqueness(
                    df=right, config=config, df_name="right DataFrame"
                )

                # Prepare DataFrames for comparison
                prepared_left, prepared_right = self.preparation_service.prepare_dataframes(
                    left=left, right=right, config=config
                )

                # Get record counts - optimized to collect once
                counts_df = pl.concat(
                    [prepared_left.select(pl.len().alias("count")), prepared_right.select(pl.len().alias("count"))]
                ).collect()

                total_left_records = counts_df["count"][0]
                total_right_records = counts_df["count"][1]

                ctx["record_counts"] = {"left": total_left_records, "right": total_right_records}

                # Execute comparison patterns
                value_differences = self.find_value_differences(left=prepared_left, right=prepared_right, config=config)
                left_only_records = self.find_left_only_records(left=prepared_left, right=prepared_right, config=config)
                right_only_records = self.find_right_only_records(
                    left=prepared_left, right=prepared_right, config=config
                )

                # Create summary
                summary = ComparisonSummary.create(
                    total_left_records=total_left_records,
                    total_right_records=total_right_records,
                    value_differences=value_differences,
                    left_only_records=left_only_records,
                    right_only_records=right_only_records,
                )

                # Add comparison results to performance context
                ctx["comparison_results"] = {
                    "matching": summary.matching_records,
                    "differences": summary.value_differences_count,
                    "left_only": summary.left_only_count,
                    "right_only": summary.right_only_count,
                }

                return ComparisonResult(
                    summary=summary,
                    value_differences=value_differences,
                    left_only_records=left_only_records,
                    right_only_records=right_only_records,
                    config=config,
                )

        except Exception as e:
            self._handle_error(e, {"operation": "comparison_execution"})
            raise
            raise

    def find_value_differences(
        self, *, left: pl.LazyFrame, right: pl.LazyFrame, config: ComparisonConfig
    ) -> pl.LazyFrame:
        """Find records with same keys but different values.

        Args:
            left: Prepared left LazyFrame.
            right: Prepared right LazyFrame.
            config: Comparison configuration.

        Returns:
            LazyFrame containing records with value differences with alternating Left/Right columns.
        """
        try:
            # Get primary key columns with PK_ prefix
            pk_columns = [f"{PRIMARY_KEY_PREFIX}{pk}" for pk in config.pk_columns]

            # Join on primary key columns
            joined = left.join(
                right,
                on=pk_columns,
                how=cast(Literal["inner", "left", "right", "full", "semi", "anti", "cross", "outer"], JOIN_INNER),
            )

            # Create difference conditions for each mapped column
            diff_conditions = []
            for mapping in config.column_mappings:
                if mapping.name in config.pk_columns:
                    # Primary key columns use PK_ prefix
                    left_col = f"{PRIMARY_KEY_PREFIX}{mapping.name}"
                    right_col = left_col  # Same column since they're joined on PK
                else:
                    # Non-primary key columns use L_ and R_ prefixes
                    left_col = f"{LEFT_PREFIX}{mapping.name}"
                    right_col = f"{RIGHT_PREFIX}{mapping.name}"

                # Handle null comparisons and tolerance together
                if config.null_equals_null:
                    # null == null, but null != value (always different when one is null and one isn't)
                    null_condition = ~pl.col(left_col).eq_missing(pl.col(right_col))
                else:
                    # null != null and null != value - handle null comparisons explicitly
                    # When null_equals_null=False, any null comparison should be a difference
                    null_condition = (
                        # Both are null: always different when null_equals_null=False
                        (pl.col(left_col).is_null() & pl.col(right_col).is_null())
                        |
                        # One is null, one is not: always different
                        (pl.col(left_col).is_null() & pl.col(right_col).is_not_null())
                        | (pl.col(left_col).is_not_null() & pl.col(right_col).is_null())
                        |
                        # Both are non-null: use standard inequality
                        (
                            pl.col(left_col).is_not_null()
                            & pl.col(right_col).is_not_null()
                            & (pl.col(left_col) != pl.col(right_col))
                        )
                    )

                # Apply tolerance for numeric columns if specified
                if config.tolerance and mapping.name in config.tolerance:
                    tolerance = config.tolerance[mapping.name]

                    # Use null_condition as base, but override the non-null case to use tolerance
                    condition = (
                        pl.when(
                            # Both values are non-null: use tolerance comparison
                            pl.col(left_col).is_not_null() & pl.col(right_col).is_not_null()
                        )
                        .then(
                            # Check if difference exceeds tolerance
                            (pl.col(left_col) - pl.col(right_col)).abs() > tolerance
                        )
                        .otherwise(
                            # Use the null_condition for all null cases
                            null_condition
                        )
                    )
                else:
                    # No tolerance, use null condition as-is
                    condition = null_condition

                diff_conditions.append(condition)

            # Filter rows with any differences
            if diff_conditions:
                filtered_joined = joined.filter(pl.any_horizontal(diff_conditions))

                # Reorder columns to show alternating Left/Right for non-primary key columns
                reordered_columns = self.preparation_service.get_alternating_column_order(config=config)

                return filtered_joined.select(reordered_columns)
            else:
                # No columns to compare, return empty result
                return joined.limit(ZERO_THRESHOLD)

        except Exception as e:
            self._handle_error(e, {"operation": "value_differences"})

    def find_left_only_records(
        self, *, left: pl.LazyFrame, right: pl.LazyFrame, config: ComparisonConfig
    ) -> pl.LazyFrame:
        """Find records that exist only in left DataFrame.

        Args:
            left: Prepared left LazyFrame.
            right: Prepared right LazyFrame.
            config: Comparison configuration.

        Returns:
            LazyFrame containing left-only records with right columns dropped.
        """
        try:
            # Get primary key columns with PK_ prefix
            pk_columns = [f"{PRIMARY_KEY_PREFIX}{pk}" for pk in config.pk_columns]

            # Join and filter for left-only records (anti-join)
            left_only_joined = left.join(right.select(pk_columns), on=pk_columns, how="anti")

            # Select only primary key columns and left columns (drop right columns which are null)
            left_only_columns = self.preparation_service.get_left_only_column_order(config=config)

            return left_only_joined.select(left_only_columns)

        except Exception as e:
            self._handle_error(e, {"operation": "left_only_records"})

    def find_right_only_records(
        self, *, left: pl.LazyFrame, right: pl.LazyFrame, config: ComparisonConfig
    ) -> pl.LazyFrame:
        """Find records that exist only in right DataFrame.

        Args:
            left: Prepared left LazyFrame.
            right: Prepared right LazyFrame.
            config: Comparison configuration.

        Returns:
            LazyFrame containing right-only records with left columns dropped.
        """
        try:
            # Get primary key columns with PK_ prefix
            pk_columns = [f"{PRIMARY_KEY_PREFIX}{pk}" for pk in config.pk_columns]

            # Join and filter for right-only records (anti-join)
            right_only_joined = right.join(left.select(pk_columns), on=pk_columns, how="anti")

            # Select only primary key columns and right columns (drop left columns which are null)
            right_only_columns = self.preparation_service.get_right_only_column_order(config=config)

            return right_only_joined.select(right_only_columns)

        except Exception as e:
            self._handle_error(e, {"operation": "right_only_records"})
