"""Main comparison engine for Polars LazyFrames."""

from typing import Dict, List, Optional, Tuple
import polars as pl

from splurge_lazyframe_compare.core.schema import ComparisonConfig, ComparisonSchema
from splurge_lazyframe_compare.core.results import ComparisonResults, ComparisonSummary
from splurge_lazyframe_compare.exceptions.comparison_exceptions import (
    PrimaryKeyViolationError,
    SchemaValidationError,
)


class LazyFrameComparator:
    """Main comparison engine for Polars LazyFrames.

    This class provides the core functionality for comparing two Polars LazyFrames
    with configurable schemas, primary keys, and column mappings.
    """

    def __init__(self, config: ComparisonConfig) -> None:
        """Initialize the comparator with configuration.

        Args:
            config: Comparison configuration defining schemas and mappings.
        """
        self.config = config
        self._validate_config()

    def compare(
        self, *, left: pl.LazyFrame, right: pl.LazyFrame
    ) -> ComparisonResults:
        """Execute complete comparison between two LazyFrames.

        Args:
            left: Left LazyFrame to compare.
            right: Right LazyFrame to compare.

        Returns:
            ComparisonResults containing all comparison results.

        Raises:
            SchemaValidationError: If DataFrames don't match their schemas.
            PrimaryKeyViolationError: If primary key constraints are violated.
        """
        # Validate DataFrames against schemas
        self._validate_dataframes(left=left, right=right)

        # Validate primary key uniqueness
        self._validate_primary_key_uniqueness(df=left, schema_name="left DataFrame")
        self._validate_primary_key_uniqueness(df=right, schema_name="right DataFrame")

        # Prepare DataFrames for comparison
        prepared_left, prepared_right = self._prepare_dataframes(left=left, right=right)

        # Get record counts
        total_left_records = prepared_left.select(pl.len()).collect().item()
        total_right_records = prepared_right.select(pl.len()).collect().item()

        # Execute comparison patterns
        value_differences = self._find_value_differences(left=prepared_left, right=prepared_right)
        left_only_records = self._find_left_only_records(left=prepared_left, right=prepared_right)
        right_only_records = self._find_right_only_records(left=prepared_left, right=prepared_right)

        # Create summary
        summary = ComparisonSummary.create(
            total_left_records=total_left_records,
            total_right_records=total_right_records,
            value_differences=value_differences,
            left_only_records=left_only_records,
            right_only_records=right_only_records,
        )

        return ComparisonResults(
            summary=summary,
            value_differences=value_differences,
            left_only_records=left_only_records,
            right_only_records=right_only_records,
            config=self.config,
        )

    def _find_value_differences(
        self, *, left: pl.LazyFrame, right: pl.LazyFrame
    ) -> pl.LazyFrame:
        """Find records with same keys but different values.

        Args:
            left: Prepared left LazyFrame.
            right: Prepared right LazyFrame.

        Returns:
            LazyFrame containing records with value differences.
        """
        # Get primary key columns with PK_ prefix
        pk_columns = [f"PK_{pk}" for pk in self.config.primary_key_columns]
        
        # Join on primary key columns
        joined = left.join(
            right, on=pk_columns, how="inner"
        )

        # Create difference conditions for each mapped column
        diff_conditions = []
        for mapping in self.config.column_mappings:
            if mapping.comparison_name in self.config.primary_key_columns:
                # Primary key columns use PK_ prefix
                left_col = f"PK_{mapping.comparison_name}"
                right_col = left_col  # Same column since they're joined on PK
            else:
                # Non-primary key columns use L_ and R_ prefixes
                left_col = f"L_{mapping.comparison_name}"
                right_col = f"R_{mapping.comparison_name}"

            # Handle null comparisons based on config
            if self.config.null_equals_null:
                condition = ~pl.col(left_col).eq_missing(pl.col(right_col))
            else:
                condition = pl.col(left_col) != pl.col(right_col)

            # Apply tolerance for numeric columns if specified
            if (
                self.config.tolerance
                and mapping.comparison_name in self.config.tolerance
            ):
                tolerance = self.config.tolerance[mapping.comparison_name]
                condition = (pl.col(left_col) - pl.col(right_col)).abs() > tolerance

            diff_conditions.append(condition)

        # Filter rows with any differences
        if diff_conditions:
            return joined.filter(pl.any_horizontal(diff_conditions))
        else:
            # No columns to compare, return empty result
            return joined.limit(0)

    def _find_left_only_records(
        self, *, left: pl.LazyFrame, right: pl.LazyFrame
    ) -> pl.LazyFrame:
        """Find records that exist only in left DataFrame.

        Args:
            left: Prepared left LazyFrame.
            right: Prepared right LazyFrame.

        Returns:
            LazyFrame containing left-only records.
        """
        # Get primary key columns with PK_ prefix
        pk_columns = [f"PK_{pk}" for pk in self.config.primary_key_columns]
        
        # For left-only records, we need to check if any non-primary key column from right is null
        # This indicates no match was found in the right DataFrame
        non_pk_columns = [f"R_{mapping.comparison_name}" for mapping in self.config.column_mappings if mapping.comparison_name not in self.config.primary_key_columns]
        return left.join(
            right, on=pk_columns, how="left"
        ).filter(pl.all_horizontal([pl.col(col).is_null() for col in non_pk_columns]))

    def _find_right_only_records(
        self, *, left: pl.LazyFrame, right: pl.LazyFrame
    ) -> pl.LazyFrame:
        """Find records that exist only in right DataFrame.

        Args:
            left: Prepared left LazyFrame.
            right: Prepared right LazyFrame.

        Returns:
            LazyFrame containing right-only records.
        """
        # Get primary key columns with PK_ prefix
        pk_columns = [f"PK_{pk}" for pk in self.config.primary_key_columns]
        
        # For right-only records, we need to check if any non-primary key column from left is null
        # This indicates no match was found in the left DataFrame
        non_pk_columns = [f"L_{mapping.comparison_name}" for mapping in self.config.column_mappings if mapping.comparison_name not in self.config.primary_key_columns]
        return right.join(
            left, on=pk_columns, how="left"
        ).filter(pl.all_horizontal([pl.col(col).is_null() for col in non_pk_columns]))

    def _prepare_dataframes(
        self, *, left: pl.LazyFrame, right: pl.LazyFrame
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """Prepare and validate DataFrames for comparison.

        Args:
            left: Original left LazyFrame.
            right: Original right LazyFrame.

        Returns:
            Tuple of (prepared_left, prepared_right) LazyFrames.
        """
        # Apply column mappings and create standardized column names
        prepared_left, prepared_right = self._apply_column_mappings(left=left, right=right)

        # Apply case sensitivity settings
        if self.config.ignore_case:
            prepared_left = self._apply_case_insensitive(prepared_left)
            prepared_right = self._apply_case_insensitive(prepared_right)

        return prepared_left, prepared_right

    def _apply_column_mappings(
        self, *, left: pl.LazyFrame, right: pl.LazyFrame
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """Apply column mappings and create standardized column names.

        Args:
            left: Original left LazyFrame.
            right: Original right LazyFrame.

        Returns:
            Tuple of (mapped_left, mapped_right) LazyFrames with standardized column names.
        """
        # Create mapping dictionaries with appropriate prefixes
        left_mapping = {}
        right_mapping = {}
        
        for mapping in self.config.column_mappings:
            if mapping.comparison_name in self.config.primary_key_columns:
                # Primary key columns get PK_ prefix
                left_mapping[mapping.left_column] = f"PK_{mapping.comparison_name}"
                right_mapping[mapping.right_column] = f"PK_{mapping.comparison_name}"
            else:
                # Non-primary key columns get L_ and R_ prefixes
                left_mapping[mapping.left_column] = f"L_{mapping.comparison_name}"
                right_mapping[mapping.right_column] = f"R_{mapping.comparison_name}"

        # Apply mappings
        mapped_left = left.rename(left_mapping)
        mapped_right = right.rename(right_mapping)

        # Select only the mapped columns
        left_columns = []
        right_columns = []
        
        for mapping in self.config.column_mappings:
            if mapping.comparison_name in self.config.primary_key_columns:
                left_columns.append(f"PK_{mapping.comparison_name}")
                right_columns.append(f"PK_{mapping.comparison_name}")
            else:
                left_columns.append(f"L_{mapping.comparison_name}")
                right_columns.append(f"R_{mapping.comparison_name}")
                
        mapped_left = mapped_left.select(left_columns)
        mapped_right = mapped_right.select(right_columns)

        return mapped_left, mapped_right

    def _apply_case_insensitive(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Apply case-insensitive transformations to string columns.

        Args:
            df: LazyFrame to transform.

        Returns:
            LazyFrame with case-insensitive transformations applied.
        """
        # Get string columns
        string_columns = []
        for col in df.columns:
            if df.select(pl.col(col)).dtypes[0] == pl.Utf8:
                string_columns.append(col)

        # Apply lowercase transformation to string columns
        if string_columns:
            transformations = [pl.col(col).str.to_lowercase().alias(col) for col in string_columns]
            return df.with_columns(transformations)

        return df

    def _validate_dataframes(
        self, *, left: pl.LazyFrame, right: pl.LazyFrame
    ) -> None:
        """Validate DataFrames against their schemas.

        Args:
            left: Left LazyFrame to validate.
            right: Right LazyFrame to validate.

        Raises:
            SchemaValidationError: If validation fails.
        """
        # Validate left DataFrame
        left_errors = self.config.left_schema.validate_schema(left)
        if left_errors:
            raise SchemaValidationError(
                "Left DataFrame validation failed", validation_errors=left_errors
            )

        # Validate right DataFrame
        right_errors = self.config.right_schema.validate_schema(right)
        if right_errors:
            raise SchemaValidationError(
                "Right DataFrame validation failed", validation_errors=right_errors
            )

    def _validate_primary_key_uniqueness(
        self, *, df: pl.LazyFrame, schema_name: str
    ) -> None:
        """Validate that primary key columns are unique.

        Args:
            df: LazyFrame to validate.
            schema_name: Name of the schema for error reporting.

        Raises:
            PrimaryKeyViolationError: If primary key constraints are violated.
        """
        # Get the actual primary key columns for this DataFrame
        if schema_name == "left DataFrame":
            pk_columns = [
                mapping.left_column
                for mapping in self.config.column_mappings
                if mapping.comparison_name in self.config.primary_key_columns
            ]
        else:
            pk_columns = [
                mapping.right_column
                for mapping in self.config.column_mappings
                if mapping.comparison_name in self.config.primary_key_columns
            ]

        # Check for duplicates
        duplicates = (
            df.group_by(pk_columns)
            .len()
            .filter(pl.col("len") > 1)
        )

        duplicate_count = duplicates.select(pl.len()).collect().item()

        if duplicate_count > 0:
            raise PrimaryKeyViolationError(
                f"Duplicate primary keys found in {schema_name}: {duplicate_count} duplicates"
            )

    def _validate_config(self) -> None:
        """Validate comparison configuration.

        This method is called during initialization to ensure the configuration
        is valid before any comparisons are performed.
        """
        # Configuration validation is handled in ComparisonConfig.__post_init__
        # This method is kept for potential future validation logic
        pass
