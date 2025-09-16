"""Data preparation service for the comparison framework.

Copyright (c) 2025 Jim Schilling.
Licensed under the MIT License. See the LICENSE file for details.
"""

DOMAINS: list[str] = ["services", "data_preparation", "transformation"]


import polars as pl

from splurge_lazyframe_compare.models.schema import ComparisonConfig
from splurge_lazyframe_compare.services.base_service import BaseService
from splurge_lazyframe_compare.utils.constants import (
    LEFT_PREFIX,
    PRIMARY_KEY_PREFIX,
    RIGHT_PREFIX,
)


class DataPreparationService(BaseService):
    """Data preparation service for LazyFrames.

    This service handles the preparation and mapping of DataFrames
    for comparison, including column mapping, case sensitivity,
    and data standardization.
    """

    def __init__(self) -> None:
        """Initialize the data preparation service."""
        super().__init__("DataPreparationService")

    def _validate_inputs(self, **kwargs) -> None:
        """Validate service inputs."""
        # Implementation will be added as needed
        pass

    def prepare_dataframes(
        self,
        *,
        left: pl.LazyFrame,
        right: pl.LazyFrame,
        config: ComparisonConfig
    ) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """Prepare and validate DataFrames for comparison.

        Args:
            left: Original left LazyFrame.
            right: Original right LazyFrame.
            config: Comparison configuration.

        Returns:
            Tuple of (prepared_left, prepared_right) LazyFrames.
        """
        try:
            # Apply column mappings and create standardized column names
            prepared_left, prepared_right = self.apply_column_mappings(
                left=left, right=right, config=config
            )

            # Apply case sensitivity settings
            if config.ignore_case:
                prepared_left = self.apply_case_insensitive(prepared_left)
                prepared_right = self.apply_case_insensitive(prepared_right)

            return prepared_left, prepared_right

        except Exception as e:
            self._handle_error(e, {"operation": "dataframe_preparation"})

    def apply_column_mappings(
        self,
        *,
        left: pl.LazyFrame,
        right: pl.LazyFrame,
        config: ComparisonConfig
    ) -> tuple[pl.LazyFrame, pl.LazyFrame]:
        """Apply column mappings and create standardized column names.

        Args:
            left: Original left LazyFrame.
            right: Original right LazyFrame.
            config: Comparison configuration.

        Returns:
            Tuple of (mapped_left, mapped_right) LazyFrames with standardized column names.
        """
        try:
            # Create mapping dictionaries with appropriate prefixes
            left_mapping = {}
            right_mapping = {}

            for mapping in config.column_mappings:
                if mapping.name in config.pk_columns:
                    # Primary key columns get PK_ prefix
                    left_mapping[mapping.left] = f"{PRIMARY_KEY_PREFIX}{mapping.name}"
                    right_mapping[mapping.right] = f"{PRIMARY_KEY_PREFIX}{mapping.name}"
                else:
                    # Non-primary key columns get L_ and R_ prefixes
                    left_mapping[mapping.left] = f"{LEFT_PREFIX}{mapping.name}"
                    right_mapping[mapping.right] = f"{RIGHT_PREFIX}{mapping.name}"

            # Apply mappings
            mapped_left = left.rename(left_mapping)
            mapped_right = right.rename(right_mapping)

            # Select only the mapped columns
            left_columns = []
            right_columns = []

            for mapping in config.column_mappings:
                if mapping.name in config.pk_columns:
                    left_columns.append(f"{PRIMARY_KEY_PREFIX}{mapping.name}")
                    right_columns.append(f"{PRIMARY_KEY_PREFIX}{mapping.name}")
                else:
                    left_columns.append(f"{LEFT_PREFIX}{mapping.name}")
                    right_columns.append(f"{RIGHT_PREFIX}{mapping.name}")

            mapped_left = mapped_left.select(left_columns)
            mapped_right = mapped_right.select(right_columns)

            return mapped_left, mapped_right

        except Exception as e:
            self._handle_error(e, {"operation": "column_mapping"})

    def apply_case_insensitive(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Apply case-insensitive transformations to string columns.

        Args:
            df: LazyFrame to transform.

        Returns:
            LazyFrame with case-insensitive transformations applied.
        """
        try:
            # Get string columns
            string_columns = []
            schema = df.collect_schema()
            for col in schema.names():
                if schema.dtypes()[schema.names().index(col)] == pl.Utf8:
                    string_columns.append(col)

            # Apply lowercase transformation to string columns
            if string_columns:
                transformations = [pl.col(col).str.to_lowercase().alias(col) for col in string_columns]
                return df.with_columns(transformations)

            return df

        except Exception as e:
            self._handle_error(e, {"operation": "case_insensitive_transformation"})

    def get_alternating_column_order(self, *, config: ComparisonConfig) -> list[str]:
        """Get column order with alternating Left/Right columns for non-primary key columns.

        Args:
            config: Comparison configuration.

        Returns:
            List of column names in the desired order: primary keys first, then alternating Left/Right.
        """
        # Start with primary key columns
        column_order = [f"{PRIMARY_KEY_PREFIX}{pk}" for pk in config.pk_columns]

        # Add non-primary key columns in alternating Left/Right order
        non_pk_mappings = [
            mapping for mapping in config.column_mappings
            if mapping.name not in config.pk_columns
        ]

        for mapping in non_pk_mappings:
            left_col = f"{LEFT_PREFIX}{mapping.name}"
            right_col = f"{RIGHT_PREFIX}{mapping.name}"
            column_order.extend([left_col, right_col])

        return column_order

    def get_left_only_column_order(self, *, config: ComparisonConfig) -> list[str]:
        """Get column order for left-only records (primary keys + left columns only).

        Args:
            config: Comparison configuration.

        Returns:
            List of column names in the desired order: primary keys first, then left columns.
        """
        # Start with primary key columns
        column_order = [f"{PRIMARY_KEY_PREFIX}{pk}" for pk in config.pk_columns]

        # Add only left columns for non-primary key columns
        non_pk_mappings = [
            mapping for mapping in config.column_mappings
            if mapping.name not in config.pk_columns
        ]

        for mapping in non_pk_mappings:
            left_col = f"{LEFT_PREFIX}{mapping.name}"
            column_order.append(left_col)

        return column_order

    def get_right_only_column_order(self, *, config: ComparisonConfig) -> list[str]:
        """Get column order for right-only records (primary keys + right columns only).

        Args:
            config: Comparison configuration.

        Returns:
            List of column names in the desired order: primary keys first, then right columns.
        """
        # Start with primary key columns
        column_order = [f"{PRIMARY_KEY_PREFIX}{pk}" for pk in config.pk_columns]

        # Add only right columns for non-primary key columns
        non_pk_mappings = [
            mapping for mapping in config.column_mappings
            if mapping.name not in config.pk_columns
        ]

        for mapping in non_pk_mappings:
            right_col = f"{RIGHT_PREFIX}{mapping.name}"
            column_order.append(right_col)

        return column_order
