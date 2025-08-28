"""Schema definition models for the comparison framework."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import polars as pl

from splurge_lazyframe_compare.exceptions.comparison_exceptions import SchemaValidationError


@dataclass
class SchemaConstants:
    """Constants for schema operations."""

    # Default values
    DEFAULT_NULLABLE: bool = False
    DEFAULT_IGNORE_CASE: bool = False

    # Validation messages
    EMPTY_SCHEMA_MSG: str = "Schema has no columns defined"
    MISSING_COLUMNS_MSG: str = "Missing columns: {}"
    WRONG_DTYPE_MSG: str = "Column {}: expected {}, got {}"
    NULL_VIOLATION_MSG: str = "Column {}: {} null values found but column defined as non-nullable"
    PK_NOT_DEFINED_MSG: str = "Primary key column '{}' not defined in schema"
    NO_PK_MSG: str = "No primary key columns defined"


@dataclass
class ColumnDefinition:
    """Defines a column with metadata for comparison.

    Attributes:
        column_name: The actual column name in the DataFrame.
        friendly_name: Human-readable name for the column.
        polars_dtype: Expected Polars data type for the column.
        nullable: Whether the column can contain null values.
    """

    column_name: str
    friendly_name: str
    polars_dtype: pl.DataType
    nullable: bool = SchemaConstants.DEFAULT_NULLABLE

    def validate_column_exists(self, df: pl.LazyFrame) -> bool:
        """Check if column exists in DataFrame.

        Args:
            df: LazyFrame to check for column existence.

        Returns:
            True if column exists, False otherwise.
        """
        return self.column_name in df.collect_schema().names()

    def validate_data_type(self, df: pl.LazyFrame) -> bool:
        """Validate column data type matches definition.

        Args:
            df: LazyFrame to validate data type against.

        Returns:
            True if data type matches, False otherwise.
        """
        if not self.validate_column_exists(df):
            return False

        schema = df.collect_schema()
        col_index = schema.names().index(self.column_name)
        actual_dtype = schema.dtypes()[col_index]
        return actual_dtype == self.polars_dtype


@dataclass
class ColumnMapping:
    """Maps columns between left and right DataFrames.

    Attributes:
        left_column: Column name in the left DataFrame.
        right_column: Column name in the right DataFrame.
        comparison_name: Standardized name for comparison.
    """

    left_column: str
    right_column: str
    comparison_name: str


@dataclass
class ComparisonSchema:
    """Schema definition for a LazyFrame in comparison.

    Attributes:
        columns: Dictionary mapping column names to their definitions.
        primary_key_columns: List of column names that form the primary key.
    """

    columns: Dict[str, ColumnDefinition]
    primary_key_columns: List[str]

    def validate_schema(self, df: pl.LazyFrame) -> List[str]:
        """Validate DataFrame against schema, return validation errors.

        Args:
            df: LazyFrame to validate against the schema.

        Returns:
            List of validation error messages. Empty list if valid.
        """
        errors = []

        # Collect schema once to avoid multiple expensive operations
        df_schema = df.collect_schema()
        df_column_names = df_schema.names()
        df_dtypes = df_schema.dtypes()
        df_columns = set(df_column_names)
        schema_columns = set(self.columns.keys())

        missing_columns = schema_columns - df_columns

        if missing_columns:
            errors.append(SchemaConstants.MISSING_COLUMNS_MSG.format(missing_columns))

        # Check data types for existing columns
        for col_name, col_def in self.columns.items():
            if col_name in df_columns:
                actual_dtype = df_dtypes[df_column_names.index(col_name)]
                # Allow Null dtype for empty DataFrames
                if actual_dtype != col_def.polars_dtype and actual_dtype != pl.Null:
                    errors.append(
                        SchemaConstants.WRONG_DTYPE_MSG.format(col_name, col_def.polars_dtype, actual_dtype)
                    )

        # Validate nullable constraints
        for col_name, col_def in self.columns.items():
            if col_name in df_columns and not col_def.nullable:
                null_count = df.select(pl.col(col_name).is_null().sum()).collect().item()
                if null_count > 0:
                    errors.append(SchemaConstants.NULL_VIOLATION_MSG.format(col_name, null_count))

        # Validate primary key columns exist
        for pk_col in self.primary_key_columns:
            if pk_col not in self.columns:
                errors.append(SchemaConstants.PK_NOT_DEFINED_MSG.format(pk_col))

        return errors

    def get_primary_key_definition(self) -> List[ColumnDefinition]:
        """Get column definitions for primary key columns.

        Returns:
            List of ColumnDefinition objects for primary key columns.
        """
        return [self.columns[col] for col in self.primary_key_columns]

    def get_compare_columns(self) -> List[str]:
        """Get non-primary-key columns for comparison.

        Returns:
            List of column names that are not part of the primary key.
        """
        return [col for col in self.columns.keys() if col not in self.primary_key_columns]


@dataclass
class ComparisonConfig:
    """Configuration for comparing two LazyFrames.

    Attributes:
        left_schema: Schema definition for the left DataFrame.
        right_schema: Schema definition for the right DataFrame.
        column_mappings: List of column mappings between datasets.
        primary_key_columns: List of primary key column names (standardized).
        ignore_case: Whether to ignore case in string comparisons.
        null_equals_null: Whether null values should be considered equal.
        tolerance: Dictionary mapping column names to tolerance values for numeric comparisons.
    """

    left_schema: ComparisonSchema
    right_schema: ComparisonSchema
    column_mappings: List[ColumnMapping]
    primary_key_columns: List[str]
    ignore_case: bool = SchemaConstants.DEFAULT_IGNORE_CASE
    null_equals_null: bool = True
    tolerance: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate comparison configuration."""
        errors = []

        # Validate schemas
        if not self.left_schema.columns:
            errors.append(SchemaConstants.EMPTY_SCHEMA_MSG)

        if not self.right_schema.columns:
            errors.append(SchemaConstants.EMPTY_SCHEMA_MSG)

        # Validate primary key columns
        if not self.primary_key_columns:
            errors.append(SchemaConstants.NO_PK_MSG)

        # Validate column mappings
        left_mapped_columns = {mapping.left_column for mapping in self.column_mappings}
        right_mapped_columns = {mapping.right_column for mapping in self.column_mappings}

        # Check that mapped columns exist in schemas
        left_schema_columns = set(self.left_schema.columns.keys())
        right_schema_columns = set(self.right_schema.columns.keys())

        missing_left = left_mapped_columns - left_schema_columns
        missing_right = right_mapped_columns - right_schema_columns

        if missing_left:
            errors.append(f"Left schema missing mapped columns: {missing_left}")

        if missing_right:
            errors.append(f"Right schema missing mapped columns: {missing_right}")

        # Validate primary key columns are mapped
        for pk_col in self.primary_key_columns:
            if not any(mapping.comparison_name == pk_col for mapping in self.column_mappings):
                errors.append(f"Primary key column '{pk_col}' not found in column mappings")

        if errors:
            raise SchemaValidationError(
                "Configuration validation failed", validation_errors=errors
            )
