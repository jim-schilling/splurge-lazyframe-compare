"""Schema definitions and validation for the comparison framework."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import polars as pl

from ..exceptions.comparison_exceptions import SchemaValidationError


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
    nullable: bool

    def validate_column_exists(self, df: pl.LazyFrame) -> bool:
        """Check if column exists in DataFrame.

        Args:
            df: LazyFrame to check for column existence.

        Returns:
            True if column exists, False otherwise.
        """
        return self.column_name in df.columns

    def validate_data_type(self, df: pl.LazyFrame) -> bool:
        """Validate column data type matches definition.

        Args:
            df: LazyFrame to validate data type against.

        Returns:
            True if data type matches, False otherwise.
        """
        if not self.validate_column_exists(df):
            return False

        actual_dtype = df.select(pl.col(self.column_name)).dtypes[0]
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

        # Check column existence
        df_columns = set(df.columns)
        schema_columns = set(self.columns.keys())

        missing_columns = schema_columns - df_columns
        extra_columns = df_columns - schema_columns

        if missing_columns:
            errors.append(f"Missing columns: {missing_columns}")

        # Check data types for existing columns
        for col_name, col_def in self.columns.items():
            if col_name in df_columns:
                actual_dtype = df.select(pl.col(col_name)).dtypes[0]
                # Allow Null dtype for empty DataFrames
                if actual_dtype != col_def.polars_dtype and actual_dtype != pl.Null:
                    errors.append(
                        f"Column {col_name}: expected {col_def.polars_dtype}, "
                        f"got {actual_dtype}"
                    )

        # Validate nullable constraints
        for col_name, col_def in self.columns.items():
            if col_name in df_columns and not col_def.nullable:
                null_count = df.select(pl.col(col_name).is_null().sum()).collect().item()
                if null_count > 0:
                    errors.append(
                        f"Column {col_name}: {null_count} null values found "
                        f"but column defined as non-nullable"
                    )

        # Validate primary key columns exist
        for pk_col in self.primary_key_columns:
            if pk_col not in self.columns:
                errors.append(f"Primary key column '{pk_col}' not defined in schema")

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
    ignore_case: bool = False
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
            errors.append("Left schema has no columns defined")

        if not self.right_schema.columns:
            errors.append("Right schema has no columns defined")

        # Validate primary key columns
        if not self.primary_key_columns:
            errors.append("No primary key columns defined")

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
