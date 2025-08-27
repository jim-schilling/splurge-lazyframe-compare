"""Schema definitions and validation for the comparison framework."""

from dataclasses import dataclass

import polars as pl

from splurge_lazyframe_compare.exceptions.comparison_exceptions import SchemaValidationError

# Private constants
_MISSING_COLUMNS_MSG = "Missing columns: {}"
_WRONG_DTYPE_MSG = "Column {}: expected {}, got {}"
_NULL_VIOLATION_MSG = "Column {}: {} null values found but column defined as non-nullable"
_PK_NOT_DEFINED_MSG = "Primary key column '{}' not defined in schema"
_EMPTY_LEFT_SCHEMA_MSG = "Left schema has no columns defined"
_EMPTY_RIGHT_SCHEMA_MSG = "Right schema has no columns defined"
_NO_PK_MSG = "No primary key columns defined"
_MISSING_LEFT_MAPPED_MSG = "Left schema missing mapped columns: {}"
_MISSING_RIGHT_MAPPED_MSG = "Right schema missing mapped columns: {}"
_PK_NOT_MAPPED_MSG = "Primary key column '{}' not found in column mappings"


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

    columns: dict[str, ColumnDefinition]
    primary_key_columns: list[str]

    def validate_schema(self, df: pl.LazyFrame) -> list[str]:
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
            errors.append(_MISSING_COLUMNS_MSG.format(missing_columns))

        # Check data types for existing columns
        for col_name, col_def in self.columns.items():
            if col_name in df_columns:
                actual_dtype = df_dtypes[df_column_names.index(col_name)]
                # Allow Null dtype for empty DataFrames
                if actual_dtype != col_def.polars_dtype and actual_dtype != pl.Null:
                    errors.append(
                        _WRONG_DTYPE_MSG.format(col_name, col_def.polars_dtype, actual_dtype)
                    )

        # Validate nullable constraints
        for col_name, col_def in self.columns.items():
            if col_name in df_columns and not col_def.nullable:
                null_count = df.select(pl.col(col_name).is_null().sum()).collect().item()
                if null_count > 0:
                    errors.append(_NULL_VIOLATION_MSG.format(col_name, null_count))

        # Validate primary key columns exist
        for pk_col in self.primary_key_columns:
            if pk_col not in self.columns:
                errors.append(_PK_NOT_DEFINED_MSG.format(pk_col))

        return errors

    def get_primary_key_definition(self) -> list[ColumnDefinition]:
        """Get column definitions for primary key columns.

        Returns:
            List of ColumnDefinition objects for primary key columns.
        """
        return [self.columns[col] for col in self.primary_key_columns]

    def get_compare_columns(self) -> list[str]:
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
    column_mappings: list[ColumnMapping]
    primary_key_columns: list[str]
    ignore_case: bool = False
    null_equals_null: bool = True
    tolerance: dict[str, float] | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate comparison configuration."""
        errors = []

        # Validate schemas
        if not self.left_schema.columns:
            errors.append(_EMPTY_LEFT_SCHEMA_MSG)

        if not self.right_schema.columns:
            errors.append(_EMPTY_RIGHT_SCHEMA_MSG)

        # Validate primary key columns
        if not self.primary_key_columns:
            errors.append(_NO_PK_MSG)

        # Validate column mappings
        left_mapped_columns = {mapping.left_column for mapping in self.column_mappings}
        right_mapped_columns = {mapping.right_column for mapping in self.column_mappings}

        # Check that mapped columns exist in schemas
        left_schema_columns = set(self.left_schema.columns.keys())
        right_schema_columns = set(self.right_schema.columns.keys())

        missing_left = left_mapped_columns - left_schema_columns
        missing_right = right_mapped_columns - right_schema_columns

        if missing_left:
            errors.append(_MISSING_LEFT_MAPPED_MSG.format(missing_left))

        if missing_right:
            errors.append(_MISSING_RIGHT_MAPPED_MSG.format(missing_right))

        # Validate primary key columns are mapped
        for pk_col in self.primary_key_columns:
            if not any(mapping.comparison_name == pk_col for mapping in self.column_mappings):
                errors.append(_PK_NOT_MAPPED_MSG.format(pk_col))

        if errors:
            raise SchemaValidationError(
                "Configuration validation failed", validation_errors=errors
            )
