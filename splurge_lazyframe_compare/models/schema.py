"""Schema definition models for the comparison framework."""

from dataclasses import dataclass

import polars as pl

from splurge_lazyframe_compare.exceptions.comparison_exceptions import SchemaValidationError
from splurge_lazyframe_compare.utils.type_helpers import (
    get_polars_datatype_name,
    get_polars_datatype_type,
)


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
    DUPLICATE_COLUMN_NAMES_MSG: str = "Duplicate column names found: {}"
    DUPLICATE_COLUMN_ALIASES_MSG: str = "Duplicate column aliases found: {}"
    RESERVED_COLUMN_NAME_MSG: str = "Column name '{}' is reserved and cannot be used"
    INCOMPATIBLE_NULLABILITY_MSG: str = "Column '{}' has incompatible nullability with data type {}"
    PK_DUPLICATES_MSG: str = "Primary key columns contain duplicates: {}"
    PK_NULL_VALUES_MSG: str = "Primary key column '{}' contains null values but is defined as non-nullable"
    EMPTY_DATAFRAME_MSG: str = "DataFrame is empty, cannot validate against schema"


@dataclass(kw_only=True)
class ColumnDefinition:
    """Defines a column with metadata for comparison.

    Attributes:
        name: The actual column name in the DataFrame.
        alias: Human-readable name for the column.
        datatype: Expected Polars datatype or datatype-name for the column.
        nullable: Whether the column can contain null values.
    """

    name: str
    alias: str
    datatype: pl.DataType | str
    nullable: bool = SchemaConstants.DEFAULT_NULLABLE

    def __post_init__(self) -> None:
        """Validate that all attributes are valid."""
        if not self.name or not self.name.strip():
            raise ValueError("ColumnDefinition.name cannot be None, empty, or whitespace-only")
        if not self.alias or not self.alias.strip():
            raise ValueError("ColumnDefinition.alias cannot be None, empty, or whitespace-only")

        # Check for reserved column names
        reserved_names = {"_polars_reserved", "index", "__index_level_0__"}
        if self.name.lower() in reserved_names:
            raise ValueError(SchemaConstants.RESERVED_COLUMN_NAME_MSG.format(self.name))

        # Check for problematic characters in column names
        problematic_chars = {".", "[", "]", "(", ")", "{", "}", "*", "+", "?", "^", "$", "|", "\\"}
        if any(char in self.name for char in problematic_chars):
            raise ValueError(
                f"ColumnDefinition.name '{self.name}' contains problematic characters "
                "that may cause issues with regex or other operations"
            )

        if isinstance(self.datatype, str):
            self.datatype = get_polars_datatype_type(self.datatype)

        # Validate nullability compatibility with data type
        self._validate_nullability_compatibility()

        # Check for unparameterized complex types and provide helpful error messages
        try:
            type_name = get_polars_datatype_name(self.datatype)
            if type_name == 'List':
                # Check if this is an unparameterized List by seeing if it has an inner type
                if not hasattr(self.datatype, 'inner'):
                    raise ValueError(
                        f"ColumnDefinition.datatype cannot be unparameterized {type_name}. "
                        f"Use pl.List(inner_type) instead, e.g., pl.List(pl.Utf8) for a list of strings."
                    )
            elif type_name == 'Struct':
                # Check if this is an unparameterized Struct by seeing if it has fields
                if not hasattr(self.datatype, 'fields') or self.datatype.fields is None:
                    raise ValueError(
                        f"ColumnDefinition.datatype cannot be unparameterized {type_name}. "
                        f"Use pl.Struct(fields) instead, e.g., pl.Struct([]) for an empty struct "
                        f"or pl.Struct({{'field': pl.Utf8}}) for a struct with fields."
                    )
        except (TypeError, AttributeError):
            # If we can't get the type name, skip this validation
            pass

        # Check if datatype is a valid polars data type
        try:
            # Check if it's a valid polars data type by trying to create a schema
            pl.Schema({"test": self.datatype})
        except Exception as e:
            # Provide more specific error message for complex types
            try:
                type_name = get_polars_datatype_name(self.datatype)
                if type_name in ['List', 'Struct']:
                    # Check if this is actually parameterized
                    if type_name == 'List' and hasattr(self.datatype, 'inner'):
                        # This is a parameterized List, so the error is not about parameterization
                        pass
                    elif (
                        type_name == 'Struct'
                        and hasattr(self.datatype, 'fields')
                        and self.datatype.fields is not None
                    ):
                        # This is a parameterized Struct, so the error is not about parameterization
                        pass
                    else:
                        # This is truly unparameterized
                        raise ValueError(
                            f"ColumnDefinition.datatype {type_name} requires parameters. "
                            f"Use proper instantiation: pl.{type_name}(...) instead of pl.{type_name}"
                        ) from e
            except (TypeError, AttributeError):
                # If we can't get the type name, fall back to generic error
                pass
            raise ValueError("ColumnDefinition.datatype must be a valid polars data type") from e

    def _validate_nullability_compatibility(self) -> None:
        """Validate that nullability setting is compatible with the data type."""
        # Some data types inherently cannot be null
        if not self.nullable:
            # For now, we allow all data types to be non-nullable since Polars handles this at runtime
            # This could be extended to check for data types that are inherently nullable
            pass

    def validate_column_exists(self, df: pl.LazyFrame) -> bool:
        """Check if column exists in DataFrame.

        Args:
            df: LazyFrame to check for column existence.

        Returns:
            True if column exists, False otherwise.
        """
        return self.name in df.collect_schema().names()

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
        col_index = schema.names().index(self.name)
        actual_dtype = schema.dtypes()[col_index]
        return actual_dtype == self.datatype


@dataclass(kw_only=True)
class ColumnMapping:
    """Maps columns between left and right DataFrames.

    Attributes:
        name: Standardized name for comparison.
        left: Column name in the left DataFrame.
        right: Column name in the right DataFrame.
    """

    name: str
    left: str
    right: str

    def __post_init__(self) -> None:
        """Validate that all attributes are non-empty and non-whitespace."""
        if not self.name or not self.name.strip():
            raise ValueError("ColumnMapping.name cannot be None, empty, or whitespace-only")
        if not self.left or not self.left.strip():
            raise ValueError("ColumnMapping.left cannot be None, empty, or whitespace-only")
        if not self.right or not self.right.strip():
            raise ValueError("ColumnMapping.right cannot be None, empty, or whitespace-only")


@dataclass(kw_only=True)
class ComparisonSchema:
    """Schema definition for a LazyFrame in comparison.

    Attributes:
        columns: Dictionary mapping column names to their definitions.
        pk_columns: List of column names that form the primary key.
    """

    columns: dict[str, ColumnDefinition]
    pk_columns: list[str]

    def validate_schema(
        self,
        df: pl.LazyFrame
    ) -> list[str]:
        """Validate DataFrame against schema, return validation errors.

        Args:
            df: LazyFrame to validate against the schema.

        Returns:
            List of validation error messages. Empty list if valid.
        """
        errors = []

        # Check if DataFrame is empty - allow empty DataFrames to pass basic validation
        try:
            row_count = df.select(pl.len()).collect().item()
            if row_count == 0:
                # For empty DataFrames, we can only validate schema structure, not data
                errors.extend(self._validate_schema_integrity())
                return errors
        except Exception:
            # If we can't get row count, continue with validation
            pass

        # Collect schema once to avoid multiple expensive operations
        df_schema = df.collect_schema()
        df_column_names = df_schema.names()
        df_dtypes = df_schema.dtypes()
        df_columns = set(df_column_names)
        schema_columns = set(self.columns.keys())

        # Validate schema integrity first
        schema_errors = self._validate_schema_integrity()
        errors.extend(schema_errors)

        # Early return for empty DataFrames to avoid further validation issues
        if errors and any("empty" in error.lower() for error in errors):
            return errors

        missing_columns = schema_columns - df_columns

        if missing_columns:
            errors.append(SchemaConstants.MISSING_COLUMNS_MSG.format(missing_columns))

        # Check data types for existing columns
        for col_name, col_def in self.columns.items():
            if col_name in df_columns:
                actual_dtype = df_dtypes[df_column_names.index(col_name)]
                # Allow Null dtype for empty DataFrames
                if actual_dtype != col_def.datatype and actual_dtype != pl.Null:
                    errors.append(
                        SchemaConstants.WRONG_DTYPE_MSG.format(col_name, col_def.datatype, actual_dtype)
                    )

        # Validate nullable constraints - optimized to collect once
        non_nullable_columns = [
            col_name for col_name, col_def in self.columns.items()
            if col_name in df_columns and not col_def.nullable
        ]

        if non_nullable_columns:
            # Collect null counts for all non-nullable columns in one operation
            null_counts_df = df.select([
                pl.col(col).is_null().sum().alias(col)
                for col in non_nullable_columns
            ]).collect()

            # Check each column for null violations
            for col_name in non_nullable_columns:
                null_count = null_counts_df[col_name][0]
                if null_count > 0:
                    errors.append(SchemaConstants.NULL_VIOLATION_MSG.format(col_name, null_count))

        # Note: Primary key validation is now handled in _validate_schema_integrity()
        # to avoid duplication

        return errors

    def get_primary_key_definition(self) -> list[ColumnDefinition]:
        """Get column definitions for primary key columns.

        Returns:
            List of ColumnDefinition objects for primary key columns.
        """
        return [self.columns[col] for col in self.pk_columns]

    def get_compare_columns(self) -> list[str]:
        """Get non-primary-key columns for comparison.

        Returns:
            List of column names that are not part of the primary key.
        """
        return [col for col in self.columns.keys() if col not in self.pk_columns]

    def _validate_schema_integrity(self) -> list[str]:
        """Validate the internal consistency of the schema itself.

        Returns:
            List of validation error messages. Empty list if schema is internally consistent.
        """
        errors = []

        # Check for empty schema
        if not self.columns:
            errors.append(SchemaConstants.EMPTY_SCHEMA_MSG)
            return errors

        # Check for duplicate column names
        column_names = list(self.columns.keys())
        duplicate_names = {name for name in column_names if column_names.count(name) > 1}
        if duplicate_names:
            errors.append(SchemaConstants.DUPLICATE_COLUMN_NAMES_MSG.format(duplicate_names))

        # Check for duplicate column aliases
        aliases = [col_def.alias for col_def in self.columns.values()]
        duplicate_aliases = {alias for alias in aliases if aliases.count(alias) > 1}
        if duplicate_aliases:
            errors.append(SchemaConstants.DUPLICATE_COLUMN_ALIASES_MSG.format(duplicate_aliases))

        # Check primary key columns
        if not self.pk_columns:
            errors.append(SchemaConstants.NO_PK_MSG)

        # Validate primary key columns exist in schema (data validation happens in main method)
        for pk_col in self.pk_columns:
            if pk_col not in self.columns:
                errors.append(SchemaConstants.PK_NOT_DEFINED_MSG.format(pk_col))

        return errors


@dataclass(kw_only=True)
class ComparisonConfig:
    """Configuration for comparing two LazyFrames.

    Attributes:
        left_schema: Schema definition for the left DataFrame.
        right_schema: Schema definition for the right DataFrame.
        column_mappings: List of column mappings between datasets.
        pk_columns: List of primary key column names (standardized).
        ignore_case: Whether to ignore case in string comparisons.
        null_equals_null: Whether null values should be considered equal.
        tolerance: Dictionary mapping column names to tolerance values for numeric comparisons.
    """

    left_schema: ComparisonSchema
    right_schema: ComparisonSchema
    column_mappings: list[ColumnMapping]
    pk_columns: list[str]
    ignore_case: bool = SchemaConstants.DEFAULT_IGNORE_CASE
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
            errors.append(SchemaConstants.EMPTY_SCHEMA_MSG)

        if not self.right_schema.columns:
            errors.append(SchemaConstants.EMPTY_SCHEMA_MSG)

        # Validate primary key columns
        if not self.pk_columns:
            errors.append(SchemaConstants.NO_PK_MSG)

        # Validate column mappings
        left_mapped_columns = {mapping.left for mapping in self.column_mappings}
        right_mapped_columns = {mapping.right for mapping in self.column_mappings}

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
        for pk_col in self.pk_columns:
            if not any(mapping.name == pk_col for mapping in self.column_mappings):
                errors.append(f"Primary key column '{pk_col}' not found in column mappings")

        # Validate tolerance settings
        if self.tolerance:
            for col_name, tolerance_val in self.tolerance.items():
                # Check that tolerance column exists in mappings
                if not any(mapping.name == col_name for mapping in self.column_mappings):
                    errors.append(f"Tolerance specified for unmapped column '{col_name}'")

                # Check that tolerance value is reasonable
                if tolerance_val < 0:
                    errors.append(f"Tolerance for column '{col_name}' cannot be negative: {tolerance_val}")

                # Check that tolerance makes sense for the data type
                if col_name in self.left_schema.columns:
                    left_dtype = self.left_schema.columns[col_name].datatype
                    if not self._is_numeric_dtype(left_dtype):
                        errors.append(f"Tolerance specified for non-numeric column '{col_name}' with type {left_dtype}")

        # Validate that column mappings are unique
        mapping_names = [mapping.name for mapping in self.column_mappings]
        if len(mapping_names) != len(set(mapping_names)):
            duplicates = {name for name in mapping_names if mapping_names.count(name) > 1}
            errors.append(f"Duplicate column mapping names found: {duplicates}")

        # Validate that left and right column references are unique within their respective schemas
        left_columns = [mapping.left for mapping in self.column_mappings]
        right_columns = [mapping.right for mapping in self.column_mappings]

        if len(left_columns) != len(set(left_columns)):
            duplicates = {col for col in left_columns if left_columns.count(col) > 1}
            errors.append(f"Left schema has duplicate column references: {duplicates}")

        if len(right_columns) != len(set(right_columns)):
            duplicates = {col for col in right_columns if right_columns.count(col) > 1}
            errors.append(f"Right schema has duplicate column references: {duplicates}")

        if errors:
            raise SchemaValidationError(
                "Configuration validation failed", validation_errors=errors
            )

    def _is_numeric_dtype(self, dtype: pl.DataType) -> bool:
        """Check if a Polars data type is numeric and supports tolerance comparisons.

        Args:
            dtype: Polars data type to check

        Returns:
            True if the data type supports numeric tolerance comparisons
        """
        numeric_types = {
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
            pl.Decimal
        }
        return dtype in numeric_types
