"""Data validation service for the comparison framework."""


import polars as pl

from splurge_lazyframe_compare.exceptions.comparison_exceptions import (
    PrimaryKeyViolationError,
    SchemaValidationError,
)
from splurge_lazyframe_compare.models.schema import ComparisonConfig, ComparisonSchema
from splurge_lazyframe_compare.models.validation import ValidationResult
from splurge_lazyframe_compare.services.base_service import BaseService
from splurge_lazyframe_compare.utils.constants import (
    ALL_COLUMNS_PRESENT_MSG,
    ALL_DTYPES_CORRECT_MSG,
    ALL_PATTERNS_CORRECT_MSG,
    ALL_RANGES_CORRECT_MSG,
    ALL_UNIQUE_MSG,
    COMPLETENESS_FAILED_MSG,
    DTYPE_FAILED_MSG,
    DUPLICATE_PK_MSG,
    DUPLICATE_THRESHOLD,
    LEFT_DF_NAME,
    LEN_COLUMN,
    PATTERN_FAILED_MSG,
    RANGE_FAILED_MSG,
    UNIQUENESS_FAILED_MSG,
    ZERO_THRESHOLD,
)
from splurge_lazyframe_compare.utils.type_helpers import is_numeric_datatype


class ValidationService(BaseService):
    """Data validation service for LazyFrames.

    This service provides various data quality checks that can be performed
    on LazyFrames before comparison, including schema validation, primary key
    uniqueness checks, and data quality validations.
    """

    def __init__(self) -> None:
        """Initialize the validation service."""
        super().__init__("ValidationService")

    def _validate_inputs(self, **kwargs) -> None:
        """Validate service inputs."""
        # Implementation will be added as needed
        pass

    def validate_dataframe_schema(
        self,
        *,
        df: pl.LazyFrame,
        schema: ComparisonSchema,
        df_name: str,
    ) -> None:
        """Validate DataFrame against schema.

        Args:
            df: LazyFrame to validate.
            schema: Schema to validate against.
            df_name: Name of the DataFrame for error reporting.

        Raises:
            SchemaValidationError: If validation fails.
        """
        try:
            errors = schema.validate_schema(df)
            if errors:
                raise SchemaValidationError(
                    f"{df_name} validation failed", validation_errors=errors
                )
        except Exception as e:
            self._handle_error(e, {"operation": "schema_validation", "df_name": df_name})

    def validate_primary_key_uniqueness(
        self,
        *,
        df: pl.LazyFrame,
        config: ComparisonConfig,
        df_name: str
    ) -> None:
        """Validate that primary key columns are unique.

        Args:
            df: LazyFrame to validate.
            config: Comparison configuration.
            df_name: Name of the DataFrame for error reporting.

        Raises:
            PrimaryKeyViolationError: If primary key constraints are violated.
        """
        try:
            # Get the actual primary key columns for this DataFrame
            if df_name == LEFT_DF_NAME:
                pk_columns = [
                    mapping.left
                    for mapping in config.column_mappings
                    if mapping.name in config.pk_columns
                ]
            else:
                pk_columns = [
                    mapping.right
                    for mapping in config.column_mappings
                    if mapping.name in config.pk_columns
                ]

            # Check for duplicates
            duplicates = (
                df.group_by(pk_columns)
                .len()
                .filter(pl.col(LEN_COLUMN) > DUPLICATE_THRESHOLD)
            )

            duplicate_count = duplicates.select(pl.len()).collect().item()

            if duplicate_count > 0:
                raise PrimaryKeyViolationError(
                    DUPLICATE_PK_MSG.format(df_name, duplicate_count)
                )

        except Exception as e:
            self._handle_error(e, {"operation": "primary_key_validation", "df_name": df_name})

    def validate_completeness(
        self,
        *,
        df: pl.LazyFrame,
        required_columns: list[str]
    ) -> ValidationResult:
        """Validate that required columns are present and not entirely null.

        Args:
            df: LazyFrame to validate.
            required_columns: List of column names that must be present and non-null.

        Returns:
            ValidationResult indicating whether validation passed.
        """
        try:
            missing_columns = []
            null_columns = []

            # Check for missing columns
            schema = df.collect_schema()
            all_columns = schema.names()
            df_columns = set(all_columns)

            for col in required_columns:
                if col not in all_columns:
                    missing_columns.append(col)

            # Check for entirely null columns - optimized to collect once per column
            if required_columns:
                # Get total count once for all columns
                total_count = df.select(pl.len()).collect().item()

                # Check each required column for null percentage
                for col in required_columns:
                    if col in df_columns:
                        null_count = df.select(pl.col(col).is_null().sum()).collect().item()
                        if null_count == total_count:
                            null_columns.append(col)

            if missing_columns or null_columns:
                details = {
                    "missing_columns": missing_columns,
                    "null_columns": null_columns,
                }
                message = COMPLETENESS_FAILED_MSG.format(len(missing_columns), len(null_columns))
                return ValidationResult(is_valid=False, message=message, details=details)

            return ValidationResult(
                is_valid=True,
                message=ALL_COLUMNS_PRESENT_MSG,
            )

        except Exception as e:
            self._handle_error(e, {"operation": "completeness_validation"})

    def validate_data_types(
        self,
        *,
        df: pl.LazyFrame,
        expected_types: dict[str, pl.DataType]
    ) -> ValidationResult:
        """Validate that columns have expected data types.

        Args:
            df: LazyFrame to validate.
            expected_types: Dictionary mapping column names to expected Polars data types.

        Returns:
            ValidationResult indicating whether validation passed.
        """
        try:
            type_mismatches = []

            schema = df.collect_schema()
            schema_names = schema.names()
            schema_dtypes = schema.dtypes()

            for col_name, expected_type in expected_types.items():
                if col_name in schema_names:
                    col_index = schema_names.index(col_name)
                    actual_type = schema_dtypes[col_index]
                    if actual_type != expected_type:
                        type_mismatches.append({
                            "column": col_name,
                            "expected": str(expected_type),
                            "actual": str(actual_type),
                        })

            if type_mismatches:
                details = {"type_mismatches": type_mismatches}
                message = DTYPE_FAILED_MSG.format(len(type_mismatches))
                return ValidationResult(is_valid=False, message=message, details=details)

            return ValidationResult(
                is_valid=True,
                message=ALL_DTYPES_CORRECT_MSG,
            )

        except Exception as e:
            self._handle_error(e, {"operation": "data_type_validation"})

    def validate_numeric_ranges(
        self,
        *,
        df: pl.LazyFrame,
        column_ranges: dict[str, dict[str, float]]
    ) -> ValidationResult:
        """Validate that numeric columns fall within expected ranges.

        Args:
            df: LazyFrame to validate.
            column_ranges: Dictionary mapping column names to range constraints.
                Each range should have 'min' and/or 'max' keys.

        Returns:
            ValidationResult indicating whether validation passed.
        """
        try:
            range_violations = []
            schema = df.collect_schema()

            for col_name, range_constraints in column_ranges.items():
                if col_name not in schema.names():
                    continue

                col_index = schema.names().index(col_name)
                col_type = schema.dtypes()[col_index]
                if not is_numeric_datatype(col_type):
                    continue

                # Check minimum value
                if "min" in range_constraints:
                    min_value = range_constraints["min"]
                    below_min = df.filter(pl.col(col_name) < min_value).select(pl.len()).collect().item()
                    if below_min > ZERO_THRESHOLD:
                        range_violations.append({
                            "column": col_name,
                            "constraint": f"min >= {min_value}",
                            "violations": below_min,
                        })

                # Check maximum value
                if "max" in range_constraints:
                    max_value = range_constraints["max"]
                    above_max = df.filter(pl.col(col_name) > max_value).select(pl.len()).collect().item()
                    if above_max > ZERO_THRESHOLD:
                        range_violations.append({
                            "column": col_name,
                            "constraint": f"max <= {max_value}",
                            "violations": above_max,
                        })

            if range_violations:
                details = {"range_violations": range_violations}
                message = RANGE_FAILED_MSG.format(len(range_violations))
                return ValidationResult(is_valid=False, message=message, details=details)

            return ValidationResult(
                is_valid=True,
                message=ALL_RANGES_CORRECT_MSG,
            )

        except Exception as e:
            self._handle_error(e, {"operation": "range_validation"})

    def validate_string_patterns(
        self,
        *,
        df: pl.LazyFrame,
        column_patterns: dict[str, str]
    ) -> ValidationResult:
        """Validate that string columns match expected patterns.

        Args:
            df: LazyFrame to validate.
            column_patterns: Dictionary mapping column names to regex patterns.

        Returns:
            ValidationResult indicating whether validation passed.
        """
        try:
            pattern_violations = []
            schema = df.collect_schema()

            for col_name, pattern in column_patterns.items():
                if col_name not in schema.names():
                    continue

                col_index = schema.names().index(col_name)
                col_type = schema.dtypes()[col_index]
                if col_type != pl.Utf8:
                    continue

                # Count rows that don't match the pattern
                non_matching = (
                    df.filter(~pl.col(col_name).str.contains(pattern))
                    .select(pl.len())
                    .collect()
                    .item()
                )

                if non_matching > ZERO_THRESHOLD:
                    pattern_violations.append({
                        "column": col_name,
                        "pattern": pattern,
                        "violations": non_matching,
                    })

            if pattern_violations:
                details = {"pattern_violations": pattern_violations}
                message = PATTERN_FAILED_MSG.format(len(pattern_violations))
                return ValidationResult(is_valid=False, message=message, details=details)

            return ValidationResult(
                is_valid=True,
                message=ALL_PATTERNS_CORRECT_MSG,
            )

        except Exception as e:
            self._handle_error(e, {"operation": "pattern_validation"})

    def validate_uniqueness(
        self,
        *,
        df: pl.LazyFrame,
        unique_columns: list[str]
    ) -> ValidationResult:
        """Validate that specified columns contain unique values.

        Args:
            df: LazyFrame to validate.
            unique_columns: List of column names that should contain unique values.

        Returns:
            ValidationResult indicating whether validation passed.
        """
        try:
            uniqueness_violations = []
            schema = df.collect_schema()

            for col_name in unique_columns:
                if col_name not in schema.names():
                    continue

                # Count duplicates
                duplicates = (
                    df.group_by(col_name)
                    .len()
                    .filter(pl.col(LEN_COLUMN) > DUPLICATE_THRESHOLD)
                    .select(pl.len())
                    .collect()
                    .item()
                )

                if duplicates > ZERO_THRESHOLD:
                    uniqueness_violations.append({
                        "column": col_name,
                        "duplicate_groups": duplicates,
                    })

            if uniqueness_violations:
                details = {"uniqueness_violations": uniqueness_violations}
                message = UNIQUENESS_FAILED_MSG.format(len(uniqueness_violations))
                return ValidationResult(is_valid=False, message=message, details=details)

            return ValidationResult(
                is_valid=True,
                message=ALL_UNIQUE_MSG,
            )

        except Exception as e:
            self._handle_error(e, {"operation": "uniqueness_validation"})

    def run_comprehensive_validation(
        self,
        *,
        df: pl.LazyFrame,
        required_columns: list[str] | None = None,
        expected_types: dict[str, pl.DataType] | None = None,
        column_ranges: dict[str, dict[str, float]] | None = None,
        column_patterns: dict[str, str] | None = None,
        unique_columns: list[str] | None = None,
    ) -> list[ValidationResult]:
        """Run a comprehensive set of data quality validations.

        Args:
            df: LazyFrame to validate.
            required_columns: List of required columns for completeness check.
            expected_types: Dictionary of expected data types.
            column_ranges: Dictionary of numeric range constraints.
            column_patterns: Dictionary of string pattern constraints.
            unique_columns: List of columns that should be unique.

        Returns:
            List of ValidationResult objects for each validation performed.
        """
        results = []

        # Completeness validation
        if required_columns:
            results.append(self.validate_completeness(df=df, required_columns=required_columns))

        # Data type validation
        if expected_types:
            results.append(self.validate_data_types(df=df, expected_types=expected_types))

        # Range validation
        if column_ranges:
            results.append(self.validate_numeric_ranges(df=df, column_ranges=column_ranges))

        # Pattern validation
        if column_patterns:
            results.append(self.validate_string_patterns(df=df, column_patterns=column_patterns))

        # Uniqueness validation
        if unique_columns:
            results.append(self.validate_uniqueness(df=df, unique_columns=unique_columns))

        return results
