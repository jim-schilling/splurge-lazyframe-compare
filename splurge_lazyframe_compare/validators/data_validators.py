"""Data quality validation for the comparison framework."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import polars as pl

from splurge_lazyframe_compare.utils.type_helpers import is_numeric_dtype


@dataclass
class ValidationResult:
    """Result of a data quality validation check.

    Attributes:
        is_valid: Whether the validation passed.
        message: Description of the validation result.
        details: Additional details about the validation.
    """

    is_valid: bool
    message: str
    details: Optional[Dict] = None


class DataQualityValidator:
    """Data quality validation for LazyFrames.

    This class provides various data quality checks that can be performed
    on LazyFrames before comparison.
    """

    def __init__(self) -> None:
        """Initialize the data quality validator."""
        pass

    def validate_completeness(
        self, *, df: pl.LazyFrame, required_columns: List[str]
    ) -> ValidationResult:
        """Validate that required columns are present and not entirely null.

        Args:
            df: LazyFrame to validate.
            required_columns: List of column names that must be present and non-null.

        Returns:
            ValidationResult indicating whether validation passed.
        """
        missing_columns = []
        null_columns = []

        # Check for missing columns
        df_columns = set(df.columns)
        for col in required_columns:
            if col not in df_columns:
                missing_columns.append(col)

        # Check for entirely null columns
        for col in required_columns:
            if col in df_columns:
                null_count = df.select(pl.col(col).is_null().sum()).collect().item()
                total_count = df.select(pl.len()).collect().item()
                if null_count == total_count:
                    null_columns.append(col)

        if missing_columns or null_columns:
            details = {
                "missing_columns": missing_columns,
                "null_columns": null_columns,
            }
            message = f"Completeness validation failed: {len(missing_columns)} missing columns, {len(null_columns)} null columns"
            return ValidationResult(is_valid=False, message=message, details=details)

        return ValidationResult(
            is_valid=True,
            message="All required columns are present and contain data",
        )

    def validate_data_types(
        self, *, df: pl.LazyFrame, expected_types: Dict[str, pl.DataType]
    ) -> ValidationResult:
        """Validate that columns have expected data types.

        Args:
            df: LazyFrame to validate.
            expected_types: Dictionary mapping column names to expected Polars data types.

        Returns:
            ValidationResult indicating whether validation passed.
        """
        type_mismatches = []

        for col_name, expected_type in expected_types.items():
            if col_name in df.columns:
                actual_type = df.select(pl.col(col_name)).dtypes[0]
                if actual_type != expected_type:
                    type_mismatches.append({
                        "column": col_name,
                        "expected": str(expected_type),
                        "actual": str(actual_type),
                    })

        if type_mismatches:
            details = {"type_mismatches": type_mismatches}
            message = f"Data type validation failed: {len(type_mismatches)} mismatches found"
            return ValidationResult(is_valid=False, message=message, details=details)

        return ValidationResult(
            is_valid=True,
            message="All columns have expected data types",
        )

    def validate_numeric_ranges(
        self, *, df: pl.LazyFrame, column_ranges: Dict[str, Dict[str, float]]
    ) -> ValidationResult:
        """Validate that numeric columns fall within expected ranges.

        Args:
            df: LazyFrame to validate.
            column_ranges: Dictionary mapping column names to range constraints.
                Each range should have 'min' and/or 'max' keys.

        Returns:
            ValidationResult indicating whether validation passed.
        """
        range_violations = []

        for col_name, range_constraints in column_ranges.items():
            if col_name not in df.columns:
                continue

            col_type = df.select(pl.col(col_name)).dtypes[0]
            if not is_numeric_dtype(col_type):
                continue

            # Check minimum value
            if "min" in range_constraints:
                min_value = range_constraints["min"]
                below_min = df.filter(pl.col(col_name) < min_value).select(pl.len()).collect().item()
                if below_min > 0:
                    range_violations.append({
                        "column": col_name,
                        "constraint": f"min >= {min_value}",
                        "violations": below_min,
                    })

            # Check maximum value
            if "max" in range_constraints:
                max_value = range_constraints["max"]
                above_max = df.filter(pl.col(col_name) > max_value).select(pl.len()).collect().item()
                if above_max > 0:
                    range_violations.append({
                        "column": col_name,
                        "constraint": f"max <= {max_value}",
                        "violations": above_max,
                    })

        if range_violations:
            details = {"range_violations": range_violations}
            message = f"Range validation failed: {len(range_violations)} violations found"
            return ValidationResult(is_valid=False, message=message, details=details)

        return ValidationResult(
            is_valid=True,
            message="All numeric columns fall within expected ranges",
        )

    def validate_string_patterns(
        self, *, df: pl.LazyFrame, column_patterns: Dict[str, str]
    ) -> ValidationResult:
        """Validate that string columns match expected patterns.

        Args:
            df: LazyFrame to validate.
            column_patterns: Dictionary mapping column names to regex patterns.

        Returns:
            ValidationResult indicating whether validation passed.
        """
        pattern_violations = []

        for col_name, pattern in column_patterns.items():
            if col_name not in df.columns:
                continue

            col_type = df.select(pl.col(col_name)).dtypes[0]
            if col_type != pl.Utf8:
                continue

            # Count rows that don't match the pattern
            non_matching = (
                df.filter(~pl.col(col_name).str.contains(pattern))
                .select(pl.len())
                .collect()
                .item()
            )

            if non_matching > 0:
                pattern_violations.append({
                    "column": col_name,
                    "pattern": pattern,
                    "violations": non_matching,
                })

        if pattern_violations:
            details = {"pattern_violations": pattern_violations}
            message = f"Pattern validation failed: {len(pattern_violations)} violations found"
            return ValidationResult(is_valid=False, message=message, details=details)

        return ValidationResult(
            is_valid=True,
            message="All string columns match expected patterns",
        )

    def validate_uniqueness(
        self, *, df: pl.LazyFrame, unique_columns: List[str]
    ) -> ValidationResult:
        """Validate that specified columns contain unique values.

        Args:
            df: LazyFrame to validate.
            unique_columns: List of column names that should contain unique values.

        Returns:
            ValidationResult indicating whether validation passed.
        """
        uniqueness_violations = []

        for col_name in unique_columns:
            if col_name not in df.columns:
                continue

            # Count duplicates
            duplicates = (
                df.group_by(col_name)
                .len()
                .filter(pl.col("len") > 1)
                .select(pl.len())
                .collect()
                .item()
            )

            if duplicates > 0:
                uniqueness_violations.append({
                    "column": col_name,
                    "duplicate_groups": duplicates,
                })

        if uniqueness_violations:
            details = {"uniqueness_violations": uniqueness_violations}
            message = f"Uniqueness validation failed: {len(uniqueness_violations)} violations found"
            return ValidationResult(is_valid=False, message=message, details=details)

        return ValidationResult(
            is_valid=True,
            message="All specified columns contain unique values",
        )

    def run_comprehensive_validation(
        self,
        *,
        df: pl.LazyFrame,
        required_columns: Optional[List[str]] = None,
        expected_types: Optional[Dict[str, pl.DataType]] = None,
        column_ranges: Optional[Dict[str, Dict[str, float]]] = None,
        column_patterns: Optional[Dict[str, str]] = None,
        unique_columns: Optional[List[str]] = None,
    ) -> List[ValidationResult]:
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
