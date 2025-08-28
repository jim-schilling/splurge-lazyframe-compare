"""Validation result models for the comparison framework."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationConstants:
    """Constants for validation operations."""

    # Validation messages
    VALIDATION_SUCCESS_MSG: str = "All validations passed"
    COMPLETENESS_FAILED_MSG: str = "Completeness validation failed: {} missing columns, {} null columns"
    ALL_COLUMNS_PRESENT_MSG: str = "All required columns are present and contain data"
    DTYPE_FAILED_MSG: str = "Data type validation failed: {} mismatches found"
    ALL_DTYPES_CORRECT_MSG: str = "All columns have expected data types"
    RANGE_FAILED_MSG: str = "Range validation failed: {} violations found"
    ALL_RANGES_CORRECT_MSG: str = "All numeric columns fall within expected ranges"
    PATTERN_FAILED_MSG: str = "Pattern validation failed: {} violations found"
    ALL_PATTERNS_CORRECT_MSG: str = "All string columns match expected patterns"
    UNIQUENESS_FAILED_MSG: str = "Uniqueness validation failed: {} violations found"
    ALL_UNIQUE_MSG: str = "All specified columns contain unique values"

    # Validation thresholds
    COMPLETENESS_THRESHOLD: float = 0.95
    ACCURACY_THRESHOLD: float = 0.99


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
    details: dict[str, Any] | None = None
