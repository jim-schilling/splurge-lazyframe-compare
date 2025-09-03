"""Shared constants for the comparison framework.

This module contains constants used across the framework,
organized by category and following naming conventions.

Copyright (c) 2025 Jim Schilling.
Licensed under the MIT License. See the LICENSE file for details.
"""

from typing import Final

# Column prefixes and naming patterns
PRIMARY_KEY_PREFIX: Final[str] = "PK_"
LEFT_PREFIX: Final[str] = "L_"
RIGHT_PREFIX: Final[str] = "R_"
LEFT_DF_NAME: Final[str] = "left DataFrame"
RIGHT_DF_NAME: Final[str] = "right DataFrame"

# Join and comparison constants
JOIN_INNER: Final[str] = "inner"
JOIN_LEFT: Final[str] = "left"
LEN_COLUMN: Final[str] = "len"

# Thresholds and limits
DUPLICATE_THRESHOLD: Final[int] = 1
ZERO_THRESHOLD: Final[int] = 0
DEFAULT_MAX_SAMPLES: Final[int] = 10
PERCENTAGE_MULTIPLIER: Final[int] = 100
PERCENTAGE_FORMAT: Final[str] = ".1f"
JSON_INDENT: Final[int] = 2

# Versioned schema for summary export
SUMMARY_SCHEMA_VERSION: Final[str] = "1.0"

# File and output constants
DEFAULT_OUTPUT_DIR: Final[str] = "."
DEFAULT_FORMAT: Final[str] = "parquet"
FORMAT_PARQUET: Final[str] = "parquet"
FORMAT_CSV: Final[str] = "csv"
FORMAT_JSON: Final[str] = "json"

# Filename patterns
VALUE_DIFFERENCES_FILENAME: Final[str] = "value_differences_{}.{}"
LEFT_ONLY_FILENAME: Final[str] = "left_only_records_{}.{}"
RIGHT_ONLY_FILENAME: Final[str] = "right_only_records_{}.{}"
SUMMARY_FILENAME: Final[str] = "comparison_summary_{}.json"
TIMESTAMP_FORMAT: Final[str] = "%Y%m%d_%H%M%S"

# Report formatting constants
REPORT_HEADER_LENGTH: Final[int] = 60
SECTION_SEPARATOR_LENGTH: Final[int] = 40
REPORT_HEADER: Final[str] = "=" * REPORT_HEADER_LENGTH
REPORT_TITLE: Final[str] = "SPLURGE LAZYFRAME COMPARISON SUMMARY"
SECTION_SEPARATOR: Final[str] = "-" * SECTION_SEPARATOR_LENGTH

# Report section headers
RECORD_COUNTS_SECTION: Final[str] = "RECORD COUNTS:"
COMPARISON_RESULTS_SECTION: Final[str] = "COMPARISON RESULTS:"
PERCENTAGES_SECTION: Final[str] = "PERCENTAGES (of Left DataFrame):"
VALUE_DIFFERENCES_SECTION: Final[str] = "VALUE DIFFERENCES SAMPLES:"
LEFT_ONLY_SECTION: Final[str] = "LEFT-ONLY RECORDS SAMPLES:"
RIGHT_ONLY_SECTION: Final[str] = "RIGHT-ONLY RECORDS SAMPLES:"

# Error message patterns
MISSING_COLUMNS_MSG: Final[str] = "Missing columns: {}"
WRONG_DTYPE_MSG: Final[str] = "Column {}: expected {}, got {}"
NULL_VIOLATION_MSG: Final[str] = "Column {}: {} null values found but column defined as non-nullable"
PK_NOT_DEFINED_MSG: Final[str] = "Primary key column '{}' not defined in schema"
EMPTY_LEFT_SCHEMA_MSG: Final[str] = "Left schema has no columns defined"
EMPTY_RIGHT_SCHEMA_MSG: Final[str] = "Right schema has no columns defined"
NO_PK_MSG: Final[str] = "No primary key columns defined"
MISSING_LEFT_MAPPED_MSG: Final[str] = "Left schema missing mapped columns: {}"
MISSING_RIGHT_MAPPED_MSG: Final[str] = "Right schema missing mapped columns: {}"
PK_NOT_MAPPED_MSG: Final[str] = "Primary key column '{}' not found in column mappings"
DUPLICATE_PK_MSG: Final[str] = "Duplicate primary keys found in {}: {} duplicates"

# Validation constants
COMPLETENESS_FAILED_MSG: Final[str] = "Completeness validation failed: {} missing columns, {} null columns"
ALL_COLUMNS_PRESENT_MSG: Final[str] = "All required columns are present and contain data"
DTYPE_FAILED_MSG: Final[str] = "Data type validation failed: {} mismatches found"
ALL_DTYPES_CORRECT_MSG: Final[str] = "All columns have expected data types"
RANGE_FAILED_MSG: Final[str] = "Range validation failed: {} violations found"
ALL_RANGES_CORRECT_MSG: Final[str] = "All numeric columns fall within expected ranges"
PATTERN_FAILED_MSG: Final[str] = "Pattern validation failed: {} violations found"
ALL_PATTERNS_CORRECT_MSG: Final[str] = "All string columns match expected patterns"
UNIQUENESS_FAILED_MSG: Final[str] = "Uniqueness validation failed: {} violations found"
ALL_UNIQUE_MSG: Final[str] = "All specified columns contain unique values"
VALIDATION_SUCCESS_MSG: Final[str] = "All validations passed"
FIRST_COLUMN_INDEX: Final[int] = 0
