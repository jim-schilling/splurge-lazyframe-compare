"""Custom exceptions for the comparison framework.

Copyright (c) 2025 Jim Schilling.
Licensed under the MIT License. See the LICENSE file for details.
"""

from .comparison_exceptions import (
    ColumnMappingError,
    ComparisonError,
    ConfigError,
    DataSourceError,
    ExportError,
    PrimaryKeyViolationError,
    ReportError,
    SchemaValidationError,
)

__all__ = [
    "ComparisonError",
    "ConfigError",
    "DataSourceError",
    "ExportError",
    "SchemaValidationError",
    "PrimaryKeyViolationError",
    "ColumnMappingError",
    "ReportError",
]

DOMAINS: list[str] = ["exceptions", "errors"]
