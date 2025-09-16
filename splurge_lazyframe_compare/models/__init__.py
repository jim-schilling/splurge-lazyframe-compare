"""Data model definitions for the comparison framework.

This package contains pure data models and dataclasses that represent
the core data structures used throughout the framework.

Copyright (c) 2025 Jim Schilling.
Licensed under the MIT License. See the LICENSE file for details.
"""

from .comparison import ComparisonResult, ComparisonSummary
from .schema import (
    ColumnDefinition,
    ColumnMapping,
    ComparisonConfig,
    ComparisonSchema,
)
from .validation import ValidationResult

__all__ = [
    "ComparisonResult",
    "ComparisonSummary",
    "ColumnDefinition",
    "ColumnMapping",
    "ComparisonConfig",
    "ComparisonSchema",
    "ValidationResult",
]

DOMAINS: list[str] = ["models", "data", "schema"]
