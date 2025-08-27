"""Polars LazyFrame Comparison Framework.

A comprehensive Python framework for comparing two Polars LazyFrames with
configurable schemas, primary keys, and column mappings.
"""

from .core.comparator import LazyFrameComparator
from .core.results import ComparisonReport, ComparisonResults
from .core.schema import (
    ColumnDefinition,
    ColumnMapping,
    ComparisonConfig,
    ComparisonSchema,
)
from .exceptions.comparison_exceptions import (
    ColumnMappingError,
    ComparisonError,
    PrimaryKeyViolationError,
    SchemaValidationError,
)

__version__ = "2025.1.0"

__all__ = [
    "LazyFrameComparator",
    "ComparisonConfig",
    "ComparisonSchema",
    "ColumnDefinition",
    "ColumnMapping",
    "ComparisonResults",
    "ComparisonReport",
    "ComparisonError",
    "SchemaValidationError",
    "PrimaryKeyViolationError",
    "ColumnMappingError",
]
