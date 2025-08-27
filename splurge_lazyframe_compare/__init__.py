"""Polars LazyFrame Comparison Framework.

A comprehensive Python framework for comparing two Polars LazyFrames with
configurable schemas, primary keys, and column mappings.
"""

from .core.comparator import LazyFrameComparator
from .core.schema import (
    ColumnDefinition,
    ColumnMapping,
    ComparisonConfig,
    ComparisonSchema,
)
from .core.results import ComparisonResults, ComparisonReport
from .exceptions.comparison_exceptions import (
    ComparisonError,
    SchemaValidationError,
    PrimaryKeyViolationError,
    ColumnMappingError,
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
