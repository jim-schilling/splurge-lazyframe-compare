"""Core components of the Polars LazyFrame comparison framework."""

from .comparator import LazyFrameComparator
from .results import ComparisonReport, ComparisonResults
from .schema import (
    ColumnDefinition,
    ColumnMapping,
    ComparisonConfig,
    ComparisonSchema,
)

__all__ = [
    "LazyFrameComparator",
    "ComparisonConfig",
    "ComparisonSchema",
    "ColumnDefinition",
    "ColumnMapping",
    "ComparisonResults",
    "ComparisonReport",
]
