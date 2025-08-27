"""Core components of the Polars LazyFrame comparison framework."""

from .comparator import LazyFrameComparator
from .schema import (
    ColumnDefinition,
    ColumnMapping,
    ComparisonConfig,
    ComparisonSchema,
)
from .results import ComparisonResults, ComparisonReport

__all__ = [
    "LazyFrameComparator",
    "ComparisonConfig",
    "ComparisonSchema",
    "ColumnDefinition",
    "ColumnMapping",
    "ComparisonResults",
    "ComparisonReport",
]
