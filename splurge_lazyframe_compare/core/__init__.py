"""Core components of the Polars LazyFrame comparison framework.

Note: Core functionality has been moved to the service-based architecture.
This module now only contains the main LazyFrameComparator interface.
"""

from .comparator import ComparisonReport, LazyFrameComparator

__all__ = [
    "LazyFrameComparator",
    "ComparisonReport",
]
