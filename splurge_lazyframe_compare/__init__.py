"""Polars LazyFrame Comparison Framework.

A comprehensive Python framework for comparing two Polars LazyFrames with
configurable schemas, primary keys, and column mappings.

Copyright (c) 2025 Jim Schilling.
Licensed under the MIT License. See the LICENSE file for details.
"""

# Service-based architecture
from .core.comparator import ComparisonReport, LazyFrameComparator
from .models.comparison import ComparisonResult, ComparisonSummary
from .models.schema import (
    ColumnDefinition,
    ColumnMapping,
    ComparisonConfig,
    ComparisonSchema,
)
from .models.validation import ValidationResult
from .services import (
    ComparisonService,
    DataPreparationService,
    ReportingService,
    ValidationService,
)
from .services.orchestrator import ComparisonOrchestrator

__version__ = "2025.2.0"

# High-level domains for the package. This helps tooling and consumers
# understand the primary responsibilities provided by this library.
__domains__: list[str] = [
    "cli",
    "comparison",
    "config",
    "constants",
    "core",
    "data",
    "data_preparation",
    "data_quality",
    "entrypoint",
    "errors",
    "exceptions",
    "export",
    "file_ops",
    "formatting",
    "framework",
    "helpers",
    "io",
    "lazyframe",
    "logging",
    "monitoring",
    "models",
    "orchestration",
    "performance",
    "polars",
    "presentation",
    "processing",
    "reporting",
    "schema",
    "services",
    "tools",
    "types",
    "validation",
    "utils",
]

__all__ = [
    # Service-based architecture
    "LazyFrameComparator",
    "ComparisonReport",
    "ComparisonResult",
    "ComparisonSummary",
    "ComparisonConfig",
    "ComparisonSchema",
    "ColumnDefinition",
    "ColumnMapping",
    "ValidationResult",
    "ComparisonOrchestrator",
    "ComparisonService",
    "DataPreparationService",
    "ReportingService",
    "ValidationService",
]
