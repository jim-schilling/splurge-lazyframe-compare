"""Service layer for the comparison framework.

This package contains service classes that implement the business logic
of the framework, following the single responsibility principle.

Copyright (c) 2025 Jim Schilling.
Licensed under the MIT License. See the LICENSE file for details.
"""

from .base_service import BaseService, ServiceConstants
from .comparison_service import ComparisonService
from .orchestrator import ComparisonOrchestrator
from .preparation_service import DataPreparationService
from .reporting_service import ReportingService
from .validation_service import ValidationService

__all__ = [
    "BaseService",
    "ServiceConstants",
    "ComparisonService",
    "ComparisonOrchestrator",
    "DataPreparationService",
    "ReportingService",
    "ValidationService",
]


DOMAINS: list[str] = ["services", "comparison", "orchestration"]
