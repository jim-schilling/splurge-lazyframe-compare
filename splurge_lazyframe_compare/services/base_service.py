"""Base service class for the comparison framework.

Copyright (c) 2025 Jim Schilling.
Licensed under the MIT License. See the LICENSE file for details.
"""

from abc import ABC, abstractmethod
from typing import Any

from splurge_lazyframe_compare.utils.logging_helpers import (
    log_service_initialization,
    log_service_operation,
)

DOMAINS: list[str] = ["services", "framework"]


class ServiceConstants:
    """Constants for service operations."""

    # Common service messages
    SERVICE_INITIALIZED_MSG: str = "Service initialized successfully"
    SERVICE_OPERATION_COMPLETED_MSG: str = "Operation completed successfully"
    SERVICE_OPERATION_FAILED_MSG: str = "Operation failed: {}"

    # Validation messages
    INVALID_INPUT_MSG: str = "Invalid input: {}"
    MISSING_REQUIRED_PARAMETER_MSG: str = "Missing required parameter: {}"
    INVALID_PARAMETER_TYPE_MSG: str = "Invalid parameter type for {}: expected {}, got {}"


class BaseService(ABC):
    """Base class for all services in the comparison framework.

    This class provides common functionality and patterns that all services
    should follow, including error handling, logging, and validation.
    """

    def __init__(self, service_name: str) -> None:
        """Initialize the service.

        Args:
            service_name: Name of the service for identification.
        """
        self.service_name = service_name
        self._validate_service_initialization()

        # Log service initialization
        log_service_initialization(self.service_name)


    def _validate_service_initialization(self) -> None:
        """Validate service initialization parameters."""
        if not self.service_name:
            raise ValueError("Service name cannot be empty")

    @abstractmethod
    def _validate_inputs(self, **kwargs: Any) -> None:
        """Validate service inputs.

        Args:
            **kwargs: Input parameters to validate.

        Raises:
            ValueError: If inputs are invalid.
        """
        pass

    def _handle_error(self, error: Exception, context: dict[str, Any] | None = None) -> None:
        """Handle errors with context information.

        Args:
            error: The exception that occurred.
            context: Additional context information.
        """
        # Log the error
        context_info = f" in {context}" if context else ""
        error_msg = f"{self.service_name}: {error}{context_info}"

        log_service_operation(
            service_name=self.service_name,
            operation="error_handling",
            status="error",
            message=str(error),
            details=context
        )

        # Re-raise the error with augmented context while preserving type and cause
        context_str = f" in {context}" if context else ""
        augmented_message = f"{self.service_name}: {error}{context_str}"
        raise type(error)(augmented_message) from error
