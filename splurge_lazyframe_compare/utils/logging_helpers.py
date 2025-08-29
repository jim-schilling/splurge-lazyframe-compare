"""Logging and monitoring helpers for the comparison framework."""

import logging
import time
from contextlib import contextmanager
from typing import Any

from splurge_lazyframe_compare.utils.constants import TIMESTAMP_FORMAT


# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Only add handler if none exists to avoid duplicate logs
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s] %(message)s",
        datefmt=TIMESTAMP_FORMAT
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.

    Args:
        name: Logger name (typically __name__ from calling module).

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(f"splurge_lazyframe_compare.{name}")


class LoggingConstants:
    """Constants for logging operations."""

    # Log levels
    DEBUG: str = "DEBUG"
    INFO: str = "INFO"
    WARNING: str = "WARNING"
    ERROR: str = "ERROR"

    # Performance thresholds (ms)
    SLOW_OPERATION_THRESHOLD: int = 1000
    VERY_SLOW_OPERATION_THRESHOLD: int = 5000

    # Monitoring
    ENABLE_PERFORMANCE_LOGGING: bool = True
    ENABLE_MEMORY_LOGGING: bool = False


def create_log_message(
    level: str,
    service_name: str,
    operation: str,
    message: str,
    details: dict[str, Any] | None = None,
    duration_ms: float | None = None
) -> str:
    """Create a standardized log message.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        service_name: Name of the service.
        operation: Operation being performed.
        message: Log message.
        details: Additional details dictionary.
        duration_ms: Operation duration in milliseconds.

    Returns:
        Formatted log message.
    """
    timestamp = time.strftime(TIMESTAMP_FORMAT)

    log_parts = [
        f"[{timestamp}]",
        f"[{level}]",
        f"[{service_name}]",
        f"[{operation}]",
        message
    ]

    if duration_ms is not None:
        log_parts.append(f"({duration_ms:.2f}ms)")

    if details:
        details_str = ", ".join(f"{k}={v}" for k, v in details.items())
        log_parts.append(f"Details: {details_str}")

    return " ".join(log_parts)


def log_performance(
    service_name: str,
    operation: str,
    duration_ms: float,
    details: dict[str, Any] | None = None
) -> None:
    """Log performance information for an operation.

    Args:
        service_name: Name of the service.
        operation: Operation name.
        duration_ms: Duration in milliseconds.
        details: Additional performance details.
    """
    if not LoggingConstants.ENABLE_PERFORMANCE_LOGGING:
        return

    if duration_ms > LoggingConstants.VERY_SLOW_OPERATION_THRESHOLD:
        level = LoggingConstants.ERROR
        message = f"VERY SLOW OPERATION: {duration_ms:.2f}ms"
    elif duration_ms > LoggingConstants.SLOW_OPERATION_THRESHOLD:
        level = LoggingConstants.WARNING
        message = f"SLOW OPERATION: {duration_ms:.2f}ms"
    else:
        level = LoggingConstants.DEBUG
        message = f"Operation completed in {duration_ms:.2f}ms"

    # Get logger for the service
    service_logger = get_logger(service_name)

    # Create structured log message
    log_message = create_log_message(
        level=level,
        service_name=service_name,
        operation=operation,
        message=message,
        details=details,
        duration_ms=duration_ms
    )

    # Log using appropriate level
    if level == LoggingConstants.DEBUG:
        service_logger.debug(log_message)
    elif level == LoggingConstants.INFO:
        service_logger.info(log_message)
    elif level == LoggingConstants.WARNING:
        service_logger.warning(log_message)
    elif level == LoggingConstants.ERROR:
        service_logger.error(log_message)


@contextmanager
def performance_monitor(service_name: str, operation: str):
    """Context manager for monitoring operation performance.

    Args:
        service_name: Name of the service performing the operation.
        operation: Name of the operation.

    Yields:
        Performance context information.

    Example:
        with performance_monitor("ComparisonService", "find_differences") as ctx:
            # Perform your actual operation here
            result = perform_comparison()  # Your actual function call
            # Add custom metrics (result may not always have len() method)
            if hasattr(result, '__len__'):
                ctx["records_processed"] = len(result)
            ctx["operation_status"] = "completed"
    """
    start_time = time.perf_counter()
    context = {}

    try:
        yield context
    finally:
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        # Add performance context
        context.setdefault("duration_ms", duration_ms)

        log_performance(
            service_name=service_name,
            operation=operation,
            duration_ms=duration_ms,
            details=context
        )


def log_service_initialization(service_name: str, config: dict[str, Any] | None = None) -> None:
    """Log service initialization.

    Args:
        service_name: Name of the service being initialized.
        config: Optional configuration information.
    """
    message = "Service initialized successfully"
    details = {"config": config} if config else None

    # Get logger for the service
    service_logger = get_logger(service_name)

    # Create structured log message
    log_message = create_log_message(
        level=LoggingConstants.INFO,
        service_name=service_name,
        operation="initialization",
        message=message,
        details=details
    )

    service_logger.info(log_message)


def log_service_operation(
    service_name: str,
    operation: str,
    status: str,
    message: str | None = None,
    details: dict[str, Any] | None = None
) -> None:
    """Log a service operation with status.

    Args:
        service_name: Name of the service.
        operation: Operation being performed.
        status: Operation status (success, error, warning).
        message: Optional status message.
        details: Optional additional details.
    """
    level_map = {
        "success": LoggingConstants.INFO,
        "error": LoggingConstants.ERROR,
        "warning": LoggingConstants.WARNING,
        "info": LoggingConstants.INFO,
    }

    level = level_map.get(status.lower(), LoggingConstants.INFO)
    log_message = message or f"Operation {status}"

    # Get logger for the service
    service_logger = get_logger(service_name)

    # Create structured log message
    log_details = create_log_message(
        level=level,
        service_name=service_name,
        operation=operation,
        message=log_message,
        details=details
    )

    # Log using appropriate level
    if level == LoggingConstants.DEBUG:
        service_logger.debug(log_details)
    elif level == LoggingConstants.INFO:
        service_logger.info(log_details)
    elif level == LoggingConstants.WARNING:
        service_logger.warning(log_details)
    elif level == LoggingConstants.ERROR:
        service_logger.error(log_details)


def create_operation_context(
    operation_name: str,
    input_params: dict[str, Any] | None = None,
    expected_output: str | None = None
) -> dict[str, Any]:
    """Create a context dictionary for operation tracking.

    Args:
        operation_name: Name of the operation.
        input_params: Input parameters for the operation.
        expected_output: Description of expected output.

    Returns:
        Context dictionary for operation tracking.
    """
    return {
        "operation": operation_name,
        "timestamp": time.strftime(TIMESTAMP_FORMAT),
        "input_params": input_params or {},
        "expected_output": expected_output,
        "status": "started",
    }


def update_operation_context(
    context: dict[str, Any],
    status: str,
    result: Any | None = None,
    error: str | None = None,
    additional_info: dict[str, Any] | None = None
) -> None:
    """Update operation context with results.

    Args:
        context: Operation context dictionary to update.
        status: New status for the operation.
        result: Operation result if successful.
        error: Error message if operation failed.
        additional_info: Additional information to include.
    """
    context.update({
        "status": status,
        "completed_at": time.strftime(TIMESTAMP_FORMAT),
    })

    if result is not None:
        context["result"] = str(result)[:200] + "..." if len(str(result)) > 200 else str(result)

    if error:
        context["error"] = error

    if additional_info:
        context.update(additional_info)


def log_dataframe_stats(
    service_name: str,
    operation: str,
    df_info: dict[str, Any],
    stage: str = "input"
) -> None:
    """Log DataFrame statistics for monitoring.

    Args:
        service_name: Name of the service.
        operation: Operation being performed.
        df_info: DataFrame information dictionary.
        stage: Stage of processing (input, output, intermediate).
    """
    if not LoggingConstants.ENABLE_MEMORY_LOGGING:
        return

    message = f"{stage.capitalize()} DataFrame: {df_info.get('row_count', 0)} rows, {df_info.get('column_count', 0)} cols"

    details = {
        "stage": stage,
        "memory_mb": df_info.get("memory_estimate_mb", 0),
        "has_nulls": df_info.get("has_nulls", False),
        "column_types": df_info.get("column_types", []),
    }

    # Get logger for the service
    service_logger = get_logger(service_name)

    # Create structured log message
    log_message = create_log_message(
        level=LoggingConstants.DEBUG,
        service_name=service_name,
        operation=operation,
        message=message,
        details=details
    )

    service_logger.debug(log_message)


def create_service_health_check(service_name: str) -> dict[str, Any]:
    """Create a health check dictionary for service monitoring.

    Args:
        service_name: Name of the service.

    Returns:
        Health check dictionary.
    """
    return {
        "service": service_name,
        "timestamp": time.strftime(TIMESTAMP_FORMAT),
        "status": "healthy",
        "uptime_seconds": 0,  # Would be tracked in a real implementation
        "memory_usage_mb": 0,  # Would be measured in a real implementation
        "active_operations": 0,
    }


def log_service_health(service_name: str, health_data: dict[str, Any]) -> None:
    """Log service health information.

    Args:
        service_name: Name of the service.
        health_data: Health check data dictionary.
    """
    status = health_data.get("status", "unknown")
    level = LoggingConstants.INFO if status == "healthy" else LoggingConstants.WARNING

    message = f"Health check: {status}"

    # Get logger for the service
    service_logger = get_logger(service_name)

    # Create structured log message
    log_message = create_log_message(
        level=level,
        service_name=service_name,
        operation="health_check",
        message=message,
        details=health_data
    )

    # Log using appropriate level
    if level == LoggingConstants.INFO:
        service_logger.info(log_message)
    else:
        service_logger.warning(log_message)
