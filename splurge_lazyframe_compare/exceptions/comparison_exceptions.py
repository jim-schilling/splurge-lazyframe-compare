"""Custom exceptions for the Polars LazyFrame comparison framework."""


class ComparisonError(Exception):
    """Base exception for comparison framework errors."""

    def __init__(self, message: str) -> None:
        """Initialize the exception with a message.

        Args:
            message: Error message describing the issue.
        """
        super().__init__(message)
        self.message = message


class SchemaValidationError(ComparisonError):
    """Raised when schema validation fails."""

    def __init__(self, message: str, validation_errors: list[str] | None = None) -> None:
        """Initialize schema validation error.

        Args:
            message: Error message describing the validation failure.
            validation_errors: List of specific validation error messages.
        """
        if validation_errors:
            full_message = f"{message}: {', '.join(validation_errors)}"
        else:
            full_message = message
        super().__init__(full_message)
        self.validation_errors = validation_errors or []


class PrimaryKeyViolationError(ComparisonError):
    """Raised when primary key constraints are violated."""

    def __init__(self, message: str, duplicate_keys: list | None = None) -> None:
        """Initialize primary key violation error.

        Args:
            message: Error message describing the violation.
            duplicate_keys: List of duplicate key values found.
        """
        super().__init__(message)
        self.duplicate_keys = duplicate_keys or []


class ColumnMappingError(ComparisonError):
    """Raised when column mapping configuration is invalid."""

    def __init__(self, message: str, mapping_errors: list[str] | None = None) -> None:
        """Initialize column mapping error.

        Args:
            message: Error message describing the mapping issue.
            mapping_errors: List of specific mapping error messages.
        """
        super().__init__(message)
        self.mapping_errors = mapping_errors or []
