"""Formatting utilities for the comparison framework.

Copyright (c) 2025 Jim Schilling.
Licensed under the MIT License. See the LICENSE file for details.
"""

DOMAINS: list[str] = ["utils", "formatting", "presentation"]

from typing import Any

import polars as pl


class FormattingConstants:
    """Constants for formatting operations."""

    # Number formatting
    DEFAULT_FLOAT_PRECISION: int = 2
    PERCENTAGE_MULTIPLIER: int = 100
    PERCENTAGE_FORMAT: str = ".1f"

    # Table formatting
    DEFAULT_TABLE_FORMAT: str = "grid"
    MAX_COLUMN_WIDTH: int = 50
    TRUNCATION_SUFFIX: str = "..."

    # Display formatting
    THOUSAND_SEPARATOR: str = ","
    DECIMAL_SEPARATOR: str = "."


def format_number(value: float, precision: int = FormattingConstants.DEFAULT_FLOAT_PRECISION) -> str:
    """Format a number with specified precision.

    Args:
        value: The number to format.
        precision: Number of decimal places.

    Returns:
        Formatted number string.
    """
    return f"{value:.{precision}f}"


def format_percentage(value: float, include_symbol: bool = True) -> str:
    """Format a value as a percentage.

    Args:
        value: The value to format (0.0 to 1.0).
        include_symbol: Whether to include the % symbol.

    Returns:
        Formatted percentage string.
    """
    percentage = value * FormattingConstants.PERCENTAGE_MULTIPLIER
    formatted = f"{percentage:{FormattingConstants.PERCENTAGE_FORMAT}}"
    return f"{formatted}%" if include_symbol else formatted


def format_large_number(value: int) -> str:
    """Format a large number with thousand separators.

    Args:
        value: The number to format.

    Returns:
        Formatted number string with thousand separators.
    """
    return f"{value:,}"


def truncate_string(text: str, max_length: int = FormattingConstants.MAX_COLUMN_WIDTH) -> str:
    """Truncate a string if it's too long.

    Args:
        text: The string to truncate.
        max_length: Maximum length before truncation.

    Returns:
        Truncated string with suffix if needed.
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(FormattingConstants.TRUNCATION_SUFFIX)] + FormattingConstants.TRUNCATION_SUFFIX


def format_dataframe_sample(
    df: pl.LazyFrame,
    max_rows: int = 10,
    max_cols: int | None = None,
    truncate_values: bool = True
) -> str:
    """Format a DataFrame sample for display.

    Args:
        df: LazyFrame to format.
        max_rows: Maximum number of rows to show.
        max_cols: Maximum number of columns to show (None for all).
        truncate_values: Whether to truncate long values.

    Returns:
        Formatted DataFrame sample string.
    """
    # Collect sample data
    sample_df = df.limit(max_rows).collect()

    if sample_df.is_empty():
        return "No data to display"

    # Limit columns if specified
    if max_cols is not None and sample_df.width > max_cols:
        columns_to_show = sample_df.columns[:max_cols]
        sample_df = sample_df.select(columns_to_show)

    # Convert to string representation
    return str(sample_df)


def format_column_list(columns: list[str], max_items: int = 10) -> str:
    """Format a list of column names for display.

    Args:
        columns: List of column names.
        max_items: Maximum number of items to show before truncating.

    Returns:
        Formatted column list string.
    """
    if len(columns) <= max_items:
        return ", ".join(f"`{col}`" for col in columns)

    visible_cols = columns[:max_items]
    remaining_count = len(columns) - max_items

    return ", ".join(f"`{col}`" for col in visible_cols) + f" ... (+{remaining_count} more)"


def format_validation_errors(errors: list[str]) -> str:
    """Format a list of validation errors for display.

    Args:
        errors: List of error messages.

    Returns:
        Formatted error string.
    """
    if not errors:
        return "No errors found"

    if len(errors) == 1:
        return errors[0]

    formatted_errors = "\n".join(f"  • {error}" for error in errors)
    return f"Multiple errors found:\n{formatted_errors}"


def create_summary_table(
    data: dict[str, Any],
    headers: list[str] | None = None,
    table_format: str = FormattingConstants.DEFAULT_TABLE_FORMAT
) -> str:
    """Create a formatted table from dictionary data.

    Args:
        data: Dictionary containing table data.
        headers: Optional custom headers.
        table_format: Table format for tabulate.

    Returns:
        Formatted table string.
    """
    try:
        from tabulate import tabulate

        if not headers:
            headers = ["Metric", "Value"]

        table_data = [[key, value] for key, value in data.items()]

        return tabulate(
            table_data,
            headers=headers,
            tablefmt=table_format,
            numalign='right',
            stralign='left'
        )
    except ImportError:
        # Fallback to simple formatting if tabulate is not available
        lines = []
        for key, value in data.items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)
