"""File operation utilities for the comparison framework."""

from pathlib import Path
from typing import Dict, Optional

import polars as pl


class FileOperationConstants:
    """Constants for file operations."""

    # Default formats
    DEFAULT_FORMAT: str = "parquet"
    SUPPORTED_FORMATS: tuple = ("parquet", "csv", "json")

    # File extensions
    PARQUET_EXT: str = ".parquet"
    CSV_EXT: str = ".csv"
    JSON_EXT: str = ".json"

    # Buffer sizes
    DEFAULT_BUFFER_SIZE: int = 8192


def ensure_directory_exists(path: Path) -> None:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists.
    """
    path.mkdir(parents=True, exist_ok=True)


def get_file_extension(format_name: str) -> str:
    """Get the file extension for a given format.

    Args:
        format_name: Format name (parquet, csv, json).

    Returns:
        File extension including the dot.

    Raises:
        ValueError: If format is not supported.
    """
    format_map = {
        "parquet": FileOperationConstants.PARQUET_EXT,
        "csv": FileOperationConstants.CSV_EXT,
        "json": FileOperationConstants.JSON_EXT,
    }

    if format_name not in format_map:
        raise ValueError(f"Unsupported format: {format_name}. Supported formats: {FileOperationConstants.SUPPORTED_FORMATS}")

    return format_map[format_name]


def export_lazyframe(
    lazyframe: pl.LazyFrame,
    file_path: Path,
    format_name: str = FileOperationConstants.DEFAULT_FORMAT,
    **kwargs
) -> None:
    """Export a LazyFrame to a file.

    Args:
        lazyframe: LazyFrame to export.
        file_path: Path to save the file.
        format_name: Export format (parquet, csv, json).
        **kwargs: Additional format-specific arguments.
    """
    # Ensure parent directory exists
    ensure_directory_exists(file_path.parent)

    if format_name == "parquet":
        lazyframe.sink_parquet(file_path, **kwargs)
    elif format_name == "csv":
        lazyframe.sink_csv(file_path, **kwargs)
    elif format_name == "json":
        lazyframe.sink_json(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format_name}")


def import_lazyframe(
    file_path: Path,
    format_name: Optional[str] = None,
    **kwargs
) -> pl.LazyFrame:
    """Import a LazyFrame from a file.

    Args:
        file_path: Path to the file to import.
        format_name: Import format (auto-detected if None).
        **kwargs: Additional format-specific arguments.

    Returns:
        Imported LazyFrame.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Auto-detect format if not specified
    if format_name is None:
        format_name = file_path.suffix[1:]  # Remove the dot

    if format_name == "parquet":
        return pl.scan_parquet(file_path, **kwargs)
    elif format_name == "csv":
        return pl.scan_csv(file_path, **kwargs)
    elif format_name == "json":
        return pl.scan_json(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format_name}")


def get_export_file_paths(
    base_name: str,
    output_dir: Path,
    formats: Optional[list] = None
) -> Dict[str, Path]:
    """Generate file paths for multiple export formats.

    Args:
        base_name: Base name for the files.
        output_dir: Output directory.
        formats: List of formats to generate paths for.

    Returns:
        Dictionary mapping format names to file paths.
    """
    if formats is None:
        formats = [FileOperationConstants.DEFAULT_FORMAT]

    file_paths = {}
    for format_name in formats:
        extension = get_file_extension(format_name)
        file_paths[format_name] = output_dir / f"{base_name}{extension}"

    return file_paths


def validate_file_path(file_path: Path, create_parent: bool = True) -> None:
    """Validate a file path for writing.

    Args:
        file_path: File path to validate.
        create_parent: Whether to create parent directories.

    Raises:
        ValueError: If the path is invalid.
    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    # Check if parent directory exists or can be created
    if create_parent:
        ensure_directory_exists(file_path.parent)
    elif not file_path.parent.exists():
        raise ValueError(f"Parent directory does not exist: {file_path.parent}")

    # Check if we have write permissions
    try:
        # Try to create the parent directory to test permissions
        if create_parent:
            file_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise ValueError(f"No write permission for directory: {file_path.parent}")


def list_files_by_pattern(directory: Path, pattern: str) -> list[Path]:
    """List files in a directory matching a pattern.

    Args:
        directory: Directory to search in.
        pattern: Glob pattern to match.

    Returns:
        List of matching file paths.
    """
    if not directory.exists():
        return []

    return list(directory.glob(pattern))
