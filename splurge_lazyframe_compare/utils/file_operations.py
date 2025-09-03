"""File operation utilities for the comparison framework."""

import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import polars as pl
from splurge_lazyframe_compare.utils.constants import (
    DEFAULT_FORMAT,
    FORMAT_CSV,
    FORMAT_JSON,
    FORMAT_PARQUET,
)


class FileOperationConstants:
    """Constants for file operations."""

    # Default formats
    DEFAULT_FORMAT: str = DEFAULT_FORMAT
    SUPPORTED_FORMATS: tuple = (FORMAT_PARQUET, FORMAT_CSV, FORMAT_JSON)

    # File extensions
    PARQUET_EXT: str = ".parquet"
    CSV_EXT: str = ".csv"
    JSON_EXT: str = ".json"

    # Buffer sizes
    DEFAULT_BUFFER_SIZE: int = 8192


@contextmanager
def atomic_write(file_path: Path) -> Generator[Path, None, None]:
    """Context manager for atomic file writes.

    Creates a temporary file and only renames it to the target path
    if the operation succeeds. This ensures that partial writes don't
    corrupt the target file.

    Args:
        file_path: Target file path

    Yields:
        Path to temporary file to write to
    """
    # Ensure parent directory exists
    ensure_directory_exists(file_path.parent)

    # Create temporary file in the same directory as target
    with tempfile.NamedTemporaryFile(
        dir=file_path.parent,
        prefix=f"{file_path.stem}_tmp_",
        suffix=file_path.suffix,
        delete=False
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        yield temp_path
        # Atomic rename only if successful
        temp_path.replace(file_path)
    except Exception:
        # Clean up temp file on failure
        if temp_path.exists():
            temp_path.unlink()
        raise
    finally:
        # Ensure cleanup in case of any other issues
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass  # Ignore cleanup errors


def validate_file_path(file_path: Path, require_writable: bool = False) -> None:
    """Validate file path for security and accessibility.

    Args:
        file_path: File path to validate
        require_writable: Whether to check if path is writable

    Raises:
        ValueError: If path is invalid or insecure
        PermissionError: If path is not accessible
    """
    if not file_path:
        raise ValueError("File path cannot be empty")

    # Convert to absolute path and resolve any symlinks
    resolved_path = file_path.resolve()

    # Check for directory traversal attempts
    try:
        resolved_path.relative_to(file_path.parent.resolve())
    except ValueError as err:
        # Path goes outside the intended directory
        raise ValueError(f"Invalid path: {file_path} (directory traversal detected)") from err

    # Ensure parent directory exists
    ensure_directory_exists(file_path.parent)

    # Check if parent directory is writable (if required)
    if require_writable:
        parent = resolved_path.parent
        if not os.access(parent, os.W_OK):
            raise PermissionError(f"Cannot write to directory: {parent}")


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
        FORMAT_PARQUET: FileOperationConstants.PARQUET_EXT,
        FORMAT_CSV: FileOperationConstants.CSV_EXT,
        FORMAT_JSON: FileOperationConstants.JSON_EXT,
    }

    if format_name not in format_map:
        raise ValueError(
            f"Unsupported format: {format_name}. Supported formats: "
            f"{FileOperationConstants.SUPPORTED_FORMATS}"
        )

    return format_map[format_name]


def export_lazyframe(
    lazyframe: pl.LazyFrame,
    file_path: Path,
    format_name: str = FileOperationConstants.DEFAULT_FORMAT,
    **kwargs: Any
) -> None:
    """Export a LazyFrame to a file with atomic writes and proper cleanup.

    Args:
        lazyframe: LazyFrame to export.
        file_path: Path to save the file.
        format_name: Export format (parquet, csv, json).
        **kwargs: Additional format-specific arguments.

    Raises:
        ValueError: If format is not supported.
        PermissionError: If target location is not writable.
        OSError: If file operation fails.
    """
    # Validate file path for security
    validate_file_path(file_path, require_writable=True)

    # Use atomic writes for data integrity
    with atomic_write(file_path) as temp_path:
        try:
            if format_name == FORMAT_PARQUET:
                # Safe default compression for parquet unless explicitly provided
                if "compression" not in kwargs:
                    kwargs["compression"] = "zstd"
                lazyframe.sink_parquet(temp_path, **kwargs)
            elif format_name == FORMAT_CSV:
                # Include header by default for CSV unless overridden
                kwargs.setdefault("include_header", True)
                lazyframe.sink_csv(temp_path, **kwargs)
            elif format_name == FORMAT_JSON:
                lazyframe.sink_ndjson(temp_path, **kwargs)
            else:
                raise ValueError(f"Unsupported format: {format_name}")
        except ValueError:
            # Re-raise ValueError as-is for format validation errors
            raise
        except Exception as e:
            # Clean up temp file on any error
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise OSError(f"Failed to export LazyFrame to {file_path}: {e}") from e


def import_lazyframe(
    file_path: Path,
    format_name: str | None = None,
    **kwargs: Any
) -> pl.LazyFrame:
    """Import a LazyFrame from a file with proper validation and error handling.

    Args:
        file_path: Path to the file to import.
        format_name: Import format (auto-detected if None).
        **kwargs: Additional format-specific arguments.

    Returns:
        Imported LazyFrame.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If format is not supported or file path is invalid.
        PermissionError: If file is not readable.
        OSError: If file operation fails.
    """
    # Validate file path for security
    validate_file_path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"Cannot read file: {file_path}")

    # Check file size (prevent reading extremely large files)
    file_size = file_path.stat().st_size
    if file_size == 0:
        raise ValueError(f"File is empty: {file_path}")

    # Auto-detect format if not specified
    if format_name is None:
        suffix = file_path.suffix.lower()
        if suffix == FileOperationConstants.PARQUET_EXT:
            format_name = FORMAT_PARQUET
        elif suffix == FileOperationConstants.CSV_EXT:
            format_name = FORMAT_CSV
        elif suffix in (FileOperationConstants.JSON_EXT, ".ndjson"):
            format_name = FORMAT_JSON
        else:
            raise ValueError(f"Cannot auto-detect format for extension: {suffix}")

    try:
        if format_name == FORMAT_PARQUET:
            return pl.scan_parquet(file_path, **kwargs)
        elif format_name == FORMAT_CSV:
            return pl.scan_csv(file_path, **kwargs)
        elif format_name == FORMAT_JSON:
            return pl.scan_ndjson(file_path, **kwargs)
        else:
            raise ValueError(
                f"Unsupported format: {format_name}. Supported formats: "
                f"{list(FileOperationConstants.SUPPORTED_FORMATS)}"
            )
    except ValueError:
        # Re-raise ValueError as-is for format validation errors
        raise
    except Exception as e:
        raise OSError(f"Failed to import LazyFrame from {file_path}: {e}") from e


def get_export_file_paths(
    base_name: str,
    output_dir: Path,
    formats: list | None = None
) -> dict[str, Path]:
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
