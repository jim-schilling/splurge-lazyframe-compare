"""Type helper functions for the comparison framework."""

import polars as pl


def is_numeric_dtype(dtype: pl.DataType) -> bool:
    """Check if a Polars data type is numeric.

    Args:
        dtype: Polars data type to check.

    Returns:
        True if the data type is numeric, False otherwise.
    """
    try:
        return dtype.is_numeric()
    except AttributeError:
        # Fallback for older Polars versions or different data types
        return False


def get_friendly_dtype_name(dtype: pl.DataType) -> str:
    """Get a human-readable name for a Polars data type.

    Args:
        dtype: Polars data type.

    Returns:
        Human-readable name for the data type.
    """
    dtype_map = {
        pl.Int8: "8-bit integer",
        pl.Int16: "16-bit integer",
        pl.Int32: "32-bit integer",
        pl.Int64: "64-bit integer",
        pl.UInt8: "8-bit unsigned integer",
        pl.UInt16: "16-bit unsigned integer",
        pl.UInt32: "32-bit unsigned integer",
        pl.UInt64: "64-bit unsigned integer",
        pl.Float32: "32-bit float",
        pl.Float64: "64-bit float",
        pl.Utf8: "string",
        pl.Boolean: "boolean",
        pl.Date: "date",
        pl.Datetime: "datetime",
        pl.Time: "time",
        pl.Duration: "duration",
        pl.Categorical: "categorical",
        pl.List: "list",
        pl.Struct: "struct",
        pl.Null: "null",
    }

    base_type = type(dtype)
    return dtype_map.get(base_type, str(dtype))
