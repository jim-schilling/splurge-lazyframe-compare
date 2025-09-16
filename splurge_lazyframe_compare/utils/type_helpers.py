"""Type helper functions for the comparison framework.

Copyright (c) 2025 Jim Schilling.
Licensed under the MIT License. See the LICENSE file for details.
"""

DOMAINS: list[str] = ["utils", "types", "polars"]

import polars as pl

# Constants
DEFAULT_DECIMAL_PRECISION: int = 38
DEFAULT_DECIMAL_SCALE: int = 9
DEFAULT_TIME_UNIT: str = "us"
DEFAULT_LIST_INNER_TYPE: pl.DataType = pl.Int64


def is_numeric_datatype(datatype: pl.DataType) -> bool:
    """Check if a Polars data type is numeric.

    Args:
        datatype: Polars data type to check.

    Returns:
        True if the data type is numeric, False otherwise.

    Raises:
        TypeError: If datatype is None.
    """
    if datatype is None:
        raise TypeError("datatype cannot be None")

    return datatype.is_numeric()


def _create_decimal_type(decimal_attr: type) -> pl.DataType:
    """Create a Decimal data type with appropriate precision and scale.

    Args:
        decimal_attr: The Decimal class from the polars module.

    Returns:
        Polars Decimal data type with appropriate parameters.
    """
    # Create Decimal with sensible defaults
    return decimal_attr(precision=DEFAULT_DECIMAL_PRECISION, scale=DEFAULT_DECIMAL_SCALE)


def get_polars_datatype_name(datatype: pl.DataType) -> str:
    """Get a human-readable name for a Polars data type.

    Args:
        datatype: Polars data type.

    Returns:
        Human-readable name for the data type (class name without parameters).

    Raises:
        TypeError: If datatype is None.
    """
    if datatype is None:
        raise TypeError("datatype cannot be None")

    # Type mapping for reliable name resolution
    type_mappings = {
        # Handle both Utf8 and String (they're the same type)
        pl.Utf8: "String",
        pl.String: "String",
        # Handle other common types
        pl.Int8: "Int8",
        pl.Int16: "Int16",
        pl.Int32: "Int32",
        pl.Int64: "Int64",
        pl.UInt8: "UInt8",
        pl.UInt16: "UInt16",
        pl.UInt32: "UInt32",
        pl.UInt64: "UInt64",
        pl.Float32: "Float32",
        pl.Float64: "Float64",
        pl.Boolean: "Boolean",
        pl.Datetime: "Datetime",
        pl.Date: "Date",
        pl.Time: "Time",
        pl.Duration: "Duration",
        pl.Categorical: "Categorical",
        pl.Binary: "Binary",
        pl.Null: "Null",
    }

    # First, try direct type mapping for known types
    for type_class, name in type_mappings.items():
        if datatype == type_class:
            return name

    # Try __name__ for simple types (works for class constants like pl.Int64)
    if hasattr(datatype, '__name__'):
        name = datatype.__name__
        # Handle the Utf8 -> String alias consistently
        if name == "Utf8":
            return "String"
        return name

    # Fall back to __class__.__name__ (works for instances like Struct, List)
    class_name = datatype.__class__.__name__
    # Handle the Utf8 -> String alias consistently
    if class_name == "Utf8":
        return "String"
    return class_name


def get_polars_datatype_type(datatype_name: str) -> pl.DataType:
    """Get a Polars data type from a human-readable classname.

    Args:
        datatype_name: Human-readable classname for the data type.

    Returns:
        Polars data type.

    Raises:
        TypeError: If datatype_name is None.
        AttributeError: If the datatype name is not valid or empty.
        ValueError: If the datatype cannot be instantiated.
    """
    if datatype_name is None:
        raise TypeError("datatype_name cannot be None")

    if not datatype_name or not datatype_name.strip():
        raise AttributeError("datatype_name cannot be empty or whitespace-only")

    # Get the attribute from polars module
    try:
        datatype_attr = getattr(pl, datatype_name)
    except AttributeError as err:
        raise AttributeError(f"'{datatype_name}' is not a valid Polars data type") from err

    # Check if it's already a DataType instance (simple types)
    if isinstance(datatype_attr, pl.DataType):
        return datatype_attr

    # Handle complex types that need instantiation
    if callable(datatype_attr):
        # These are classes that need to be instantiated
        if datatype_name == "Datetime":
            # Default to microsecond precision with no timezone
            return datatype_attr(time_unit=DEFAULT_TIME_UNIT, time_zone=None)
        elif datatype_name == "Categorical":
            # Default categorical with physical ordering
            return datatype_attr(ordering="physical")
        elif datatype_name == "List":
            # List needs an inner type - default to Int64
            return datatype_attr(DEFAULT_LIST_INNER_TYPE)
        elif datatype_name == "Struct":
            # Struct needs fields - default to empty struct
            return datatype_attr([])
        elif datatype_name == "Duration":
            # Duration needs time unit - default to microseconds
            return datatype_attr(time_unit=DEFAULT_TIME_UNIT)
        elif datatype_name == "Decimal":
            return _create_decimal_type(datatype_attr)
        else:
            # Try to instantiate with no arguments as fallback
            try:
                return datatype_attr()
            except TypeError as err:
                raise ValueError(f"Cannot instantiate {datatype_name} without parameters") from err

    # If we get here, it's not a valid datatype
    raise ValueError(f"'{datatype_name}' is not a valid Polars data type")
