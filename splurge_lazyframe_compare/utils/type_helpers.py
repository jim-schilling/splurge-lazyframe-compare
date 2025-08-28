"""Type helper functions for the comparison framework."""

import polars as pl


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

    try:
        return datatype.is_numeric()
    except AttributeError:
        # Fallback for older Polars versions or different data types
        return False


def get_polars_datatype_name(datatype: pl.DataType) -> str:
    """Get a human-readable name for a Polars data type.

    Args:
        datatype: Polars data type.

    Returns:
        Human-readable classname for the data type.

    Raises:
        TypeError: If datatype is None.
    """
    if datatype is None:
        raise TypeError("datatype cannot be None")

    return repr(datatype)


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
    except AttributeError:
        raise AttributeError(f"'{datatype_name}' is not a valid Polars data type")

    # Check if it's already a DataType instance (simple types)
    if isinstance(datatype_attr, pl.DataType):
        return datatype_attr

    # Handle complex types that need instantiation
    if hasattr(datatype_attr, '__call__'):
        # These are classes that need to be instantiated
        if datatype_name == "Datetime":
            # Default to microsecond precision with no timezone
            return datatype_attr(time_unit="us", time_zone=None)
        elif datatype_name == "Categorical":
            # Default categorical with no ordering
            return datatype_attr(ordering="physical")
        elif datatype_name == "List":
            # List needs an inner type - default to Int64
            return datatype_attr(pl.Int64)
        elif datatype_name == "Struct":
            # Struct needs fields - default to empty struct
            return datatype_attr([])
        elif datatype_name == "Duration":
            # Duration needs time unit
            return datatype_attr(time_unit="us")
        elif datatype_name == "Decimal":
            # Decimal needs precision and scale
            return datatype_attr(precision=None, scale=None)
        else:
            # Try to instantiate with no arguments as fallback
            try:
                return datatype_attr()
            except TypeError:
                raise ValueError(f"Cannot instantiate {datatype_name} without parameters")

    # If we get here, it's not a valid datatype
    raise ValueError(f"'{datatype_name}' is not a valid Polars data type")
