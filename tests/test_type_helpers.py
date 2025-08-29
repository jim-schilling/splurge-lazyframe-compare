"""Tests for the type_helpers module."""

import polars as pl
import pytest

from splurge_lazyframe_compare.utils.type_helpers import (
    get_polars_datatype_name,
    get_polars_datatype_type,
    is_numeric_datatype,
)


class TestIsNumericDatatype:
    """Test is_numeric_datatype function."""

    def test_numeric_types(self) -> None:
        """Test that numeric data types return True."""
        # Integer types
        assert is_numeric_datatype(pl.Int8) is True
        assert is_numeric_datatype(pl.Int16) is True
        assert is_numeric_datatype(pl.Int32) is True
        assert is_numeric_datatype(pl.Int64) is True

        # Unsigned integer types
        assert is_numeric_datatype(pl.UInt8) is True
        assert is_numeric_datatype(pl.UInt16) is True
        assert is_numeric_datatype(pl.UInt32) is True
        assert is_numeric_datatype(pl.UInt64) is True

        # Float types
        assert is_numeric_datatype(pl.Float32) is True
        assert is_numeric_datatype(pl.Float64) is True

    def test_non_numeric_types(self) -> None:
        """Test that non-numeric data types return False."""
        assert is_numeric_datatype(pl.Utf8) is False
        assert is_numeric_datatype(pl.Boolean) is False
        assert is_numeric_datatype(pl.Date) is False
        assert is_numeric_datatype(pl.Datetime) is False
        assert is_numeric_datatype(pl.Time) is False
        assert is_numeric_datatype(pl.Duration) is False
        assert is_numeric_datatype(pl.Categorical) is False
        assert is_numeric_datatype(pl.List) is False
        assert is_numeric_datatype(pl.Struct) is False
        assert is_numeric_datatype(pl.Null) is False

    def test_edge_cases(self) -> None:
        """Test edge cases and error handling."""
        # Test with None (should raise TypeError)
        with pytest.raises(TypeError):
            is_numeric_datatype(None)


class TestGetPolarsDatatypeName:
    """Test get_polars_datatype_name function."""

    def test_integer_types(self) -> None:
        """Test integer data type names."""
        assert get_polars_datatype_name(pl.Int8) == "Int8"
        assert get_polars_datatype_name(pl.Int16) == "Int16"
        assert get_polars_datatype_name(pl.Int32) == "Int32"
        assert get_polars_datatype_name(pl.Int64) == "Int64"

    def test_unsigned_integer_types(self) -> None:
        """Test unsigned integer data type names."""
        assert get_polars_datatype_name(pl.UInt8) == "UInt8"
        assert get_polars_datatype_name(pl.UInt16) == "UInt16"
        assert get_polars_datatype_name(pl.UInt32) == "UInt32"
        assert get_polars_datatype_name(pl.UInt64) == "UInt64"

    def test_float_types(self) -> None:
        """Test float data type names."""
        assert get_polars_datatype_name(pl.Float32) == "Float32"
        assert get_polars_datatype_name(pl.Float64) == "Float64"

    def test_string_and_text_types(self) -> None:
        """Test string and text data type names."""
        assert get_polars_datatype_name(pl.Utf8) == "String"

    def test_boolean_type(self) -> None:
        """Test boolean data type name."""
        assert get_polars_datatype_name(pl.Boolean) == "Boolean"

    def test_temporal_types(self) -> None:
        """Test temporal data type names."""
        assert get_polars_datatype_name(pl.Date) == "Date"
        assert get_polars_datatype_name(pl.Datetime) == "Datetime"
        assert get_polars_datatype_name(pl.Time) == "Time"
        assert get_polars_datatype_name(pl.Duration) == "Duration"

    def test_categorical_type(self) -> None:
        """Test categorical data type name."""
        assert get_polars_datatype_name(pl.Categorical) == "Categorical"

    def test_complex_types(self) -> None:
        """Test complex data type names."""
        assert get_polars_datatype_name(pl.List) == "List"
        assert get_polars_datatype_name(pl.Struct) == "Struct"
        assert get_polars_datatype_name(pl.Null) == "Null"

    def test_custom_datatype_instances(self) -> None:
        """Test with actual datatype instances."""
        schema = pl.Schema({"col": pl.Int64})
        assert get_polars_datatype_name(schema["col"]) == "Int64"

        schema = pl.Schema({"col": pl.Utf8})
        assert get_polars_datatype_name(schema["col"]) == "String"

    def test_edge_cases(self) -> None:
        """Test edge cases."""
        # Test with None (should raise TypeError)
        with pytest.raises(TypeError):
            get_polars_datatype_name(None)


class TestGetPolarsDatatypeType:
    """Test get_polars_datatype_type function."""

    def test_integer_types(self) -> None:
        """Test getting integer data types from names."""
        assert get_polars_datatype_type("Int8") == pl.Int8
        assert get_polars_datatype_type("Int16") == pl.Int16
        assert get_polars_datatype_type("Int32") == pl.Int32
        assert get_polars_datatype_type("Int64") == pl.Int64

    def test_unsigned_integer_types(self) -> None:
        """Test getting unsigned integer data types from names."""
        assert get_polars_datatype_type("UInt8") == pl.UInt8
        assert get_polars_datatype_type("UInt16") == pl.UInt16
        assert get_polars_datatype_type("UInt32") == pl.UInt32
        assert get_polars_datatype_type("UInt64") == pl.UInt64

    def test_float_types(self) -> None:
        """Test getting float data types from names."""
        assert get_polars_datatype_type("Float32") == pl.Float32
        assert get_polars_datatype_type("Float64") == pl.Float64

    def test_string_and_text_types(self) -> None:
        """Test getting string and text data types from names."""
        assert get_polars_datatype_type("String") == pl.Utf8

    def test_boolean_type(self) -> None:
        """Test getting boolean data type from name."""
        assert get_polars_datatype_type("Boolean") == pl.Boolean

    def test_temporal_types(self) -> None:
        """Test getting temporal data types from names."""
        assert get_polars_datatype_type("Date") == pl.Date
        assert get_polars_datatype_type("Datetime") == pl.Datetime
        assert get_polars_datatype_type("Time") == pl.Time
        assert get_polars_datatype_type("Duration") == pl.Duration

    def test_categorical_type(self) -> None:
        """Test getting categorical data type from name."""
        assert get_polars_datatype_type("Categorical") == pl.Categorical

    def test_complex_types(self) -> None:
        """Test getting complex data types from names."""
        assert get_polars_datatype_type("List") == pl.List
        assert get_polars_datatype_type("Struct") == pl.Struct
        assert get_polars_datatype_type("Null") == pl.Null

    def test_exact_case_conversion(self) -> None:
        """Test that the function handles exact case input correctly."""
        # Since we removed .title(), we need exact case matching
        assert get_polars_datatype_type("Int8") == pl.Int8
        assert get_polars_datatype_type("String") == pl.Utf8
        assert get_polars_datatype_type("Boolean") == pl.Boolean
        assert get_polars_datatype_type("Float64") == pl.Float64

        # Test that incorrect case fails
        with pytest.raises(AttributeError):
            get_polars_datatype_type("int64")
        with pytest.raises(AttributeError):
            get_polars_datatype_type("uInt32")
        with pytest.raises(AttributeError):
            get_polars_datatype_type("FLOAT32")

    def test_round_trip_conversion(self) -> None:
        """Test that name -> type -> name conversion works correctly."""
        test_types = [
            pl.Int8, pl.Int64, pl.Float32, pl.Utf8, pl.Boolean,
            pl.Date, pl.Datetime, pl.Categorical, pl.List, pl.Null
        ]

        for dtype in test_types:
            name = get_polars_datatype_name(dtype)
            converted_type = get_polars_datatype_type(name)
            assert converted_type == dtype, f"Failed for {dtype}: got {converted_type}"

    def test_invalid_datatype_names(self) -> None:
        """Test error handling for invalid datatype names."""
        with pytest.raises(AttributeError):
            get_polars_datatype_type("InvalidType")

        with pytest.raises(AttributeError):
            get_polars_datatype_type("NotARealType")

        with pytest.raises(AttributeError):
            get_polars_datatype_type("")

    def test_edge_cases(self) -> None:
        """Test edge cases and error handling."""
        # Test with None
        with pytest.raises(TypeError):
            get_polars_datatype_type(None)

        # Test with empty string
        with pytest.raises(AttributeError):
            get_polars_datatype_type("")

        # Test with whitespace
        with pytest.raises(AttributeError):
            get_polars_datatype_type("   ")

        # Test with special characters
        with pytest.raises(AttributeError):
            get_polars_datatype_type("Int64!")


class TestTypeHelpersIntegration:
    """Integration tests for type helper functions."""

    def test_complete_workflow(self) -> None:
        """Test complete workflow of name <-> type conversion."""
        # Test various data types
        test_cases = [
            (pl.Int8, "numeric"),
            (pl.Float64, "numeric"),
            (pl.Utf8, "string"),
            (pl.Boolean, "boolean"),
            (pl.Date, "temporal"),
            (pl.List, "complex"),
        ]

        for dtype, category in test_cases:
            # Get name from type
            name = get_polars_datatype_name(dtype)
            assert isinstance(name, str)
            assert len(name) > 0

            # Get type from name (should use the exact name returned)
            converted_type = get_polars_datatype_type(name)
            assert converted_type == dtype

            # Verify numeric detection works
            if category == "numeric":
                assert is_numeric_datatype(dtype) is True
            else:
                assert is_numeric_datatype(dtype) is False

    def test_schema_integration(self) -> None:
        """Test integration with Polars Schema."""
        schema = pl.Schema({
            "id": pl.Int64,
            "name": pl.Utf8,
            "active": pl.Boolean,
            "score": pl.Float32,
        })

        for _col_name, dtype in schema.items():
            # Get name from schema dtype
            name = get_polars_datatype_name(dtype)
            assert isinstance(name, str)

            # Convert back to type
            converted_type = get_polars_datatype_type(name)
            assert converted_type == dtype

        # Test Datetime separately (can't round-trip due to parameters)
        datetime_dtype = pl.Datetime(time_unit="us")
        name = get_polars_datatype_name(datetime_dtype)
        assert isinstance(name, str)
        assert "Datetime" in name  # Should contain Datetime in the name

    def test_lazyframe_integration(self) -> None:
        """Test integration with LazyFrame."""
        df = pl.LazyFrame({
            "int_col": [1, 2, 3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        })

        schema = df.collect_schema()

        for _col_name, dtype in schema.items():
            name = get_polars_datatype_name(dtype)
            converted_type = get_polars_datatype_type(name)
            assert converted_type == dtype

    def test_complex_type_defaults(self) -> None:
        """Test that complex types are created with sensible defaults."""
        # Test List defaults to Int64
        list_type = get_polars_datatype_type("List")
        assert str(list_type) == "List(Int64)"

        # Test Struct defaults to empty
        struct_type = get_polars_datatype_type("Struct")
        assert str(struct_type) == "Struct({})"

        # Test Decimal defaults to precision=38, scale=9
        decimal_type = get_polars_datatype_type("Decimal")
        assert "precision=38" in str(decimal_type)
        assert "scale=9" in str(decimal_type)

    def test_parameterized_type_behavior(self) -> None:
        """Test behavior with parameterized type instances."""
        # Test that parameterized instances work correctly
        datetime_param = pl.Datetime(time_unit="ms", time_zone="UTC")
        name = get_polars_datatype_name(datetime_param)
        assert name == "Datetime"

        # Test that we can recreate the base type
        recreated = get_polars_datatype_type(name)
        assert str(recreated) == "Datetime(time_unit='us', time_zone=None)"

    def test_numeric_detection_edge_cases(self) -> None:
        """Test numeric detection with various edge cases."""
        # Test with custom objects that might have is_numeric
        class CustomNumericType:
            def is_numeric(self):
                return True

        class CustomNonNumericType:
            def is_numeric(self):
                return False

        # These should use the is_numeric method when available
        assert is_numeric_datatype(CustomNumericType()) is True
        assert is_numeric_datatype(CustomNonNumericType()) is False

    def test_error_handling_comprehensive(self) -> None:
        """Test comprehensive error handling scenarios."""
        # Test None inputs
        with pytest.raises(TypeError):
            get_polars_datatype_name(None)

        with pytest.raises(TypeError):
            get_polars_datatype_type(None)

        with pytest.raises(TypeError):
            is_numeric_datatype(None)

        # Test invalid type names
        with pytest.raises(AttributeError):
            get_polars_datatype_type("InvalidType123")

        with pytest.raises(AttributeError):
            get_polars_datatype_type("")

        with pytest.raises(AttributeError):
            get_polars_datatype_type("   ")

    def test_type_conversion_consistency(self) -> None:
        """Test that type conversions are consistent across different scenarios."""
        # Test that string names map to the same types as direct access
        string_to_direct_mapping = {
            "Int64": pl.Int64,
            "String": pl.Utf8,
            "Boolean": pl.Boolean,
            "Float32": pl.Float32,
            "Date": pl.Date,
            "List": pl.List,  # Note: this will be List(Int64) when instantiated
        }

        for string_name, direct_type in string_to_direct_mapping.items():
            # Get type from string
            from_string = get_polars_datatype_type(string_name)

            # For simple types, they should be equal
            if string_name != "List":
                assert from_string == direct_type
            else:
                # List should be List(Int64)
                assert str(from_string) == "List(Int64)"

    def test_performance_behavior(self) -> None:
        """Test that functions perform correctly under various conditions."""
        import time

        # Test that functions don't take excessive time
        start_time = time.time()

        # Perform multiple operations
        for _i in range(100):
            assert get_polars_datatype_name(pl.Int64) == "Int64"
            assert get_polars_datatype_type("Int64") == pl.Int64

            assert is_numeric_datatype(pl.Int64) is True

        end_time = time.time()
        duration = end_time - start_time

        # Should complete in reasonable time (less than 1 second for 100 iterations of 3 operations)
        assert duration < 1.0, f"Operations took too long: {duration:.3f} seconds"

    def test_memory_behavior(self) -> None:
        """Test that functions don't leak memory or create excessive objects."""
        import gc

        # Get initial object count
        initial_objects = len(gc.get_objects())

        # Perform operations that might create objects
        for _i in range(1000):
            get_polars_datatype_name(pl.Int64)
            get_polars_datatype_type("Int64")
            is_numeric_datatype(pl.Int64)

        # Force garbage collection
        gc.collect()

        # Check that we haven't created excessive objects
        final_objects = len(gc.get_objects())
        object_growth = final_objects - initial_objects

        # Allow some growth but not excessive (should be much less than 1000)
        assert object_growth < 100, f"Excessive object creation: {object_growth} objects"
