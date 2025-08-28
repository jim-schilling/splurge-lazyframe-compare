"""Tests for the schema module."""


import polars as pl
import pytest

from splurge_lazyframe_compare.exceptions.comparison_exceptions import SchemaValidationError
from splurge_lazyframe_compare.models.schema import (
    ColumnDefinition,
    ColumnMapping,
    ComparisonConfig,
    ComparisonSchema,
)


class TestColumnDefinition:
    """Test ColumnDefinition class."""

    def test_column_definition_creation(self) -> None:
        """Test creating a ColumnDefinition."""
        col_def = ColumnDefinition(
            name="test_col",
            alias="Test Column",
            datatype=pl.Int64,
            nullable=False,
        )

        assert col_def.name == "test_col"
        assert col_def.alias == "Test Column"
        assert col_def.datatype == pl.Int64
        assert col_def.nullable is False

    def test_validate_column_exists(self) -> None:
        """Test column existence validation."""
        col_def = ColumnDefinition(
            name="test_col",
            alias="Test Column",
            datatype=pl.Int64,
            nullable=False,
        )

        # Test with existing column
        df = pl.LazyFrame({"test_col": [1, 2, 3]})
        assert col_def.validate_column_exists(df) is True

        # Test with missing column
        df = pl.LazyFrame({"other_col": [1, 2, 3]})
        assert col_def.validate_column_exists(df) is False

    def test_validate_data_type(self) -> None:
        """Test data type validation."""
        col_def = ColumnDefinition(
            name="test_col",
            alias="Test Column",
            datatype=pl.Int64,
            nullable=False,
        )

        # Test with correct data type
        df = pl.LazyFrame({"test_col": [1, 2, 3]})
        assert col_def.validate_data_type(df) is True

        # Test with wrong data type
        df = pl.LazyFrame({"test_col": ["a", "b", "c"]})
        assert col_def.validate_data_type(df) is False

        # Test with missing column
        df = pl.LazyFrame({"other_col": [1, 2, 3]})
        assert col_def.validate_data_type(df) is False


class TestColumnMapping:
    """Test ColumnMapping class."""

    def test_column_mapping_creation(self) -> None:
        """Test creating a ColumnMapping."""
        mapping = ColumnMapping(
            left="left_col",
            right="right_col",
            name="standard_name",
        )

        assert mapping.left == "left_col"
        assert mapping.right == "right_col"
        assert mapping.name == "standard_name"


class TestComparisonSchema:
    """Test ComparisonSchema class."""

    def test_schema_creation(self) -> None:
        """Test creating a ComparisonSchema."""
        columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=True),
        }
        schema = ComparisonSchema(
            columns=columns,
            pk_columns=["id"],
        )

        assert len(schema.columns) == 2
        assert "id" in schema.columns
        assert "name" in schema.columns
        assert schema.pk_columns == ["id"]

    def test_validate_schema_success(self) -> None:
        """Test successful schema validation."""
        columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=True),
        }
        schema = ComparisonSchema(
            columns=columns,
            pk_columns=["id"],
        )

        df = pl.LazyFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        })

        errors = schema.validate_schema(df)
        assert len(errors) == 0

    def test_validate_schema_missing_columns(self) -> None:
        """Test schema validation with missing columns."""
        columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=True),
        }
        schema = ComparisonSchema(
            columns=columns,
            pk_columns=["id"],
        )

        df = pl.LazyFrame({"id": [1, 2, 3]})  # Missing 'name' column

        errors = schema.validate_schema(df)
        assert len(errors) == 1
        assert "Missing columns" in errors[0]

    def test_validate_schema_wrong_data_type(self) -> None:
        """Test schema validation with wrong data types."""
        columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=True),
        }
        schema = ComparisonSchema(
            columns=columns,
            pk_columns=["id"],
        )

        df = pl.LazyFrame({
            "id": ["1", "2", "3"],  # Wrong type (string instead of int)
            "name": ["Alice", "Bob", "Charlie"],
        })

        errors = schema.validate_schema(df)
        assert len(errors) == 1
        assert "expected" in errors[0] and "got" in errors[0]

    def test_validate_schema_nullable_violation(self) -> None:
        """Test schema validation with nullable constraint violations."""
        columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=False),  # Not nullable
        }
        schema = ComparisonSchema(
            columns=columns,
            pk_columns=["id"],
        )

        df = pl.LazyFrame({
            "id": [1, 2, 3],
            "name": ["Alice", None, "Charlie"],  # Contains null
        })

        errors = schema.validate_schema(df)
        assert len(errors) == 1
        assert "null values found" in errors[0]

    def test_validate_schema_missing_primary_key(self) -> None:
        """Test schema validation with missing primary key column."""
        columns = {
            "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=True),
        }
        schema = ComparisonSchema(
            columns=columns,
            pk_columns=["id"],  # 'id' not in columns
        )

        df = pl.LazyFrame({"name": ["Alice", "Bob", "Charlie"]})

        errors = schema.validate_schema(df)
        assert len(errors) == 1
        assert "not defined in schema" in errors[0]

    def test_get_primary_key_definition(self) -> None:
        """Test getting primary key column definitions."""
        columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=True),
        }
        schema = ComparisonSchema(
            columns=columns,
            pk_columns=["id"],
        )

        pk_defs = schema.get_primary_key_definition()
        assert len(pk_defs) == 1
        assert pk_defs[0].name == "id"

    def test_get_compare_columns(self) -> None:
        """Test getting non-primary key columns."""
        columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=True),
            "age": ColumnDefinition(name="age", alias="Age", datatype=pl.Int64, nullable=True),
        }
        schema = ComparisonSchema(
            columns=columns,
            pk_columns=["id"],
        )

        compare_cols = schema.get_compare_columns()
        assert len(compare_cols) == 2
        assert "name" in compare_cols
        assert "age" in compare_cols
        assert "id" not in compare_cols


class TestComparisonConfig:
    """Test ComparisonConfig class."""

    def test_config_creation(self) -> None:
        """Test creating a ComparisonConfig."""
        left_columns = {
            "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "amount": ColumnDefinition(name="amount", alias="Amount", datatype=pl.Float64, nullable=False),
        }
        right_columns = {
            "cust_id": ColumnDefinition(name="cust_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "total": ColumnDefinition(name="total", alias="Amount", datatype=pl.Float64, nullable=False),
        }

        left_schema = ComparisonSchema(
            columns=left_columns,
            pk_columns=["customer_id"],
        )
        right_schema = ComparisonSchema(
            columns=right_columns,
            pk_columns=["cust_id"],
        )

        mappings = [
            ColumnMapping(left="customer_id", right="cust_id", name="customer_id"),
            ColumnMapping(left="amount", right="total", name="amount"),
        ]

        config = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=mappings,
            primary_key_columns=["customer_id"],
        )

        assert config.left_schema == left_schema
        assert config.right_schema == right_schema
        assert len(config.column_mappings) == 2
        assert config.primary_key_columns == ["customer_id"]
        assert config.ignore_case is False
        assert config.null_equals_null is True

    def test_config_validation_empty_schemas(self) -> None:
        """Test config validation with empty schemas."""
        left_schema = ComparisonSchema(columns={}, pk_columns=[])
        right_schema = ComparisonSchema(columns={}, pk_columns=[])

        with pytest.raises(SchemaValidationError) as exc_info:
            ComparisonConfig(
                left_schema=left_schema,
                right_schema=right_schema,
                column_mappings=[],
                primary_key_columns=[],
            )

        assert "no columns defined" in str(exc_info.value)

    def test_config_validation_no_primary_keys(self) -> None:
        """Test config validation with no primary keys."""
        left_columns = {
            "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
        }
        right_columns = {
            "cust_id": ColumnDefinition(name="cust_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
        }

        left_schema = ComparisonSchema(
            columns=left_columns,
            pk_columns=["customer_id"],
        )
        right_schema = ComparisonSchema(
            columns=right_columns,
            pk_columns=["cust_id"],
        )

        mappings = [
            ColumnMapping(left="customer_id", right="cust_id", name="customer_id"),
        ]

        with pytest.raises(SchemaValidationError) as exc_info:
            ComparisonConfig(
                left_schema=left_schema,
                right_schema=right_schema,
                column_mappings=mappings,
                primary_key_columns=[],  # No primary keys
            )

        assert "No primary key columns defined" in str(exc_info.value)

    def test_config_validation_missing_mapped_columns(self) -> None:
        """Test config validation with missing mapped columns."""
        left_columns = {
            "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
        }
        right_columns = {
            "cust_id": ColumnDefinition(name="cust_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
        }

        left_schema = ComparisonSchema(
            columns=left_columns,
            pk_columns=["customer_id"],
        )
        right_schema = ComparisonSchema(
            columns=right_columns,
            pk_columns=["cust_id"],
        )

        # Mapping references non-existent column
        mappings = [
            ColumnMapping(left="customer_id", right="cust_id", name="customer_id"),
            ColumnMapping(left="missing_col", right="cust_id", name="missing"),  # 'missing_col' not in left schema
        ]

        with pytest.raises(SchemaValidationError) as exc_info:
            ComparisonConfig(
                left_schema=left_schema,
                right_schema=right_schema,
                column_mappings=mappings,
                primary_key_columns=["customer_id"],
            )

        assert "missing mapped columns" in str(exc_info.value)

    def test_config_validation_missing_primary_key_mapping(self) -> None:
        """Test config validation with missing primary key mapping."""
        left_columns = {
            "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "amount": ColumnDefinition(name="amount", alias="Amount", datatype=pl.Float64, nullable=False),
        }
        right_columns = {
            "cust_id": ColumnDefinition(name="cust_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "total": ColumnDefinition(name="total", alias="Amount", datatype=pl.Float64, nullable=False),
        }

        left_schema = ComparisonSchema(
            columns=left_columns,
            pk_columns=["customer_id"],
        )
        right_schema = ComparisonSchema(
            columns=right_columns,
            pk_columns=["cust_id"],
        )

        # Mapping doesn't include primary key
        mappings = [
            ColumnMapping(left="amount", right="total", name="amount"),
        ]

        with pytest.raises(SchemaValidationError) as exc_info:
            ComparisonConfig(
                left_schema=left_schema,
                right_schema=right_schema,
                column_mappings=mappings,
                primary_key_columns=["customer_id"],
            )

        assert "not found in column mappings" in str(exc_info.value)
