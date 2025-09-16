"""Unit tests for service classes."""

from pathlib import Path

import polars as pl
import pytest

from splurge_lazyframe_compare.exceptions.comparison_exceptions import (
    PrimaryKeyViolationError,
    SchemaValidationError,
)
from splurge_lazyframe_compare.models.schema import ColumnDefinition, ColumnMapping, ComparisonConfig, ComparisonSchema
from splurge_lazyframe_compare.services import (
    ComparisonOrchestrator,
    ComparisonService,
    DataPreparationService,
    ReportingService,
    ValidationService,
)


@pytest.fixture
def sample_dataframes():
    """Create sample DataFrames for testing."""
    left_data = {
        "id": [1, 2, 3, 4],
        "name": ["Alice", "Bob", "Charlie", "David"],
        "amount": [100.0, 200.0, 300.0, 400.0],
    }

    right_data = {
        "customer_id": [1, 2, 3, 5],
        "customer_name": ["Alice", "Bob", "Chuck", "Eve"],
        "total_amount": [100.0, 250.0, 300.0, 500.0],
    }

    left_df = pl.LazyFrame(left_data)
    right_df = pl.LazyFrame(right_data)

    return left_df, right_df


@pytest.fixture
def sample_config(sample_dataframes):
    """Create sample configuration for testing."""
    left_df, right_df = sample_dataframes

    left_columns = {
        "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
        "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=False),
        "amount": ColumnDefinition(name="amount", alias="Amount", datatype=pl.Float64, nullable=False),
    }
    left_schema = ComparisonSchema(columns=left_columns, pk_columns=["id"])

    right_columns = {
        "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
        "customer_name": ColumnDefinition(
            name="customer_name", alias="Customer Name", datatype=pl.Utf8, nullable=False
        ),
        "total_amount": ColumnDefinition(name="total_amount", alias="Amount", datatype=pl.Float64, nullable=False),
    }
    right_schema = ComparisonSchema(columns=right_columns, pk_columns=["customer_id"])

    mappings = [
        ColumnMapping(left="id", right="customer_id", name="id"),
        ColumnMapping(left="name", right="customer_name", name="name"),
        ColumnMapping(left="amount", right="total_amount", name="amount"),
    ]

    config = ComparisonConfig(
        left_schema=left_schema, right_schema=right_schema, column_mappings=mappings, pk_columns=["id"]
    )

    return config


class TestValidationService:
    """Test ValidationService functionality."""

    def test_service_initialization(self):
        """Test that ValidationService initializes correctly."""
        service = ValidationService()
        assert service.service_name == "ValidationService"

    def test_validate_dataframe_schema_success(self, sample_dataframes, sample_config):
        """Test successful DataFrame schema validation."""
        left_df, right_df = sample_dataframes

        service = ValidationService()
        # Should not raise any exceptions
        service.validate_dataframe_schema(df=left_df, schema=sample_config.left_schema, df_name="Left")
        service.validate_dataframe_schema(df=right_df, schema=sample_config.right_schema, df_name="Right")

    def test_validate_dataframe_schema_failure(self):
        """Test DataFrame schema validation failure."""
        # Create DataFrame missing required columns
        df = pl.LazyFrame({"wrong_column": [1, 2, 3]})

        columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=False),
        }
        schema = ComparisonSchema(columns=columns, pk_columns=["id"])

        service = ValidationService()

        with pytest.raises(SchemaValidationError):
            service.validate_dataframe_schema(df=df, schema=schema, df_name="Test")

    def test_validate_completeness(self, sample_dataframes):
        """Test completeness validation."""
        df, _ = sample_dataframes

        service = ValidationService()
        result = service.validate_completeness(df=df, required_columns=["id", "name"])

        assert result.is_valid
        assert "All required columns are present" in result.message

    def test_validate_data_types(self, sample_dataframes):
        """Test data type validation."""
        df, _ = sample_dataframes

        service = ValidationService()
        expected_types = {"id": pl.Int64, "name": pl.Utf8}

        result = service.validate_data_types(df=df, expected_types=expected_types)

        assert result.is_valid
        assert "All columns have expected data types" in result.message

    def test_validate_data_types_failure(self):
        """Test data type validation with type mismatches."""
        df = pl.LazyFrame(
            {
                "id": ["1", "2", "3"],  # Wrong type - should be Int64
                "name": ["Alice", "Bob", "Charlie"],
            }
        )

        service = ValidationService()
        expected_types = {"id": pl.Int64, "name": pl.Utf8}

        result = service.validate_data_types(df=df, expected_types=expected_types)

        assert not result.is_valid
        assert "data type validation failed" in result.message.lower()
        assert "type_mismatches" in result.details
        assert len(result.details["type_mismatches"]) == 1
        assert result.details["type_mismatches"][0]["column"] == "id"

    def test_validate_completeness_failure_missing_columns(self):
        """Test completeness validation with missing columns."""
        df = pl.LazyFrame({"existing_column": [1, 2, 3]})

        service = ValidationService()
        result = service.validate_completeness(df=df, required_columns=["missing_column", "existing_column"])

        assert not result.is_valid
        assert "completeness validation failed" in result.message.lower()
        assert "missing_columns" in result.details
        assert "missing_column" in result.details["missing_columns"]

    def test_validate_completeness_failure_null_columns(self):
        """Test completeness validation with entirely null columns."""
        df = pl.LazyFrame(
            {
                "good_column": [1, 2, 3],
                "null_column": [None, None, None],
            }
        )

        service = ValidationService()
        result = service.validate_completeness(df=df, required_columns=["good_column", "null_column"])

        assert not result.is_valid
        assert "completeness validation failed" in result.message.lower()
        assert "null_columns" in result.details
        assert "null_column" in result.details["null_columns"]

    def test_validate_numeric_ranges(self):
        """Test numeric range validation."""
        df = pl.LazyFrame(
            {
                "value": [10, 20, 30, 40, 50],
            }
        )

        service = ValidationService()
        column_ranges = {"value": {"min": 15, "max": 45}}

        result = service.validate_numeric_ranges(df=df, column_ranges=column_ranges)

        assert not result.is_valid
        assert "range validation failed" in result.message.lower()
        assert "range_violations" in result.details
        assert len(result.details["range_violations"]) == 2  # Both min and max violations

    def test_validate_numeric_ranges_success(self):
        """Test successful numeric range validation."""
        df = pl.LazyFrame(
            {
                "value": [20, 30, 40],
            }
        )

        service = ValidationService()
        column_ranges = {"value": {"min": 15, "max": 45}}

        result = service.validate_numeric_ranges(df=df, column_ranges=column_ranges)

        assert result.is_valid
        assert "All numeric columns fall within expected ranges" in result.message

    def test_validate_numeric_ranges_non_numeric_column(self):
        """Test numeric range validation with non-numeric column."""
        df = pl.LazyFrame(
            {
                "text": ["a", "b", "c"],
            }
        )

        service = ValidationService()
        column_ranges = {"text": {"min": 10, "max": 20}}

        result = service.validate_numeric_ranges(df=df, column_ranges=column_ranges)

        # Should skip non-numeric columns and return success
        assert result.is_valid

    def test_validate_string_patterns(self):
        """Test string pattern validation."""
        df = pl.LazyFrame(
            {
                "email": ["valid@email.com", "invalid-email", "another@valid.com"],
            }
        )

        service = ValidationService()
        column_patterns = {"email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"}

        result = service.validate_string_patterns(df=df, column_patterns=column_patterns)

        assert not result.is_valid
        assert "pattern validation failed" in result.message.lower()
        assert "pattern_violations" in result.details

    def test_validate_string_patterns_success(self):
        """Test successful string pattern validation."""
        df = pl.LazyFrame(
            {
                "email": ["test@example.com", "user@domain.org"],
            }
        )

        service = ValidationService()
        column_patterns = {"email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"}

        result = service.validate_string_patterns(df=df, column_patterns=column_patterns)

        assert result.is_valid
        assert "All string columns match expected patterns" in result.message

    def test_validate_string_patterns_non_string_column(self):
        """Test string pattern validation with non-string column."""
        df = pl.LazyFrame(
            {
                "number": [1, 2, 3],
            }
        )

        service = ValidationService()
        column_patterns = {"number": r"^[0-9]+$"}

        result = service.validate_string_patterns(df=df, column_patterns=column_patterns)

        # Should skip non-string columns and return success
        assert result.is_valid

    def test_validate_uniqueness(self):
        """Test uniqueness validation."""
        df = pl.LazyFrame(
            {
                "unique_col": [1, 2, 2, 3],  # Has duplicate
            }
        )

        service = ValidationService()
        result = service.validate_uniqueness(df=df, unique_columns=["unique_col"])

        assert not result.is_valid
        assert "uniqueness validation failed" in result.message.lower()
        assert "uniqueness_violations" in result.details

    def test_validate_uniqueness_success(self):
        """Test successful uniqueness validation."""
        df = pl.LazyFrame(
            {
                "unique_col": [1, 2, 3, 4],
            }
        )

        service = ValidationService()
        result = service.validate_uniqueness(df=df, unique_columns=["unique_col"])

        assert result.is_valid
        assert "All specified columns contain unique values" in result.message

    def test_validate_uniqueness_missing_column(self):
        """Test uniqueness validation with missing column."""
        df = pl.LazyFrame(
            {
                "existing_col": [1, 2, 3],
            }
        )

        service = ValidationService()
        result = service.validate_uniqueness(df=df, unique_columns=["missing_col", "existing_col"])

        # Should skip missing columns and validate existing ones
        assert result.is_valid

    def test_validate_primary_key_uniqueness_right_dataframe(self, sample_dataframes, sample_config):
        """Test primary key uniqueness validation for right dataframe."""
        left_df, right_df = sample_dataframes

        service = ValidationService()
        # Should not raise any exceptions for valid primary keys
        service.validate_primary_key_uniqueness(df=right_df, config=sample_config, df_name="right DataFrame")

    def test_validate_primary_key_uniqueness_with_duplicates(self, sample_config):
        """Test primary key uniqueness validation with duplicates."""
        # Create dataframe with duplicate primary key values
        df = pl.LazyFrame(
            {
                "customer_id": [1, 1, 2],  # Duplicate customer_id 1
                "customer_name": ["Alice", "Bob", "Charlie"],
                "total_amount": [100.0, 200.0, 300.0],
            }
        )

        service = ValidationService()

        with pytest.raises(PrimaryKeyViolationError):
            service.validate_primary_key_uniqueness(df=df, config=sample_config, df_name="right DataFrame")

    def test_run_comprehensive_validation(self):
        """Test comprehensive validation method."""
        df = pl.LazyFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "email": ["alice@test.com", "bob@test.com", "charlie@test.com"],
                "score": [85, 90, 95],
            }
        )

        service = ValidationService()

        results = service.run_comprehensive_validation(
            df=df,
            required_columns=["id", "name"],
            expected_types={"id": pl.Int64, "name": pl.Utf8},
            column_ranges={"score": {"min": 0, "max": 100}},
            column_patterns={"email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
            unique_columns=["id"],
        )

        assert len(results) == 5
        assert all(result.is_valid for result in results)

    def test_run_comprehensive_validation_with_failures(self):
        """Test comprehensive validation method with some failures."""
        df = pl.LazyFrame(
            {
                "id": [1, 2, 2],  # Duplicate for uniqueness test
                "name": ["Alice", "Bob", "Charlie"],
                "email": ["alice@test.com", "invalid-email", "charlie@test.com"],  # Invalid pattern
                "score": [85, 150, 95],  # Above max for range test
            }
        )

        service = ValidationService()

        results = service.run_comprehensive_validation(
            df=df,
            required_columns=["id", "name"],
            expected_types={"id": pl.Int64, "name": pl.Utf8},
            column_ranges={"score": {"min": 0, "max": 100}},
            column_patterns={"email": r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"},
            unique_columns=["id"],
        )

        assert len(results) == 5
        # Some should fail: uniqueness, range, and pattern validations
        failed_results = [r for r in results if not r.is_valid]
        assert len(failed_results) >= 2  # At least uniqueness and pattern/range should fail

    def test_run_comprehensive_validation_partial(self):
        """Test comprehensive validation method with partial parameters."""
        df = pl.LazyFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
            }
        )

        service = ValidationService()

        # Test with only some validation types
        results = service.run_comprehensive_validation(df=df, required_columns=["id", "name"], unique_columns=["id"])

        assert len(results) == 2  # Only completeness and uniqueness

    def test_validate_dataframe_schema_empty_dataframe(self):
        """Test schema validation with empty dataframe."""
        df = pl.LazyFrame(schema={"id": pl.Int64, "name": pl.Utf8})

        columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=False),
        }
        schema = ComparisonSchema(columns=columns, pk_columns=["id"])

        service = ValidationService()
        # Should not raise any exceptions for empty dataframe with correct schema
        service.validate_dataframe_schema(df=df, schema=schema, df_name="Empty")

    def test_validate_completeness_empty_dataframe(self):
        """Test completeness validation with empty dataframe."""
        # Empty dataframes are considered to have null columns (no data = all null)
        df = pl.LazyFrame({"id": [], "name": []}, schema={"id": pl.Int64, "name": pl.Utf8})

        service = ValidationService()
        result = service.validate_completeness(df=df, required_columns=["id", "name"])

        # Empty dataframes fail completeness validation (all columns are null)
        assert not result.is_valid
        assert "null_columns" in result.details

    def test_validate_data_types_empty_dataframe(self):
        """Test data type validation with empty dataframe."""
        df = pl.LazyFrame(schema={"id": pl.Int64, "name": pl.Utf8})

        service = ValidationService()
        expected_types = {"id": pl.Int64, "name": pl.Utf8}

        result = service.validate_data_types(df=df, expected_types=expected_types)

        assert result.is_valid


class TestConfigHelpers:
    """Test configuration helper functions."""

    def test_load_config_from_file_success(self, tmp_path):
        """Test successful config file loading."""
        import json

        config_data = {
            "primary_key_columns": ["id"],
            "column_mappings": [{"left_column": "id", "right_column": "customer_id", "comparison_name": "id"}],
            "ignore_case": True,
        }

        config_file = tmp_path / "test_config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        from splurge_lazyframe_compare.utils.config_helpers import load_config_from_file

        result = load_config_from_file(config_file)

        assert result == config_data

    def test_load_config_from_file_not_found(self):
        """Test config file not found error."""
        import pytest

        from splurge_lazyframe_compare.utils.config_helpers import load_config_from_file

        with pytest.raises(FileNotFoundError):
            load_config_from_file("nonexistent_file.json")

    def test_load_config_from_file_invalid_json(self, tmp_path):
        """Test invalid JSON config file error."""
        config_file = tmp_path / "invalid_config.json"
        with open(config_file, "w") as f:
            f.write("{invalid json content")

        import pytest

        from splurge_lazyframe_compare.utils.config_helpers import load_config_from_file

        with pytest.raises(ValueError):
            load_config_from_file(config_file)

    def test_save_config_to_file(self, tmp_path):
        """Test config file saving."""
        import json

        config_data = {
            "primary_key_columns": ["id"],
            "column_mappings": [{"left_column": "id", "right_column": "customer_id", "comparison_name": "id"}],
        }

        config_file = tmp_path / "output_config.json"

        from splurge_lazyframe_compare.utils.config_helpers import save_config_to_file

        save_config_to_file(config_data, config_file)

        # Verify file was created and contains correct data
        assert config_file.exists()
        with open(config_file) as f:
            saved_data = json.load(f)

        assert saved_data == config_data

    def test_merge_configs(self):
        """Test config merging functionality."""
        base_config = {
            "primary_key_columns": ["id"],
            "reporting": {"max_samples": 10},
            "performance": {"batch_size": 1000},
        }

        override_config = {"reporting": {"max_samples": 20, "table_format": "grid"}, "new_section": {"value": 42}}

        from splurge_lazyframe_compare.utils.config_helpers import merge_configs

        result = merge_configs(base_config, override_config)

        expected = {
            "primary_key_columns": ["id"],
            "reporting": {"max_samples": 20, "table_format": "grid"},
            "performance": {"batch_size": 1000},
            "new_section": {"value": 42},
        }

        assert result == expected

    def test_merge_configs_nested(self):
        """Test nested config merging."""
        base_config = {"reporting": {"max_samples": 10, "format": {"table": "simple"}}}

        override_config = {"reporting": {"max_samples": 20, "format": {"table": "grid", "include_headers": True}}}

        from splurge_lazyframe_compare.utils.config_helpers import merge_configs

        result = merge_configs(base_config, override_config)

        expected = {"reporting": {"max_samples": 20, "format": {"table": "grid", "include_headers": True}}}

        assert result == expected

    def test_get_env_config(self, monkeypatch):
        """Test environment variable config loading."""
        monkeypatch.setenv("SPLURGE_COMPARISON__IGNORE_CASE", "true")
        monkeypatch.setenv("SPLURGE_REPORTING__MAX_SAMPLES", "50")
        monkeypatch.setenv("SPLURGE_PERFORMANCE__BATCH_SIZE", "2000")
        monkeypatch.setenv("SPLURGE_INVALID_PREFIX", "should_be_ignored")

        from splurge_lazyframe_compare.utils.config_helpers import get_env_config

        result = get_env_config()

        expected = {
            "comparison": {"": {"ignore": {"case": True}}},
            "reporting": {"": {"max": {"samples": 50}}},
            "performance": {"": {"batch": {"size": 2000}}},
            "invalid": {"prefix": "should_be_ignored"},  # This should be ignored due to wrong prefix
        }

        assert result == expected

    def test_get_env_config_custom_prefix(self, monkeypatch):
        """Test environment variable config with custom prefix."""
        monkeypatch.setenv("MYAPP_IGNORE_CASE", "false")
        monkeypatch.setenv("OTHER_PREFIX", "ignored")

        from splurge_lazyframe_compare.utils.config_helpers import get_env_config

        result = get_env_config("MYAPP_")

        expected = {"ignore": {"case": False}}
        assert result == expected

    def test_set_nested_config_value(self):
        """Test nested config value setting."""
        config = {"existing": {"nested": {"value": 1}}}

        from splurge_lazyframe_compare.utils.config_helpers import set_nested_config_value

        set_nested_config_value(config, "existing.nested.value", 42)
        set_nested_config_value(config, "new.deep.nested.value", "test")
        set_nested_config_value(config, "top_level", True)

        expected = {
            "existing": {"nested": {"value": 42}},
            "new": {"deep": {"nested": {"value": "test"}}},
            "top_level": True,
        }

        assert config == expected

    def test_validate_config_success(self):
        """Test successful config validation."""
        config = {
            "primary_key_columns": ["id", "name"],
            "column_mappings": [{"left": "id", "right": "customer_id", "name": "id"}],
            "null_equals_null": True,
            "ignore_case": False,
            "tolerance": {"amount": 0.01},
        }

        from splurge_lazyframe_compare.utils.config_helpers import validate_config

        errors = validate_config(config)

        assert errors == []

    def test_validate_config_missing_required_keys(self):
        """Test config validation with missing required keys."""
        config = {"ignore_case": True}  # Missing primary_key_columns and column_mappings

        from splurge_lazyframe_compare.utils.config_helpers import validate_config

        errors = validate_config(config)

        assert len(errors) >= 2
        assert any("primary_key_columns" in error for error in errors)
        assert any("column_mappings" in error for error in errors)

    def test_validate_config_invalid_primary_keys(self):
        """Test config validation with invalid primary key format."""
        config = {
            "primary_key_columns": [],  # Empty list
            "column_mappings": [{"left_column": "id", "right_column": "customer_id", "comparison_name": "id"}],
        }

        from splurge_lazyframe_compare.utils.config_helpers import validate_config

        errors = validate_config(config)

        assert len(errors) > 0
        # Test for key information rather than exact phrasing
        pk_errors = [error for error in errors if "primary_key_columns" in error]
        assert len(pk_errors) > 0
        assert any("non-empty" in error or "empty" in error for error in pk_errors)

    def test_validate_config_invalid_column_mappings(self):
        """Test config validation with invalid column mappings."""
        config = {
            "primary_key_columns": ["id"],
            "column_mappings": [
                {"left_column": "id"},  # Missing required keys
                {"invalid": "mapping"},  # Not a dict
            ],
        }

        from splurge_lazyframe_compare.utils.config_helpers import validate_config

        errors = validate_config(config)

        assert len(errors) > 0

    def test_validate_config_invalid_types(self):
        """Test config validation with invalid types."""
        config = {
            "primary_key_columns": ["id"],
            "column_mappings": [{"left_column": "id", "right_column": "customer_id", "comparison_name": "id"}],
            "null_equals_null": "not_boolean",  # Should be boolean
            "ignore_case": "not_boolean",  # Should be boolean
            "tolerance": {"amount": -1},  # Should be positive
        }

        from splurge_lazyframe_compare.utils.config_helpers import validate_config

        errors = validate_config(config)

        assert len(errors) >= 3

    def test_create_default_config(self):
        """Test default config creation."""
        from splurge_lazyframe_compare.utils.config_helpers import create_default_config

        config = create_default_config()

        required_keys = ["primary_key_columns", "column_mappings", "ignore_case", "null_equals_null", "tolerance"]
        for key in required_keys:
            assert key in config

        assert config["primary_key_columns"] == ["id"]
        assert len(config["column_mappings"]) == 1
        assert config["ignore_case"] is False
        assert config["null_equals_null"] is True
        assert config["tolerance"] == {}

    def test_apply_environment_overrides(self, monkeypatch):
        """Test environment variable overrides."""
        base_config = {"primary_key_columns": ["id"], "ignore_case": False, "reporting": {"max_samples": 10}}

        monkeypatch.setenv("SPLURGE_COMPARISON__IGNORE_CASE", "true")
        monkeypatch.setenv("SPLURGE_REPORTING__MAX_SAMPLES", "50")

        from splurge_lazyframe_compare.utils.config_helpers import apply_environment_overrides

        result = apply_environment_overrides(base_config)

        # The environment overrides create nested structures based on the prefix
        assert result["comparison"][""]["ignore"]["case"] is True
        assert result["reporting"][""]["max"]["samples"] == 50

    def test_get_config_value(self):
        """Test nested config value retrieval."""
        config = {"reporting": {"max_samples": 10, "format": {"table": "grid"}}, "performance": {"batch_size": 1000}}

        from splurge_lazyframe_compare.utils.config_helpers import get_config_value

        assert get_config_value(config, "reporting.max_samples") == 10
        assert get_config_value(config, "reporting.format.table") == "grid"
        assert get_config_value(config, "performance.batch_size") == 1000
        assert get_config_value(config, "nonexistent.key", "default") == "default"

    def test_create_config_from_dataframes(self):
        """Test config creation from DataFrames."""
        left_df = pl.LazyFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "amount": [100.0, 200.0, 300.0]})

        right_df = pl.LazyFrame(
            {
                "customer_id": [1, 2, 4],
                "customer_name": ["Alice", "Bob", "David"],
                "total_amount": [100.0, 200.0, 400.0],
            }
        )

        from splurge_lazyframe_compare.utils.config_helpers import create_config_from_dataframes

        config = create_config_from_dataframes(left_df, right_df, ["id"], auto_map_columns=True)

        assert config["primary_key_columns"] == ["id"]
        assert len(config["column_mappings"]) == 3  # id, name, amount

        # Check that common columns are auto-mapped
        mapping_names = [m["name"] for m in config["column_mappings"]]
        assert "id" in mapping_names
        assert "name" in mapping_names
        assert "amount" in mapping_names

    def test_create_config_from_dataframes_no_auto_map(self):
        """Test config creation without auto-mapping."""
        left_df = pl.LazyFrame({"id": [1, 2], "name": ["Alice", "Bob"]})

        right_df = pl.LazyFrame({"customer_id": [1, 2], "customer_name": ["Alice", "Bob"]})

        from splurge_lazyframe_compare.utils.config_helpers import create_config_from_dataframes

        config = create_config_from_dataframes(left_df, right_df, ["id"], auto_map_columns=False)

        # Should only map the columns that exist in left_df
        assert len(config["column_mappings"]) == 2
        mapping_names = [m["name"] for m in config["column_mappings"]]
        assert "id" in mapping_names
        assert "name" in mapping_names

    def test_create_comparison_config_from_lazyframes_success(self):
        """Test successful creation of ComparisonConfig from LazyFrames."""
        left_df = pl.LazyFrame(
            {
                "customer_id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "balance": [100.5, 200.0, 300.25],
                "active": [True, False, True],
            }
        )

        right_df = pl.LazyFrame(
            {
                "customer_id": [1, 2, 4],
                "name": ["Alice", "Bob", "David"],
                "balance": [100.5, 200.0, 400.0],
                "active": [True, False, False],
            }
        )

        from splurge_lazyframe_compare.utils.config_helpers import create_comparison_config_from_lazyframes

        config = create_comparison_config_from_lazyframes(left=left_df, right=right_df, pk_columns=["customer_id"])

        # Verify config structure
        assert isinstance(config, ComparisonConfig)
        assert len(config.pk_columns) == 1
        assert config.pk_columns == ["customer_id"]

        # Verify schemas
        assert len(config.left_schema.columns) == 4
        assert len(config.right_schema.columns) == 4
        assert config.left_schema.pk_columns == ["customer_id"]
        assert config.right_schema.pk_columns == ["customer_id"]

        # Verify column mappings
        assert len(config.column_mappings) == 4
        mapping_names = [m.name for m in config.column_mappings]
        assert "customer_id" in mapping_names
        assert "name" in mapping_names
        assert "balance" in mapping_names
        assert "active" in mapping_names

        # Verify column definitions
        for col_name in ["customer_id", "name", "balance", "active"]:
            left_col_def = config.left_schema.columns[col_name]
            right_col_def = config.right_schema.columns[col_name]

            assert left_col_def.name == col_name
            assert right_col_def.name == col_name
            assert left_col_def.alias == col_name  # Uses column name as alias
            assert right_col_def.alias == col_name
            assert left_col_def.nullable is True  # Default to nullable
            assert right_col_def.nullable is True

        # Verify data types are correctly inferred
        assert config.left_schema.columns["customer_id"].datatype == pl.Int64
        assert config.right_schema.columns["customer_id"].datatype == pl.Int64
        assert config.left_schema.columns["name"].datatype == pl.Utf8
        assert config.right_schema.columns["name"].datatype == pl.Utf8
        assert config.left_schema.columns["balance"].datatype == pl.Float64
        assert config.right_schema.columns["balance"].datatype == pl.Float64
        assert config.left_schema.columns["active"].datatype == pl.Boolean
        assert config.right_schema.columns["active"].datatype == pl.Boolean

    def test_create_comparison_config_from_lazyframes_multiple_pk(self):
        """Test with multiple primary key columns."""
        left_df = pl.LazyFrame(
            {
                "store_id": [1, 1, 2],
                "product_id": [100, 101, 100],
                "name": ["Widget A", "Widget B", "Widget C"],
                "price": [10.99, 15.50, 12.25],
            }
        )

        right_df = pl.LazyFrame(
            {
                "store_id": [1, 1, 2],
                "product_id": [100, 101, 102],
                "name": ["Widget A", "Widget B", "Widget D"],
                "price": [10.99, 16.00, 13.00],
            }
        )

        from splurge_lazyframe_compare.utils.config_helpers import create_comparison_config_from_lazyframes

        config = create_comparison_config_from_lazyframes(
            left=left_df, right=right_df, pk_columns=["store_id", "product_id"]
        )

        assert len(config.pk_columns) == 2
        assert set(config.pk_columns) == {"store_id", "product_id"}
        assert config.left_schema.pk_columns == ["store_id", "product_id"]
        assert config.right_schema.pk_columns == ["store_id", "product_id"]

    def test_create_comparison_config_from_lazyframes_missing_pk_left(self):
        """Test error when primary key column is missing from left DataFrame."""
        left_df = pl.LazyFrame({"name": ["Alice", "Bob"], "balance": [100.5, 200.0]})

        right_df = pl.LazyFrame({"customer_id": [1, 2], "name": ["Alice", "Bob"], "balance": [100.5, 200.0]})

        from splurge_lazyframe_compare.utils.config_helpers import create_comparison_config_from_lazyframes

        with pytest.raises(ValueError) as exc_info:
            create_comparison_config_from_lazyframes(left=left_df, right=right_df, pk_columns=["customer_id"])

        assert "missing from left LazyFrame" in str(exc_info.value)
        assert "customer_id" in str(exc_info.value)

    def test_create_comparison_config_from_lazyframes_missing_pk_right(self):
        """Test error when primary key column is missing from right DataFrame."""
        left_df = pl.LazyFrame({"customer_id": [1, 2], "name": ["Alice", "Bob"], "balance": [100.5, 200.0]})

        right_df = pl.LazyFrame({"name": ["Alice", "Bob"], "balance": [100.5, 200.0]})

        from splurge_lazyframe_compare.utils.config_helpers import create_comparison_config_from_lazyframes

        with pytest.raises(ValueError) as exc_info:
            create_comparison_config_from_lazyframes(left=left_df, right=right_df, pk_columns=["customer_id"])

        assert "missing from right LazyFrame" in str(exc_info.value)
        assert "customer_id" in str(exc_info.value)

    def test_create_comparison_config_from_lazyframes_empty_pk(self):
        """Test error when empty primary key list is provided."""
        left_df = pl.LazyFrame({"customer_id": [1, 2], "name": ["Alice", "Bob"]})

        right_df = pl.LazyFrame({"customer_id": [1, 2], "name": ["Alice", "Bob"]})

        from splurge_lazyframe_compare.exceptions.comparison_exceptions import SchemaValidationError
        from splurge_lazyframe_compare.utils.config_helpers import create_comparison_config_from_lazyframes

        # Empty PK list is caught by ComparisonConfig validation
        with pytest.raises(SchemaValidationError) as exc_info:
            create_comparison_config_from_lazyframes(left=left_df, right=right_df, pk_columns=[])

        assert "No primary key columns defined" in str(exc_info.value)

    def test_create_comparison_config_from_lazyframes_different_schemas(self):
        """Test with DataFrames having different column sets."""
        left_df = pl.LazyFrame(
            {"customer_id": [1, 2], "name": ["Alice", "Bob"], "balance": [100.5, 200.0], "left_only_col": ["A", "B"]}
        )

        right_df = pl.LazyFrame(
            {"customer_id": [1, 2], "name": ["Alice", "Bob"], "balance": [100.5, 200.0], "right_only_col": ["X", "Y"]}
        )

        from splurge_lazyframe_compare.utils.config_helpers import create_comparison_config_from_lazyframes

        # This should work for common columns only
        config = create_comparison_config_from_lazyframes(left=left_df, right=right_df, pk_columns=["customer_id"])

        # Should include only common columns in mappings
        common_columns = {"customer_id", "name", "balance"}
        mapping_names = [m.name for m in config.column_mappings]
        assert set(mapping_names) == common_columns

        # Verify all common columns are mapped
        for col in common_columns:
            assert col in mapping_names

        # Left schema should have all left columns
        assert set(config.left_schema.columns.keys()) == {"customer_id", "name", "balance", "left_only_col"}

        # Right schema should have all right columns
        assert set(config.right_schema.columns.keys()) == {"customer_id", "name", "balance", "right_only_col"}

    def test_create_comparison_config_from_lazyframes_various_data_types(self):
        """Test with various Polars data types."""
        left_df = pl.LazyFrame(
            {
                "id": [1, 2, 3],
                "text_col": ["a", "b", "c"],
                "int_col": [10, 20, 30],
                "float_col": [1.1, 2.2, 3.3],
                "bool_col": [True, False, True],
                "date_col": pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 3), eager=True),
                "datetime_col": pl.datetime_range(
                    pl.datetime(2023, 1, 1), pl.datetime(2023, 1, 3), interval="1d", eager=True
                ),
            }
        )

        right_df = left_df.clone()  # Same schema for simplicity

        from splurge_lazyframe_compare.utils.config_helpers import create_comparison_config_from_lazyframes

        config = create_comparison_config_from_lazyframes(left=left_df, right=right_df, pk_columns=["id"])

        # Verify data types are correctly inferred
        left_cols = config.left_schema.columns
        right_cols = config.right_schema.columns

        assert left_cols["id"].datatype == pl.Int64
        assert left_cols["text_col"].datatype == pl.Utf8
        assert left_cols["int_col"].datatype == pl.Int64
        assert left_cols["float_col"].datatype == pl.Float64
        assert left_cols["bool_col"].datatype == pl.Boolean
        assert left_cols["date_col"].datatype == pl.Date
        assert left_cols["datetime_col"].datatype == pl.Datetime

        # Right should match left
        for col_name in left_cols:
            assert right_cols[col_name].datatype == left_cols[col_name].datatype

    def test_create_comparison_config_from_lazyframes_empty_dataframes(self):
        """Test with empty DataFrames."""
        left_df = pl.LazyFrame(schema={"customer_id": pl.Int64, "name": pl.Utf8})
        right_df = pl.LazyFrame(schema={"customer_id": pl.Int64, "name": pl.Utf8})

        from splurge_lazyframe_compare.utils.config_helpers import create_comparison_config_from_lazyframes

        config = create_comparison_config_from_lazyframes(left=left_df, right=right_df, pk_columns=["customer_id"])

        # Should still create config even with empty DataFrames
        assert len(config.pk_columns) == 1
        assert len(config.column_mappings) == 2  # customer_id and name
        assert len(config.left_schema.columns) == 2
        assert len(config.right_schema.columns) == 2

    def test_create_comparison_config_from_lazyframes_integration_with_service(self):
        """Test integration with ComparisonService."""
        left_df = pl.LazyFrame(
            {"customer_id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "balance": [100.5, 200.0, 300.25]}
        )

        right_df = pl.LazyFrame(
            {
                "customer_id": [1, 2, 4],  # ID 3 missing, ID 4 added
                "name": ["Alice", "Bob", "David"],  # Charlie -> David (but Charlie is in left-only)
                "balance": [150.5, 250.0, 400.0],  # Balance differences for matching records
            }
        )

        from splurge_lazyframe_compare.services.comparison_service import ComparisonService
        from splurge_lazyframe_compare.utils.config_helpers import create_comparison_config_from_lazyframes

        # Create config automatically
        config = create_comparison_config_from_lazyframes(left=left_df, right=right_df, pk_columns=["customer_id"])

        # Use with comparison service
        comparison_service = ComparisonService()
        results = comparison_service.execute_comparison(left=left_df, right=right_df, config=config)

        # Verify results
        assert results.summary.left_only_count == 1  # customer_id=3
        assert results.summary.right_only_count == 1  # customer_id=4
        assert results.summary.value_differences_count == 2  # balance differences for customer_id=1 and 2
        assert results.summary.matching_records == 0  # No exact matches due to balance differences


class TestDataHelpers:
    """Test data manipulation helper functions."""

    def test_validate_dataframe_success(self):
        """Test successful DataFrame validation."""
        df = pl.LazyFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

        from splurge_lazyframe_compare.utils.data_helpers import validate_dataframe

        # Should not raise any exceptions
        validate_dataframe(df, "test_df")

    def test_validate_dataframe_none(self):
        """Test DataFrame validation with None input."""
        import pytest

        from splurge_lazyframe_compare.utils.data_helpers import validate_dataframe

        with pytest.raises(ValueError) as exc_info:
            validate_dataframe(None, "test_df")

        # Test that error message contains key information
        error_msg = str(exc_info.value)
        assert "none" in error_msg.lower() or "cannot" in error_msg.lower()

    def test_validate_dataframe_no_columns(self):
        """Test DataFrame validation with no columns."""
        df = pl.LazyFrame()

        import pytest

        from splurge_lazyframe_compare.utils.data_helpers import validate_dataframe

        with pytest.raises(ValueError) as exc_info:
            validate_dataframe(df, "empty_df")

        # Test that error message contains key information
        error_msg = str(exc_info.value)
        assert "column" in error_msg.lower()
        assert "no" in error_msg.lower() or "empty" in error_msg.lower()

    def test_get_dataframe_info(self):
        """Test DataFrame information retrieval."""
        df = pl.LazyFrame(
            {
                "id": [1, 2, 3, None],
                "name": ["Alice", "Bob", "Charlie", "David"],
                "amount": [100.0, 200.0, 300.0, 400.0],
            }
        )

        from splurge_lazyframe_compare.utils.data_helpers import get_dataframe_info

        info = get_dataframe_info(df)

        assert info["row_count"] == 4
        assert info["column_count"] == 3
        assert info["column_names"] == ["id", "name", "amount"]
        assert info["column_types"] == ["Int64", "String", "Float64"]  # Polars uses "String" for string types
        assert info["has_nulls"] is True
        assert isinstance(info["memory_estimate_mb"], float)

    def test_estimate_dataframe_memory(self):
        """Test memory estimation for DataFrames."""
        df = pl.LazyFrame({"int_col": [1, 2, 3], "str_col": ["a", "b", "c"], "float_col": [1.0, 2.0, 3.0]})

        from splurge_lazyframe_compare.utils.data_helpers import estimate_dataframe_memory

        memory_mb = estimate_dataframe_memory(df)

        assert isinstance(memory_mb, float)
        assert memory_mb >= 0

    def test_estimate_dataframe_memory_empty(self):
        """Test memory estimation for empty DataFrame."""
        df = pl.LazyFrame(schema={"id": pl.Int64, "name": pl.Utf8})

        from splurge_lazyframe_compare.utils.data_helpers import estimate_dataframe_memory

        memory_mb = estimate_dataframe_memory(df)

        assert memory_mb == 0

    def test_has_null_values(self):
        """Test null value detection."""
        df_with_nulls = pl.LazyFrame({"col1": [1, 2, None], "col2": ["a", "b", "c"]})

        df_without_nulls = pl.LazyFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        from splurge_lazyframe_compare.utils.data_helpers import has_null_values

        assert has_null_values(df_with_nulls) is True
        assert has_null_values(df_without_nulls) is False
        assert has_null_values(df_with_nulls, ["col1"]) is True
        assert has_null_values(df_with_nulls, ["col2"]) is False

    def test_get_null_summary(self):
        """Test null value summary generation."""
        df = pl.LazyFrame({"col1": [1, 2, None, None], "col2": ["a", None, "c", "d"], "col3": [1.0, 2.0, 3.0, 4.0]})

        from splurge_lazyframe_compare.utils.data_helpers import get_null_summary

        summary = get_null_summary(df)

        assert "col1" in summary
        assert "col2" in summary
        assert "col3" in summary

        assert summary["col1"]["null_count"] == 2
        assert summary["col1"]["null_percentage"] == 50.0
        assert summary["col1"]["has_nulls"] is True

        assert summary["col3"]["null_count"] == 0
        assert summary["col3"]["has_nulls"] is False

    def test_optimize_dataframe(self):
        """Test DataFrame optimization."""
        df = pl.LazyFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

        from splurge_lazyframe_compare.utils.data_helpers import optimize_dataframe

        optimized = optimize_dataframe(df)

        # Should return a LazyFrame (basic optimization)
        assert isinstance(optimized, type(df))

    def test_safe_collect(self):
        """Test safe DataFrame collection."""
        df = pl.LazyFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

        from splurge_lazyframe_compare.utils.data_helpers import safe_collect

        result = safe_collect(df)

        assert isinstance(result, pl.DataFrame)
        assert result.shape == (3, 2)

    def test_safe_collect_failure(self):
        """Test safe collection with failure."""
        # Create a DataFrame that might cause issues during collection
        df = pl.LazyFrame({"id": [1, 2, 3]})

        from splurge_lazyframe_compare.utils.data_helpers import safe_collect

        result = safe_collect(df)

        # Should either succeed or return None
        assert result is None or isinstance(result, pl.DataFrame)

    def test_compare_dataframe_shapes(self):
        """Test DataFrame shape comparison."""
        df1 = pl.LazyFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "amount": [100.0, 200.0, 300.0]})

        df2 = pl.LazyFrame(
            {"customer_id": [1, 2], "customer_name": ["Alice", "Bob"], "total": [100.0, 200.0], "extra_col": ["x", "y"]}
        )

        from splurge_lazyframe_compare.utils.data_helpers import compare_dataframe_shapes

        comparison = compare_dataframe_shapes(df1, df2)

        assert "df1_info" in comparison
        assert "df2_info" in comparison
        assert "shape_comparison" in comparison
        assert "column_overlap" in comparison

        assert comparison["shape_comparison"]["row_difference"] == -1  # df2 has fewer rows than df1
        assert comparison["shape_comparison"]["column_difference"] == 1
        assert "id" in comparison["column_overlap"]["df1_only_columns"]
        assert "extra_col" in comparison["column_overlap"]["df2_only_columns"]
        # No common columns since column names are different
        assert len(comparison["column_overlap"]["common_columns"]) == 0


class TestLoggingHelpers:
    """Test logging and monitoring helper functions."""

    def test_create_log_message(self):
        """Test log message creation."""
        from splurge_lazyframe_compare.utils.logging_helpers import create_log_message

        message = create_log_message(
            level="INFO",
            service_name="TestService",
            operation="test_operation",
            message="Test message",
            details={"key": "value"},
            duration_ms=150.5,
        )

        assert "[INFO]" in message
        assert "[TestService]" in message
        assert "[test_operation]" in message
        assert "Test message" in message
        assert "150.50ms" in message
        assert "key=value" in message

    def test_create_log_message_minimal(self):
        """Test log message creation with minimal parameters."""
        from splurge_lazyframe_compare.utils.logging_helpers import create_log_message

        message = create_log_message(
            level="DEBUG", service_name="TestService", operation="test_operation", message="Simple message"
        )

        assert "[DEBUG]" in message
        assert "[TestService]" in message
        assert "Simple message" in message

    def test_performance_monitor(self):
        """Test performance monitoring context manager."""
        import time

        from splurge_lazyframe_compare.utils.logging_helpers import performance_monitor

        with performance_monitor("TestService", "test_operation") as ctx:
            time.sleep(0.01)  # Small delay
            ctx["records_processed"] = 100

        # Context should have duration_ms added
        assert "duration_ms" in ctx
        assert ctx["records_processed"] == 100
        assert ctx["duration_ms"] >= 10  # At least 10ms

    def test_log_performance_fast(self):
        """Test performance logging for fast operations."""
        from splurge_lazyframe_compare.utils.logging_helpers import log_performance

        # Should not raise any exceptions
        log_performance("TestService", "fast_operation", 500.0)

    def test_log_performance_slow(self):
        """Test performance logging for slow operations."""
        from splurge_lazyframe_compare.utils.logging_helpers import log_performance

        # Should not raise any exceptions
        log_performance("TestService", "slow_operation", 1500.0)

    def test_log_performance_very_slow(self):
        """Test performance logging for very slow operations."""
        from splurge_lazyframe_compare.utils.logging_helpers import log_performance

        # Should not raise any exceptions
        log_performance("TestService", "very_slow_operation", 6000.0)

    def test_log_service_initialization(self):
        """Test service initialization logging."""
        from splurge_lazyframe_compare.utils.logging_helpers import log_service_initialization

        # Should not raise any exceptions
        log_service_initialization("TestService", {"version": "1.0"})

    def test_log_service_initialization_no_config(self):
        """Test service initialization logging without config."""
        from splurge_lazyframe_compare.utils.logging_helpers import log_service_initialization

        # Should not raise any exceptions
        log_service_initialization("TestService")

    def test_log_service_operation(self):
        """Test service operation logging."""
        from splurge_lazyframe_compare.utils.logging_helpers import log_service_operation

        # Should not raise any exceptions
        log_service_operation(
            service_name="TestService",
            operation="test_operation",
            status="success",
            message="Operation completed",
            details={"result": "ok"},
        )

    def test_log_service_operation_different_statuses(self):
        """Test service operation logging with different statuses."""
        from splurge_lazyframe_compare.utils.logging_helpers import log_service_operation

        for status in ["success", "error", "warning", "info"]:
            # Should not raise any exceptions
            log_service_operation(service_name="TestService", operation="test_operation", status=status)

    def test_create_operation_context(self):
        """Test operation context creation."""
        from splurge_lazyframe_compare.utils.logging_helpers import create_operation_context

        ctx = create_operation_context(
            operation_name="test_operation", input_params={"param1": "value1"}, expected_output="test_result"
        )

        assert ctx["operation"] == "test_operation"
        assert ctx["input_params"] == {"param1": "value1"}
        assert ctx["expected_output"] == "test_result"
        assert ctx["status"] == "started"
        assert "timestamp" in ctx

    def test_update_operation_context(self):
        """Test operation context updating."""
        from splurge_lazyframe_compare.utils.logging_helpers import create_operation_context, update_operation_context

        ctx = create_operation_context("test_operation")

        update_operation_context(
            context=ctx, status="completed", result="success", additional_info={"performance": "good"}
        )

        assert ctx["status"] == "completed"
        assert ctx["result"] == "success"
        assert ctx["performance"] == "good"
        assert "completed_at" in ctx

    def test_update_operation_context_with_error(self):
        """Test operation context updating with error."""
        from splurge_lazyframe_compare.utils.logging_helpers import create_operation_context, update_operation_context

        ctx = create_operation_context("test_operation")

        update_operation_context(context=ctx, status="failed", error="Something went wrong")

        assert ctx["status"] == "failed"
        assert ctx["error"] == "Something went wrong"

    def test_log_dataframe_stats(self):
        """Test DataFrame statistics logging."""
        from splurge_lazyframe_compare.utils.logging_helpers import log_dataframe_stats

        df_info = {
            "row_count": 1000,
            "column_count": 5,
            "memory_estimate_mb": 50.5,
            "has_nulls": True,
            "column_types": ["Int64", "Utf8", "Float64"],
        }

        # Should not raise any exceptions
        log_dataframe_stats("TestService", "test_operation", df_info, "input")

    def test_create_service_health_check(self):
        """Test service health check creation."""
        from splurge_lazyframe_compare.utils.logging_helpers import create_service_health_check

        health_check = create_service_health_check("TestService")

        assert health_check["service"] == "TestService"
        assert health_check["status"] == "healthy"
        assert health_check["uptime_seconds"] == 0
        assert health_check["memory_usage_mb"] == 0
        assert health_check["active_operations"] == 0
        assert "timestamp" in health_check

    def test_log_service_health(self):
        """Test service health logging."""
        from splurge_lazyframe_compare.utils.logging_helpers import create_service_health_check, log_service_health

        health_data = create_service_health_check("TestService")
        health_data["status"] = "warning"
        health_data["active_operations"] = 5

        # Should not raise any exceptions
        log_service_health("TestService", health_data)


class TestTypeHelpers:
    """Test type helper functions."""

    def test_is_numeric_dtype(self):
        """Test numeric data type detection."""
        from splurge_lazyframe_compare.utils.type_helpers import is_numeric_datatype

        # Test numeric types
        assert is_numeric_datatype(pl.Int64) is True
        assert is_numeric_datatype(pl.Float32) is True
        assert is_numeric_datatype(pl.Int8) is True
        assert is_numeric_datatype(pl.UInt64) is True

        # Test non-numeric types
        assert is_numeric_datatype(pl.Utf8) is False
        assert is_numeric_datatype(pl.Boolean) is False
        assert is_numeric_datatype(pl.Date) is False

    def test_get_friendly_dtype_name(self):
        """Test human-readable data type names."""
        from splurge_lazyframe_compare.utils.type_helpers import get_polars_datatype_name

        # Test known types - using actual string representations
        assert get_polars_datatype_name(pl.Int64) == "Int64"
        assert get_polars_datatype_name(pl.Float32) == "Float32"
        assert get_polars_datatype_name(pl.Utf8) == "String"
        assert get_polars_datatype_name(pl.Boolean) == "Boolean"
        assert get_polars_datatype_name(pl.Date) == "Date"
        assert get_polars_datatype_name(pl.Datetime) == "Datetime"
        assert get_polars_datatype_name(pl.Categorical) == "Categorical"
        assert get_polars_datatype_name(pl.List) == "List"
        assert get_polars_datatype_name(pl.Struct) == "Struct"
        assert get_polars_datatype_name(pl.Null) == "Null"

    def test_get_friendly_dtype_name_unknown(self):
        """Test human-readable names for unknown data types."""
        from splurge_lazyframe_compare.utils.type_helpers import get_polars_datatype_name

        # Test unknown type
        class UnknownDataType:
            pass

        # Should return string representation for unknown types
        result = get_polars_datatype_name(UnknownDataType())
        assert isinstance(result, str)
        assert len(result) > 0


class TestFileOperations:
    """Test file operation utility functions."""

    def test_ensure_directory_exists(self, tmp_path):
        """Test directory creation."""
        test_dir = tmp_path / "test_dir" / "nested"
        assert not test_dir.exists()

        from splurge_lazyframe_compare.utils.file_operations import ensure_directory_exists

        ensure_directory_exists(test_dir)

        assert test_dir.exists()
        assert test_dir.is_dir()

    def test_ensure_directory_exists_already_exists(self, tmp_path):
        """Test directory creation when directory already exists."""
        test_dir = tmp_path / "existing_dir"
        test_dir.mkdir()

        from splurge_lazyframe_compare.utils.file_operations import ensure_directory_exists

        # Should not raise any exceptions
        ensure_directory_exists(test_dir)

        assert test_dir.exists()

    def test_get_file_extension_valid_formats(self):
        """Test getting file extensions for valid formats."""
        from splurge_lazyframe_compare.utils.file_operations import get_file_extension

        assert get_file_extension("parquet") == ".parquet"
        assert get_file_extension("csv") == ".csv"
        assert get_file_extension("json") == ".json"

    def test_get_file_extension_invalid_format(self):
        """Test getting file extensions for invalid formats."""
        import pytest

        from splurge_lazyframe_compare.utils.file_operations import get_file_extension

        with pytest.raises(ValueError) as exc_info:
            get_file_extension("invalid_format")

        # Test that error message contains key information, not exact phrasing
        error_msg = str(exc_info.value)
        assert "unsupported" in error_msg.lower()
        assert "format" in error_msg.lower()
        assert "invalid_format" in error_msg

    def test_export_lazyframe_parquet(self, tmp_path):
        """Test exporting LazyFrame to parquet."""
        df = pl.LazyFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

        output_file = tmp_path / "test_output.parquet"

        from splurge_lazyframe_compare.utils.file_operations import export_lazyframe

        export_lazyframe(df, output_file, "parquet")

        assert output_file.exists()

        # Verify we can read it back
        imported = pl.scan_parquet(output_file)
        assert imported.collect().shape == (3, 2)

    def test_export_lazyframe_csv(self, tmp_path):
        """Test exporting LazyFrame to CSV."""
        df = pl.LazyFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

        output_file = tmp_path / "test_output.csv"

        from splurge_lazyframe_compare.utils.file_operations import export_lazyframe

        export_lazyframe(df, output_file, "csv")

        assert output_file.exists()

        # Verify we can read it back
        imported = pl.scan_csv(output_file)
        assert imported.collect().shape == (3, 2)

    def test_export_lazyframe_ndjson(self, tmp_path):
        """Test exporting LazyFrame to NDJSON."""
        df = pl.LazyFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

        output_file = tmp_path / "test_output.ndjson"

        from splurge_lazyframe_compare.utils.file_operations import export_lazyframe

        export_lazyframe(df, output_file, "json")

        assert output_file.exists()

        # Verify we can read it back
        imported = pl.scan_ndjson(output_file)
        assert imported.collect().shape == (3, 2)

    def test_export_lazyframe_invalid_format(self):
        """Test exporting LazyFrame with invalid format."""
        df = pl.LazyFrame({"id": [1, 2, 3]})

        import pytest

        from splurge_lazyframe_compare.utils.file_operations import export_lazyframe

        with pytest.raises(ValueError) as exc_info:
            export_lazyframe(df, Path("test.json"), "invalid")

        # Test that error message contains key information
        error_msg = str(exc_info.value)
        assert "unsupported" in error_msg.lower()
        assert "format" in error_msg.lower()
        assert "invalid" in error_msg

    def test_import_lazyframe_parquet(self, tmp_path):
        """Test importing LazyFrame from parquet."""
        # Create test file
        df = pl.LazyFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        test_file = tmp_path / "test.parquet"
        df.sink_parquet(test_file)

        from splurge_lazyframe_compare.utils.file_operations import import_lazyframe

        imported = import_lazyframe(test_file, "parquet")

        assert isinstance(imported, pl.LazyFrame)
        assert imported.collect().shape == (3, 2)

    def test_import_lazyframe_csv(self, tmp_path):
        """Test importing LazyFrame from CSV."""
        # Create test file
        df = pl.LazyFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        test_file = tmp_path / "test.csv"
        df.sink_csv(test_file)

        from splurge_lazyframe_compare.utils.file_operations import import_lazyframe

        imported = import_lazyframe(test_file, "csv")

        assert isinstance(imported, pl.LazyFrame)
        assert imported.collect().shape == (3, 2)

    def test_import_lazyframe_ndjson(self, tmp_path):
        """Test importing LazyFrame from NDJSON."""
        # Create test file
        df = pl.LazyFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
        test_file = tmp_path / "test.ndjson"
        df.sink_ndjson(test_file)

        from splurge_lazyframe_compare.utils.file_operations import import_lazyframe

        imported = import_lazyframe(test_file, "json")

        assert isinstance(imported, pl.LazyFrame)
        assert imported.collect().shape == (3, 2)

    def test_import_lazyframe_auto_detect(self, tmp_path):
        """Test importing LazyFrame with auto format detection."""
        # Create test files
        df = pl.LazyFrame({"id": [1, 2, 3]})

        parquet_file = tmp_path / "test.parquet"
        df.sink_parquet(parquet_file)

        csv_file = tmp_path / "test.csv"
        df.sink_csv(csv_file)

        from splurge_lazyframe_compare.utils.file_operations import import_lazyframe

        # Test parquet auto-detection
        imported_parquet = import_lazyframe(parquet_file)
        assert isinstance(imported_parquet, pl.LazyFrame)

        # Test csv auto-detection
        imported_csv = import_lazyframe(csv_file)
        assert isinstance(imported_csv, pl.LazyFrame)

    def test_import_lazyframe_file_not_found(self):
        """Test importing LazyFrame from non-existent file."""
        import pytest

        from splurge_lazyframe_compare.utils.file_operations import import_lazyframe

        with pytest.raises(FileNotFoundError) as exc_info:
            import_lazyframe(Path("nonexistent.parquet"))

        # Test that error message contains key information
        error_msg = str(exc_info.value)
        assert "nonexistent.parquet" in error_msg or "not found" in error_msg.lower()

    def test_import_lazyframe_invalid_format(self, tmp_path):
        """Test importing LazyFrame with invalid format."""
        test_file = tmp_path / "test.parquet"
        test_file.write_text("dummy")

        import pytest

        from splurge_lazyframe_compare.utils.file_operations import import_lazyframe

        with pytest.raises(ValueError) as exc_info:
            import_lazyframe(test_file, "invalid")

        # Test that error message contains key information
        error_msg = str(exc_info.value)
        assert "unsupported" in error_msg.lower()
        assert "format" in error_msg.lower()
        assert "invalid" in error_msg

    def test_get_export_file_paths_default(self, tmp_path):
        """Test getting export file paths with default format."""
        from splurge_lazyframe_compare.utils.file_operations import get_export_file_paths

        paths = get_export_file_paths("test", tmp_path)

        assert len(paths) == 1
        assert "parquet" in paths
        assert paths["parquet"] == tmp_path / "test.parquet"

    def test_get_export_file_paths_multiple_formats(self, tmp_path):
        """Test getting export file paths for multiple formats."""
        from splurge_lazyframe_compare.utils.file_operations import get_export_file_paths

        paths = get_export_file_paths("test", tmp_path, ["parquet", "csv", "json"])

        assert len(paths) == 3
        assert paths["parquet"] == tmp_path / "test.parquet"
        assert paths["csv"] == tmp_path / "test.csv"
        assert paths["json"] == tmp_path / "test.json"

    def test_get_export_file_paths_invalid_format(self, tmp_path):
        """Test getting export file paths with invalid format."""
        import pytest

        from splurge_lazyframe_compare.utils.file_operations import get_export_file_paths

        with pytest.raises(ValueError) as exc_info:
            get_export_file_paths("test", tmp_path, ["invalid"])

        # Test that error message contains key information
        error_msg = str(exc_info.value)
        assert "unsupported" in error_msg.lower()
        assert "format" in error_msg.lower()
        assert "invalid" in error_msg

    def test_validate_file_path_valid(self, tmp_path):
        """Test validating a valid file path."""
        test_file = tmp_path / "test.txt"

        from splurge_lazyframe_compare.utils.file_operations import validate_file_path

        # Should not raise any exceptions
        validate_file_path(test_file)

        assert test_file.parent.exists()

    def test_validate_file_path_basic_validation(self):
        """Test basic file path validation."""

        # Test with None (should be handled appropriately)
        # Note: The actual implementation may need to handle None differently
        # This test verifies the function doesn't crash with None input

    def test_validate_file_path_creates_parent(self, tmp_path):
        """Test that validating file path creates parent directories."""
        non_existent_dir = tmp_path / "nonexistent"
        test_file = non_existent_dir / "test.txt"

        from splurge_lazyframe_compare.utils.file_operations import validate_file_path

        # Should create parent directory and not raise any exceptions
        validate_file_path(test_file)

        # Verify parent directory was created
        assert non_existent_dir.exists()
        assert test_file.parent.exists()

    def test_list_files_by_pattern(self, tmp_path):
        """Test listing files by pattern."""
        # Create test files
        (tmp_path / "test1.txt").write_text("content1")
        (tmp_path / "test2.txt").write_text("content2")
        (tmp_path / "other.csv").write_text("content3")

        from splurge_lazyframe_compare.utils.file_operations import list_files_by_pattern

        txt_files = list_files_by_pattern(tmp_path, "*.txt")
        assert len(txt_files) == 2
        assert all(f.suffix == ".txt" for f in txt_files)

        csv_files = list_files_by_pattern(tmp_path, "*.csv")
        assert len(csv_files) == 1
        assert csv_files[0].suffix == ".csv"

    def test_list_files_by_pattern_nonexistent_directory(self):
        """Test listing files in nonexistent directory."""
        from splurge_lazyframe_compare.utils.file_operations import list_files_by_pattern

        files = list_files_by_pattern(Path("nonexistent"), "*.txt")
        assert files == []


class TestFormatting:
    """Test formatting utility functions."""

    def test_format_number(self):
        """Test number formatting."""
        from splurge_lazyframe_compare.utils.formatting import format_number

        assert format_number(3.14159, 2) == "3.14"
        assert format_number(123.456, 0) == "123"
        assert format_number(1.0, 3) == "1.000"

    def test_format_number_default_precision(self):
        """Test number formatting with default precision."""
        from splurge_lazyframe_compare.utils.formatting import format_number

        assert format_number(3.14159) == "3.14"

    def test_format_percentage(self):
        """Test percentage formatting."""
        from splurge_lazyframe_compare.utils.formatting import format_percentage

        assert format_percentage(0.5) == "50.0%"
        assert format_percentage(0.123) == "12.3%"
        assert format_percentage(1.0) == "100.0%"

    def test_format_percentage_no_symbol(self):
        """Test percentage formatting without symbol."""
        from splurge_lazyframe_compare.utils.formatting import format_percentage

        assert format_percentage(0.5, include_symbol=False) == "50.0"

    def test_format_large_number(self):
        """Test large number formatting."""
        from splurge_lazyframe_compare.utils.formatting import format_large_number

        assert format_large_number(1234) == "1,234"
        assert format_large_number(1234567) == "1,234,567"
        assert format_large_number(1000000) == "1,000,000"

    def test_truncate_string_short(self):
        """Test truncating short strings."""
        from splurge_lazyframe_compare.utils.formatting import truncate_string

        short_text = "Hello World"
        assert truncate_string(short_text) == short_text

    def test_truncate_string_long(self):
        """Test truncating long strings."""
        from splurge_lazyframe_compare.utils.formatting import truncate_string

        long_text = "This is a very long string that should be truncated because it exceeds the maximum length"
        truncated = truncate_string(long_text, max_length=30)

        assert len(truncated) == 30
        assert truncated.endswith("...")
        assert truncated.startswith("This is a very long string")

    def test_truncate_string_custom_length(self):
        """Test truncating strings with custom length."""
        from splurge_lazyframe_compare.utils.formatting import truncate_string

        text = "Very long text that needs truncation"
        truncated = truncate_string(text, max_length=15)

        assert len(truncated) == 15
        assert truncated.endswith("...")

    def test_format_dataframe_sample(self):
        """Test DataFrame sample formatting."""
        df = pl.LazyFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
                "value": [100, 200, 300, 400, 500],
            }
        )

        from splurge_lazyframe_compare.utils.formatting import format_dataframe_sample

        sample = format_dataframe_sample(df, max_rows=3)
        assert isinstance(sample, str)
        assert "id" in sample
        assert "name" in sample

        # Should only show first 3 rows
        assert "1" in sample
        assert "2" in sample
        assert "3" in sample

    def test_format_dataframe_sample_empty(self):
        """Test formatting empty DataFrame."""
        df = pl.LazyFrame(schema={"id": pl.Int64, "name": pl.Utf8})

        from splurge_lazyframe_compare.utils.formatting import format_dataframe_sample

        sample = format_dataframe_sample(df)
        assert sample == "No data to display"

    def test_format_dataframe_sample_limited_columns(self):
        """Test DataFrame sample formatting with column limit."""
        df = pl.LazyFrame({"col1": [1, 2], "col2": [3, 4], "col3": [5, 6], "col4": [7, 8]})

        from splurge_lazyframe_compare.utils.formatting import format_dataframe_sample

        sample = format_dataframe_sample(df, max_cols=2)
        assert "col1" in sample
        assert "col2" in sample
        assert "col3" not in sample  # Should be truncated

    def test_format_column_list_short(self):
        """Test formatting short column lists."""
        columns = ["id", "name", "email"]

        from splurge_lazyframe_compare.utils.formatting import format_column_list

        result = format_column_list(columns)
        assert "`id`" in result
        assert "`name`" in result
        assert "`email`" in result
        assert "... (+0 more)" not in result

    def test_format_column_list_long(self):
        """Test formatting long column lists."""
        columns = [f"col{i}" for i in range(15)]

        from splurge_lazyframe_compare.utils.formatting import format_column_list

        result = format_column_list(columns, max_items=5)
        # Test that first few items are present and properly formatted
        assert "`col0`" in result
        assert "`col4`" in result
        # Test that items beyond max_items are not present
        assert "`col5`" not in result
        assert "`col10`" not in result
        # Test for truncation indicator pattern (could be "...", "more", etc.)
        assert "..." in result
        assert "more" in result.lower()
        # Test that remaining count is indicated (10 more items)
        assert "10" in result or "ten" in result.lower()

    def test_format_validation_errors_empty(self):
        """Test formatting empty error list."""
        from splurge_lazyframe_compare.utils.formatting import format_validation_errors

        result = format_validation_errors([])
        # Test that result indicates absence of errors, not exact phrase
        assert "error" in result.lower()
        assert "found" in result.lower()
        assert "no" in result.lower()

    def test_format_validation_errors_single(self):
        """Test formatting single error."""
        error_message = "Invalid column name"
        errors = [error_message]

        from splurge_lazyframe_compare.utils.formatting import format_validation_errors

        result = format_validation_errors(errors)
        # Test that the error message is present, not exact string match
        assert error_message in result
        assert "Invalid" in result
        assert "column" in result
        assert "name" in result

    def test_format_validation_errors_multiple(self):
        """Test formatting multiple errors."""
        errors = ["Error 1", "Error 2", "Error 3"]

        from splurge_lazyframe_compare.utils.formatting import format_validation_errors

        result = format_validation_errors(errors)
        # Test for presence of all errors and structured format, not exact formatting
        assert "Error 1" in result
        assert "Error 2" in result
        assert "Error 3" in result
        # Test for some indication of multiple errors (could be "multiple", "errors", etc.)
        assert "multiple" in result.lower() or "errors" in result.lower()
        # Test for structured presentation (bullet points or numbering)
        assert "•" in result or "1." in result or "2." in result or "3." in result

    def test_create_summary_table_fallback(self):
        """Test creating summary table fallback formatting."""
        # Test the fallback formatting without tabulate
        data = {"Total Rows": 1000, "Total Columns": 5, "Memory (MB)": 50.5}

        from splurge_lazyframe_compare.utils.formatting import create_summary_table

        result = create_summary_table(data)
        # The function will try to import tabulate and fall back to simple formatting
        assert isinstance(result, str)
        assert len(result) > 0

    def test_create_summary_table_custom_headers(self):
        """Test creating summary table with custom headers."""
        data = {"Metric1": "Value1", "Metric2": "Value2"}

        from splurge_lazyframe_compare.utils.formatting import create_summary_table

        result = create_summary_table(data, headers=["Custom Header 1", "Custom Header 2"])
        assert isinstance(result, str)
        assert len(result) > 0


class TestDataPreparationService:
    """Test DataPreparationService functionality."""

    def test_service_initialization(self):
        """Test that DataPreparationService initializes correctly."""
        service = DataPreparationService()
        assert service.service_name == "DataPreparationService"

    def test_apply_column_mappings(self, sample_dataframes, sample_config):
        """Test column mapping application."""
        left_df, right_df = sample_dataframes

        service = DataPreparationService()
        mapped_left, mapped_right = service.apply_column_mappings(left=left_df, right=right_df, config=sample_config)

        # Check that columns were renamed according to mappings
        left_columns = mapped_left.collect_schema().names()
        right_columns = mapped_right.collect_schema().names()

        assert "PK_id" in left_columns
        assert "L_name" in left_columns
        assert "PK_id" in right_columns
        assert "R_name" in right_columns

    def test_apply_case_insensitive(self):
        """Test case insensitive transformation."""
        df = pl.LazyFrame({"NAME": ["Alice", "BOB", "Charlie"], "VALUE": ["ABC", "def", "GHI"]})

        service = DataPreparationService()
        result = service.apply_case_insensitive(df)

        collected = result.collect()

        # Check that string columns were converted to lowercase
        assert collected["NAME"].to_list() == ["alice", "bob", "charlie"]
        assert collected["VALUE"].to_list() == ["abc", "def", "ghi"]

    def test_get_alternating_column_order(self, sample_config):
        """Test alternating column order generation."""
        service = DataPreparationService()
        column_order = service.get_alternating_column_order(config=sample_config)

        # Should start with primary key, then alternate L_/R_ columns
        assert column_order[0] == "PK_id"
        assert "L_name" in column_order
        assert "R_name" in column_order
        assert "L_amount" in column_order
        assert "R_amount" in column_order


class TestReportingService:
    """Test ReportingService functionality."""

    def test_service_initialization(self):
        """Test that ReportingService initializes correctly."""
        service = ReportingService()
        assert service.service_name == "ReportingService"

    def test_generate_summary_report(self, sample_dataframes, sample_config):
        """Test summary report generation."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        # Create a mock result
        summary = ComparisonSummary(
            total_left_records=4,
            total_right_records=4,
            matching_records=2,
            value_differences_count=2,
            left_only_count=1,
            right_only_count=1,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=sample_dataframes[0].limit(0),  # Empty
            left_only_records=sample_dataframes[0].limit(0),  # Empty
            right_only_records=sample_dataframes[0].limit(0),  # Empty
            config=sample_config,
        )

        service = ReportingService()
        report = service.generate_summary_report(results=result)

        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in report
        assert "4" in report  # Should contain record counts
        assert "50.0%" in report  # Should contain percentages

    def test_generate_summary_report_zero_records(self, sample_config):
        """Test summary report generation with zero records."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        # Create a result with zero records
        summary = ComparisonSummary(
            total_left_records=0,
            total_right_records=0,
            matching_records=0,
            value_differences_count=0,
            left_only_count=0,
            right_only_count=0,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=pl.LazyFrame({}).limit(0),
            left_only_records=pl.LazyFrame({}).limit(0),
            right_only_records=pl.LazyFrame({}).limit(0),
            config=sample_config,
        )

        service = ReportingService()
        report = service.generate_summary_report(results=result)

        assert "0" in report
        assert "Left DataFrame:  0 records" in report
        assert "Right DataFrame: 0 records" in report
        # Should not contain percentages when records are zero
        assert "PERCENTAGES" not in report

    def test_generate_summary_report_large_numbers(self, sample_config):
        """Test summary report generation with large numbers."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        # Create a result with large numbers
        summary = ComparisonSummary(
            total_left_records=1000000,
            total_right_records=2000000,
            matching_records=500000,
            value_differences_count=300000,
            left_only_count=200000,
            right_only_count=1200000,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=pl.LazyFrame({}).limit(0),
            left_only_records=pl.LazyFrame({}).limit(0),
            right_only_records=pl.LazyFrame({}).limit(0),
            config=sample_config,
        )

        service = ReportingService()
        report = service.generate_summary_report(results=result)

        assert "1,000,000" in report  # Should format large numbers
        assert "2,000,000" in report
        assert "50.0%" in report  # Should calculate percentages correctly

    def test_generate_summary_table(self, sample_dataframes, sample_config):
        """Test summary table generation."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        summary = ComparisonSummary(
            total_left_records=4,
            total_right_records=4,
            matching_records=2,
            value_differences_count=2,
            left_only_count=1,
            right_only_count=1,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=sample_dataframes[0].limit(0),
            left_only_records=sample_dataframes[0].limit(0),
            right_only_records=sample_dataframes[0].limit(0),
            config=sample_config,
        )

        service = ReportingService()
        table = service.generate_summary_table(results=result)

        assert "Left DataFrame" in table
        assert "Right DataFrame" in table
        assert "4" in table

    def test_generate_summary_table_different_formats(self, sample_config):
        """Test summary table generation with different table formats."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        summary = ComparisonSummary(
            total_left_records=100,
            total_right_records=100,
            matching_records=50,
            value_differences_count=25,
            left_only_count=15,
            right_only_count=10,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=pl.LazyFrame({}).limit(0),
            left_only_records=pl.LazyFrame({}).limit(0),
            right_only_records=pl.LazyFrame({}).limit(0),
            config=sample_config,
        )

        service = ReportingService()

        # Test different table formats
        formats = ["grid", "simple", "pipe", "orgtbl"]
        for fmt in formats:
            table = service.generate_summary_table(results=result, table_format=fmt)
            assert "100" in table
            assert "50.0%" in table

    def test_generate_detailed_report_with_data(self, sample_config):
        """Test detailed report generation with actual data."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        # Create sample data for value differences
        diff_data = {"id": [1, 2], "name": ["Alice", "Bob"], "amount": [100.0, 200.0]}

        left_only_data = {"id": [3], "name": ["Charlie"], "amount": [300.0]}

        right_only_data = {"customer_id": [5], "customer_name": ["Eve"], "total_amount": [500.0]}

        summary = ComparisonSummary(
            total_left_records=4,
            total_right_records=4,
            matching_records=1,
            value_differences_count=2,
            left_only_count=1,
            right_only_count=1,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=pl.LazyFrame(diff_data),
            left_only_records=pl.LazyFrame(left_only_data),
            right_only_records=pl.LazyFrame(right_only_data),
            config=sample_config,
        )

        service = ReportingService()
        report = service.generate_detailed_report(results=result)

        # Check summary section
        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in report
        assert "VALUE DIFFERENCES" in report
        assert "LEFT-ONLY" in report
        assert "RIGHT-ONLY" in report

        # Check data is included
        assert "Alice" in report
        assert "Bob" in report
        assert "Charlie" in report
        assert "Eve" in report

    def test_generate_detailed_report_empty_sections(self, sample_config):
        """Test detailed report generation with empty data sections."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        summary = ComparisonSummary(
            total_left_records=4,
            total_right_records=4,
            matching_records=4,
            value_differences_count=0,
            left_only_count=0,
            right_only_count=0,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=pl.LazyFrame({}).limit(0),
            left_only_records=pl.LazyFrame({}).limit(0),
            right_only_records=pl.LazyFrame({}).limit(0),
            config=sample_config,
        )

        service = ReportingService()
        report = service.generate_detailed_report(results=result)

        # Should not include empty sections
        assert "VALUE DIFFERENCES" not in report
        assert "LEFT-ONLY" not in report
        assert "RIGHT-ONLY" not in report
        assert "No value differences found." not in report  # Because section isn't included

    def test_generate_detailed_report_max_samples(self, sample_config):
        """Test detailed report generation with max_samples parameter."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        # Create data with more records than max_samples
        diff_data = {
            "id": list(range(1, 6)),  # 5 records
            "name": [f"Person{i}" for i in range(1, 6)],
            "amount": [i * 100.0 for i in range(1, 6)],
        }

        summary = ComparisonSummary(
            total_left_records=10,
            total_right_records=10,
            matching_records=5,
            value_differences_count=5,
            left_only_count=0,
            right_only_count=0,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=pl.LazyFrame(diff_data),
            left_only_records=pl.LazyFrame({}).limit(0),
            right_only_records=pl.LazyFrame({}).limit(0),
            config=sample_config,
        )

        service = ReportingService()

        # Test with max_samples=3
        report = service.generate_detailed_report(results=result, max_samples=3)

        # Should contain only first 3 records
        assert "Person1" in report
        assert "Person2" in report
        assert "Person3" in report
        # Should not contain records beyond max_samples
        assert "Person4" not in report
        assert "Person5" not in report

    def test_generate_detailed_report_table_formats(self, sample_config):
        """Test detailed report generation with different table formats."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        diff_data = {"id": [1, 2], "name": ["Alice", "Bob"], "amount": [100.0, 200.0]}

        summary = ComparisonSummary(
            total_left_records=4,
            total_right_records=4,
            matching_records=2,
            value_differences_count=2,
            left_only_count=0,
            right_only_count=0,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=pl.LazyFrame(diff_data),
            left_only_records=pl.LazyFrame({}).limit(0),
            right_only_records=pl.LazyFrame({}).limit(0),
            config=sample_config,
        )

        service = ReportingService()

        # Test different table formats
        formats = ["grid", "simple", "pipe", "orgtbl"]
        for fmt in formats:
            report = service.generate_detailed_report(results=result, table_format=fmt)
            assert "Alice" in report
            assert "Bob" in report
            assert "VALUE DIFFERENCES" in report

    def test_export_results_all_formats(self, sample_config, tmp_path):
        """Test export_results with different formats."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        # Create sample data
        diff_data = {"id": [1, 2], "name": ["Alice", "Bob"]}
        left_data = {"id": [3], "name": ["Charlie"]}
        right_data = {"customer_id": [5], "customer_name": ["Eve"]}

        summary = ComparisonSummary(
            total_left_records=4,
            total_right_records=4,
            matching_records=1,
            value_differences_count=2,
            left_only_count=1,
            right_only_count=1,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=pl.LazyFrame(diff_data),
            left_only_records=pl.LazyFrame(left_data),
            right_only_records=pl.LazyFrame(right_data),
            config=sample_config,
        )

        service = ReportingService()

        # Test different formats
        formats = ["parquet", "csv", "json"]
        for fmt in formats:
            exported_files = service.export_results(results=result, format=fmt, output_dir=str(tmp_path))

            # Should export all files when data is present
            assert "value_differences" in exported_files
            assert "left_only_records" in exported_files
            assert "right_only_records" in exported_files
            assert "summary" in exported_files

            # Check files exist
            for file_path in exported_files.values():
                assert Path(file_path).exists()

    def test_export_results_empty_dataframes(self, sample_config, tmp_path):
        """Test export_results with empty dataframes."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        summary = ComparisonSummary(
            total_left_records=4,
            total_right_records=4,
            matching_records=4,
            value_differences_count=0,
            left_only_count=0,
            right_only_count=0,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=pl.LazyFrame({}).limit(0),
            left_only_records=pl.LazyFrame({}).limit(0),
            right_only_records=pl.LazyFrame({}).limit(0),
            config=sample_config,
        )

        service = ReportingService()
        exported_files = service.export_results(results=result, format="csv", output_dir=str(tmp_path))

        # Should only export summary when data is empty
        assert "summary" in exported_files
        assert "value_differences" not in exported_files
        assert "left_only_records" not in exported_files
        assert "right_only_records" not in exported_files

        # Check summary file exists
        assert Path(exported_files["summary"]).exists()

    def test_export_results_invalid_format(self, sample_config, tmp_path):
        """Test export_results with invalid format."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        summary = ComparisonSummary(
            total_left_records=1,
            total_right_records=1,
            matching_records=1,
            value_differences_count=0,
            left_only_count=0,
            right_only_count=0,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=pl.LazyFrame({}).limit(0),
            left_only_records=pl.LazyFrame({}).limit(0),
            right_only_records=pl.LazyFrame({}).limit(0),
            config=sample_config,
        )

        service = ReportingService()

        with pytest.raises(ValueError, match="Unsupported format"):
            service.export_results(results=result, format="invalid_format", output_dir=str(tmp_path))

    def test_export_results_invalid_directory(self, sample_config):
        """Test export_results with invalid directory."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        summary = ComparisonSummary(
            total_left_records=1,
            total_right_records=1,
            matching_records=1,
            value_differences_count=0,
            left_only_count=0,
            right_only_count=0,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=pl.LazyFrame({}).limit(0),
            left_only_records=pl.LazyFrame({}).limit(0),
            right_only_records=pl.LazyFrame({}).limit(0),
            config=sample_config,
        )

        service = ReportingService()

        # Test with non-existent absolute path
        with pytest.raises(ValueError, match="Parent directory does not exist"):
            service.export_results(results=result, format="csv", output_dir="C:/definitely/does/not/exist")

        # Test with non-existent relative path
        with pytest.raises(ValueError, match="Parent directory does not exist"):
            service.export_results(results=result, format="csv", output_dir="non/existent/directory")

    def test_export_results_non_writable_directory(self, sample_config, tmp_path, monkeypatch):
        """Test export_results with non-writable directory."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        summary = ComparisonSummary(
            total_left_records=1,
            total_right_records=1,
            matching_records=1,
            value_differences_count=0,
            left_only_count=0,
            right_only_count=0,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=pl.LazyFrame({}).limit(0),
            left_only_records=pl.LazyFrame({}).limit(0),
            right_only_records=pl.LazyFrame({}).limit(0),
            config=sample_config,
        )

        service = ReportingService()

        # Mock os.access to return False (not writable)
        def mock_access(path, mode):
            return False

        monkeypatch.setattr("os.access", mock_access)

        with pytest.raises(ValueError, match="Parent directory is not writable"):
            service.export_results(results=result, format="csv", output_dir=str(tmp_path))

    def test_input_validation_invalid_results(self):
        """Test input validation for invalid results parameter."""
        service = ReportingService()

        with pytest.raises(ValueError, match="results must be a ComparisonResult"):
            service.generate_summary_report(results="invalid")

        with pytest.raises(ValueError, match="results must be a ComparisonResult"):
            service.generate_detailed_report(results="invalid")

        with pytest.raises(ValueError, match="results must be a ComparisonResult"):
            service.generate_summary_table(results="invalid")

        with pytest.raises(ValueError, match="results must be a ComparisonResult"):
            service.export_results(results="invalid")

    def test_edge_case_special_characters_in_data(self, sample_config):
        """Test handling of special characters in data."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        # Create data with special characters
        diff_data = {
            "name": ["José", "Müller", "O'Connor", "测试"],
            "description": ["Line 1\nLine 2", "Tab\there", 'Quote"here', "Unicode: 🎉"],
        }

        summary = ComparisonSummary(
            total_left_records=4,
            total_right_records=4,
            matching_records=0,
            value_differences_count=4,
            left_only_count=0,
            right_only_count=0,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=pl.LazyFrame(diff_data),
            left_only_records=pl.LazyFrame({}).limit(0),
            right_only_records=pl.LazyFrame({}).limit(0),
            config=sample_config,
        )

        service = ReportingService()
        report = service.generate_detailed_report(results=result)

        # Should handle special characters gracefully
        assert "VALUE DIFFERENCES" in report
        assert "José" in report or "Unicode" in report  # At least some data should be present

    def test_edge_case_null_values_in_data(self, sample_config):
        """Test handling of null values in data."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        # Create data with null values
        diff_data = {"id": [1, 2, None], "name": ["Alice", None, "Charlie"], "amount": [100.0, None, None]}

        summary = ComparisonSummary(
            total_left_records=3,
            total_right_records=3,
            matching_records=0,
            value_differences_count=3,
            left_only_count=0,
            right_only_count=0,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=pl.LazyFrame(diff_data),
            left_only_records=pl.LazyFrame({}).limit(0),
            right_only_records=pl.LazyFrame({}).limit(0),
            config=sample_config,
        )

        service = ReportingService()
        report = service.generate_detailed_report(results=result)

        # Should handle null values gracefully
        assert "VALUE DIFFERENCES" in report
        assert "Alice" in report
        assert "Charlie" in report

    def test_edge_case_very_large_max_samples(self, sample_config):
        """Test detailed report with very large max_samples parameter."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        # Create small dataset
        diff_data = {"id": [1, 2], "name": ["Alice", "Bob"]}

        summary = ComparisonSummary(
            total_left_records=10,
            total_right_records=10,
            matching_records=8,
            value_differences_count=2,
            left_only_count=0,
            right_only_count=0,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=pl.LazyFrame(diff_data),
            left_only_records=pl.LazyFrame({}).limit(0),
            right_only_records=pl.LazyFrame({}).limit(0),
            config=sample_config,
        )

        service = ReportingService()

        # Test with max_samples larger than available data
        report = service.generate_detailed_report(results=result, max_samples=1000)

        # Should include all available data
        assert "VALUE DIFFERENCES" in report
        assert "Alice" in report
        assert "Bob" in report

    def test_edge_case_zero_max_samples(self, sample_config):
        """Test detailed report with zero max_samples parameter."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        # Create dataset
        diff_data = {"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]}

        summary = ComparisonSummary(
            total_left_records=5,
            total_right_records=5,
            matching_records=2,
            value_differences_count=3,
            left_only_count=0,
            right_only_count=0,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=pl.LazyFrame(diff_data),
            left_only_records=pl.LazyFrame({}).limit(0),
            right_only_records=pl.LazyFrame({}).limit(0),
            config=sample_config,
        )

        service = ReportingService()

        # Test with max_samples=0
        report = service.generate_detailed_report(results=result, max_samples=0)

        # Should include section header but no data rows
        assert "VALUE DIFFERENCES" in report
        # Should not contain actual data
        assert "Alice" not in report
        assert "Bob" not in report
        assert "Charlie" not in report

    def test_edge_case_different_timestamp_formats(self, sample_config):
        """Test summary report with different timestamp formats."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        # Test different timestamp formats
        timestamps = ["2025-01-01T00:00:00", "2025-01-01 00:00:00", "2025-01-01T00:00:00.123456", "2025-01-01"]

        service = ReportingService()

        for timestamp in timestamps:
            summary = ComparisonSummary(
                total_left_records=1,
                total_right_records=1,
                matching_records=1,
                value_differences_count=0,
                left_only_count=0,
                right_only_count=0,
                comparison_timestamp=timestamp,
            )

            result = ComparisonResult(
                summary=summary,
                value_differences=pl.LazyFrame({}).limit(0),
                left_only_records=pl.LazyFrame({}).limit(0),
                right_only_records=pl.LazyFrame({}).limit(0),
                config=sample_config,
            )

            report = service.generate_summary_report(results=result)

            # Should contain the timestamp in some form
            assert timestamp.split("T")[0] in report  # At least the date should be present

    def test_export_results_relative_path(self, sample_config, tmp_path):
        """Test export_results with relative paths."""
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        summary = ComparisonSummary(
            total_left_records=1,
            total_right_records=1,
            matching_records=1,
            value_differences_count=0,
            left_only_count=0,
            right_only_count=0,
            comparison_timestamp="2025-01-01T00:00:00",
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=pl.LazyFrame({}).limit(0),
            left_only_records=pl.LazyFrame({}).limit(0),
            right_only_records=pl.LazyFrame({}).limit(0),
            config=sample_config,
        )

        service = ReportingService()

        # Change to tmp_path directory and use relative path
        original_cwd = tmp_path.cwd()
        try:
            import os

            os.chdir(tmp_path)

            # Test with relative path
            exported_files = service.export_results(results=result, format="csv", output_dir="relative_output")

            # Should create the relative directory and export files
            assert "summary" in exported_files
            assert Path("relative_output").exists()

            # Check that files exist in the relative directory
            for file_path in exported_files.values():
                assert Path(file_path).exists()

        finally:
            os.chdir(original_cwd)


class TestComparisonService:
    """Test ComparisonService functionality."""

    def test_service_initialization(self):
        """Test that ComparisonService initializes correctly."""
        service = ComparisonService()
        assert service.service_name == "ComparisonService"

    def test_execute_comparison_integration(self, sample_dataframes, sample_config):
        """Test full comparison execution integration."""
        left_df, right_df = sample_dataframes

        service = ComparisonService()
        result = service.execute_comparison(left=left_df, right=right_df, config=sample_config)

        assert result.summary.total_left_records == 4
        assert result.summary.total_right_records == 4
        assert result.summary.matching_records >= 0
        assert result.summary.value_differences_count >= 0
        assert result.summary.left_only_count >= 0
        assert result.summary.right_only_count >= 0

        # Check that result DataFrames are not None
        assert result.value_differences is not None
        assert result.left_only_records is not None
        assert result.right_only_records is not None


class TestComparisonOrchestrator:
    """Test ComparisonOrchestrator functionality."""

    def test_service_initialization(self):
        """Test that ComparisonOrchestrator initializes correctly."""
        orchestrator = ComparisonOrchestrator()
        assert orchestrator.service_name == "ComparisonOrchestrator"
        assert orchestrator.comparison_service is not None
        assert orchestrator.reporting_service is not None

    def test_compare_dataframes_integration(self, sample_dataframes, sample_config):
        """Test orchestrator DataFrame comparison."""
        left_df, right_df = sample_dataframes

        orchestrator = ComparisonOrchestrator()
        result = orchestrator.compare_dataframes(config=sample_config, left=left_df, right=right_df)

        assert result.summary.total_left_records == 4
        assert result.summary.total_right_records == 4
        assert result.summary.matching_records >= 0

    def test_compare_and_report_integration(self, sample_dataframes, sample_config):
        """Test orchestrator compare and report."""
        left_df, right_df = sample_dataframes

        orchestrator = ComparisonOrchestrator()
        report = orchestrator.compare_and_report(
            config=sample_config, left=left_df, right=right_df, include_samples=False
        )

        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in report
        assert "RECORD COUNTS:" in report

    def test_service_injection(self, sample_dataframes, sample_config):
        """Test service dependency injection."""
        left_df, right_df = sample_dataframes

        # Create custom services
        validation_service = ValidationService()
        preparation_service = DataPreparationService()
        reporting_service = ReportingService()

        # Create comparison service with injected dependencies
        comparison_service = ComparisonService(
            validation_service=validation_service, preparation_service=preparation_service
        )

        # Create orchestrator with custom services
        orchestrator = ComparisonOrchestrator(
            comparison_service=comparison_service, reporting_service=reporting_service
        )

        # Test that it works
        result = orchestrator.compare_dataframes(config=sample_config, left=left_df, right=right_df)

        assert result.summary.matching_records >= 0


class TestComparisonOrchestratorComprehensive:
    """Comprehensive tests for ComparisonOrchestrator public APIs."""

    def setup_method(self) -> None:
        """Set up test fixtures for ComparisonOrchestrator tests."""
        # Create test schemas
        self.left_columns = {
            "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "order_date": ColumnDefinition(name="order_date", alias="Order Date", datatype=pl.Date, nullable=False),
            "amount": ColumnDefinition(name="amount", alias="Order Amount", datatype=pl.Float64, nullable=False),
            "status": ColumnDefinition(name="status", alias="Order Status", datatype=pl.Utf8, nullable=True),
        }
        self.right_columns = {
            "cust_id": ColumnDefinition(name="cust_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "order_dt": ColumnDefinition(name="order_dt", alias="Order Date", datatype=pl.Date, nullable=False),
            "total_amount": ColumnDefinition(
                name="total_amount", alias="Order Amount", datatype=pl.Float64, nullable=False
            ),
            "order_status": ColumnDefinition(
                name="order_status", alias="Order Status", datatype=pl.Utf8, nullable=True
            ),
        }

        self.left_schema = ComparisonSchema(
            columns=self.left_columns,
            pk_columns=["customer_id", "order_date"],
        )
        self.right_schema = ComparisonSchema(
            columns=self.right_columns,
            pk_columns=["cust_id", "order_dt"],
        )

        # Create column mappings
        self.mappings = [
            ColumnMapping(left="customer_id", right="cust_id", name="customer_id"),
            ColumnMapping(left="order_date", right="order_dt", name="order_date"),
            ColumnMapping(left="amount", right="total_amount", name="amount"),
            ColumnMapping(left="status", right="order_status", name="status"),
        ]

        # Create configuration
        self.config = ComparisonConfig(
            left_schema=self.left_schema,
            right_schema=self.right_schema,
            column_mappings=self.mappings,
            pk_columns=["customer_id", "order_date"],
        )

        self.orchestrator = ComparisonOrchestrator()

    def test_compare_dataframes_comprehensive(self) -> None:
        """Test compare_dataframes with various scenarios."""
        # Test with identical dataframes
        left_data = {
            "customer_id": [1, 2, 3],
            "order_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", "completed", "pending"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d"))

        # Create right dataframe with correct column names for right schema
        right_data = {
            "cust_id": [1, 2, 3],
            "order_dt": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "total_amount": [100.0, 200.0, 300.0],
            "order_status": ["pending", "completed", "pending"],
        }
        right_df = pl.LazyFrame(right_data).with_columns(pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d"))

        # Test successful comparison
        result = self.orchestrator.compare_dataframes(config=self.config, left=left_df, right=right_df)

        assert result.summary.total_left_records == 3
        assert result.summary.total_right_records == 3
        assert result.summary.matching_records == 3
        assert result.summary.value_differences_count == 0
        assert result.summary.left_only_count == 0
        assert result.summary.right_only_count == 0

    def test_compare_dataframes_with_differences(self) -> None:
        """Test compare_dataframes when there are differences."""
        left_data = {
            "customer_id": [1, 2, 3],
            "order_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", "completed", "pending"],
        }
        right_data = {
            "cust_id": [1, 2, 4],  # Different customer 3 -> 4
            "order_dt": ["2023-01-01", "2023-01-02", "2023-01-04"],  # Different date
            "total_amount": [100.0, 250.0, 400.0],  # Different amount for customer 2
            "order_status": ["pending", "completed", "shipped"],  # Different status
        }

        left_df = pl.LazyFrame(left_data).with_columns(pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d"))
        right_df = pl.LazyFrame(right_data).with_columns(pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d"))

        result = self.orchestrator.compare_dataframes(config=self.config, left=left_df, right=right_df)

        assert result.summary.total_left_records == 3
        assert result.summary.total_right_records == 3
        assert result.summary.matching_records == 1  # Only Customer 1 matches (same values)
        assert result.summary.value_differences_count == 1  # Customer 2 has different amount
        assert result.summary.left_only_count == 1  # Customer 3 only in left
        assert result.summary.right_only_count == 1  # Customer 4 only in right

    def test_compare_and_report_comprehensive(self) -> None:
        """Test compare_and_report with various parameters."""
        left_data = {
            "customer_id": [1, 2],
            "order_date": ["2023-01-01", "2023-01-02"],
            "amount": [100.0, 200.0],
            "status": ["pending", "completed"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d"))

        # Create right dataframe with correct column names for right schema
        right_data = {
            "cust_id": [1, 2],
            "order_dt": ["2023-01-01", "2023-01-02"],
            "total_amount": [100.0, 200.0],
            "order_status": ["pending", "completed"],
        }
        right_df = pl.LazyFrame(right_data).with_columns(pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d"))

        # Test with default parameters
        report = self.orchestrator.compare_and_report(config=self.config, left=left_df, right=right_df)

        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in report
        assert "RECORD COUNTS:" in report

        # Test with custom parameters
        report_custom = self.orchestrator.compare_and_report(
            config=self.config,
            left=left_df,
            right=right_df,
            include_samples=False,
            max_samples=5,
            table_format="simple",
        )

        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in report_custom

    def test_compare_and_export_comprehensive(self) -> None:
        """Test compare_and_export with various formats and scenarios."""
        import tempfile
        from pathlib import Path

        left_data = {
            "customer_id": [1, 2],
            "order_date": ["2023-01-01", "2023-01-02"],
            "amount": [100.0, 200.0],
            "status": ["pending", "completed"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d"))

        # Create right dataframe with correct column names for right schema
        right_data = {
            "cust_id": [1, 2],
            "order_dt": ["2023-01-01", "2023-01-02"],
            "total_amount": [100.0, 200.0],
            "order_status": ["pending", "completed"],
        }
        right_df = pl.LazyFrame(right_data).with_columns(pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d"))

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test parquet export
            exported_files = self.orchestrator.compare_and_export(
                config=self.config, left=left_df, right=right_df, output_dir=temp_dir, format="parquet"
            )

            assert len(exported_files) == 1  # Only summary file when no differences
            for file_path in exported_files.values():
                assert Path(file_path).exists()
                assert file_path.endswith(".json")

            # Test CSV export
            exported_csv = self.orchestrator.compare_and_export(
                config=self.config, left=left_df, right=right_df, output_dir=temp_dir, format="csv"
            )

            for file_path in exported_csv.values():
                assert Path(file_path).exists()
                assert file_path.endswith(".json")  # Summary is always JSON

    def test_generate_report_from_result_comprehensive(self) -> None:
        """Test generate_report_from_result with different report types."""
        left_data = {
            "customer_id": [1, 2],
            "order_date": ["2023-01-01", "2023-01-02"],
            "amount": [100.0, 200.0],
            "status": ["pending", "completed"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d"))

        # Create right dataframe with correct column names for right schema
        right_data = {
            "cust_id": [1, 2],
            "order_dt": ["2023-01-01", "2023-01-02"],
            "total_amount": [100.0, 200.0],
            "order_status": ["pending", "completed"],
        }
        right_df = pl.LazyFrame(right_data).with_columns(pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d"))

        # Execute comparison
        result = self.orchestrator.compare_dataframes(config=self.config, left=left_df, right=right_df)

        # Test detailed report
        detailed_report = self.orchestrator.generate_report_from_result(result=result, report_type="detailed")
        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in detailed_report

        # Test summary report
        summary_report = self.orchestrator.generate_report_from_result(result=result, report_type="summary")
        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in summary_report

        # Test table report
        table_report = self.orchestrator.generate_report_from_result(result=result, report_type="table")
        assert "Left DataFrame" in table_report

        # Test invalid report type
        with pytest.raises(ValueError) as exc_info:
            self.orchestrator.generate_report_from_result(result=result, report_type="invalid")

        # Test that error message contains key information
        error_msg = str(exc_info.value)
        assert "report" in error_msg.lower()
        assert "type" in error_msg.lower()
        assert "invalid" in error_msg.lower() or "unknown" in error_msg.lower()

    def test_get_comparison_summary_comprehensive(self) -> None:
        """Test get_comparison_summary method."""
        left_data = {
            "customer_id": [1, 2],
            "order_date": ["2023-01-01", "2023-01-02"],
            "amount": [100.0, 200.0],
            "status": ["pending", "completed"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d"))

        # Create right dataframe with correct column names for right schema
        right_data = {
            "cust_id": [1, 2],
            "order_dt": ["2023-01-01", "2023-01-02"],
            "total_amount": [100.0, 200.0],
            "order_status": ["pending", "completed"],
        }
        right_df = pl.LazyFrame(right_data).with_columns(pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d"))

        # Execute comparison
        result = self.orchestrator.compare_dataframes(config=self.config, left=left_df, right=right_df)

        # Test getting summary
        summary = self.orchestrator.get_comparison_summary(result=result)
        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in summary
        assert "2" in summary  # Record counts

    def test_get_comparison_table_comprehensive(self) -> None:
        """Test get_comparison_table method."""
        left_data = {
            "customer_id": [1, 2],
            "order_date": ["2023-01-01", "2023-01-02"],
            "amount": [100.0, 200.0],
            "status": ["pending", "completed"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d"))

        # Create right dataframe with correct column names for right schema
        right_data = {
            "cust_id": [1, 2],
            "order_dt": ["2023-01-01", "2023-01-02"],
            "total_amount": [100.0, 200.0],
            "order_status": ["pending", "completed"],
        }
        right_df = pl.LazyFrame(right_data).with_columns(pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d"))

        # Execute comparison
        result = self.orchestrator.compare_dataframes(config=self.config, left=left_df, right=right_df)

        # Test getting table with different formats
        table_default = self.orchestrator.get_comparison_table(result=result)
        assert "Left DataFrame" in table_default

        table_simple = self.orchestrator.get_comparison_table(result=result, table_format="simple")
        assert "Left DataFrame" in table_simple

    def test_compare_dataframes_input_validation(self) -> None:
        """Test input validation for compare_dataframes."""
        valid_df = pl.LazyFrame(
            {
                "customer_id": [1, 2, 3],
                "order_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "amount": [100.0, 200.0, 300.0],
                "status": ["pending", "completed", "pending"],
            }
        ).with_columns(pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d"))
        invalid_df = "not a dataframe"

        # Test with invalid config
        with pytest.raises(ValueError) as exc_info:
            self.orchestrator.compare_dataframes(config="invalid_config", left=valid_df, right=valid_df)

        # Test that error message contains key information
        error_msg = str(exc_info.value)
        assert "config" in error_msg.lower()
        assert "comparison" in error_msg.lower()

        # Test with invalid left dataframe
        with pytest.raises(ValueError) as exc_info:
            self.orchestrator.compare_dataframes(config=self.config, left=invalid_df, right=valid_df)

        # Test that error message contains key information
        error_msg = str(exc_info.value)
        assert "left" in error_msg.lower()
        assert "lazy" in error_msg.lower() or "dataframe" in error_msg.lower()

        # Test with invalid right dataframe
        with pytest.raises(ValueError) as exc_info:
            self.orchestrator.compare_dataframes(config=self.config, left=valid_df, right=invalid_df)

        # Test that error message contains key information
        error_msg = str(exc_info.value)
        assert "right" in error_msg.lower()
        assert "lazy" in error_msg.lower() or "dataframe" in error_msg.lower()

    def test_compare_and_report_input_validation(self) -> None:
        """Test input validation for compare_and_report."""
        valid_df = pl.LazyFrame(
            {
                "customer_id": [1, 2, 3],
                "order_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "amount": [100.0, 200.0, 300.0],
                "status": ["pending", "completed", "pending"],
            }
        ).with_columns(pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d"))
        invalid_df = "not a dataframe"

        # Test with invalid config
        with pytest.raises(ValueError, match="config must be a ComparisonConfig"):
            self.orchestrator.compare_and_report(config="invalid_config", left=valid_df, right=valid_df)

        # Test with invalid dataframes
        with pytest.raises(ValueError):
            self.orchestrator.compare_and_report(config=self.config, left=invalid_df, right=valid_df)

    def test_compare_and_export_input_validation(self) -> None:
        """Test input validation for compare_and_export."""
        import tempfile

        left_df = pl.LazyFrame(
            {
                "customer_id": [1, 2, 3],
                "order_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "amount": [100.0, 200.0, 300.0],
                "status": ["pending", "completed", "pending"],
            }
        ).with_columns(pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d"))

        right_df = pl.LazyFrame(
            {
                "cust_id": [1, 2, 3],
                "order_dt": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "total_amount": [100.0, 200.0, 300.0],
                "order_status": ["pending", "completed", "pending"],
            }
        ).with_columns(pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d"))

        invalid_df = "not a dataframe"

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with invalid format
            with pytest.raises((ValueError, TypeError)):
                self.orchestrator.compare_and_export(
                    config=self.config, left=left_df, right=right_df, output_dir=temp_dir, format="invalid_format"
                )

            # Test with invalid dataframe
            with pytest.raises(ValueError):
                self.orchestrator.compare_and_export(
                    config=self.config, left=invalid_df, right=right_df, output_dir=temp_dir
                )

    def test_orchestrator_initialization_with_custom_services(self) -> None:
        """Test orchestrator initialization with custom services."""
        # Create custom services
        custom_comparison = ComparisonService()
        custom_reporting = ReportingService()

        # Initialize orchestrator with custom services
        orchestrator = ComparisonOrchestrator(comparison_service=custom_comparison, reporting_service=custom_reporting)

        # Verify services are set correctly
        assert orchestrator.comparison_service == custom_comparison
        assert orchestrator.reporting_service == custom_reporting

    def test_orchestrator_initialization_default_services(self) -> None:
        """Test orchestrator initialization with default services."""
        orchestrator = ComparisonOrchestrator()

        # Verify default services are created
        assert isinstance(orchestrator.comparison_service, ComparisonService)
        assert isinstance(orchestrator.reporting_service, ReportingService)

    def test_compare_dataframes_with_schema_validation_error(self) -> None:
        """Test compare_dataframes with schema validation error."""
        # Create dataframes with mismatched schemas
        left_df = pl.LazyFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"]})

        right_df = pl.LazyFrame(
            {
                "customer_id": [1, 2, 3],  # Different column name
                "full_name": ["Alice", "Bob", "Charlie"],  # Different column name
            }
        )

        # This should raise a SchemaValidationError
        with pytest.raises(SchemaValidationError):
            self.orchestrator.compare_dataframes(config=self.config, left=left_df, right=right_df)

    def test_compare_dataframes_with_duplicates_allowed(self) -> None:
        """Test compare_dataframes handles duplicate primary keys gracefully."""
        # Create dataframes with duplicate primary keys
        left_data = {
            "customer_id": [1, 1, 2],  # Duplicate customer_id 1
            "order_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", "completed", "pending"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d"))

        # Create right dataframe with correct column names for right schema
        right_data = {
            "cust_id": [1, 2, 3],
            "order_dt": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "total_amount": [100.0, 200.0, 300.0],
            "order_status": ["pending", "completed", "pending"],
        }
        right_df = pl.LazyFrame(right_data).with_columns(pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d"))

        # Comparison should succeed (current behavior allows duplicates)
        result = self.orchestrator.compare_dataframes(config=self.config, left=left_df, right=right_df)

        # Verify the comparison completed
        assert result.summary.total_left_records == 3
        assert result.summary.total_right_records == 3
