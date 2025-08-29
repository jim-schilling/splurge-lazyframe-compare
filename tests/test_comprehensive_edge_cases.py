"""Comprehensive edge case and behavior validation tests for the comparison framework.

These tests focus on achieving 85%+ code coverage by testing edge cases,
error conditions, boundary scenarios, and real-world usage patterns.
"""

import copy
import tempfile
from pathlib import Path
from typing import Any, Dict

import polars as pl
import pytest

from splurge_lazyframe_compare import (
    ColumnDefinition,
    ColumnMapping,
    ComparisonConfig,
    ComparisonOrchestrator,
    ComparisonSchema,
    ComparisonSummary,
    LazyFrameComparator,
)
from splurge_lazyframe_compare.exceptions.comparison_exceptions import (
    PrimaryKeyViolationError,
    SchemaValidationError,
)


class TestEdgeCasesEmptyDataFrames:
    """Test edge cases with empty DataFrames."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.left_columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=False),
        }
        self.right_columns = {
            "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "full_name": ColumnDefinition(name="full_name", alias="Full Name", datatype=pl.Utf8, nullable=False),
        }

        self.left_schema = ComparisonSchema(
            columns=self.left_columns,
            pk_columns=["id"],
        )
        self.right_schema = ComparisonSchema(
            columns=self.right_columns,
            pk_columns=["customer_id"],
        )

        self.config = ComparisonConfig(
            left_schema=self.left_schema,
            right_schema=self.right_schema,
            column_mappings=[
                ColumnMapping(left="id", right="customer_id", name="id"),
                ColumnMapping(left="name", right="full_name", name="name"),
            ],
            pk_columns=["id"],
        )

        self.orchestrator = ComparisonOrchestrator()
        self.comparator = LazyFrameComparator(self.config)

    def test_empty_left_dataframe(self) -> None:
        """Test comparison with empty left DataFrame."""
        left_df = pl.LazyFrame(schema={"id": pl.Int64, "name": pl.Utf8})
        right_df = pl.LazyFrame({
            "customer_id": [1, 2],
            "full_name": ["Alice", "Bob"]
        })

        result = self.orchestrator.compare_dataframes(
            config=self.config,
            left=left_df,
            right=right_df
        )

        assert result.summary.total_left_records == 0
        assert result.summary.total_right_records == 2
        assert result.summary.matching_records == 0
        assert result.summary.left_only_count == 0
        assert result.summary.right_only_count == 2

    def test_empty_right_dataframe(self) -> None:
        """Test comparison with empty right DataFrame."""
        left_df = pl.LazyFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"]
        })
        right_df = pl.LazyFrame(schema={"customer_id": pl.Int64, "full_name": pl.Utf8})

        result = self.orchestrator.compare_dataframes(
            config=self.config,
            left=left_df,
            right=right_df
        )

        assert result.summary.total_left_records == 2
        assert result.summary.total_right_records == 0
        assert result.summary.matching_records == 0
        assert result.summary.left_only_count == 2
        assert result.summary.right_only_count == 0

    def test_both_empty_dataframes(self) -> None:
        """Test comparison with both DataFrames empty."""
        left_df = pl.LazyFrame(schema={"id": pl.Int64, "name": pl.Utf8})
        right_df = pl.LazyFrame(schema={"customer_id": pl.Int64, "full_name": pl.Utf8})

        result = self.orchestrator.compare_dataframes(
            config=self.config,
            left=left_df,
            right=right_df
        )

        assert result.summary.total_left_records == 0
        assert result.summary.total_right_records == 0
        assert result.summary.matching_records == 0
        assert result.summary.left_only_count == 0
        assert result.summary.right_only_count == 0

    def test_empty_dataframe_export(self) -> None:
        """Test exporting results with empty DataFrames."""
        left_df = pl.LazyFrame(schema={"id": pl.Int64, "name": pl.Utf8})
        right_df = pl.LazyFrame(schema={"customer_id": pl.Int64, "full_name": pl.Utf8})

        with tempfile.TemporaryDirectory() as temp_dir:
            exported_files = self.orchestrator.compare_and_export(
                config=self.config,
                left=left_df,
                right=right_df,
                output_dir=temp_dir,
                format="csv"
            )

            # Should export summary file even when dataframes are empty
            # Other files (value_differences, left_only, right_only) are not created when empty
            assert len(exported_files) == 1  # Only summary file
            for file_path in exported_files.values():
                assert Path(file_path).exists()
                assert file_path.endswith('.json')  # Summary is always exported as JSON

    def test_empty_dataframe_report(self) -> None:
        """Test generating reports with empty DataFrames."""
        left_df = pl.LazyFrame(schema={"id": pl.Int64, "name": pl.Utf8})
        right_df = pl.LazyFrame(schema={"customer_id": pl.Int64, "full_name": pl.Utf8})

        report = self.orchestrator.compare_and_report(
            config=self.config,
            left=left_df,
            right=right_df
        )

        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in report
        assert "0" in report  # Record counts should be 0


class TestDataTypeEdgeCases:
    """Test edge cases with various Polars data types."""

    def test_datetime_with_timezone_handling(self) -> None:
        """Test datetime columns with timezone information."""
        left_data = {
            "id": [1, 2],
            "timestamp": ["2023-01-01T10:00:00Z", "2023-01-02T15:30:00Z"],
        }
        right_data = {
            "customer_id": [1, 2],
            "event_time": ["2023-01-01T10:00:00Z", "2023-01-02T15:30:00Z"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ")
        )
        right_df = pl.LazyFrame(right_data).with_columns(
            pl.col("event_time").str.strptime(pl.Datetime, "%Y-%m-%dT%H:%M:%SZ")
        )

        left_schema = ComparisonSchema(
            columns={
                "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
                "timestamp": ColumnDefinition(name="timestamp", alias="Timestamp", datatype=pl.Datetime(time_unit="us"), nullable=False),
            },
            pk_columns=["id"],
        )
        right_schema = ComparisonSchema(
            columns={
                "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
                "event_time": ColumnDefinition(name="event_time", alias="Event Time", datatype=pl.Datetime(time_unit="us"), nullable=False),
            },
            pk_columns=["customer_id"],
        )

        config = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=[
                ColumnMapping(left="id", right="customer_id", name="id"),
                ColumnMapping(left="timestamp", right="event_time", name="timestamp"),
            ],
            pk_columns=["id"],
        )

        orchestrator = ComparisonOrchestrator()
        result = orchestrator.compare_dataframes(
            config=config,
            left=left_df,
            right=right_df
        )

        assert result.summary.matching_records == 2

    def test_list_column_handling(self) -> None:
        """Test columns containing lists/arrays."""
        left_data = {
            "id": [1, 2],
            "tags": [["red", "blue"], ["green", "yellow"]],
        }
        right_data = {
            "customer_id": [1, 2],
            "categories": [["red", "blue"], ["green", "yellow"]],
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        left_schema = ComparisonSchema(
            columns={
                "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
                "tags": ColumnDefinition(name="tags", alias="Tags", datatype=pl.List(pl.Utf8), nullable=False),
            },
            pk_columns=["id"],
        )
        right_schema = ComparisonSchema(
            columns={
                "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
                "categories": ColumnDefinition(name="categories", alias="Categories", datatype=pl.List(pl.Utf8), nullable=False),
            },
            pk_columns=["customer_id"],
        )

        config = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=[
                ColumnMapping(left="id", right="customer_id", name="id"),
                ColumnMapping(left="tags", right="categories", name="tags"),
            ],
            pk_columns=["id"],
        )

        orchestrator = ComparisonOrchestrator()
        result = orchestrator.compare_dataframes(
            config=config,
            left=left_df,
            right=right_df
        )

        assert result.summary.matching_records == 2

    def test_struct_column_handling(self) -> None:
        """Test columns containing struct/nested data."""
        left_data = {
            "id": [1, 2],
            "address": [
                {"street": "123 Main St", "city": "Anytown"},
                {"street": "456 Oak Ave", "city": "Somewhere"}
            ],
        }
        right_data = {
            "customer_id": [1, 2],
            "location": [
                {"street": "123 Main St", "city": "Anytown"},
                {"street": "456 Oak Ave", "city": "Somewhere"}
            ],
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        # Define struct schema
        address_schema = {"street": pl.Utf8, "city": pl.Utf8}

        left_schema = ComparisonSchema(
            columns={
                "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
                "address": ColumnDefinition(name="address", alias="Address", datatype=pl.Struct(address_schema), nullable=False),
            },
            pk_columns=["id"],
        )
        right_schema = ComparisonSchema(
            columns={
                "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
                "location": ColumnDefinition(name="location", alias="Location", datatype=pl.Struct(address_schema), nullable=False),
            },
            pk_columns=["customer_id"],
        )

        config = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=[
                ColumnMapping(left="id", right="customer_id", name="id"),
                ColumnMapping(left="address", right="location", name="address"),
            ],
            pk_columns=["id"],
        )

        orchestrator = ComparisonOrchestrator()
        result = orchestrator.compare_dataframes(
            config=config,
            left=left_df,
            right=right_df
        )

        assert result.summary.matching_records == 2

    def test_nullable_column_edge_cases(self) -> None:
        """Test edge cases with nullable columns."""
        left_data = {
            "id": [1, 2, 3],
            "optional_field": ["value1", None, "value3"],
        }
        right_data = {
            "customer_id": [1, 2, 3],
            "optional_data": ["value1", None, "value3"],
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        left_schema = ComparisonSchema(
            columns={
                "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
                "optional_field": ColumnDefinition(name="optional_field", alias="Optional Field", datatype=pl.Utf8, nullable=True),
            },
            pk_columns=["id"],
        )
        right_schema = ComparisonSchema(
            columns={
                "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
                "optional_data": ColumnDefinition(name="optional_data", alias="Optional Data", datatype=pl.Utf8, nullable=True),
            },
            pk_columns=["customer_id"],
        )

        config = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=[
                ColumnMapping(left="id", right="customer_id", name="id"),
                ColumnMapping(left="optional_field", right="optional_data", name="optional_field"),
            ],
            pk_columns=["id"],
            null_equals_null=True,
        )

        orchestrator = ComparisonOrchestrator()
        result = orchestrator.compare_dataframes(
            config=config,
            left=left_df,
            right=right_df
        )

        assert result.summary.matching_records == 3  # All should match including null values

    def test_numeric_precision_edge_cases(self) -> None:
        """Test edge cases with numeric precision and floating point comparisons."""
        left_data = {
            "id": [1, 2, 3],
            "value": [1.23456789012345, 2.78901234567890, 3.45678901234567],
        }
        right_data = {
            "customer_id": [1, 2, 3],
            "amount": [1.23456789012346, 2.78901234567891, 3.45678901234568],  # Slightly different
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        left_schema = ComparisonSchema(
            columns={
                "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
                "value": ColumnDefinition(name="value", alias="Value", datatype=pl.Float64, nullable=False),
            },
            pk_columns=["id"],
        )
        right_schema = ComparisonSchema(
            columns={
                "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
                "amount": ColumnDefinition(name="amount", alias="Amount", datatype=pl.Float64, nullable=False),
            },
            pk_columns=["customer_id"],
        )

        config = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=[
                ColumnMapping(left="id", right="customer_id", name="id"),
                ColumnMapping(left="value", right="amount", name="value"),
            ],
            pk_columns=["id"],
            tolerance={"value": 1e-10},  # Very small tolerance
        )

        orchestrator = ComparisonOrchestrator()
        result = orchestrator.compare_dataframes(
            config=config,
            left=left_df,
            right=right_df
        )

        # With small tolerance, these should match
        assert result.summary.matching_records == 3
        assert result.summary.value_differences_count == 0


class TestConfigurationValidationEdgeCases:
    """Test edge cases in configuration validation."""

    def test_config_with_empty_mappings(self) -> None:
        """Test configuration validation with empty column mappings."""
        left_schema = ComparisonSchema(
            columns={
                "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
                "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=False),
            },
            pk_columns=["id"],
        )
        right_schema = ComparisonSchema(
            columns={
                "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
                "full_name": ColumnDefinition(name="full_name", alias="Full Name", datatype=pl.Utf8, nullable=False),
            },
            pk_columns=["customer_id"],
        )

        with pytest.raises((ValueError, SchemaValidationError)):
            ComparisonConfig(
                left_schema=left_schema,
                right_schema=right_schema,
                column_mappings=[],  # Empty mappings
                pk_columns=["id"],
            )

    def test_config_with_invalid_primary_key_mapping(self) -> None:
        """Test configuration with primary key not in mappings."""
        left_schema = ComparisonSchema(
            columns={
                "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
                "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=False),
            },
            pk_columns=["id"],
        )
        right_schema = ComparisonSchema(
            columns={
                "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
                "full_name": ColumnDefinition(name="full_name", alias="Full Name", datatype=pl.Utf8, nullable=False),
            },
            pk_columns=["customer_id"],
        )

        with pytest.raises((ValueError, SchemaValidationError)):
            ComparisonConfig(
                left_schema=left_schema,
                right_schema=right_schema,
                column_mappings=[
                    ColumnMapping(left="name", right="full_name", name="name"),
                    # Missing primary key mapping
                ],
                pk_columns=["id"],
            )

    def test_config_with_duplicate_comparison_names(self) -> None:
        """Test configuration with duplicate comparison names in mappings."""
        left_schema = ComparisonSchema(
            columns={
                "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
                "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=False),
                "email": ColumnDefinition(name="email", alias="Email", datatype=pl.Utf8, nullable=False),
            },
            pk_columns=["id"],
        )
        right_schema = ComparisonSchema(
            columns={
                "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
                "full_name": ColumnDefinition(name="full_name", alias="Full Name", datatype=pl.Utf8, nullable=False),
                "email_addr": ColumnDefinition(name="email_addr", alias="Email Address", datatype=pl.Utf8, nullable=False),
            },
            pk_columns=["customer_id"],
        )

        # Duplicate comparison names should now be rejected as they cause ambiguity
        with pytest.raises(SchemaValidationError) as exc_info:
            config = ComparisonConfig(
                left_schema=left_schema,
                right_schema=right_schema,
                column_mappings=[
                    ColumnMapping(left="id", right="customer_id", name="id"),
                    ColumnMapping(left="name", right="full_name", name="name"),
                    ColumnMapping(left="email", right="email_addr", name="name"),  # Same comparison name - should fail
                ],
                pk_columns=["id"],
            )

        # Verify the error message mentions duplicate names
        error_msg = str(exc_info.value)
        assert "duplicate column mapping names" in error_msg.lower()
        assert "name" in error_msg


class TestPerformanceAndScalability:
    """Test performance and scalability scenarios."""

    def test_large_dataframe_comparison(self) -> None:
        """Test comparison with larger DataFrames."""
        import random

        # Create larger test data
        num_records = 1000

        left_data = {
            "id": list(range(1, num_records + 1)),
            "value": [random.uniform(0, 1000) for _ in range(num_records)],
            "category": [f"Category_{random.randint(1, 10)}" for _ in range(num_records)],
            "active": [random.choice([True, False]) for _ in range(num_records)],
        }

        # Create right data with some differences (within tolerance)
        right_data = left_data.copy()
        right_data["value"] = [v + random.uniform(-2, 2) for v in right_data["value"]]  # Small variation within tolerance

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        left_schema = ComparisonSchema(
            columns={
                "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
                "value": ColumnDefinition(name="value", alias="Value", datatype=pl.Float64, nullable=False),
                "category": ColumnDefinition(name="category", alias="Category", datatype=pl.Utf8, nullable=False),
                "active": ColumnDefinition(name="active", alias="Active", datatype=pl.Boolean, nullable=False),
            },
            pk_columns=["id"],
        )

        right_schema = ComparisonSchema(
            columns={
                "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
                "value": ColumnDefinition(name="value", alias="Value", datatype=pl.Float64, nullable=False),
                "category": ColumnDefinition(name="category", alias="Category", datatype=pl.Utf8, nullable=False),
                "active": ColumnDefinition(name="active", alias="Active", datatype=pl.Boolean, nullable=False),
            },
            pk_columns=["id"],
        )

        config = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=[
                ColumnMapping(left="id", right="id", name="id"),
                ColumnMapping(left="value", right="value", name="value"),
                ColumnMapping(left="category", right="category", name="category"),
                ColumnMapping(left="active", right="active", name="active"),
            ],
            pk_columns=["id"],
            tolerance={"value": 5.0},  # Allow some tolerance for floating point differences
        )

        orchestrator = ComparisonOrchestrator()
        result = orchestrator.compare_dataframes(
            config=config,
            left=left_df,
            right=right_df
        )

        assert result.summary.total_left_records == num_records
        assert result.summary.total_right_records == num_records
        assert result.summary.matching_records >= num_records * 0.3  # At least 30% should match within tolerance

    def test_memory_efficient_large_comparison(self) -> None:
        """Test memory efficiency with large DataFrames."""
        import tempfile

        # Create moderately large test data
        num_records = 5000

        left_data = {
            "id": list(range(1, num_records + 1)),
            "data": [f"value_{i}" for i in range(1, num_records + 1)],
        }

        left_df = pl.LazyFrame(left_data)
        right_df = left_df.clone()

        left_schema = ComparisonSchema(
            columns={
                "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
                "data": ColumnDefinition(name="data", alias="Data", datatype=pl.Utf8, nullable=False),
            },
            pk_columns=["id"],
        )

        right_schema = left_schema

        config = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=[
                ColumnMapping(left="id", right="id", name="id"),
                ColumnMapping(left="data", right="data", name="data"),
            ],
            pk_columns=["id"],
        )

        orchestrator = ComparisonOrchestrator()

        # Test that comparison completes without memory issues
        result = orchestrator.compare_dataframes(
            config=config,
            left=left_df,
            right=right_df
        )

        assert result.summary.matching_records == num_records

        # Test export functionality with large data
        with tempfile.TemporaryDirectory() as temp_dir:
            exported_files = orchestrator.compare_and_export(
                config=config,
                left=left_df,
                right=right_df,
                output_dir=temp_dir,
                format="parquet"
            )

            assert len(exported_files) == 1  # Only summary file when data is identical
            for file_path in exported_files.values():
                assert Path(file_path).exists()
                assert file_path.endswith('.json')  # Summary is always JSON


class TestRealWorldUsagePatterns:
    """Test real-world usage patterns and scenarios."""

    def test_customer_data_comparison_scenario(self) -> None:
        """Test a complete customer data comparison scenario."""
        # Simulate customer data from two different systems
        left_data = {
            "customer_id": [1001, 1002, 1003, 1004, 1005],
            "first_name": ["John", "Jane", "Bob", "Alice", "Charlie"],
            "last_name": ["Doe", "Smith", "Johnson", "Williams", "Brown"],
            "email": ["john.doe@email.com", "jane.smith@email.com", "bob.johnson@email.com", "alice.w@email.com", "charlie.brown@email.com"],
            "phone": ["555-0101", "555-0102", "555-0103", "555-0104", "555-0105"],
            "signup_date": ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05", "2023-05-12"],
            "last_login": ["2024-01-10", "2024-01-08", "2024-01-12", "2024-01-05", "2024-01-11"],
            "account_balance": [1250.50, 890.25, 2100.75, 450.00, 3200.00],
            "is_active": ["active", "active", "inactive", "active", "active"],
        }

        # Right system has some differences and missing data
        right_data = {
            "cust_id": [1001, 1002, 1003, 1006, 1007],  # Different ID format, missing 1004, extra customers
            "fname": ["John", "Jane", "Bob", "David", "Eve"],  # Different column names
            "lname": ["Doe", "Smith", "Johnson", "Miller", "Davis"],
            "email_addr": ["john.doe@email.com", "jane.smith@email.com", "bob.johnson@email.com", "david.m@email.com", "eve.davis@email.com"],
            "contact_phone": ["555-0101", "555-0102", "555-0103", "555-0106", "555-0107"],  # Different column name
            "registration_date": ["2023-01-15", "2023-02-20", "2023-03-10", "2023-06-01", "2023-07-15"],  # Different column name
            "balance": [1250.50, 950.25, 2100.75, 1800.00, 750.00],  # Different column name, different balance for Jane
            "status": ["active", "active", "inactive", "active", "active"],  # Different representation
        }

        left_df = pl.LazyFrame(left_data).with_columns(
            pl.col("signup_date").str.strptime(pl.Date, "%Y-%m-%d"),
            pl.col("last_login").str.strptime(pl.Date, "%Y-%m-%d")
        )
        right_df = pl.LazyFrame(right_data).with_columns(
            pl.col("registration_date").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # Define schemas
        left_schema = ComparisonSchema(
            columns={
                "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
                "first_name": ColumnDefinition(name="first_name", alias="First Name", datatype=pl.Utf8, nullable=False),
                "last_name": ColumnDefinition(name="last_name", alias="Last Name", datatype=pl.Utf8, nullable=False),
                "email": ColumnDefinition(name="email", alias="Email", datatype=pl.Utf8, nullable=False),
                "phone": ColumnDefinition(name="phone", alias="Phone", datatype=pl.Utf8, nullable=False),
                "signup_date": ColumnDefinition(name="signup_date", alias="Signup Date", datatype=pl.Date, nullable=False),
                "last_login": ColumnDefinition(name="last_login", alias="Last Login", datatype=pl.Date, nullable=False),
                "account_balance": ColumnDefinition(name="account_balance", alias="Account Balance", datatype=pl.Float64, nullable=False),
                "is_active": ColumnDefinition(name="is_active", alias="Is Active", datatype=pl.Utf8, nullable=False),
            },
            pk_columns=["customer_id"],
        )

        right_schema = ComparisonSchema(
            columns={
                "cust_id": ColumnDefinition(name="cust_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
                "fname": ColumnDefinition(name="fname", alias="First Name", datatype=pl.Utf8, nullable=False),
                "lname": ColumnDefinition(name="lname", alias="Last Name", datatype=pl.Utf8, nullable=False),
                "email_addr": ColumnDefinition(name="email_addr", alias="Email", datatype=pl.Utf8, nullable=False),
                "contact_phone": ColumnDefinition(name="contact_phone", alias="Phone", datatype=pl.Utf8, nullable=False),
                "registration_date": ColumnDefinition(name="registration_date", alias="Registration Date", datatype=pl.Date, nullable=False),
                "balance": ColumnDefinition(name="balance", alias="Balance", datatype=pl.Float64, nullable=False),
                "status": ColumnDefinition(name="status", alias="Status", datatype=pl.Utf8, nullable=False),
            },
            pk_columns=["cust_id"],
        )

        # Define mappings
        config = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=[
                ColumnMapping(left="customer_id", right="cust_id", name="customer_id"),
                ColumnMapping(left="first_name", right="fname", name="first_name"),
                ColumnMapping(left="last_name", right="lname", name="last_name"),
                ColumnMapping(left="email", right="email_addr", name="email"),
                ColumnMapping(left="phone", right="contact_phone", name="phone"),
                ColumnMapping(left="signup_date", right="registration_date", name="signup_date"),
                ColumnMapping(left="account_balance", right="balance", name="account_balance"),
                ColumnMapping(left="is_active", right="status", name="status"),
            ],
            pk_columns=["customer_id"],
            ignore_case=True,
            tolerance={"account_balance": 10.0},  # Allow $10 tolerance for balances
        )

        # Execute comprehensive comparison
        orchestrator = ComparisonOrchestrator()

        # 1. Compare dataframes
        result = orchestrator.compare_dataframes(
            config=config,
            left=left_df,
            right=right_df
        )

        # Validate results
        assert result.summary.total_left_records == 5
        assert result.summary.total_right_records == 5
        assert result.summary.matching_records == 2  # John and Bob should match (Jane has balance difference)
        assert result.summary.value_differences_count >= 1  # Jane's balance differs
        assert result.summary.left_only_count >= 1  # Alice (1004) missing from right
        assert result.summary.right_only_count >= 2  # David (1006) and Eve (1007) are new

        # 2. Generate comprehensive report
        report = orchestrator.compare_and_report(
            config=config,
            left=left_df,
            right=right_df,
            include_samples=True,
            max_samples=3
        )

        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in report
        assert "VALUE DIFFERENCES" in report
        assert "LEFT-ONLY RECORDS" in report
        assert "RIGHT-ONLY RECORDS" in report

        # 3. Export results
        with tempfile.TemporaryDirectory() as temp_dir:
            exported_files = orchestrator.compare_and_export(
                config=config,
                left=left_df,
                right=right_df,
                output_dir=temp_dir,
                format="csv"
            )

            assert len(exported_files) == 4
            for file_path in exported_files.values():
                assert Path(file_path).exists()

        # 4. Test report generation from result
        detailed_report = orchestrator.generate_report_from_result(
            result=result,
            report_type="detailed"
        )
        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in detailed_report

        # 5. Test summary generation
        summary = orchestrator.get_comparison_summary(result=result)
        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in summary

        # This comprehensive test validates the entire workflow from data preparation
        # to result export, ensuring all public APIs work correctly together.

    def test_incremental_data_loading_scenario(self) -> None:
        """Test scenario where data is loaded and compared incrementally."""
        # Create base dataset
        base_left = {
            "id": [1, 2, 3],
            "value": ["A", "B", "C"],
        }
        base_right = {
            "id": [1, 2, 3],
            "value": ["A", "B", "C"],
        }

        # Define schema once
        schema = ComparisonSchema(
            columns={
                "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
                "value": ColumnDefinition(name="value", alias="Value", datatype=pl.Utf8, nullable=False),
            },
            pk_columns=["id"],
        )

        config = ComparisonConfig(
            left_schema=schema,
            right_schema=schema,
            column_mappings=[
                ColumnMapping(left="id", right="id", name="id"),
                ColumnMapping(left="value", right="value", name="value"),
            ],
            pk_columns=["id"],
        )

        orchestrator = ComparisonOrchestrator()

        # First comparison - identical data
        left_df1 = pl.LazyFrame(base_left)
        right_df1 = pl.LazyFrame(base_right)

        result1 = orchestrator.compare_dataframes(
            config=config,
            left=left_df1,
            right=right_df1
        )

        assert result1.summary.matching_records == 3

        # Second comparison - add new records
        updated_left = {
            "id": [1, 2, 3, 4, 5],
            "value": ["A", "B", "C", "D", "E"],
        }
        updated_right = {
            "id": [1, 2, 3, 4],
            "value": ["A", "B", "X", "D"],  # Changed value for ID 3
        }

        left_df2 = pl.LazyFrame(updated_left)
        right_df2 = pl.LazyFrame(updated_right)

        result2 = orchestrator.compare_dataframes(
            config=config,
            left=left_df2,
            right=right_df2
        )

        assert result2.summary.total_left_records == 5
        assert result2.summary.total_right_records == 4
        assert result2.summary.matching_records == 3  # IDs 1, 2, 4 match
        assert result2.summary.value_differences_count == 1  # ID 3 has different value
        assert result2.summary.left_only_count == 1  # ID 5 only in left
        assert result2.summary.right_only_count == 0  # No right-only records

    def test_error_recovery_and_reporting(self) -> None:
        """Test error recovery and comprehensive error reporting."""
        # Create invalid data that will cause schema validation errors
        left_df = pl.LazyFrame({
            "wrong_column": [1, 2, 3]
        })
        right_df = pl.LazyFrame({
            "customer_id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })

        # Create config that expects different columns
        left_schema = ComparisonSchema(
            columns={
                "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),  # This column doesn't exist
                "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=False),
            },
            pk_columns=["id"],
        )
        right_schema = ComparisonSchema(
            columns={
                "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
                "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=False),
            },
            pk_columns=["customer_id"],
        )

        config = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=[
                ColumnMapping(left="id", right="customer_id", name="id"),
                ColumnMapping(left="name", right="name", name="name"),
            ],
            pk_columns=["id"],
        )

        orchestrator = ComparisonOrchestrator()

        # Test that schema validation errors are properly caught and reported
        with pytest.raises(SchemaValidationError) as exc_info:
            orchestrator.compare_dataframes(
                config=config,
                left=left_df,
                right=right_df
            )

        # Verify error message contains useful information
        error_msg = str(exc_info.value)
        assert "validation failed" in error_msg.lower()

        # Test that other operations handle errors gracefully
        try:
            orchestrator.compare_and_report(
                config=config,
                left=left_df,
                right=right_df
            )
            assert False, "Should have raised an exception"
        except SchemaValidationError:
            pass  # Expected behavior

    def test_configuration_flexibility(self) -> None:
        """Test various configuration options and their effects."""
        left_data = {
            "id": [1, 2, 3],
            "value": ["Hello", "WORLD", "Test"],
            "number": [1.1, 2.2, 3.3],
        }
        right_data = {
            "id": [1, 2, 3],
            "value": ["hello", "world", "test"],  # Different case
            "number": [1.1001, 2.1999, 3.3001],  # Slightly different numbers
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        left_schema = ComparisonSchema(
            columns={
                "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
                "value": ColumnDefinition(name="value", alias="Value", datatype=pl.Utf8, nullable=False),
                "number": ColumnDefinition(name="number", alias="Number", datatype=pl.Float64, nullable=False),
            },
            pk_columns=["id"],
        )

        right_schema = left_schema

        # Test with ignore_case=True
        config_case_sensitive = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=[
                ColumnMapping(left="id", right="id", name="id"),
                ColumnMapping(left="value", right="value", name="value"),
                ColumnMapping(left="number", right="number", name="number"),
            ],
            pk_columns=["id"],
            ignore_case=False,
        )

        # Test with ignore_case=False
        config_case_insensitive = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=[
                ColumnMapping(left="id", right="id", name="id"),
                ColumnMapping(left="value", right="value", name="value"),
                ColumnMapping(left="number", right="number", name="number"),
            ],
            pk_columns=["id"],
            ignore_case=True,
            tolerance={"number": 0.01},  # Small tolerance for floating point
        )

        orchestrator = ComparisonOrchestrator()

        # Case sensitive comparison - should find differences
        result_sensitive = orchestrator.compare_dataframes(
            config=config_case_sensitive,
            left=left_df,
            right=right_df
        )

        # Should find differences in string case and number precision
        assert result_sensitive.summary.value_differences_count >= 2

        # Case insensitive comparison with tolerance - should match more
        result_insensitive = orchestrator.compare_dataframes(
            config=config_case_insensitive,
            left=left_df,
            right=right_df
        )

        # Should match all records due to ignore_case and tolerance
        assert result_insensitive.summary.matching_records == 3
        assert result_insensitive.summary.value_differences_count == 0


class TestToleranceNullLogicFix:
    """Test the fix for tolerance + null comparison logic bug.

    This test class verifies that tolerance and null handling work correctly together,
    addressing the critical bug where they were mutually exclusive.
    """

    def test_tolerance_with_null_equals_null_true(self):
        """Test tolerance combined with null_equals_null=True."""
        left_data = {
            "id": [1, 2, 3, 4],
            "amount": [100.0, 200.0, None, 400.0],
        }

        right_data = {
            "customer_id": [1, 2, 3, 4],
            "total": [100.5, 199.8, None, 410.0],  # Within tolerance: 0.5, 0.2, null, over tolerance: 10.0
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        # Define schemas
        left_columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "amount": ColumnDefinition(name="amount", alias="Amount", datatype=pl.Float64, nullable=True),
        }
        left_schema = ComparisonSchema(columns=left_columns, pk_columns=["id"])

        right_columns = {
            "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "total": ColumnDefinition(name="total", alias="Total", datatype=pl.Float64, nullable=True),
        }
        right_schema = ComparisonSchema(columns=right_columns, pk_columns=["customer_id"])

        mappings = [
            ColumnMapping(left="id", right="customer_id", name="id"),
            ColumnMapping(left="amount", right="total", name="amount"),
        ]

        config = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=mappings,
            pk_columns=["id"],
            null_equals_null=True,  # Nulls should be considered equal
            tolerance={"amount": 1.0}  # 1.0 tolerance for amount
        )

        comparator = LazyFrameComparator(config)
        result = comparator.compare(left=left_df, right=right_df)

        # With null_equals_null=True and tolerance=1.0:
        # - Record 1: |100.0 - 100.5| = 0.5 <= 1.0 → No difference
        # - Record 2: |200.0 - 199.8| = 0.2 <= 1.0 → No difference
        # - Record 3: Both null → No difference (null_equals_null=True)
        # - Record 4: |400.0 - 410.0| = 10.0 > 1.0 → Difference
        assert result.summary.matching_records == 3
        assert result.summary.value_differences_count == 1

    def test_tolerance_with_null_equals_null_false(self):
        """Test tolerance combined with null_equals_null=False."""
        left_data = {
            "id": [1, 2, 3],
            "amount": [100.0, None, 300.0],
        }

        right_data = {
            "customer_id": [1, 2, 3],
            "total": [100.5, 200.0, None],  # Within tolerance: 0.5, null vs non-null, both null
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        # Define schemas
        left_columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "amount": ColumnDefinition(name="amount", alias="Amount", datatype=pl.Float64, nullable=True),
        }
        left_schema = ComparisonSchema(columns=left_columns, pk_columns=["id"])

        right_columns = {
            "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "total": ColumnDefinition(name="total", alias="Total", datatype=pl.Float64, nullable=True),
        }
        right_schema = ComparisonSchema(columns=right_columns, pk_columns=["customer_id"])

        mappings = [
            ColumnMapping(left="id", right="customer_id", name="id"),
            ColumnMapping(left="amount", right="total", name="amount"),
        ]

        config = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=mappings,
            pk_columns=["id"],
            null_equals_null=False,  # Nulls should be considered different
            tolerance={"amount": 1.0}  # 1.0 tolerance for amount
        )

        comparator = LazyFrameComparator(config)
        result = comparator.compare(left=left_df, right=right_df)

        # With null_equals_null=False and tolerance=1.0:
        # - Record 1: |100.0 - 100.5| = 0.5 <= 1.0 → No difference
        # - Record 2: null vs 200.0 → Difference (null_equals_null=False)
        # - Record 3: 300.0 vs null → Difference (null_equals_null=False)
        # DEBUG: Let's see what the actual differences are
        print("Value differences count:", result.summary.value_differences_count)
        print("Matching records:", result.summary.matching_records)
        if result.summary.value_differences_count > 0:
            diff_df = result.value_differences.collect()
            print("Value differences DataFrame:")
            print(diff_df)

        assert result.summary.matching_records == 1
        assert result.summary.value_differences_count == 2




class TestToleranceNullLogicFix:
    """Test the fix for tolerance + null comparison logic bug."""

    def test_null_only_no_tolerance(self):
        """Test null handling without tolerance configuration."""
        left_data = {
            "id": [1, 2, 3],
            "amount": [100.0, None, 300.0],
        }

        right_data = {
            "customer_id": [1, 2, 3],
            "total": [100.0, None, 350.0],  # Exact match, both null, different value
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        # Define schemas
        left_columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "amount": ColumnDefinition(name="amount", alias="Amount", datatype=pl.Float64, nullable=True),
        }
        left_schema = ComparisonSchema(columns=left_columns, pk_columns=["id"])

        right_columns = {
            "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "total": ColumnDefinition(name="total", alias="Total", datatype=pl.Float64, nullable=True),
        }
        right_schema = ComparisonSchema(columns=right_columns, pk_columns=["customer_id"])

        mappings = [
            ColumnMapping(left="id", right="customer_id", name="id"),
            ColumnMapping(left="amount", right="total", name="amount"),
        ]

        config = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=mappings,
            pk_columns=["id"],
            null_equals_null=True  # Nulls should be considered equal
        )

        comparator = LazyFrameComparator(config)
        result = comparator.compare(left=left_df, right=right_df)

        # With null_equals_null=True:
        # - Record 1: 100.0 == 100.0 → No difference
        # - Record 2: null == null → No difference
        # - Record 3: 300.0 != 350.0 → Difference
        assert result.summary.matching_records == 2
        assert result.summary.value_differences_count == 1

    def test_tolerance_with_null_equals_null_true(self):
        """Test tolerance combined with null_equals_null=True."""
        left_data = {
            "id": [1, 2, 3, 4],
            "amount": [100.0, 200.0, None, 400.0],
        }

        right_data = {
            "customer_id": [1, 2, 3, 4],
            "total": [100.5, 199.8, None, 410.0],  # Within tolerance: 0.5, 0.2, null, over tolerance: 10.0
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        # Define schemas
        left_columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "amount": ColumnDefinition(name="amount", alias="Amount", datatype=pl.Float64, nullable=True),
        }
        left_schema = ComparisonSchema(columns=left_columns, pk_columns=["id"])

        right_columns = {
            "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "total": ColumnDefinition(name="total", alias="Total", datatype=pl.Float64, nullable=True),
        }
        right_schema = ComparisonSchema(columns=right_columns, pk_columns=["customer_id"])

        mappings = [
            ColumnMapping(left="id", right="customer_id", name="id"),
            ColumnMapping(left="amount", right="total", name="amount"),
        ]

        config = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=mappings,
            pk_columns=["id"],
            null_equals_null=True,  # Nulls should be considered equal
            tolerance={"amount": 1.0}  # 1.0 tolerance for amount
        )

        comparator = LazyFrameComparator(config)
        result = comparator.compare(left=left_df, right=right_df)

        # With null_equals_null=True and tolerance=1.0:
        # - Record 1: |100.0 - 100.5| = 0.5 <= 1.0 → No difference
        # - Record 2: |200.0 - 199.8| = 0.2 <= 1.0 → No difference
        # - Record 3: Both null → No difference (null_equals_null=True)
        # - Record 4: |400.0 - 410.0| = 10.0 > 1.0 → Difference
        assert result.summary.matching_records == 3
        assert result.summary.value_differences_count == 1

    def test_tolerance_with_null_equals_null_false(self):
        """Test tolerance combined with null_equals_null=False."""
        left_data = {
            "id": [1, 2, 3],
            "amount": [100.0, None, 300.0],
        }

        right_data = {
            "customer_id": [1, 2, 3],
            "total": [100.5, 200.0, None],  # Within tolerance: 0.5, null vs non-null, both null
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        # Define schemas
        left_columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "amount": ColumnDefinition(name="amount", alias="Amount", datatype=pl.Float64, nullable=True),
        }
        left_schema = ComparisonSchema(columns=left_columns, pk_columns=["id"])

        right_columns = {
            "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "total": ColumnDefinition(name="total", alias="Total", datatype=pl.Float64, nullable=True),
        }
        right_schema = ComparisonSchema(columns=right_columns, pk_columns=["customer_id"])

        mappings = [
            ColumnMapping(left="id", right="customer_id", name="id"),
            ColumnMapping(left="amount", right="total", name="amount"),
        ]

        config = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=mappings,
            pk_columns=["id"],
            null_equals_null=False,  # Nulls should be considered different
            tolerance={"amount": 1.0}  # 1.0 tolerance for amount
        )

        comparator = LazyFrameComparator(config)
        result = comparator.compare(left=left_df, right=right_df)

        # With null_equals_null=False and tolerance=1.0:
        # - Record 1: |100.0 - 100.5| = 0.5 <= 1.0 → No difference
        # - Record 2: null vs 200.0 → Difference (null_equals_null=False)
        # - Record 3: 300.0 vs null → Difference (null_equals_null=False)
        assert result.summary.matching_records == 1
        assert result.summary.value_differences_count == 2
