"""Integration tests for complete comparison workflows."""

import pytest
import polars as pl
from pathlib import Path
import tempfile
import os

from splurge_lazyframe_compare import (
    LazyFrameComparator,
    ComparisonResult,
    ComparisonConfig,
    ComparisonSchema,
    ColumnDefinition,
    ColumnMapping,
    ComparisonOrchestrator,
)


@pytest.fixture
def complex_dataframes():
    """Create complex DataFrames with various data types and relationships."""
    # Create left DataFrame with various data types
    left_data = {
        "customer_id": [1, 2, 3, 4, 5, 6],
        "first_name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank"],
        "last_name": ["Smith", "Johnson", "Brown", "Wilson", "Davis", "Miller"],
        "email": ["alice@example.com", "bob@example.com", "charlie@example.com",
                 "david@example.com", "eve@example.com", "frank@example.com"],
        "balance": [1000.50, 2500.75, 750.25, 3200.00, 1800.90, 950.60],
        "active": [True, True, False, True, True, False],
        "signup_date": ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05", "2023-05-12", "2023-06-08"],
    }

    # Create right DataFrame with differences
    right_data = {
        "cust_id": [1, 2, 3, 4, 7],  # Missing customer 5, 6, added customer 7
        "fname": ["Alice", "Bob", "Chuck", "David", "Grace"],  # Charlie -> Chuck, added Grace
        "lname": ["Smith", "Johnson", "Brown", "Wilson", "Parker"],  # Eve -> Grace Parker
        "email_addr": ["alice@example.com", "bob@example.com", "chuck@example.com",
                      "david@example.com", "grace@example.com"],  # Charlie email updated
        "account_balance": [1000.50, 2750.75, 750.25, 3200.00, 2100.00],  # Bob balance changed, added Grace
        "is_active": [True, True, False, True, True],  # No Frank (customer 6)
        "registration_date": ["2023-01-15", "2023-02-20", "2023-03-10", "2023-04-05", "2023-07-01"],
    }

    left_df = pl.LazyFrame(left_data).with_columns(
        pl.col("signup_date").str.strptime(pl.Date, "%Y-%m-%d")
    )
    right_df = pl.LazyFrame(right_data).with_columns(
        pl.col("registration_date").str.strptime(pl.Date, "%Y-%m-%d")
    )

    return left_df, right_df


@pytest.fixture
def complex_config(complex_dataframes):
    """Create complex configuration for testing."""
    left_df, right_df = complex_dataframes

    # Define left schema
    left_columns = {
        "customer_id": ColumnDefinition("customer_id", "Customer ID", pl.Int64, False),
        "first_name": ColumnDefinition("first_name", "First Name", pl.Utf8, False),
        "last_name": ColumnDefinition("last_name", "Last Name", pl.Utf8, False),
        "email": ColumnDefinition("email", "Email", pl.Utf8, False),
        "balance": ColumnDefinition("balance", "Balance", pl.Float64, False),
        "active": ColumnDefinition("active", "Active Status", pl.Boolean, False),
        "signup_date": ColumnDefinition("signup_date", "Signup Date", pl.Date, False),
    }
    left_schema = ComparisonSchema(
        columns=left_columns,
        primary_key_columns=["customer_id"]
    )

    # Define right schema
    right_columns = {
        "cust_id": ColumnDefinition("cust_id", "Customer ID", pl.Int64, False),
        "fname": ColumnDefinition("fname", "First Name", pl.Utf8, False),
        "lname": ColumnDefinition("lname", "Last Name", pl.Utf8, False),
        "email_addr": ColumnDefinition("email_addr", "Email", pl.Utf8, False),
        "account_balance": ColumnDefinition("account_balance", "Balance", pl.Float64, False),
        "is_active": ColumnDefinition("is_active", "Active Status", pl.Boolean, False),
        "registration_date": ColumnDefinition("registration_date", "Registration Date", pl.Date, False),
    }
    right_schema = ComparisonSchema(
        columns=right_columns,
        primary_key_columns=["cust_id"]
    )

    # Define column mappings
    mappings = [
        ColumnMapping("customer_id", "cust_id", "customer_id"),
        ColumnMapping("first_name", "fname", "first_name"),
        ColumnMapping("last_name", "lname", "last_name"),
        ColumnMapping("email", "email_addr", "email"),
        ColumnMapping("balance", "account_balance", "balance"),
        ColumnMapping("active", "is_active", "active"),
        ColumnMapping("signup_date", "registration_date", "signup_date"),
    ]

    config = ComparisonConfig(
        left_schema=left_schema,
        right_schema=right_schema,
        column_mappings=mappings,
        primary_key_columns=["customer_id"],
        ignore_case=False,
        null_equals_null=True,
    )

    return config


class TestCompleteWorkflow:
    """Test complete comparison workflows."""

    def test_full_lazyframe_comparator_workflow(self, complex_dataframes, complex_config):
        """Test complete workflow using LazyFrameComparator."""
        left_df, right_df = complex_dataframes

        comparator = LazyFrameComparator(complex_config)

        # Execute comparison
        result = comparator.compare(left=left_df, right=right_df)

        # Validate results
        assert result.summary.total_left_records == 6
        assert result.summary.total_right_records == 5
        assert result.summary.matching_records >= 1  # At least Alice should match
        assert result.summary.value_differences_count >= 1  # Bob's balance changed
        assert result.summary.left_only_count >= 1  # Customers 5 and 6 missing from right
        assert result.summary.right_only_count >= 1  # Customer 7 added to right

        # Test report generation
        report = comparator.compare_and_report(left=left_df, right=right_df)
        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in report
        assert "RECORD COUNTS:" in report
        assert "6" in report  # Left record count
        assert "5" in report  # Right record count

    def test_orchestrator_full_workflow(self, complex_dataframes, complex_config):
        """Test complete workflow using ComparisonOrchestrator."""
        left_df, right_df = complex_dataframes

        orchestrator = ComparisonOrchestrator()

        # Execute comparison
        result = orchestrator.compare_dataframes(
            config=complex_config,
            left=left_df,
            right=right_df
        )

        # Validate results
        assert result.summary.total_left_records == 6
        assert result.summary.total_right_records == 5

        # Generate detailed report
        report = orchestrator.generate_report_from_result(
            result=result,
            report_type="detailed",
            max_samples=3
        )

        assert "VALUE DIFFERENCES" in report or "No value differences" in report
        assert "LEFT-ONLY RECORDS" in report or "No left-only records" in report

    def test_file_export_workflow(self, complex_dataframes, complex_config):
        """Test complete workflow with file export."""
        left_df, right_df = complex_dataframes

        with tempfile.TemporaryDirectory() as temp_dir:
            orchestrator = ComparisonOrchestrator()

            # Compare and export
            exported_files = orchestrator.compare_and_export(
                config=complex_config,
                left=left_df,
                right=right_df,
                output_dir=temp_dir,
                format="csv"
            )

            # Check that files were created
            assert "value_differences" in exported_files
            assert "left_only_records" in exported_files
            assert "right_only_records" in exported_files
            assert "summary" in exported_files

            # Verify files exist
            for file_path in exported_files.values():
                assert os.path.exists(file_path)
                assert os.path.getsize(file_path) > 0

    def test_configuration_workflow(self, complex_dataframes):
        """Test configuration creation and validation workflow."""
        left_df, right_df = complex_dataframes

        from splurge_lazyframe_compare.utils import (
            create_config_from_dataframes,
            validate_config,
            create_default_config
        )

        # Create config from DataFrames
        config = create_config_from_dataframes(
            left_df=left_df,
            right_df=right_df,
            primary_keys=["customer_id"],
            auto_map_columns=True
        )

        # Validate configuration
        errors = validate_config(config)
        assert len(errors) == 0, f"Configuration validation failed: {errors}"

        # Test with default config
        default_config = create_default_config()
        assert "primary_key_columns" in default_config
        assert "column_mappings" in default_config

    def test_error_handling_workflow(self, complex_config):
        """Test error handling in complete workflow."""
        # Create invalid DataFrame (missing required columns)
        invalid_df = pl.LazyFrame({"wrong_column": [1, 2, 3]})

        orchestrator = ComparisonOrchestrator()

        # Should handle errors gracefully
        with pytest.raises(Exception):  # Should raise validation error
            orchestrator.compare_dataframes(
                config=complex_config,
                left=invalid_df,
                right=invalid_df
            )

    def test_service_customization_workflow(self, complex_dataframes, complex_config):
        """Test workflow with customized services."""
        left_df, right_df = complex_dataframes

        from splurge_lazyframe_compare.services import (
            ComparisonService,
            ValidationService,
            DataPreparationService,
            ReportingService,
        )

        # Create customized services
        validation_service = ValidationService()
        preparation_service = DataPreparationService()

        # Create comparison service with custom validation
        comparison_service = ComparisonService(
            validation_service=validation_service,
            preparation_service=preparation_service
        )

        # Create orchestrator with custom services
        orchestrator = ComparisonOrchestrator(comparison_service=comparison_service)

        # Execute workflow
        result = orchestrator.compare_dataframes(
            config=complex_config,
            left=left_df,
            right=right_df
        )

        # Verify it works with custom services
        assert result.summary.total_left_records == 6
        assert result.summary.total_right_records == 5

    def test_performance_monitoring_workflow(self, complex_dataframes, complex_config):
        """Test that performance monitoring works in complete workflow."""
        left_df, right_df = complex_dataframes

        orchestrator = ComparisonOrchestrator()

        # The comparison should include performance monitoring
        # We can't easily test the log output, but we can ensure it doesn't crash
        result = orchestrator.compare_dataframes(
            config=complex_config,
            left=left_df,
            right=right_df
        )

        assert result is not None
        assert result.summary is not None


class TestDataQualityWorkflow:
    """Test data quality validation workflows."""

    def test_data_quality_validation_workflow(self, complex_dataframes, complex_config):
        """Test data quality validation in complete workflow."""
        left_df, right_df = complex_dataframes

        from splurge_lazyframe_compare.services import ValidationService

        service = ValidationService()

        # Test completeness validation
        result = service.validate_completeness(
            df=left_df,
            required_columns=["customer_id", "first_name", "balance"]
        )
        assert result.is_valid

        # Test data type validation
        expected_types = {
            "customer_id": pl.Int64,
            "first_name": pl.Utf8,
            "balance": pl.Float64
        }
        result = service.validate_data_types(
            df=left_df,
            expected_types=expected_types
        )
        assert result.is_valid

    def test_comprehensive_validation_workflow(self, complex_dataframes):
        """Test comprehensive validation workflow."""
        df, _ = complex_dataframes

        from splurge_lazyframe_compare.services import ValidationService

        service = ValidationService()

        # Run comprehensive validation
        results = service.run_comprehensive_validation(
            df=df,
            required_columns=["customer_id", "first_name"],
            expected_types={"customer_id": pl.Int64, "first_name": pl.Utf8},
        )

        # Should have results for each validation type
        assert len(results) >= 2  # At least completeness and data types

        # All results should be validation results
        for result in results:
            assert hasattr(result, 'is_valid')
            assert hasattr(result, 'message')


class TestUtilityIntegrationWorkflow:
    """Test utility integration in complete workflows."""

    def test_dataframe_analysis_workflow(self, complex_dataframes):
        """Test DataFrame analysis utilities."""
        df, _ = complex_dataframes

        from splurge_lazyframe_compare.utils import (
            get_dataframe_info,
            compare_dataframe_shapes,
            estimate_dataframe_memory,
        )

        # Get DataFrame info
        info = get_dataframe_info(df)
        assert info["row_count"] == 6
        assert info["column_count"] == 7
        assert "customer_id" in info["column_names"]

        # Compare DataFrames
        df1, df2 = complex_dataframes
        shapes = compare_dataframe_shapes(df1, df2)
        assert shapes["shape_comparison"]["same_row_count"] is False
        assert "common_columns" in shapes["column_overlap"]

        # Estimate memory
        memory_mb = estimate_dataframe_memory(df)
        assert memory_mb >= 0

    def test_formatting_integration_workflow(self, complex_dataframes, complex_config):
        """Test formatting utilities in complete workflow."""
        left_df, right_df = complex_dataframes

        from splurge_lazyframe_compare.services import ComparisonOrchestrator

        orchestrator = ComparisonOrchestrator()

        # Generate report (uses formatting utilities internally)
        report = orchestrator.compare_and_report(
            config=complex_config,
            left=left_df,
            right=right_df,
            include_samples=True,
            max_samples=2
        )

        # Report should contain formatted numbers
        assert "6" in report  # Formatted record count
        assert "5" in report  # Formatted record count

        # Generate table (uses formatting utilities)
        from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary

        summary = ComparisonSummary(
            total_left_records=6,
            total_right_records=5,
            matching_records=3,
            value_differences_count=2,
            left_only_count=1,
            right_only_count=1,
            comparison_timestamp="2025-01-01T00:00:00"
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=left_df.limit(0),
            left_only_records=left_df.limit(0),
            right_only_records=left_df.limit(0),
            config=complex_config
        )

        table = orchestrator.generate_report_from_result(result=result, report_type="table")
        assert "6" in table or "5" in table  # Should contain formatted numbers
