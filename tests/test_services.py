"""Unit tests for service classes."""

import polars as pl
import pytest

from splurge_lazyframe_compare.exceptions.comparison_exceptions import SchemaValidationError
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
        "id": ColumnDefinition("id", "ID", pl.Int64, False),
        "name": ColumnDefinition("name", "Name", pl.Utf8, False),
        "amount": ColumnDefinition("amount", "Amount", pl.Float64, False),
    }
    left_schema = ComparisonSchema(columns=left_columns, primary_key_columns=["id"])

    right_columns = {
        "customer_id": ColumnDefinition("customer_id", "Customer ID", pl.Int64, False),
        "customer_name": ColumnDefinition("customer_name", "Customer Name", pl.Utf8, False),
        "total_amount": ColumnDefinition("total_amount", "Amount", pl.Float64, False),
    }
    right_schema = ComparisonSchema(columns=right_columns, primary_key_columns=["customer_id"])

    mappings = [
        ColumnMapping("id", "customer_id", "id"),
        ColumnMapping("name", "customer_name", "name"),
        ColumnMapping("amount", "total_amount", "amount"),
    ]

    config = ComparisonConfig(
        left_schema=left_schema,
        right_schema=right_schema,
        column_mappings=mappings,
        primary_key_columns=["id"]
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
            "id": ColumnDefinition("id", "ID", pl.Int64, False),
            "name": ColumnDefinition("name", "Name", pl.Utf8, False),
        }
        schema = ComparisonSchema(columns=columns, primary_key_columns=["id"])

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
        mapped_left, mapped_right = service.apply_column_mappings(
            left=left_df,
            right=right_df,
            config=sample_config
        )

        # Check that columns were renamed according to mappings
        left_columns = mapped_left.collect_schema().names()
        right_columns = mapped_right.collect_schema().names()

        assert "PK_id" in left_columns
        assert "L_name" in left_columns
        assert "PK_id" in right_columns
        assert "R_name" in right_columns

    def test_apply_case_insensitive(self):
        """Test case insensitive transformation."""
        df = pl.LazyFrame({
            "NAME": ["Alice", "BOB", "Charlie"],
            "VALUE": ["ABC", "def", "GHI"]
        })

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
            comparison_timestamp="2025-01-01T00:00:00"
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=sample_dataframes[0].limit(0),  # Empty
            left_only_records=sample_dataframes[0].limit(0),  # Empty
            right_only_records=sample_dataframes[0].limit(0),  # Empty
            config=sample_config
        )

        service = ReportingService()
        report = service.generate_summary_report(results=result)

        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in report
        assert "4" in report  # Should contain record counts
        assert "50.0%" in report  # Should contain percentages

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
            comparison_timestamp="2025-01-01T00:00:00"
        )

        result = ComparisonResult(
            summary=summary,
            value_differences=sample_dataframes[0].limit(0),
            left_only_records=sample_dataframes[0].limit(0),
            right_only_records=sample_dataframes[0].limit(0),
            config=sample_config
        )

        service = ReportingService()
        table = service.generate_summary_table(results=result)

        assert "Left DataFrame" in table
        assert "Right DataFrame" in table
        assert "4" in table


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
        result = service.execute_comparison(
            left=left_df,
            right=right_df,
            config=sample_config
        )

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
        result = orchestrator.compare_dataframes(
            config=sample_config,
            left=left_df,
            right=right_df
        )

        assert result.summary.total_left_records == 4
        assert result.summary.total_right_records == 4
        assert result.summary.matching_records >= 0

    def test_compare_and_report_integration(self, sample_dataframes, sample_config):
        """Test orchestrator compare and report."""
        left_df, right_df = sample_dataframes

        orchestrator = ComparisonOrchestrator()
        report = orchestrator.compare_and_report(
            config=sample_config,
            left=left_df,
            right=right_df,
            include_samples=False
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
            validation_service=validation_service,
            preparation_service=preparation_service
        )

        # Create orchestrator with custom services
        orchestrator = ComparisonOrchestrator(
            comparison_service=comparison_service,
            reporting_service=reporting_service
        )

        # Test that it works
        result = orchestrator.compare_dataframes(
            config=sample_config,
            left=left_df,
            right=right_df
        )

        assert result.summary.matching_records >= 0
