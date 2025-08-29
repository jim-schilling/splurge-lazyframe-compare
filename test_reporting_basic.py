"""Basic test to verify ReportingService public API behavior."""

import sys
import os
sys.path.insert(0, '.')

from splurge_lazyframe_compare.services.reporting_service import ReportingService
from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary
from splurge_lazyframe_compare.models.schema import ComparisonConfig, ComparisonSchema, ColumnDefinition, ColumnMapping
import polars as pl

def test_basic_functionality():
    """Test basic ReportingService functionality."""
    print("Testing ReportingService basic functionality...")

    # Initialize service
    service = ReportingService()
    assert service.service_name == "ReportingService"
    print("✓ Service initialization works")

    # Create minimal test data
    summary = ComparisonSummary(
        total_left_records=4,
        total_right_records=4,
        matching_records=2,
        value_differences_count=2,
        left_only_count=0,
        right_only_count=0,
        comparison_timestamp='2025-01-01T00:00:00'
    )

    # Create minimal config
    left_columns = {'id': ColumnDefinition(name='id', alias='ID', datatype=pl.Int64, nullable=False)}
    right_columns = {'id': ColumnDefinition(name='id', alias='ID', datatype=pl.Int64, nullable=False)}
    left_schema = ComparisonSchema(columns=left_columns, pk_columns=['id'])
    right_schema = ComparisonSchema(columns=right_columns, pk_columns=['id'])

    # Create column mapping for primary key
    mappings = [ColumnMapping(left='id', right='id', name='id')]
    config = ComparisonConfig(left_schema=left_schema, right_schema=right_schema, column_mappings=mappings, primary_key_columns=['id'])

    result = ComparisonResult(
        summary=summary,
        value_differences=pl.LazyFrame({}).limit(0),
        left_only_records=pl.LazyFrame({}).limit(0),
        right_only_records=pl.LazyFrame({}).limit(0),
        config=config
    )

    # Test summary report generation
    report = service.generate_summary_report(results=result)
    assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in report
    assert "4" in report
    print("✓ Summary report generation works")

    # Test detailed report generation
    detailed_report = service.generate_detailed_report(results=result)
    assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in detailed_report
    print("✓ Detailed report generation works")

    # Test summary table generation
    table = service.generate_summary_table(results=result)
    assert "Left DataFrame" in table
    assert "Right DataFrame" in table
    print("✓ Summary table generation works")

    print("\nAll basic functionality tests passed! ✓")

if __name__ == "__main__":
    test_basic_functionality()
