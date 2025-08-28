#!/usr/bin/env python3
"""Example demonstrating the service-based architecture.

This example shows how to use the ComparisonOrchestrator and
service-based LazyFrameComparator for comparing DataFrames.
"""

from datetime import date

import polars as pl

from splurge_lazyframe_compare import (
    ColumnDefinition,
    ColumnMapping,
    ComparisonConfig,
    ComparisonOrchestrator,
    ComparisonSchema,
    LazyFrameComparator,
)


def create_sample_data():
    """Create sample data for comparison demonstration."""
    # Left dataset (source system)
    left_data = {
        "customer_id": [1, 2, 3, 4, 5],
        "order_date": [
            date(2023, 1, 1),
            date(2023, 1, 2),
            date(2023, 1, 3),
            date(2023, 1, 4),
            date(2023, 1, 5),
        ],
        "amount": [100.0, 200.0, 300.0, 400.0, 500.0],
        "status": ["pending", "completed", "pending", "cancelled", "completed"],
    }

    # Right dataset (target system) - with some differences
    right_data = {
        "cust_id": [1, 2, 3, 4, 6],  # Customer 5 missing, customer 6 added
        "order_dt": [
            date(2023, 1, 1),
            date(2023, 1, 2),
            date(2023, 1, 3),
            date(2023, 1, 4),
            date(2023, 1, 6),
        ],
        "total_amount": [100.0, 250.0, 300.0, 400.0, 600.0],  # Customer 2 amount different
        "order_status": ["pending", "completed", "shipped", "cancelled", "pending"],  # Customer 3 status different
    }

    left_df = pl.LazyFrame(left_data)
    right_df = pl.LazyFrame(right_data)

    return left_df, right_df


def define_schemas():
    """Define schemas for the left and right datasets."""
    # Left schema
    left_columns = {
        "customer_id": ColumnDefinition("customer_id", "Customer ID", pl.Int64, False),
        "order_date": ColumnDefinition("order_date", "Order Date", pl.Date, False),
        "amount": ColumnDefinition("amount", "Order Amount", pl.Float64, False),
        "status": ColumnDefinition("status", "Order Status", pl.Utf8, True),
    }
    left_schema = ComparisonSchema(
        columns=left_columns,
        primary_key_columns=["customer_id", "order_date"],
    )

    # Right schema
    right_columns = {
        "cust_id": ColumnDefinition("cust_id", "Customer ID", pl.Int64, False),
        "order_dt": ColumnDefinition("order_dt", "Order Date", pl.Date, False),
        "total_amount": ColumnDefinition("total_amount", "Order Amount", pl.Float64, False),
        "order_status": ColumnDefinition("order_status", "Order Status", pl.Utf8, True),
    }
    right_schema = ComparisonSchema(
        columns=right_columns,
        primary_key_columns=["cust_id", "order_dt"],
    )

    return left_schema, right_schema


def create_column_mappings():
    """Create column mappings between the datasets."""
    return [
        ColumnMapping("customer_id", "cust_id", "customer_id"),
        ColumnMapping("order_date", "order_dt", "order_date"),
        ColumnMapping("amount", "total_amount", "amount"),
        ColumnMapping("status", "order_status", "status"),
    ]


def demonstrate_orchestrator():
    """Demonstrate using the ComparisonOrchestrator directly."""
    print("=" * 60)
    print("DEMONSTRATING COMPARISON ORCHESTRATOR")
    print("=" * 60)

    # Create sample data and configuration
    left_df, right_df = create_sample_data()
    left_schema, right_schema = define_schemas()
    mappings = create_column_mappings()

    config = ComparisonConfig(
        left_schema=left_schema,
        right_schema=right_schema,
        column_mappings=mappings,
        primary_key_columns=["customer_id", "order_date"],
        ignore_case=False,
        null_equals_null=True,
    )

    # Create orchestrator
    orchestrator = ComparisonOrchestrator()

    # Method 1: Direct comparison
    print("\n1. Executing direct comparison...")
    result = orchestrator.compare_dataframes(
        config=config,
        left=left_df,
        right=right_df
    )

    print(f"   Results: {result.summary.total_left_records} left, {result.summary.total_right_records} right")
    print(f"   Matching: {result.summary.matching_records}")
    print(f"   Differences: {result.summary.value_differences_count}")
    print(f"   Left-only: {result.summary.left_only_count}")
    print(f"   Right-only: {result.summary.right_only_count}")

    # Method 2: Compare and report
    print("\n2. Generating comparison report...")
    report = orchestrator.compare_and_report(
        config=config,
        left=left_df,
        right=right_df,
        include_samples=True,
        max_samples=3
    )

    print("   Report generated successfully!")
    print("\n" + "="*50)
    print("SAMPLE REPORT:")
    print("="*50)
    print(report[:500] + "..." if len(report) > 500 else report)


def demonstrate_new_comparator():
    """Demonstrate using the new LazyFrameComparator."""
    print("\n\n" + "=" * 60)
    print("DEMONSTRATING NEW LAZYFRAME COMPARATOR")
    print("=" * 60)

    # Create sample data and configuration
    left_df, right_df = create_sample_data()
    left_schema, right_schema = define_schemas()
    mappings = create_column_mappings()

    config = ComparisonConfig(
        left_schema=left_schema,
        right_schema=right_schema,
        column_mappings=mappings,
        primary_key_columns=["customer_id", "order_date"],
    )

    # Create new comparator
    comparator = LazyFrameComparator(config)

    # Method 1: Compare and get result
    print("\n1. Executing comparison...")
    result = comparator.compare(left=left_df, right=right_df)

    print(f"   Results: {result.summary.matching_records} matching records")

    # Method 2: Compare and report in one step
    print("\n2. Generating report...")
    report = comparator.compare_and_report(
        left=left_df,
        right=right_df,
        include_samples=False
    )

    print("   Report generated successfully!")

    # Method 3: Export results
    print("\n3. Exporting results...")
    exported_files = comparator.compare_and_export(
        left=left_df,
        right=right_df,
        output_dir="./service_example_output",
        format="csv"
    )

    print(f"   Exported files: {list(exported_files.keys())}")


def demonstrate_service_injection():
    """Demonstrate service dependency injection."""
    print("\n\n" + "=" * 60)
    print("DEMONSTRATING SERVICE DEPENDENCY INJECTION")
    print("=" * 60)

    from splurge_lazyframe_compare.services import (
        ComparisonService,
        ValidationService,
        DataPreparationService,
        ReportingService,
    )

    # Create custom services (could be customized versions)
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

    # Create sample data and configuration
    left_df, right_df = create_sample_data()
    left_schema, right_schema = define_schemas()
    mappings = create_column_mappings()

    config = ComparisonConfig(
        left_schema=left_schema,
        right_schema=right_schema,
        column_mappings=mappings,
        primary_key_columns=["customer_id", "order_date"],
    )

    print("\n1. Using orchestrator with injected services...")
    result = orchestrator.compare_dataframes(
        config=config,
        left=left_df,
        right=right_df
    )

    print(f"   Custom services working: {result.summary.matching_records} matches found")


def main():
    """Run all demonstrations."""
    print("NEW SERVICE-BASED ARCHITECTURE DEMONSTRATION")
    print("This example shows the new modular, service-oriented design")

    demonstrate_orchestrator()
    demonstrate_new_comparator()
    demonstrate_service_injection()

    print("\n\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("\nKey Benefits of the New Architecture:")
    print("• Modular design - services can be used independently")
    print("• Dependency injection - easy to test and customize")
    print("• Clean separation of concerns - each service has one job")
    print("• Extensible - new services can be easily added")
    print("• Maintainable - smaller, focused modules")


if __name__ == "__main__":
    main()
