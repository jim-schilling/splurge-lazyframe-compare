#!/usr/bin/env python3
"""Example demonstrating the service-based architecture.

This example shows how to use the ComparisonOrchestrator and
service-based LazyFrameComparator for comparing DataFrames.
"""

from datetime import date, datetime

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
        "tags": [["urgent", "premium"], ["standard"], ["express", "gift"], ["bulk"], ["repeat"]],
        "metadata": [
            {"source": "web"},
            {"source": "mobile"},
            {"source": "api"},
            {"source": "batch"},
            {"source": "web"},
        ],
        "status": ["pending", "completed", "pending", "cancelled", "completed"],
        "created_at": [
            datetime(2023, 1, 1, 10, 0, 0),
            datetime(2023, 1, 2, 11, 30, 0),
            datetime(2023, 1, 3, 9, 15, 0),
            datetime(2023, 1, 4, 14, 45, 0),
            datetime(2023, 1, 5, 16, 20, 0),
        ],
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
        "category": ["retail", "wholesale", "retail", "enterprise", "retail"],
        "tags": [["urgent", "premium"], ["standard"], ["express"], ["bulk"], ["new_customer"]],
        "order_status": ["pending", "completed", "shipped", "cancelled", "pending"],  # Customer 3 status different
        "processed_at": [
            datetime(2023, 1, 1, 10, 30, 0),
            datetime(2023, 1, 2, 12, 0, 0),
            datetime(2023, 1, 3, 10, 0, 0),
            datetime(2023, 1, 4, 15, 15, 0),
            datetime(2023, 1, 6, 9, 0, 0),
        ],
    }

    left_df = pl.LazyFrame(left_data)
    right_df = pl.LazyFrame(right_data).with_columns(pl.col("category").cast(pl.Categorical))

    return left_df, right_df


def define_schemas():
    """Define schemas for the left and right datasets."""
    # Left schema - showcasing complex datatypes with string names
    left_columns = {
        # Simple datatypes using string names
        "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype="Int64", nullable=False),
        "order_date": ColumnDefinition(name="order_date", alias="Order Date", datatype="Date", nullable=False),
        "amount": ColumnDefinition(name="amount", alias="Order Amount", datatype="Float64", nullable=False),
        # Complex datatypes using direct Polars types
        "tags": ColumnDefinition(name="tags", alias="Tags", datatype=pl.List(pl.Utf8), nullable=True),
        "metadata": ColumnDefinition(
            name="metadata", alias="Metadata", datatype=pl.Struct({"source": pl.Utf8}), nullable=True
        ),
        "status": ColumnDefinition(name="status", alias="Order Status", datatype="String", nullable=True),
        "created_at": ColumnDefinition(name="created_at", alias="Created At", datatype="Datetime", nullable=False),
    }
    left_schema = ComparisonSchema(
        columns=left_columns,
        pk_columns=["customer_id", "order_date"],
    )

    # Right schema - mixing approaches and showcasing categorical data
    right_columns = {
        # Using direct Polars datatypes
        "cust_id": ColumnDefinition(name="cust_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
        "order_dt": ColumnDefinition(name="order_dt", alias="Order Date", datatype=pl.Date, nullable=False),
        "total_amount": ColumnDefinition(
            name="total_amount", alias="Order Amount", datatype=pl.Float64, nullable=False
        ),
        # Using string names for complex types
        "category": ColumnDefinition(name="category", alias="Category", datatype="Categorical", nullable=False),
        "tags": ColumnDefinition(name="tags", alias="Tags", datatype=pl.List(pl.Utf8), nullable=True),
        "order_status": ColumnDefinition(name="order_status", alias="Order Status", datatype="String", nullable=True),
        "processed_at": ColumnDefinition(name="processed_at", alias="Processed At", datatype="Datetime", nullable=True),
    }
    right_schema = ComparisonSchema(
        columns=right_columns,
        pk_columns=["cust_id", "order_dt"],
    )

    return left_schema, right_schema


def create_column_mappings():
    """Create column mappings between the datasets."""
    return [
        # Primary key mappings
        ColumnMapping(left="customer_id", right="cust_id", name="customer_id"),
        ColumnMapping(left="order_date", right="order_dt", name="order_date"),
        # Value comparison mappings
        ColumnMapping(left="amount", right="total_amount", name="amount"),
        ColumnMapping(left="status", right="order_status", name="status"),
        # Additional complex datatype mappings
        ColumnMapping(left="tags", right="tags", name="tags"),
        ColumnMapping(left="created_at", right="processed_at", name="timestamp"),
        # Right-only columns (no left equivalent)
        # Note: "category" and "metadata" are not mapped as they don't have equivalents
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
        pk_columns=["customer_id", "order_date"],
        ignore_case=False,
        null_equals_null=True,
    )

    # Create orchestrator
    orchestrator = ComparisonOrchestrator()

    # Method 1: Direct comparison
    print("\n1. Executing direct comparison...")
    result = orchestrator.compare_dataframes(config=config, left=left_df, right=right_df)

    print(f"   Results: {result.summary.total_left_records} left, {result.summary.total_right_records} right")
    print(f"   Matching: {result.summary.matching_records}")
    print(f"   Differences: {result.summary.value_differences_count}")
    print(f"   Left-only: {result.summary.left_only_count}")
    print(f"   Right-only: {result.summary.right_only_count}")

    # Method 2: Compare and report
    print("\n2. Generating comparison report...")
    report = orchestrator.compare_and_report(
        config=config, left=left_df, right=right_df, include_samples=True, max_samples=3
    )

    print("   Report generated successfully!")
    print("\n" + "=" * 50)
    print("SAMPLE REPORT:")
    print("=" * 50)
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
        pk_columns=["customer_id", "order_date"],
    )

    # Create new comparator
    comparator = LazyFrameComparator(config)

    # Method 1: Compare and get result
    print("\n1. Executing comparison...")
    result = comparator.compare(left=left_df, right=right_df)

    print(f"   Results: {result.summary.matching_records} matching records")

    # Method 2: Compare and report in one step
    print("\n2. Generating report...")
    comparator.compare_and_report(left=left_df, right=right_df, include_samples=False)

    print("   Report generated successfully!")

    # Method 3: Export results
    print("\n3. Exporting results...")
    exported_files = comparator.compare_and_export(
        left=left_df, right=right_df, output_dir="./service_example_output", format="parquet"
    )

    print(f"   Exported files: {list(exported_files.keys())}")


def demonstrate_service_injection():
    """Demonstrate service dependency injection."""
    print("\n\n" + "=" * 60)
    print("DEMONSTRATING SERVICE DEPENDENCY INJECTION")
    print("=" * 60)

    from splurge_lazyframe_compare.services import (
        ComparisonService,
        DataPreparationService,
        ReportingService,
        ValidationService,
    )

    # Create custom services (could be customized versions)
    validation_service = ValidationService()
    preparation_service = DataPreparationService()
    reporting_service = ReportingService()

    # Create comparison service with injected dependencies
    comparison_service = ComparisonService(
        validation_service=validation_service, preparation_service=preparation_service
    )

    # Create orchestrator with custom services
    orchestrator = ComparisonOrchestrator(comparison_service=comparison_service, reporting_service=reporting_service)

    # Create sample data and configuration
    left_df, right_df = create_sample_data()
    left_schema, right_schema = define_schemas()
    mappings = create_column_mappings()

    config = ComparisonConfig(
        left_schema=left_schema,
        right_schema=right_schema,
        column_mappings=mappings,
        pk_columns=["customer_id", "order_date"],
    )

    print("\n1. Using orchestrator with injected services...")
    result = orchestrator.compare_dataframes(config=config, left=left_df, right=right_df)

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
