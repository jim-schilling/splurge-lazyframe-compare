#!/usr/bin/env python3
"""Example demonstrating the enhanced tabulated reporting functionality.

This example shows how to use the new tabulate integration in the results module
to generate beautifully formatted reports with different table styles.
"""

from datetime import date

import polars as pl

from splurge_lazyframe_compare import (
    ColumnDefinition,
    ColumnMapping,
    ComparisonConfig,
    ComparisonSchema,
    LazyFrameComparator,
    ReportingService,
)


def create_sample_data() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Create sample data for comparison demonstration.

    Returns:
        Tuple of (left_df, right_df) with sample order data.
    """
    # Left dataset (source system)
    left_data = {
        "customer_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "order_date": [
            date(2023, 1, 1),
            date(2023, 1, 2),
            date(2023, 1, 3),
            date(2023, 1, 4),
            date(2023, 1, 5),
            date(2023, 1, 6),
            date(2023, 1, 7),
            date(2023, 1, 8),
        ],
        "amount": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
        "status": ["pending", "completed", "pending", "cancelled", "completed", "shipped", "pending", "completed"],
        "priority": ["low", "medium", "high", "low", "medium", "high", "low", "medium"],
    }

    # Right dataset (target system) - with some differences
    right_data = {
        "cust_id": [1, 2, 3, 4, 6, 7, 9, 10],  # Customers 5, 8 missing; 9, 10 added
        "order_dt": [
            date(2023, 1, 1),
            date(2023, 1, 2),
            date(2023, 1, 3),
            date(2023, 1, 4),
            date(2023, 1, 6),
            date(2023, 1, 7),
            date(2023, 1, 9),
            date(2023, 1, 10),
        ],
        "total_amount": [100.0, 250.0, 300.0, 400.0, 650.0, 700.0, 900.0, 1000.0],  # Some amounts different
        "order_status": ["pending", "completed", "shipped", "cancelled", "shipped", "pending", "completed", "pending"],  # Some status different
        "priority_level": ["low", "medium", "high", "low", "high", "low", "medium", "high"],  # Some priority different
    }

    left_df = pl.LazyFrame(left_data).with_columns(
        pl.col("priority").cast(pl.Categorical)
    )
    right_df = pl.LazyFrame(right_data).with_columns(
        pl.col("priority_level").cast(pl.Categorical)
    )

    return left_df, right_df


def define_schemas() -> tuple[ComparisonSchema, ComparisonSchema]:
    """Define schemas for the left and right datasets.

    Returns:
        Tuple of (left_schema, right_schema).
    """
    # Left schema - showcasing string datatype names for complex types
    left_columns = {
        # Using string datatype names for better readability
        "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype="Int64", nullable=False),
        "order_date": ColumnDefinition(name="order_date", alias="Order Date", datatype="Date", nullable=False),
        "amount": ColumnDefinition(name="amount", alias="Order Amount", datatype="Float64", nullable=False),
        "status": ColumnDefinition(name="status", alias="Order Status", datatype="String", nullable=True),
        "priority": ColumnDefinition(name="priority", alias="Priority Level", datatype="Categorical", nullable=True),
    }
    left_schema = ComparisonSchema(
        columns=left_columns,
        pk_columns=["customer_id", "order_date"],
    )

    # Right schema - mixing direct datatypes and string names
    right_columns = {
        # Using direct Polars datatypes
        "cust_id": ColumnDefinition(name="cust_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
        "order_dt": ColumnDefinition(name="order_dt", alias="Order Date", datatype=pl.Date, nullable=False),
        "total_amount": ColumnDefinition(name="total_amount", alias="Order Amount", datatype=pl.Float64, nullable=False),
        "order_status": ColumnDefinition(name="order_status", alias="Order Status", datatype="String", nullable=True),
        "priority_level": ColumnDefinition(name="priority_level", alias="Priority Level", datatype="Categorical", nullable=True),
    }
    right_schema = ComparisonSchema(
        columns=right_columns,
        pk_columns=["cust_id", "order_dt"],
    )

    return left_schema, right_schema


def create_column_mappings() -> list[ColumnMapping]:
    """Create column mappings between the datasets.

    Returns:
        List of ColumnMapping objects.
    """
    return [
        ColumnMapping(left="customer_id", right="cust_id", name="customer_id"),
        ColumnMapping(left="order_date", right="order_dt", name="order_date"),
        ColumnMapping(left="amount", right="total_amount", name="amount"),
        ColumnMapping(left="status", right="order_status", name="status"),
        ColumnMapping(left="priority", right="priority_level", name="priority"),
    ]


def main() -> None:
    """Run the tabulated report example."""
    print("=" * 80)
    print("POLARS LAZYFRAME COMPARISON FRAMEWORK - TABULATED REPORT EXAMPLE")
    print("=" * 80)

    # Create sample data
    print("\n1. Creating sample data...")
    left_df, right_df = create_sample_data()
    print(f"   Left dataset: {left_df.select(pl.len()).collect().item()} records")
    print(f"   Right dataset: {right_df.select(pl.len()).collect().item()} records")

    # Define schemas
    print("\n2. Defining schemas...")
    left_schema, right_schema = define_schemas()
    print(f"   Left schema: {len(left_schema.columns)} columns")
    print(f"   Right schema: {len(right_schema.columns)} columns")

    # Create column mappings
    print("\n3. Creating column mappings...")
    mappings = create_column_mappings()
    print(f"   Column mappings: {len(mappings)} mappings")

    # Create comparison configuration
    print("\n4. Creating comparison configuration...")
    config = ComparisonConfig(
        left_schema=left_schema,
        right_schema=right_schema,
        column_mappings=mappings,
        pk_columns=["customer_id", "order_date"],
        ignore_case=False,
        null_equals_null=True,
    )

    # Create comparator and run comparison
    print("\n5. Running comparison...")
    comparator = LazyFrameComparator(config)
    results = comparator.compare(left=left_df, right=right_df)

    # Generate reports with different table formats using ReportingService
    print("\n6. Generating tabulated reports...")
    reporter = ReportingService()

    # Show summary table with different formats
    print("\n" + "=" * 60)
    print("SUMMARY TABLE - GRID FORMAT")
    print("=" * 60)
    summary_table_grid = reporter.generate_summary_table(
        results=results,
        table_format="grid"
    )
    print(summary_table_grid)

    print("\n" + "=" * 60)
    print("SUMMARY TABLE - SIMPLE FORMAT")
    print("=" * 60)
    summary_table_simple = reporter.generate_summary_table(
        results=results,
        table_format="simple"
    )
    print(summary_table_simple)

    print("\n" + "=" * 60)
    print("SUMMARY TABLE - PIPE FORMAT")
    print("=" * 60)
    summary_table_pipe = reporter.generate_summary_table(
        results=results,
        table_format="pipe"
    )
    print(summary_table_pipe)

    # Show detailed report with grid format
    print("\n" + "=" * 60)
    print("DETAILED REPORT - GRID FORMAT")
    print("=" * 60)
    detailed_report_grid = reporter.generate_detailed_report(
        results=results,
        max_samples=5
    )
    print(detailed_report_grid)

    # Show summary table in orgtbl format
    print("\n" + "=" * 60)
    print("SUMMARY TABLE - ORGTBL FORMAT")
    print("=" * 60)
    summary_table_orgtbl = reporter.generate_summary_table(
        results=results,
        table_format="orgtbl"
    )
    print(summary_table_orgtbl)

    # Show original detailed report for comparison
    print("\n" + "=" * 60)
    print("ORIGINAL DETAILED REPORT (for comparison)")
    print("=" * 60)
    detailed_report_original = reporter.generate_detailed_report(
        results=results,
        max_samples=3
    )
    print(detailed_report_original)

    print("\n" + "=" * 80)
    print("TABULATED REPORT EXAMPLE COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
