#!/usr/bin/env python3
"""Basic example demonstrating the Polars LazyFrame comparison framework.

This example shows how to:
1. Define schemas for two datasets with different column names
2. Create column mappings between the datasets
3. Configure and run a comparison
4. Generate and display comparison reports
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


def define_schemas() -> tuple[ComparisonSchema, ComparisonSchema]:
    """Define schemas for the left and right datasets.

    Returns:
        Tuple of (left_schema, right_schema).
    """
    # Left schema - demonstrating mixed datatype usage
    left_columns = {
        # Using string datatype names (recommended for readability)
        "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype="Int64", nullable=False),
        "order_date": ColumnDefinition(name="order_date", alias="Order Date", datatype="Date", nullable=False),
        # Using direct Polars datatypes
        "amount": ColumnDefinition(name="amount", alias="Order Amount", datatype=pl.Float64, nullable=False),
        "status": ColumnDefinition(name="status", alias="Order Status", datatype=pl.Utf8, nullable=True),
    }
    left_schema = ComparisonSchema(
        columns=left_columns,
        pk_columns=["customer_id", "order_date"],
    )

    # Right schema - mixing both approaches for demonstration
    right_columns = {
        # Using direct Polars datatypes
        "cust_id": ColumnDefinition(name="cust_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
        "order_dt": ColumnDefinition(name="order_dt", alias="Order Date", datatype=pl.Date, nullable=False),
        # Using string datatype names
        "total_amount": ColumnDefinition(name="total_amount", alias="Order Amount", datatype="Float64", nullable=False),
        "order_status": ColumnDefinition(name="order_status", alias="Order Status", datatype="String", nullable=True),
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
    ]


def main() -> None:
    """Run the basic comparison example."""
    print("=" * 60)
    print("POLARS LAZYFRAME COMPARISON FRAMEWORK - BASIC EXAMPLE")
    print("=" * 60)

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
        primary_key_columns=["customer_id", "order_date"],
        ignore_case=False,
        null_equals_null=True,
    )

    # Create comparator and run comparison
    print("\n5. Running comparison...")
    comparator = LazyFrameComparator(config)
    results = comparator.compare(left=left_df, right=right_df)

    # Display results using ReportingService
    print("\n6. Comparison Results:")
    print("-" * 40)
    reporter = ReportingService()
    summary_report = reporter.generate_summary_report(results=results)
    print(summary_report)

    # Display detailed results
    print("\n7. Detailed Results:")
    print("-" * 40)
    detailed_report = reporter.generate_detailed_report(
        results=results,
        max_samples=5
    )
    print(detailed_report)

    # Export results (optional)
    print("\n8. Exporting results...")
    exported_files = reporter.export_results(
        results=results,
        format="csv",
        output_dir="./comparison_results"
    )
    print(f"   Exported files: {list(exported_files.keys())}")

    print("\n9. Example completed successfully!")

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
