#!/usr/bin/env python3
"""Example demonstrating automatic ComparisonConfig generation from LazyFrames."""

import polars as pl

from splurge_lazyframe_compare.services.comparison_service import ComparisonService
from splurge_lazyframe_compare.utils import create_comparison_config_from_lazyframes


def main():
    """Demonstrate automatic configuration generation and comparison."""

    # Create sample left DataFrame
    left_data = {
        "customer_id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "email": ["alice@example.com", "bob@example.com", "charlie@example.com", "david@example.com", "eve@example.com"],
        "balance": [100.50, 250.00, 75.25, 300.00, 150.75],
        "active": [True, True, False, True, True]
    }

    # Create sample right DataFrame with some differences
    right_data = {
        "customer_id": [1, 2, 3, 4, 6],  # ID 5 missing, ID 6 added
        "name": ["Alice", "Bob", "Charlie", "Dave", "Frank"],  # David -> Dave, Eve -> Frank
        "email": ["alice@example.com", "bob@example.com", "charlie@example.com", "dave@example.com", "frank@example.com"],
        "balance": [100.50, 250.00, 75.25, 320.00, 200.00],  # David's balance changed, new balance for Frank
        "active": [True, True, False, True, False]  # Frank is inactive
    }

    # Create LazyFrames
    left_lf = pl.LazyFrame(left_data)
    right_lf = pl.LazyFrame(right_data)

    print("Left DataFrame schema:")
    print(left_lf.collect_schema())
    print("\nRight DataFrame schema:")
    print(right_lf.collect_schema())

    # Define primary key columns
    primary_keys = ["customer_id"]

    print(f"\nUsing primary keys: {primary_keys}")

    try:
        # Automatically generate ComparisonConfig
        config = create_comparison_config_from_lazyframes(
            left_df=left_lf,
            right_df=right_lf,
            pk_columns=primary_keys
        )

        print("\nComparisonConfig generated successfully!")
        print(f"Primary key columns: {config.pk_columns}")
        print(f"Number of column mappings: {len(config.column_mappings)}")
        print(f"Left schema columns: {list(config.left_schema.columns.keys())}")
        print(f"Right schema columns: {list(config.right_schema.columns.keys())}")

        # Perform comparison using the generated config
        comparison_service = ComparisonService()

        print("\nPerforming comparison...")
        results = comparison_service.execute_comparison(
            left=left_lf,
            right=right_lf,
            config=config
        )

        print("\nComparison completed!")
        print(f"Left-only records: {results.summary.left_only_count}")
        print(f"Right-only records: {results.summary.right_only_count}")
        print(f"Value differences: {results.summary.value_differences_count}")
        print(f"Matching records: {results.summary.matching_records}")

        # Display summary
        print("\n" + "="*50)
        print("COMPARISON SUMMARY")
        print("="*50)
        print(results.summary)

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
