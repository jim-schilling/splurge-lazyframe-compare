"""Tests for the comparator module."""

import polars as pl
import pytest

from splurge_lazyframe_compare.core.comparator import LazyFrameComparator
from splurge_lazyframe_compare.core.schema import (
    ColumnDefinition,
    ColumnMapping,
    ComparisonConfig,
    ComparisonSchema,
)
from splurge_lazyframe_compare.exceptions.comparison_exceptions import (
    PrimaryKeyViolationError,
    SchemaValidationError,
)


class TestLazyFrameComparator:
    """Test LazyFrameComparator class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create test schemas
        self.left_columns = {
            "customer_id": ColumnDefinition("customer_id", "Customer ID", pl.Int64, False),
            "order_date": ColumnDefinition("order_date", "Order Date", pl.Date, False),
            "amount": ColumnDefinition("amount", "Order Amount", pl.Float64, False),
            "status": ColumnDefinition("status", "Order Status", pl.Utf8, True),
        }
        self.right_columns = {
            "cust_id": ColumnDefinition("cust_id", "Customer ID", pl.Int64, False),
            "order_dt": ColumnDefinition("order_dt", "Order Date", pl.Date, False),
            "total_amount": ColumnDefinition("total_amount", "Order Amount", pl.Float64, False),
            "order_status": ColumnDefinition("order_status", "Order Status", pl.Utf8, True),
        }

        self.left_schema = ComparisonSchema(
            columns=self.left_columns,
            primary_key_columns=["customer_id", "order_date"],
        )
        self.right_schema = ComparisonSchema(
            columns=self.right_columns,
            primary_key_columns=["cust_id", "order_dt"],
        )

        # Create column mappings
        self.mappings = [
            ColumnMapping("customer_id", "cust_id", "customer_id"),
            ColumnMapping("order_date", "order_dt", "order_date"),
            ColumnMapping("amount", "total_amount", "amount"),
            ColumnMapping("status", "order_status", "status"),
        ]

        # Create configuration
        self.config = ComparisonConfig(
            left_schema=self.left_schema,
            right_schema=self.right_schema,
            column_mappings=self.mappings,
            primary_key_columns=["customer_id", "order_date"],
        )

        # Create comparator
        self.comparator = LazyFrameComparator(self.config)

    def test_comparator_initialization(self) -> None:
        """Test comparator initialization."""
        assert self.comparator.config == self.config

    def test_compare_identical_dataframes(self) -> None:
        """Test comparison of identical DataFrames."""
        from datetime import date

        # Create identical test data
        left_data = {
            "customer_id": [1, 2, 3],
            "order_date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", "completed", "pending"],
        }
        right_data = {
            "cust_id": [1, 2, 3],
            "order_dt": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "total_amount": [100.0, 200.0, 300.0],
            "order_status": ["pending", "completed", "pending"],
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        results = self.comparator.compare(left=left_df, right=right_df)

        # All records should match
        assert results.summary.total_left_records == 3
        assert results.summary.total_right_records == 3
        assert results.summary.matching_records == 3
        assert results.summary.value_differences_count == 0
        assert results.summary.left_only_count == 0
        assert results.summary.right_only_count == 0

    def test_value_differences_alternating_column_order(self) -> None:
        """Test that value differences show alternating Left/Right columns for non-PK columns."""
        from datetime import date

        # Create test data with differences
        left_data = {
            "customer_id": [1, 2],
            "order_date": [date(2023, 1, 1), date(2023, 1, 2)],
            "amount": [100.0, 200.0],
            "status": ["pending", "completed"],
        }
        right_data = {
            "cust_id": [1, 2],
            "order_dt": [date(2023, 1, 1), date(2023, 1, 2)],
            "total_amount": [150.0, 200.0],  # Different amount for customer 1
            "order_status": ["pending", "shipped"],  # Different status for customer 2
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        results = self.comparator.compare(left=left_df, right=right_df)

        # Should have value differences
        assert results.summary.value_differences_count == 2

        # Get the value differences DataFrame
        value_diff_df = results.value_differences.collect()
        
        # Check that columns are in the correct alternating order
        expected_columns = [
            "PK_customer_id", "PK_order_date",  # Primary key columns first
            "L_amount", "R_amount",             # Then alternating Left/Right
            "L_status", "R_status"
        ]
        
        assert list(value_diff_df.columns) == expected_columns

    def test_left_only_records_clean_columns(self) -> None:
        """Test that left-only records show only primary keys and left columns (no null right columns)."""
        from datetime import date

        # Create test data with left-only records
        left_data = {
            "customer_id": [1, 2],
            "order_date": [date(2023, 1, 1), date(2023, 1, 2)],
            "amount": [100.0, 200.0],
            "status": ["pending", "completed"],
        }
        right_data = {
            "cust_id": [1],  # Only customer 1 exists in right
            "order_dt": [date(2023, 1, 1)],
            "total_amount": [100.0],
            "order_status": ["pending"],
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        results = self.comparator.compare(left=left_df, right=right_df)

        # Should have 1 left-only record
        assert results.summary.left_only_count == 1

        # Get the left-only records DataFrame
        left_only_df = results.left_only_records.collect()
        
        # Check that columns are clean (no right columns)
        expected_columns = [
            "PK_customer_id", "PK_order_date",  # Primary key columns first
            "L_amount", "L_status"              # Then left columns only
        ]
        
        assert list(left_only_df.columns) == expected_columns

    def test_right_only_records_clean_columns(self) -> None:
        """Test that right-only records show only primary keys and right columns (no null left columns)."""
        from datetime import date

        # Create test data with right-only records
        left_data = {
            "customer_id": [1],  # Only customer 1 exists in left
            "order_date": [date(2023, 1, 1)],
            "amount": [100.0],
            "status": ["pending"],
        }
        right_data = {
            "cust_id": [1, 2],
            "order_dt": [date(2023, 1, 1), date(2023, 1, 2)],
            "total_amount": [100.0, 200.0],
            "order_status": ["pending", "completed"],
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        results = self.comparator.compare(left=left_df, right=right_df)

        # Should have 1 right-only record
        assert results.summary.right_only_count == 1

        # Get the right-only records DataFrame
        right_only_df = results.right_only_records.collect()
        
        # Check that columns are clean (no left columns)
        expected_columns = [
            "PK_customer_id", "PK_order_date",  # Primary key columns first
            "R_amount", "R_status"              # Then right columns only
        ]
        
        assert list(right_only_df.columns) == expected_columns

    def test_compare_with_value_differences(self) -> None:
        """Test comparison with value differences."""
        from datetime import date

        # Create test data with differences
        left_data = {
            "customer_id": [1, 2, 3],
            "order_date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", "completed", "pending"],
        }
        right_data = {
            "cust_id": [1, 2, 3],
            "order_dt": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "total_amount": [100.0, 250.0, 300.0],  # Different amount for customer 2
            "order_status": ["pending", "completed", "cancelled"],  # Different status for customer 3
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        results = self.comparator.compare(left=left_df, right=right_df)

        # Should have 2 value differences
        assert results.summary.total_left_records == 3
        assert results.summary.total_right_records == 3
        assert results.summary.matching_records == 1
        assert results.summary.value_differences_count == 2
        assert results.summary.left_only_count == 0
        assert results.summary.right_only_count == 0

    def test_compare_with_left_only_records(self) -> None:
        """Test comparison with left-only records."""
        from datetime import date

        # Create test data with left-only records
        left_data = {
            "customer_id": [1, 2, 3, 4],  # Extra record
            "order_date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
            "amount": [100.0, 200.0, 300.0, 400.0],
            "status": ["pending", "completed", "pending", "pending"],
        }
        right_data = {
            "cust_id": [1, 2, 3],
            "order_dt": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "total_amount": [100.0, 200.0, 300.0],
            "order_status": ["pending", "completed", "pending"],
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        results = self.comparator.compare(left=left_df, right=right_df)

        # Should have 1 left-only record
        assert results.summary.total_left_records == 4
        assert results.summary.total_right_records == 3
        assert results.summary.matching_records == 3
        assert results.summary.value_differences_count == 0
        assert results.summary.left_only_count == 1
        assert results.summary.right_only_count == 0

    def test_compare_with_right_only_records(self) -> None:
        """Test comparison with right-only records."""
        from datetime import date

        # Create test data with right-only records
        left_data = {
            "customer_id": [1, 2, 3],
            "order_date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", "completed", "pending"],
        }
        right_data = {
            "cust_id": [1, 2, 3, 4],  # Extra record
            "order_dt": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
            "total_amount": [100.0, 200.0, 300.0, 400.0],
            "order_status": ["pending", "completed", "pending", "pending"],
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        results = self.comparator.compare(left=left_df, right=right_df)

        # Should have 1 right-only record
        assert results.summary.total_left_records == 3
        assert results.summary.total_right_records == 4
        assert results.summary.matching_records == 3
        assert results.summary.value_differences_count == 0
        assert results.summary.left_only_count == 0
        assert results.summary.right_only_count == 1

    def test_compare_with_tolerance(self) -> None:
        """Test comparison with numeric tolerance."""
        from datetime import date

        # Create configuration with tolerance
        config_with_tolerance = ComparisonConfig(
            left_schema=self.left_schema,
            right_schema=self.right_schema,
            column_mappings=self.mappings,
            primary_key_columns=["customer_id", "order_date"],
            tolerance={"amount": 0.01},  # Allow 1 cent difference
        )
        comparator = LazyFrameComparator(config_with_tolerance)

        # Create test data with small differences
        left_data = {
            "customer_id": [1, 2, 3],
            "order_date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", "completed", "pending"],
        }
        right_data = {
            "cust_id": [1, 2, 3],
            "order_dt": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "total_amount": [100.005, 200.0, 300.02],  # Small differences
            "order_status": ["pending", "completed", "pending"],
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        results = comparator.compare(left=left_df, right=right_df)

        # Should have 1 value difference (300.02 vs 300.0 exceeds tolerance)
        assert results.summary.value_differences_count == 1

    def test_compare_with_case_insensitive(self) -> None:
        """Test comparison with case-insensitive string matching."""
        from datetime import date

        # Create configuration with case-insensitive matching
        config_case_insensitive = ComparisonConfig(
            left_schema=self.left_schema,
            right_schema=self.right_schema,
            column_mappings=self.mappings,
            primary_key_columns=["customer_id", "order_date"],
            ignore_case=True,
        )
        comparator = LazyFrameComparator(config_case_insensitive)

        # Create test data with case differences
        left_data = {
            "customer_id": [1, 2, 3],
            "order_date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "amount": [100.0, 200.0, 300.0],
            "status": ["PENDING", "COMPLETED", "pending"],
        }
        right_data = {
            "cust_id": [1, 2, 3],
            "order_dt": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "total_amount": [100.0, 200.0, 300.0],
            "order_status": ["pending", "completed", "PENDING"],
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        results = comparator.compare(left=left_df, right=right_df)

        # Should have no value differences due to case-insensitive matching
        assert results.summary.value_differences_count == 0

    def test_compare_with_null_handling(self) -> None:
        """Test comparison with null value handling."""
        from datetime import date

        # Create test data with null values
        left_data = {
            "customer_id": [1, 2, 3],
            "order_date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", None, "pending"],
        }
        right_data = {
            "cust_id": [1, 2, 3],
            "order_dt": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "total_amount": [100.0, 200.0, 300.0],
            "order_status": ["pending", None, "completed"],  # Different non-null value
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        results = self.comparator.compare(left=left_df, right=right_df)

        # Should have 1 value difference (customer 3 has different status)
        assert results.summary.value_differences_count == 1

    def test_compare_schema_validation_error(self) -> None:
        """Test comparison with schema validation error."""
        from datetime import date

        # Create test data with wrong data type
        left_data = {
            "customer_id": ["1", "2", "3"],  # Wrong type (string instead of int)
            "order_date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", "completed", "pending"],
        }
        right_data = {
            "cust_id": [1, 2, 3],
            "order_dt": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "total_amount": [100.0, 200.0, 300.0],
            "order_status": ["pending", "completed", "pending"],
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        with pytest.raises(SchemaValidationError):
            self.comparator.compare(left=left_df, right=right_df)

    def test_compare_primary_key_violation_error(self) -> None:
        """Test comparison with primary key violation error."""
        from datetime import date

        # Create test data with duplicate primary keys
        left_data = {
            "customer_id": [1, 1, 3],  # Duplicate customer_id
            "order_date": [date(2023, 1, 1), date(2023, 1, 1), date(2023, 1, 3)],  # Same date for duplicates
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", "completed", "pending"],
        }
        right_data = {
            "cust_id": [1, 2, 3],
            "order_dt": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "total_amount": [100.0, 200.0, 300.0],
            "order_status": ["pending", "completed", "pending"],
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        with pytest.raises(PrimaryKeyViolationError):
            self.comparator.compare(left=left_df, right=right_df)

    def test_compare_empty_dataframes(self) -> None:
        """Test comparison of empty DataFrames."""

        # Create empty test data
        left_data = {
            "customer_id": [],
            "order_date": [],
            "amount": [],
            "status": [],
        }
        right_data = {
            "cust_id": [],
            "order_dt": [],
            "total_amount": [],
            "order_status": [],
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        results = self.comparator.compare(left=left_df, right=right_df)

        # Should have no records
        assert results.summary.total_left_records == 0
        assert results.summary.total_right_records == 0
        assert results.summary.matching_records == 0
        assert results.summary.value_differences_count == 0
        assert results.summary.left_only_count == 0
        assert results.summary.right_only_count == 0
