"""Tests for the comparator module."""

import polars as pl
import pytest

from splurge_lazyframe_compare.core.comparator import LazyFrameComparator
from splurge_lazyframe_compare.exceptions.comparison_exceptions import (
    PrimaryKeyViolationError,
    SchemaValidationError,
)
from splurge_lazyframe_compare.models.schema import (
    ColumnDefinition,
    ColumnMapping,
    ComparisonConfig,
    ComparisonSchema,
)


class TestLazyFrameComparator:
    """Test LazyFrameComparator class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Create test schemas
        self.left_columns = {
            "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "order_date": ColumnDefinition(name="order_date", alias="Order Date", datatype=pl.Date, nullable=False),
            "amount": ColumnDefinition(name="amount", alias="Order Amount", datatype=pl.Float64, nullable=False),
            "status": ColumnDefinition(name="status", alias="Order Status", datatype=pl.Utf8, nullable=True),
        }
        self.right_columns = {
            "cust_id": ColumnDefinition(name="cust_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "order_dt": ColumnDefinition(name="order_dt", alias="Order Date", datatype=pl.Date, nullable=False),
            "total_amount": ColumnDefinition(name="total_amount", alias="Order Amount", datatype=pl.Float64, nullable=False),
            "order_status": ColumnDefinition(name="order_status", alias="Order Status", datatype=pl.Utf8, nullable=True),
        }

        self.left_schema = ComparisonSchema(
            columns=self.left_columns,
            pk_columns=["customer_id", "order_date"],
        )
        self.right_schema = ComparisonSchema(
            columns=self.right_columns,
            pk_columns=["cust_id", "order_dt"],
        )

        # Create column mappings
        self.mappings = [
            ColumnMapping(left="customer_id", right="cust_id", name="customer_id"),
            ColumnMapping(left="order_date", right="order_dt", name="order_date"),
            ColumnMapping(left="amount", right="total_amount", name="amount"),
            ColumnMapping(left="status", right="order_status", name="status"),
        ]

        # Create configuration
        self.config = ComparisonConfig(
            left_schema=self.left_schema,
            right_schema=self.right_schema,
            column_mappings=self.mappings,
            pk_columns=["customer_id", "order_date"],
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
            pk_columns=["customer_id", "order_date"],
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
            pk_columns=["customer_id", "order_date"],
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

    def test_compare_and_report_comprehensive(self) -> None:
        """Test compare_and_report method with various parameters."""
        # Test with identical dataframes
        left_data = {
            "customer_id": [1, 2, 3],
            "order_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", "completed", "pending"],
        }
        right_data = {
            "cust_id": [1, 2, 3],
            "order_dt": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "total_amount": [100.0, 200.0, 300.0],
            "order_status": ["pending", "completed", "pending"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(
            pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d")
        )
        right_df = pl.LazyFrame(right_data).with_columns(
            pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # Test with default parameters
        report = self.comparator.compare_and_report(left=left_df, right=right_df)
        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in report
        assert "3" in report  # Record counts

        # Test with custom parameters
        report_custom = self.comparator.compare_and_report(
            left=left_df,
            right=right_df,
            include_samples=False,
            max_samples=5,
            table_format="simple"
        )
        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in report_custom
        assert "3" in report_custom

    def test_compare_and_report_with_differences(self) -> None:
        """Test compare_and_report method when there are differences."""
        left_data = {
            "customer_id": [1, 2, 3],
            "order_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", "completed", "pending"],
        }
        right_data = {
            "cust_id": [1, 2, 4],  # Different customer 3 -> 4
            "order_dt": ["2023-01-01", "2023-01-02", "2023-01-04"],  # Different date
            "total_amount": [100.0, 250.0, 400.0],  # Different amount for customer 2
            "order_status": ["pending", "completed", "shipped"],  # Different status
        }

        left_df = pl.LazyFrame(left_data).with_columns(
            pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d")
        )
        right_df = pl.LazyFrame(right_data).with_columns(
            pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d")
        )

        report = self.comparator.compare_and_report(left=left_df, right=right_df)
        assert "VALUE DIFFERENCES" in report
        assert "LEFT-ONLY RECORDS" in report
        assert "RIGHT-ONLY RECORDS" in report
        assert "2" in report  # Matching records
        assert "1" in report  # Left only records
        assert "1" in report  # Right only records

    def test_compare_and_export_comprehensive(self) -> None:
        """Test compare_and_export method with various formats and scenarios."""
        import tempfile
        from pathlib import Path

        left_data = {
            "customer_id": [1, 2, 3],
            "order_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", "completed", "pending"],
        }
        right_data = {
            "cust_id": [1, 2, 4],
            "order_dt": ["2023-01-01", "2023-01-02", "2023-01-04"],
            "total_amount": [100.0, 250.0, 400.0],
            "order_status": ["pending", "completed", "shipped"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(
            pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d")
        )
        right_df = pl.LazyFrame(right_data).with_columns(
            pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d")
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test parquet export (default)
            exported_files = self.comparator.compare_and_export(
                left=left_df,
                right=right_df,
                output_dir=temp_dir
            )

            # Should export 4 files: value_differences, left_only, right_only, summary
            assert len(exported_files) == 4
            for file_path in exported_files.values():
                assert Path(file_path).exists()
                assert Path(file_path).stat().st_size > 0
                # Summary is always JSON, others are parquet
                if 'summary' in file_path:
                    assert file_path.endswith('.json')
                else:
                    assert file_path.endswith('.parquet')

            # Test CSV export
            exported_csv = self.comparator.compare_and_export(
                left=left_df,
                right=right_df,
                output_dir=temp_dir,
                format="csv"
            )

            for file_path in exported_csv.values():
                assert Path(file_path).exists()
                # Summary is always JSON, others are csv
                if 'summary' in file_path:
                    assert file_path.endswith('.json')
                else:
                    assert file_path.endswith('.csv')

    def test_compare_and_export_empty_results(self) -> None:
        """Test compare_and_export with empty results."""
        import tempfile
        from pathlib import Path

        # Create identical dataframes
        left_data = {
            "customer_id": [1, 2, 3],
            "order_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", "completed", "pending"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(
            pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # Create right data with correct column names
        right_data = {
            "cust_id": [1, 2, 3],
            "order_dt": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "total_amount": [100.0, 200.0, 300.0],
            "order_status": ["pending", "completed", "pending"],
        }

        right_df = pl.LazyFrame(right_data).with_columns(
            pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d")
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            exported_files = self.comparator.compare_and_export(
                left=left_df,
                right=right_df,
                output_dir=temp_dir,
                format="csv"
            )

            # Should export summary file (no differences so other files are not created)
            assert len(exported_files) == 1  # Only summary file
            for file_path in exported_files.values():
                assert Path(file_path).exists()
                assert file_path.endswith('.json')  # Summary is always exported as JSON

    def test_to_report_method(self) -> None:
        """Test to_report method returns ComparisonReport instance."""
        from splurge_lazyframe_compare.core.comparator import ComparisonReport

        report_obj = self.comparator.to_report()
        assert isinstance(report_obj, ComparisonReport)
        assert report_obj.config == self.config
        assert report_obj.orchestrator == self.comparator.orchestrator

    def test_compare_with_invalid_lazyframe(self) -> None:
        """Test compare method with invalid LazyFrame inputs."""
        valid_df = pl.LazyFrame({"id": [1, 2, 3]})
        invalid_df = "not a dataframe"

        # Test with invalid left dataframe
        with pytest.raises(ValueError, match="left must be a polars LazyFrame"):
            self.comparator.compare(left=invalid_df, right=valid_df)

        # Test with invalid right dataframe
        with pytest.raises(ValueError, match="right must be a polars LazyFrame"):
            self.comparator.compare(left=valid_df, right=invalid_df)

    def test_compare_and_report_parameter_validation(self) -> None:
        """Test compare_and_report parameter validation."""
        # Create a simple comparator for this test
        simple_schema = ComparisonSchema(
            columns={
                "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
                "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=False),
            },
            pk_columns=["id"],
        )
        simple_config = ComparisonConfig(
            left_schema=simple_schema,
            right_schema=simple_schema,
            column_mappings=[
                ColumnMapping(left="id", right="id", name="id"),
                ColumnMapping(left="name", right="name", name="name"),
            ],
            pk_columns=["id"],
        )
        simple_comparator = LazyFrameComparator(simple_config)

        # Create valid dataframes that match the simple schema
        valid_data = {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
        }
        valid_df = pl.LazyFrame(valid_data)
        invalid_df = "not a dataframe"

        # Test with invalid parameters
        with pytest.raises(ValueError, match="left must be a polars LazyFrame"):
            simple_comparator.compare_and_report(left=invalid_df, right=valid_df)

        # Test with negative max_samples
        with pytest.raises((ValueError, TypeError)):
            simple_comparator.compare_and_report(
                left=valid_df,
                right=valid_df,
                max_samples=-1
            )

    def test_compare_and_export_parameter_validation(self) -> None:
        """Test compare_and_export parameter validation."""
        import tempfile
        from datetime import date

        # Create valid dataframes that match the LEFT schema
        left_data = {
            "customer_id": [1, 2, 3],
            "order_date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", "completed", "pending"],
        }
        left_df = pl.LazyFrame(left_data)

        # Create right dataframe that matches the RIGHT schema
        right_data = {
            "cust_id": [1, 2, 3],
            "order_dt": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
            "total_amount": [100.0, 200.0, 300.0],
            "order_status": ["pending", "completed", "pending"],
        }
        right_df = pl.LazyFrame(right_data)
        invalid_df = "not a dataframe"

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with invalid output directory (try to create in a file location)
            import os
            invalid_path = os.path.join(temp_dir, "not_a_directory.txt")
            with open(invalid_path, "w") as f:
                f.write("this is a file, not a directory")

            with pytest.raises((ValueError, TypeError, OSError)):
                self.comparator.compare_and_export(
                    left=left_df,
                    right=right_df,
                    output_dir=invalid_path
                )

            # Test with invalid dataframe
            with pytest.raises(ValueError, match="left must be a polars LazyFrame"):
                self.comparator.compare_and_export(
                    left=invalid_df,
                    right=right_df,
                    output_dir=temp_dir
                )

    def test_compare_with_mismatched_schema(self) -> None:
        """Test compare method with mismatched schema."""
        # Create dataframes with different schemas
        left_df = pl.LazyFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"]
        })

        right_df = pl.LazyFrame({
            "customer_id": [1, 2, 3],  # Different column name
            "full_name": ["Alice", "Bob", "Charlie"]  # Different column name
        })

        # This should raise a SchemaValidationError due to missing columns
        with pytest.raises(SchemaValidationError):
            self.comparator.compare(left=left_df, right=right_df)


class TestComparisonReport:
    """Test ComparisonReport class functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures for ComparisonReport tests."""
        from splurge_lazyframe_compare.services.orchestrator import ComparisonOrchestrator

        # Create test schemas
        left_columns = {
            "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "order_date": ColumnDefinition(name="order_date", alias="Order Date", datatype=pl.Date, nullable=False),
            "amount": ColumnDefinition(name="amount", alias="Order Amount", datatype=pl.Float64, nullable=False),
            "status": ColumnDefinition(name="status", alias="Order Status", datatype=pl.Utf8, nullable=True),
        }
        right_columns = {
            "cust_id": ColumnDefinition(name="cust_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "order_dt": ColumnDefinition(name="order_dt", alias="Order Date", datatype=pl.Date, nullable=False),
            "total_amount": ColumnDefinition(name="total_amount", alias="Order Amount", datatype=pl.Float64, nullable=False),
            "order_status": ColumnDefinition(name="order_status", alias="Order Status", datatype=pl.Utf8, nullable=True),
        }

        left_schema = ComparisonSchema(
            columns=left_columns,
            pk_columns=["customer_id", "order_date"],
        )
        right_schema = ComparisonSchema(
            columns=right_columns,
            pk_columns=["cust_id", "order_dt"],
        )

        # Create column mappings
        mappings = [
            ColumnMapping(left="customer_id", right="cust_id", name="customer_id"),
            ColumnMapping(left="order_date", right="order_dt", name="order_date"),
            ColumnMapping(left="amount", right="total_amount", name="amount"),
            ColumnMapping(left="status", right="order_status", name="status"),
        ]

        # Create configuration
        config = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=mappings,
            pk_columns=["customer_id", "order_date"],
        )

        self.orchestrator = ComparisonOrchestrator()
        self.config = config

    def test_comparison_report_initialization(self) -> None:
        """Test ComparisonReport initialization."""
        from splurge_lazyframe_compare.core.comparator import ComparisonReport

        report = ComparisonReport(self.orchestrator, self.config)

        assert report.orchestrator == self.orchestrator
        assert report.config == self.config
        assert report._last_result is None

    def test_generate_from_result_comprehensive(self) -> None:
        """Test generate_from_result with various scenarios."""
        from splurge_lazyframe_compare.core.comparator import ComparisonReport

        # Create test data
        left_data = {
            "customer_id": [1, 2, 3],
            "order_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", "completed", "pending"],
        }
        right_data = {
            "cust_id": [1, 2, 4],
            "order_dt": ["2023-01-01", "2023-01-02", "2023-01-04"],
            "total_amount": [100.0, 250.0, 400.0],
            "order_status": ["pending", "completed", "shipped"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(
            pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d")
        )
        right_df = pl.LazyFrame(right_data).with_columns(
            pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # Execute comparison
        result = self.orchestrator.compare_dataframes(
            config=self.config,
            left=left_df,
            right=right_df
        )

        # Create report and generate from result
        report = ComparisonReport(self.orchestrator, self.config)
        report_str = report.generate_from_result(result)

        # Verify report content
        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in report_str
        assert "VALUE DIFFERENCES" in report_str
        assert "LEFT-ONLY RECORDS" in report_str
        assert "RIGHT-ONLY RECORDS" in report_str
        assert report._last_result == result

    def test_generate_summary_report_with_last_result(self) -> None:
        """Test generate_summary_report using last result."""
        from splurge_lazyframe_compare.core.comparator import ComparisonReport

        # Create test data with all required columns
        left_data = {
            "customer_id": [1, 2, 3],
            "order_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", "completed", "pending"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(
            pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # Create right data with correct column names for schema
        right_data = {
            "cust_id": [1, 2, 3],
            "order_dt": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "total_amount": [100.0, 200.0, 300.0],
            "order_status": ["pending", "completed", "pending"],
        }

        right_df = pl.LazyFrame(right_data).with_columns(
            pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # Execute comparison
        result = self.orchestrator.compare_dataframes(
            config=self.config,
            left=left_df,
            right=right_df
        )

        # Create report and generate from result first
        report = ComparisonReport(self.orchestrator, self.config)
        report.generate_from_result(result)

        # Now generate summary report (should use last result)
        summary_report = report.generate_summary_report()

        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in summary_report
        assert "3" in summary_report  # All records match

    def test_generate_summary_report_with_explicit_result(self) -> None:
        """Test generate_summary_report with explicit result parameter."""
        from splurge_lazyframe_compare.core.comparator import ComparisonReport

        # Create test data with all required columns
        left_data = {
            "customer_id": [1, 2],
            "order_date": ["2023-01-01", "2023-01-02"],
            "amount": [100.0, 200.0],
            "status": ["pending", "completed"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(
            pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # Create right data with correct column names for schema
        right_data = {
            "cust_id": [1, 2],
            "order_dt": ["2023-01-01", "2023-01-02"],
            "total_amount": [100.0, 200.0],
            "order_status": ["pending", "completed"],
        }

        right_df = pl.LazyFrame(right_data).with_columns(
            pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # Execute comparison
        result = self.orchestrator.compare_dataframes(
            config=self.config,
            left=left_df,
            right=right_df
        )

        # Create report and generate summary with explicit result
        report = ComparisonReport(self.orchestrator, self.config)
        summary_report = report.generate_summary_report(result=result)

        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in summary_report
        assert "2" in summary_report  # All records match

    def test_generate_summary_report_no_result_error(self) -> None:
        """Test generate_summary_report raises error when no result available."""
        from splurge_lazyframe_compare.core.comparator import ComparisonReport

        report = ComparisonReport(self.orchestrator, self.config)

        with pytest.raises(ValueError, match="No comparison result available"):
            report.generate_summary_report()

    def test_generate_detailed_report_comprehensive(self) -> None:
        """Test generate_detailed_report with various parameters."""
        from splurge_lazyframe_compare.core.comparator import ComparisonReport

        # Create test data with differences
        left_data = {
            "customer_id": [1, 2, 3],
            "order_date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "amount": [100.0, 200.0, 300.0],
            "status": ["pending", "completed", "pending"],
        }
        right_data = {
            "cust_id": [1, 2, 4],
            "order_dt": ["2023-01-01", "2023-01-02", "2023-01-04"],
            "total_amount": [100.0, 250.0, 400.0],
            "order_status": ["pending", "completed", "shipped"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(
            pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d")
        )
        right_df = pl.LazyFrame(right_data).with_columns(
            pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # Execute comparison
        result = self.orchestrator.compare_dataframes(
            config=self.config,
            left=left_df,
            right=right_df
        )

        # Create report
        report = ComparisonReport(self.orchestrator, self.config)

        # Test with default parameters
        detailed_report = report.generate_detailed_report(result=result)
        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in detailed_report
        assert "VALUE DIFFERENCES" in detailed_report

        # Test with custom parameters
        detailed_report_custom = report.generate_detailed_report(
            result=result,
            max_samples=2,
            table_format="simple"
        )
        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in detailed_report_custom

    def test_generate_detailed_report_no_result_error(self) -> None:
        """Test generate_detailed_report raises error when no result available."""
        from splurge_lazyframe_compare.core.comparator import ComparisonReport

        report = ComparisonReport(self.orchestrator, self.config)

        with pytest.raises(ValueError, match="No comparison result available"):
            report.generate_detailed_report()

    def test_generate_summary_table_comprehensive(self) -> None:
        """Test generate_summary_table method."""
        from splurge_lazyframe_compare.core.comparator import ComparisonReport

        # Create test data with all required columns
        left_data = {
            "customer_id": [1, 2],
            "order_date": ["2023-01-01", "2023-01-02"],
            "amount": [100.0, 200.0],
            "status": ["pending", "completed"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(
            pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # Create right data with correct column names for schema
        right_data = {
            "cust_id": [1, 2],
            "order_dt": ["2023-01-01", "2023-01-02"],
            "total_amount": [100.0, 200.0],
            "order_status": ["pending", "completed"],
        }

        right_df = pl.LazyFrame(right_data).with_columns(
            pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # Execute comparison
        result = self.orchestrator.compare_dataframes(
            config=self.config,
            left=left_df,
            right=right_df
        )

        # Create report and generate summary table
        report = ComparisonReport(self.orchestrator, self.config)
        table_report = report.generate_summary_table(result=result)

        assert "Left DataFrame" in table_report
        assert "2" in table_report  # Record counts

    def test_export_to_html_comprehensive(self) -> None:
        """Test export_to_html method with various scenarios."""
        from splurge_lazyframe_compare.core.comparator import ComparisonReport
        import tempfile

        # Create test data with all required columns
        left_data = {
            "customer_id": [1, 2],
            "order_date": ["2023-01-01", "2023-01-02"],
            "amount": [100.0, 200.0],
            "status": ["pending", "completed"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(
            pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # Create right data with correct column names for schema
        right_data = {
            "cust_id": [1, 2],
            "order_dt": ["2023-01-01", "2023-01-02"],
            "total_amount": [100.0, 200.0],
            "order_status": ["pending", "completed"],
        }

        right_df = pl.LazyFrame(right_data).with_columns(
            pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # Execute comparison
        result = self.orchestrator.compare_dataframes(
            config=self.config,
            left=left_df,
            right=right_df
        )

        # Create report
        report = ComparisonReport(self.orchestrator, self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            html_file = f"{temp_dir}/test_report.html"

            # Test export with explicit result
            report.export_to_html(result=result, filename=html_file)

            # Verify file was created
            from pathlib import Path
            assert Path(html_file).exists()
            assert Path(html_file).stat().st_size > 0

            # Test export using last result (should work after generate_from_result)
            report.generate_from_result(result)
            html_file2 = f"{temp_dir}/test_report2.html"
            report.export_to_html(filename=html_file2)

            assert Path(html_file2).exists()
            assert Path(html_file2).stat().st_size > 0

    def test_export_to_html_no_result_error(self) -> None:
        """Test export_to_html raises error when no result available."""
        from splurge_lazyframe_compare.core.comparator import ComparisonReport
        import tempfile

        report = ComparisonReport(self.orchestrator, self.config)

        with tempfile.TemporaryDirectory() as temp_dir:
            html_file = f"{temp_dir}/test_report.html"

            with pytest.raises(ValueError, match="No comparison result available"):
                report.export_to_html(filename=html_file)

    def test_report_state_management(self) -> None:
        """Test that report properly manages internal state."""
        from splurge_lazyframe_compare.core.comparator import ComparisonReport

        # Create test data
        left_data = {
            "customer_id": [1, 2],
            "order_date": ["2023-01-01", "2023-01-02"],
            "amount": [100.0, 200.0],
            "status": ["pending", "completed"],
        }

        # Create right data with different column names but same data
        right_data = {
            "cust_id": [1, 2],
            "order_dt": ["2023-01-01", "2023-01-02"],
            "total_amount": [100.0, 200.0],
            "order_status": ["pending", "completed"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(
            pl.col("order_date").str.strptime(pl.Date, "%Y-%m-%d")
        )
        right_df = pl.LazyFrame(right_data).with_columns(
            pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d")
        )

        # Execute two different comparisons
        result1 = self.orchestrator.compare_dataframes(
            config=self.config,
            left=left_df,
            right=right_df
        )

        # Modify data for second comparison
        right_data2 = right_data.copy()
        right_data2["total_amount"] = [100.0, 250.0]  # Change one amount
        right_df2 = pl.LazyFrame(right_data2).with_columns(
            pl.col("order_dt").str.strptime(pl.Date, "%Y-%m-%d")
        )

        result2 = self.orchestrator.compare_dataframes(
            config=self.config,
            left=left_df,
            right=right_df2
        )

        # Create report and test state management
        report = ComparisonReport(self.orchestrator, self.config)

        # Generate from first result
        report.generate_from_result(result1)
        assert report._last_result == result1

        # Generate summary (should use result1)
        summary1 = report.generate_summary_report()
        assert "2" in summary1  # Both records match

        # Generate from second result
        report.generate_from_result(result2)
        assert report._last_result == result2

        # Generate summary (should now use result2)
        summary2 = report.generate_summary_report()
        assert "Value Differences" in summary2  # Should show differences