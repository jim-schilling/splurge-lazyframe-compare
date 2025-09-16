"""End-to-end tests for complete user workflows."""

import tempfile
from pathlib import Path

import polars as pl
import pytest

from splurge_lazyframe_compare import (
    ColumnDefinition,
    ColumnMapping,
    ComparisonConfig,
    ComparisonSchema,
    LazyFrameComparator,
)


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""

    def test_simple_comparison_scenario(self):
        """Test a simple customer data comparison scenario."""
        # Create test data - customer information from two systems
        left_data = {
            "customer_id": [1, 2, 3, 4],
            "name": ["Alice Johnson", "Bob Smith", "Charlie Brown", "David Wilson"],
            "email": ["alice@example.com", "bob@example.com", "charlie@example.com", "david@example.com"],
            "balance": [1250.50, 890.25, 2100.75, 450.00],
        }

        # Right system has some differences
        right_data = {
            "cust_id": [1, 2, 3, 5],  # Different column name, missing customer 4, extra customer 5
            "full_name": ["Alice Johnson", "Bob Smith", "Charlie Brown", "Eve Davis"],  # Different column name
            "email_addr": ["alice@example.com", "bob@example.com", "charlie@example.com", "eve@example.com"],
            "account_balance": [1250.50, 950.25, 2100.75, 750.00],  # Bob's balance changed
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        # Step 1: Set up configuration
        # Define schemas
        left_columns = {
            "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "name": ColumnDefinition(name="name", alias="Customer Name", datatype=pl.Utf8, nullable=False),
            "email": ColumnDefinition(name="email", alias="Email Address", datatype=pl.Utf8, nullable=False),
            "balance": ColumnDefinition(name="balance", alias="Account Balance", datatype=pl.Float64, nullable=False),
        }
        left_schema = ComparisonSchema(columns=left_columns, pk_columns=["customer_id"])

        right_columns = {
            "cust_id": ColumnDefinition(name="cust_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "full_name": ColumnDefinition(name="full_name", alias="Customer Name", datatype=pl.Utf8, nullable=False),
            "email_addr": ColumnDefinition(name="email_addr", alias="Email Address", datatype=pl.Utf8, nullable=False),
            "account_balance": ColumnDefinition(
                name="account_balance", alias="Account Balance", datatype=pl.Float64, nullable=False
            ),
        }
        right_schema = ComparisonSchema(columns=right_columns, pk_columns=["cust_id"])

        # Define mappings
        mappings = [
            ColumnMapping(left="customer_id", right="cust_id", name="customer_id"),
            ColumnMapping(left="name", right="full_name", name="name"),
            ColumnMapping(left="email", right="email_addr", name="email"),
            ColumnMapping(left="balance", right="account_balance", name="balance"),
        ]

        config = ComparisonConfig(
            left_schema=left_schema, right_schema=right_schema, column_mappings=mappings, pk_columns=["customer_id"]
        )

        # Step 2: Execute comparison
        comparator = LazyFrameComparator(config)
        result = comparator.compare(left=left_df, right=right_df)

        # Step 3: Validate results
        assert result.summary.total_left_records == 4
        assert result.summary.total_right_records == 4
        assert result.summary.matching_records == 2  # Alice and Charlie match exactly
        assert result.summary.value_differences_count == 1  # Bob's balance differs
        assert result.summary.left_only_count == 1  # David missing from right
        assert result.summary.right_only_count == 1  # Eve added to right

        # Step 4: Generate report
        report = comparator.compare_and_report(left=left_df, right=right_df)
        assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in report
        assert "4" in report  # Record counts appear twice (left and right)

        # Step 5: Export results
        with tempfile.TemporaryDirectory() as temp_dir:
            exported_files = comparator.compare_and_export(
                left=left_df, right=right_df, output_dir=temp_dir, format="csv"
            )

            assert len(exported_files) == 4  # value_differences, left_only, right_only, summary

            # Verify files exist and have content
            for _, file_path in exported_files.items():
                assert Path(file_path).exists()
                assert Path(file_path).stat().st_size > 0

    def test_complex_data_types_scenario(self):
        """Test scenario with complex data types."""
        # Create test data with dates, nested structures, and various types
        left_data = {
            "id": [1, 2, 3],
            "name": ["Product A", "Product B", "Product C"],
            "price": [29.99, 49.99, 19.99],
            "in_stock": [True, False, True],
            "categories": [["electronics", "gadgets"], ["books", "education"], ["clothing", "accessories"]],
            "created_date": ["2024-01-15", "2024-02-20", "2024-03-10"],
            "metadata": [
                {"brand": "BrandA", "warranty": 12},
                {"brand": "BrandB", "warranty": 24},
                {"brand": "BrandC", "warranty": 6},
            ],
        }

        right_data = {
            "product_id": [1, 2, 4],  # Different column name, different products
            "product_name": ["Product A", "Product B", "Product D"],
            "cost": [29.99, 52.99, 25.00],  # Price changed for Product B
            "available": [True, False, True],
            "tags": [["electronics", "gadgets"], ["books", "education"], ["home", "garden"]],  # Different structure
            "release_date": ["2024-01-15", "2024-02-20", "2024-04-01"],
        }

        left_df = pl.LazyFrame(left_data).with_columns(pl.col("created_date").str.strptime(pl.Date, "%Y-%m-%d"))
        right_df = pl.LazyFrame(right_data).with_columns(pl.col("release_date").str.strptime(pl.Date, "%Y-%m-%d"))

        # Create configuration
        from splurge_lazyframe_compare import (
            ColumnDefinition,
            ColumnMapping,
            ComparisonConfig,
            ComparisonOrchestrator,
            ComparisonSchema,
        )

        left_columns = {
            "id": ColumnDefinition(name="id", alias="Product ID", datatype=pl.Int64, nullable=False),
            "name": ColumnDefinition(name="name", alias="Product Name", datatype=pl.Utf8, nullable=False),
            "price": ColumnDefinition(name="price", alias="Price", datatype=pl.Float64, nullable=False),
            "in_stock": ColumnDefinition(name="in_stock", alias="In Stock", datatype=pl.Boolean, nullable=False),
            "created_date": ColumnDefinition(
                name="created_date", alias="Created Date", datatype=pl.Date, nullable=False
            ),
        }
        left_schema = ComparisonSchema(columns=left_columns, pk_columns=["id"])

        right_columns = {
            "product_id": ColumnDefinition(name="product_id", alias="Product ID", datatype=pl.Int64, nullable=False),
            "product_name": ColumnDefinition(
                name="product_name", alias="Product Name", datatype=pl.Utf8, nullable=False
            ),
            "cost": ColumnDefinition(name="cost", alias="Cost", datatype=pl.Float64, nullable=False),
            "available": ColumnDefinition(name="available", alias="Available", datatype=pl.Boolean, nullable=False),
            "release_date": ColumnDefinition(
                name="release_date", alias="Release Date", datatype=pl.Date, nullable=False
            ),
        }
        right_schema = ComparisonSchema(columns=right_columns, pk_columns=["product_id"])

        mappings = [
            ColumnMapping(left="id", right="product_id", name="id"),
            ColumnMapping(left="name", right="product_name", name="name"),
            ColumnMapping(left="price", right="cost", name="price"),
            ColumnMapping(left="in_stock", right="available", name="in_stock"),
            ColumnMapping(left="created_date", right="release_date", name="created_date"),
        ]

        config = ComparisonConfig(
            left_schema=left_schema, right_schema=right_schema, column_mappings=mappings, pk_columns=["id"]
        )

        # Execute comparison
        orchestrator = ComparisonOrchestrator()
        result = orchestrator.compare_dataframes(config=config, left=left_df, right=right_df)

        # Validate results
        assert result.summary.total_left_records == 3
        assert result.summary.total_right_records == 3
        assert result.summary.matching_records == 1  # Product A matches exactly
        assert result.summary.value_differences_count == 1  # Product B price changed
        assert result.summary.left_only_count == 1  # Product C missing from right
        assert result.summary.right_only_count == 1  # Product D added to right

    def test_large_dataset_performance_scenario(self):
        """Test performance with larger dataset."""
        # Create larger dataset
        import random

        num_records = 1000

        left_data = {
            "id": list(range(1, num_records + 1)),
            "value": [random.uniform(0, 1000) for _ in range(num_records)],
            "category": [f"Category_{random.randint(1, 10)}" for _ in range(num_records)],
            "active": [random.choice([True, False]) for _ in range(num_records)],
        }

        # Right dataset with some differences
        right_ids = list(range(1, num_records)) + [num_records + 1]  # Missing last ID, add new one
        right_data = {
            "customer_id": right_ids,
            "amount": [random.uniform(0, 1000) for _ in range(num_records)],
            "type": [f"Type_{random.randint(1, 5)}" for _ in range(num_records)],
            "enabled": [random.choice([True, False]) for _ in range(num_records)],
        }

        left_df = pl.LazyFrame(left_data)
        right_df = pl.LazyFrame(right_data)

        # Create configuration
        from splurge_lazyframe_compare import (
            ColumnDefinition,
            ColumnMapping,
            ComparisonConfig,
            ComparisonOrchestrator,
            ComparisonSchema,
        )

        left_columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "value": ColumnDefinition(name="value", alias="Value", datatype=pl.Float64, nullable=False),
            "category": ColumnDefinition(name="category", alias="Category", datatype=pl.Utf8, nullable=False),
            "active": ColumnDefinition(name="active", alias="Active", datatype=pl.Boolean, nullable=False),
        }
        left_schema = ComparisonSchema(columns=left_columns, pk_columns=["id"])

        right_columns = {
            "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "amount": ColumnDefinition(name="amount", alias="Amount", datatype=pl.Float64, nullable=False),
            "type": ColumnDefinition(name="type", alias="Type", datatype=pl.Utf8, nullable=False),
            "enabled": ColumnDefinition(name="enabled", alias="Enabled", datatype=pl.Boolean, nullable=False),
        }
        right_schema = ComparisonSchema(columns=right_columns, pk_columns=["customer_id"])

        mappings = [
            ColumnMapping(left="id", right="customer_id", name="id"),
            ColumnMapping(left="value", right="amount", name="value"),
            ColumnMapping(left="category", right="type", name="category"),
            ColumnMapping(left="active", right="enabled", name="active"),
        ]

        config = ComparisonConfig(
            left_schema=left_schema, right_schema=right_schema, column_mappings=mappings, pk_columns=["id"]
        )

        # Execute comparison - should handle larger dataset efficiently
        orchestrator = ComparisonOrchestrator()
        result = orchestrator.compare_dataframes(config=config, left=left_df, right=right_df)

        # Validate results
        assert result.summary.total_left_records == num_records
        assert result.summary.total_right_records == num_records
        assert result.summary.matching_records >= 0
        assert result.summary.left_only_count >= 0
        assert result.summary.right_only_count >= 0

    def test_configuration_management_scenario(self):
        """Test complete configuration management workflow."""
        # Step 1: Create configuration manually
        import polars as pl

        from splurge_lazyframe_compare.models.schema import (
            ColumnDefinition,
            ColumnMapping,
            ComparisonConfig,
            ComparisonSchema,
        )

        # Create test DataFrames
        left_df = pl.LazyFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "score": [95.5, 87.2, 91.8]})

        right_df = pl.LazyFrame(
            {
                "student_id": [1, 2, 4],
                "student_name": ["Alice", "Bob", "David"],
                "grade": [95.5, 89.2, 88.5],  # Bob's grade changed
            }
        )

        # Create schemas
        left_columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=False),
            "score": ColumnDefinition(name="score", alias="Score", datatype=pl.Float64, nullable=False),
        }
        left_schema = ComparisonSchema(columns=left_columns, pk_columns=["id"])

        right_columns = {
            "student_id": ColumnDefinition(name="student_id", alias="Student ID", datatype=pl.Int64, nullable=False),
            "student_name": ColumnDefinition(
                name="student_name", alias="Student Name", datatype=pl.Utf8, nullable=False
            ),
            "grade": ColumnDefinition(name="grade", alias="Grade", datatype=pl.Float64, nullable=False),
        }
        right_schema = ComparisonSchema(columns=right_columns, pk_columns=["student_id"])

        # Create column mappings
        column_mappings = [
            ColumnMapping(left="id", right="student_id", name="id"),
            ColumnMapping(left="name", right="student_name", name="name"),
            ColumnMapping(left="score", right="grade", name="score"),
        ]

        config = ComparisonConfig(
            left_schema=left_schema,
            right_schema=right_schema,
            column_mappings=column_mappings,
            pk_columns=["id"],
            tolerance={},
        )

        # Convert config to dictionary for JSON serialization
        config_dict = {
            "primary_key_columns": config.pk_columns,
            "column_mappings": [mapping.__dict__ for mapping in config.column_mappings],
            "ignore_case": config.ignore_case,
            "null_equals_null": config.null_equals_null,
            "tolerance": config.tolerance,
        }

        # Step 2: Validate configuration
        from splurge_lazyframe_compare.utils import validate_config

        # Convert config to dictionary for validation
        config_for_validation = {
            "primary_key_columns": config.pk_columns,
            "column_mappings": [mapping.__dict__ for mapping in config.column_mappings],
            "ignore_case": config.ignore_case,
            "null_equals_null": config.null_equals_null,
            "tolerance": config.tolerance,
        }
        errors = validate_config(config_for_validation)
        assert len(errors) == 0

        # Step 3: Save configuration
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            import json

            json.dump(config_dict, f)
            config_file = f.name

        # Step 4: Load configuration
        try:
            from splurge_lazyframe_compare.utils import load_config_from_file

            loaded_config = load_config_from_file(config_file)

            # Verify config was loaded correctly
            assert loaded_config is not None
            assert isinstance(loaded_config, dict)
            assert len(loaded_config) > 0

            # Step 5: Convert loaded config back to ComparisonConfig and use it
            from splurge_lazyframe_compare import ComparisonConfig, ComparisonOrchestrator

            # The loaded config is already in the right format, just use it directly
            # But ComparisonConfig expects schema objects, not dictionaries
            # So we need to reconstruct them
            from splurge_lazyframe_compare.models.schema import ComparisonSchema

            # Create minimal schemas from DataFrames since we don't save full schema info
            left_schema_obj = left_df.collect_schema()
            right_schema_obj = right_df.collect_schema()

            left_schema = ComparisonSchema(
                columns={
                    col: ColumnDefinition(name=col, alias=col, datatype=dtype, nullable=True)
                    for col, dtype in zip(left_schema_obj.names(), left_schema_obj.dtypes(), strict=False)
                },
                pk_columns=["id"],
            )

            right_schema = ComparisonSchema(
                columns={
                    col: ColumnDefinition(name=col, alias=col, datatype=dtype, nullable=True)
                    for col, dtype in zip(right_schema_obj.names(), right_schema_obj.dtypes(), strict=False)
                },
                pk_columns=["student_id"],
            )

            # Reconstruct column mappings from loaded config
            column_mappings = [ColumnMapping(**mapping) for mapping in loaded_config["column_mappings"]]

            loaded_comparison_config = ComparisonConfig(
                left_schema=left_schema,
                right_schema=right_schema,
                column_mappings=column_mappings,
                pk_columns=loaded_config["primary_key_columns"],
                ignore_case=loaded_config.get("ignore_case", False),
                null_equals_null=loaded_config.get("null_equals_null", True),
                tolerance=loaded_config.get("tolerance", {}),
            )

            orchestrator = ComparisonOrchestrator()
            result = orchestrator.compare_dataframes(config=loaded_comparison_config, left=left_df, right=right_df)

            assert result.summary.total_left_records == 3
            assert result.summary.total_right_records == 3

        finally:
            # Clean up
            Path(config_file).unlink()

    def test_error_recovery_scenario(self):
        """Test error handling and recovery in complete workflow."""
        # Create DataFrame with issues
        bad_df = pl.LazyFrame(
            {
                "id": [1, 2, None],  # NULL in primary key
                "name": ["Alice", "Bob", None],  # NULL in required field
            }
        )

        left_good_df = pl.LazyFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
            }
        )

        right_good_df = pl.LazyFrame(
            {
                "customer_id": [1, 2, 3],
                "customer_name": ["Alice", "Bob", "Charlie"],
            }
        )

        # Create configuration
        from splurge_lazyframe_compare import (
            ColumnDefinition,
            ColumnMapping,
            ComparisonConfig,
            ComparisonOrchestrator,
            ComparisonSchema,
        )

        left_columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),  # NOT NULL
            "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=False),  # NOT NULL
        }
        left_schema = ComparisonSchema(columns=left_columns, pk_columns=["id"])

        right_columns = {
            "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "customer_name": ColumnDefinition(
                name="customer_name", alias="Customer Name", datatype=pl.Utf8, nullable=False
            ),
        }
        right_schema = ComparisonSchema(columns=right_columns, pk_columns=["customer_id"])

        mappings = [
            ColumnMapping(left="id", right="customer_id", name="id"),
            ColumnMapping(left="name", right="customer_name", name="name"),
        ]

        config = ComparisonConfig(
            left_schema=left_schema, right_schema=right_schema, column_mappings=mappings, pk_columns=["id"]
        )

        # Test error handling
        orchestrator = ComparisonOrchestrator()

        # Should raise validation error due to NULL values
        from splurge_lazyframe_compare.exceptions.comparison_exceptions import SchemaValidationError

        with pytest.raises(SchemaValidationError):
            orchestrator.compare_dataframes(config=config, left=bad_df, right=right_good_df)

        # Should work with valid data
        result = orchestrator.compare_dataframes(config=config, left=left_good_df, right=right_good_df)

        assert result.summary.matching_records == 3  # All should match

    def test_custom_service_integration_scenario(self):
        """Test integrating custom services into the workflow."""
        # Create test data
        left_df = pl.LazyFrame({"id": [1, 2, 3], "value": [100, 200, 300]})

        right_df = pl.LazyFrame(
            {
                "customer_id": [1, 2, 3],
                "amount": [100, 250, 300],  # Second value different
            }
        )

        # Create configuration
        from splurge_lazyframe_compare import ColumnDefinition, ColumnMapping, ComparisonConfig, ComparisonSchema

        left_columns = {
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "value": ColumnDefinition(name="value", alias="Value", datatype=pl.Int64, nullable=False),
        }
        left_schema = ComparisonSchema(columns=left_columns, pk_columns=["id"])

        right_columns = {
            "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
            "amount": ColumnDefinition(name="amount", alias="Amount", datatype=pl.Int64, nullable=False),
        }
        right_schema = ComparisonSchema(columns=right_columns, pk_columns=["customer_id"])

        mappings = [
            ColumnMapping(left="id", right="customer_id", name="id"),
            ColumnMapping(left="value", right="amount", name="value"),
        ]

        config = ComparisonConfig(
            left_schema=left_schema, right_schema=right_schema, column_mappings=mappings, pk_columns=["id"]
        )

        # Create custom services
        from splurge_lazyframe_compare.services import (
            ComparisonService,
            DataPreparationService,
            ReportingService,
            ValidationService,
        )

        # Custom validation service with special logic
        class CustomValidationService(ValidationService):
            def validate_dataframe_schema(self, *, df, schema, df_name):
                # Custom validation logic
                return super().validate_dataframe_schema(df=df, schema=schema, df_name=df_name)

        # Custom reporting service
        class CustomReportingService(ReportingService):
            def generate_summary_report(self, *, results):
                # Custom report formatting
                report = super().generate_summary_report(results=results)
                return f"CUSTOM REPORT:\n{report}"

        # Inject custom services
        custom_validation = CustomValidationService()
        custom_reporting = CustomReportingService()
        custom_preparation = DataPreparationService()

        comparison_service = ComparisonService(
            validation_service=custom_validation, preparation_service=custom_preparation
        )

        from splurge_lazyframe_compare import ComparisonOrchestrator

        orchestrator = ComparisonOrchestrator(comparison_service=comparison_service, reporting_service=custom_reporting)

        # Execute with custom services
        result = orchestrator.compare_dataframes(config=config, left=left_df, right=right_df)

        # Generate custom report
        report = orchestrator.generate_report_from_result(result=result, report_type="summary")

        # Validate results
        assert result.summary.total_left_records == 3
        assert result.summary.total_right_records == 3
        assert result.summary.value_differences_count == 1  # One value differs
        assert "CUSTOM REPORT:" in report  # Custom reporting worked
