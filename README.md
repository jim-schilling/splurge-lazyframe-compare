# Splurge LazyFrame Comparison Framework

A comprehensive Python framework for comparing two Polars LazyFrames with configurable schemas, primary keys, and column mappings. The framework identifies value differences, missing records, and provides detailed comparison reports.

## Features

- **Multi-column primary key support** - Compare datasets using composite keys
- **Flexible schema definition** - Define friendly names and data types for columns
- **Column-to-column mapping** - Map columns between datasets with different naming conventions
- **Three core comparison patterns**:
  - Value differences (same keys, different values)
  - Left-only records (records only in left dataset)
  - Right-only records (records only in right dataset)
- **Comprehensive reporting** - Generate summary and detailed reports
- **Type-safe implementation** - Full type annotations and validation
- **Performance optimized** - Leverages Polars' lazy evaluation for memory efficiency
- **Export capabilities** - Export results to CSV, Parquet, JSON, and HTML formats

## Installation

```bash
pip install splurge-lazyframe-compare
```

## Quick Start

```python
import polars as pl
from datetime import date
from splurge_lazyframe_compare import (
    LazyFrameComparator,
    ComparisonConfig,
    ComparisonSchema,
    ColumnDefinition,
    ColumnMapping,
)

# Define schemas
left_schema = ComparisonSchema(
    columns={
        "customer_id": ColumnDefinition("customer_id", "Customer ID", pl.Int64, False),
        "order_date": ColumnDefinition("order_date", "Order Date", pl.Date, False),
        "amount": ColumnDefinition("amount", "Order Amount", pl.Float64, False),
        "status": ColumnDefinition("status", "Order Status", pl.Utf8, True),
    },
    primary_key_columns=["customer_id", "order_date"]
)

right_schema = ComparisonSchema(
    columns={
        "cust_id": ColumnDefinition("cust_id", "Customer ID", pl.Int64, False),
        "order_dt": ColumnDefinition("order_dt", "Order Date", pl.Date, False),
        "total_amount": ColumnDefinition("total_amount", "Order Amount", pl.Float64, False),
        "order_status": ColumnDefinition("order_status", "Order Status", pl.Utf8, True),
    },
    primary_key_columns=["cust_id", "order_dt"]
)

# Define column mappings
mappings = [
    ColumnMapping("customer_id", "cust_id", "customer_id"),
    ColumnMapping("order_date", "order_dt", "order_date"),
    ColumnMapping("amount", "total_amount", "amount"),
    ColumnMapping("status", "order_status", "status"),
]

# Create configuration
config = ComparisonConfig(
    left_schema=left_schema,
    right_schema=right_schema,
    column_mappings=mappings,
    primary_key_columns=["customer_id", "order_date"]
)

# Create sample data
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

# Execute comparison
comparator = LazyFrameComparator(config)
results = comparator.compare(left=left_df, right=right_df)

# Generate report
report = results.to_report()
print(report.generate_summary_report())
```

## Advanced Usage

### Numeric Tolerance

```python
# Allow small differences in numeric columns
config = ComparisonConfig(
    left_schema=left_schema,
    right_schema=right_schema,
    column_mappings=mappings,
    primary_key_columns=["customer_id", "order_date"],
    tolerance={"amount": 0.01}  # Allow 1 cent difference
)
```

### Case-Insensitive String Comparison

```python
# Ignore case in string comparisons
config = ComparisonConfig(
    left_schema=left_schema,
    right_schema=right_schema,
    column_mappings=mappings,
    primary_key_columns=["customer_id", "order_date"],
    ignore_case=True
)
```

### Null Value Handling

```python
# Configure null value comparison behavior
config = ComparisonConfig(
    left_schema=left_schema,
    right_schema=right_schema,
    column_mappings=mappings,
    primary_key_columns=["customer_id", "order_date"],
    null_equals_null=True  # Treat null values as equal
)
```

### Export Results

```python
# Export comparison results to files
exported_files = results.export_results(
    format="csv",
    output_dir="./comparison_results"
)

# Generate HTML report
report.export_to_html("./comparison_results/report.html")
```

## API Reference

### Core Classes

#### `ComparisonSchema`
Defines the structure and constraints for a dataset.

```python
schema = ComparisonSchema(
    columns={
        "id": ColumnDefinition("id", "ID", pl.Int64, False),
        "name": ColumnDefinition("name", "Name", pl.Utf8, True),
    },
    primary_key_columns=["id"]
)
```

#### `ColumnDefinition`
Defines a column with metadata for comparison.

```python
col_def = ColumnDefinition(
    column_name="customer_id",
    friendly_name="Customer ID",
    polars_dtype=pl.Int64,
    nullable=False
)
```

#### `ColumnMapping`
Maps columns between left and right datasets.

```python
mapping = ColumnMapping(
    left_column="customer_id",
    right_column="cust_id",
    comparison_name="customer_id"
)
```

#### `ComparisonConfig`
Configuration for comparing two datasets.

```python
config = ComparisonConfig(
    left_schema=left_schema,
    right_schema=right_schema,
    column_mappings=mappings,
    primary_key_columns=["customer_id", "order_date"],
    ignore_case=False,
    null_equals_null=True,
    tolerance={"amount": 0.01}
)
```

#### `LazyFrameComparator`
Main comparison engine.

```python
comparator = LazyFrameComparator(config)
results = comparator.compare(left=left_df, right=right_df)
```

### Results and Reporting

#### `ComparisonResults`
Container for all comparison results.

```python
# Access summary statistics
print(f"Matching records: {results.summary.matching_records}")
print(f"Value differences: {results.summary.value_differences_count}")
print(f"Left-only records: {results.summary.left_only_count}")
print(f"Right-only records: {results.summary.right_only_count}")

# Access result DataFrames
value_diffs = results.value_differences.collect()
left_only = results.left_only_records.collect()
right_only = results.right_only_records.collect()
```

#### `ComparisonReport`
Generate human-readable reports.

```python
report = results.to_report()

# Summary report
print(report.generate_summary_report())

# Detailed report with samples
print(report.generate_detailed_report(max_samples=10))

# Export to HTML
report.export_to_html("comparison_report.html")
```

## Data Quality Validation

The framework includes data quality validation capabilities:

```python
from splurge_lazyframe_compare.validators import DataQualityValidator

validator = DataQualityValidator()

# Validate completeness
result = validator.validate_completeness(
    df=df,
    required_columns=["customer_id", "amount"]
)

# Validate data types
result = validator.validate_data_types(
    df=df,
    expected_types={"customer_id": pl.Int64, "amount": pl.Float64}
)

# Validate numeric ranges
result = validator.validate_numeric_ranges(
    df=df,
    column_ranges={"amount": {"min": 0, "max": 10000}}
)

# Run comprehensive validation
results = validator.run_comprehensive_validation(
    df=df,
    required_columns=["customer_id", "amount"],
    expected_types={"customer_id": pl.Int64, "amount": pl.Float64},
    column_ranges={"amount": {"min": 0}},
    unique_columns=["customer_id"]
)
```

## Error Handling

The framework provides custom exceptions for different error scenarios:

```python
from splurge_lazyframe_compare.exceptions import (
    SchemaValidationError,
    PrimaryKeyViolationError,
    ColumnMappingError,
)

try:
    results = comparator.compare(left=left_df, right=right_df)
except SchemaValidationError as e:
    print(f"Schema validation failed: {e.message}")
    for error in e.validation_errors:
        print(f"  - {error}")
except PrimaryKeyViolationError as e:
    print(f"Primary key violation: {e.message}")
except ColumnMappingError as e:
    print(f"Column mapping error: {e.message}")
```

## Examples

See the `examples/` directory for complete working examples:

- `basic_comparison_example.py` - Basic usage demonstration
- Additional examples coming soon

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=splurge_lazyframe_compare

# Run specific test file
pytest tests/test_comparator.py
```

### Code Quality

```bash
# Run linting
ruff check .

# Run formatting
ruff format .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Requirements

- Python 3.10+
- Polars >= 1.32.0
- tabulate >= 0.9.0 (for reporting)
