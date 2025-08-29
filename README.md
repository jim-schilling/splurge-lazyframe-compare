# Splurge LazyFrame Comparison Framework

A comprehensive Python framework for comparing two Polars LazyFrames with configurable schemas, primary keys, and column mappings. The framework identifies value differences, missing records, and provides detailed comparison reports.

## Features

- **Service-oriented architecture** - Modular design with separate services for comparison, validation, reporting, and data preparation
- **Multi-column primary key support** - Compare datasets using composite keys
- **Flexible schema definition** - Define friendly names and data types for columns using either direct Polars datatypes or human-readable string names
- **Column-to-column mapping** - Map columns between datasets with different naming conventions
- **Three core comparison patterns**:
  - Value differences (same keys, different values)
  - Left-only records (records only in left dataset)
  - Right-only records (records only in right dataset)
- **Comprehensive reporting** - Generate summary and detailed reports with multiple table formats
- **Data validation** - Built-in data quality validation capabilities
- **Type-safe implementation** - Full type annotations and validation
- **Performance optimized** - Leverages Polars' lazy evaluation for memory efficiency
- **Export capabilities** - Export results to CSV, Parquet, and JSON formats
- **Production-ready logging** - Structured logging with Python's logging module, configurable log levels, and performance monitoring
- **Error handling** - Robust exception handling with custom exceptions and graceful error recovery

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

# Define schemas (demonstrating mixed datatype usage)
left_schema = ComparisonSchema(
    columns={
        "customer_id": ColumnDefinition("customer_id", "Customer ID", "Int64", False),  # String name
        "order_date": ColumnDefinition("order_date", "Order Date", pl.Date, False),     # Direct datatype
        "amount": ColumnDefinition("amount", "Order Amount", "Float64", False),         # String name
        "status": ColumnDefinition("status", "Order Status", pl.Utf8, True),            # Direct datatype
    },
    primary_key_columns=["customer_id", "order_date"]
)

right_schema = ComparisonSchema(
    columns={
        "cust_id": ColumnDefinition("cust_id", "Customer ID", "Int64", False),          # String name
        "order_dt": ColumnDefinition("order_dt", "Order Date", "Date", False),          # String name
        "total_amount": ColumnDefinition("total_amount", "Order Amount", pl.Float64, False), # Direct datatype
        "order_status": ColumnDefinition("order_status", "Order Status", "String", True), # String name
    },
    primary_key_columns=["cust_id", "order_dt"]
)

# Define column mappings
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

# Generate report using ReportingService
from splurge_lazyframe_compare import ReportingService
reporter = ReportingService()
summary_report = reporter.generate_summary_report(results=results)
print(summary_report)
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
# Export comparison results to files using ReportingService
reporter = ReportingService()
exported_files = reporter.export_results(
    results=results,
    format="csv",
    output_dir="./comparison_results"
)
print(f"Exported files: {list(exported_files.keys())}")

# Generate detailed report with different table formats
detailed_report = reporter.generate_detailed_report(
    results=results,
    max_samples=5
)
print(detailed_report)

# Generate summary table in different formats
summary_grid = reporter.generate_summary_table(
    results=results,
    table_format="grid"
)
print(summary_grid)

summary_pipe = reporter.generate_summary_table(
    results=results,
    table_format="pipe"
)
print(summary_pipe)
```

## Service Architecture

The framework follows a modular service-oriented architecture with clear separation of concerns:

### Core Services

#### `ComparisonOrchestrator`
Main entry point that coordinates all comparison activities and manages service dependencies.

#### `ComparisonService`
Handles the core comparison logic, schema validation, and result generation.

#### `ValidationService`
Provides comprehensive data quality validation including schema validation, primary key checks, data type validation, and completeness checks.

#### `ReportingService`
Generates human-readable reports in multiple formats and handles data export functionality.

#### `DataPreparationService`
Manages data preprocessing, column mapping, and schema transformations.

### Logging Utilities

#### `get_logger(name: str)`
Factory function to create configured loggers with proper naming hierarchy.

#### `performance_monitor(service_name: str, operation: str)`
Context manager for automatic performance monitoring and logging.

#### `log_service_initialization(service_name: str, config: dict = None)`
Logs service initialization with optional configuration details.

#### `log_service_operation(service_name: str, operation: str, status: str, message: str = None)`
Logs service operations with status and optional details.

#### `log_performance(service_name: str, operation: str, duration_ms: float, details: dict = None)`
Logs performance metrics with automatic slow operation detection.

### Service Usage Pattern
```python
from splurge_lazyframe_compare import ComparisonOrchestrator

# Services are automatically managed by the orchestrator
orchestrator = ComparisonOrchestrator()
results = orchestrator.compare_dataframes(
    config=comparison_config,
    left=left_df,
    right=right_df
)
```

### Individual Service Usage
```python
from splurge_lazyframe_compare import (
    ValidationService,
    ReportingService
)

# Use validation service independently
validator = ValidationService()
validation_result = validator.validate_dataframe_schema(
    df=df,
    expected_schema=comparison_schema
)

# Use reporting service independently
reporter = ReportingService()
summary_report = reporter.generate_summary_report(results=results)
```

## Logging and Monitoring

The framework includes comprehensive logging and monitoring capabilities using Python's standard logging module:

### Logger Configuration

```python
import logging
from splurge_lazyframe_compare.utils.logging_helpers import get_logger

# Configure logging (typically done once at application startup)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s] %(message)s'
)

# Get a logger for your module
logger = get_logger(__name__)
```

### Performance Monitoring

```python
from splurge_lazyframe_compare.utils.logging_helpers import performance_monitor

# Automatically log performance metrics
with performance_monitor("ComparisonService", "find_differences") as ctx:
    result = perform_comparison()
    ctx["records_processed"] = len(result)
```

### Service Logging

```python
from splurge_lazyframe_compare.utils.logging_helpers import (
    log_service_initialization,
    log_service_operation
)

# Log service initialization
log_service_initialization("ComparisonService", {"version": "1.0"})

# Log service operations
log_service_operation("ComparisonService", "compare", "success", "Comparison completed")
```

### Log Output Format

```
[2025-01-29 10:30:45,123] [INFO] [splurge_lazyframe_compare.ComparisonService] [initialization] Service initialized successfully Details: config={'version': '1.0'}
[2025-01-29 10:30:45,234] [WARNING] [splurge_lazyframe_compare.ComparisonService] [find_differences] SLOW OPERATION: 150.50ms (150.50ms) Details: records=1000
[2025-01-29 10:30:45,345] [ERROR] [splurge_lazyframe_compare.ValidationService] [validate_schema] Schema validation failed: Invalid column type
```

### Log Levels

- **DEBUG**: Detailed debugging information and performance metrics
- **INFO**: General information about operations and service lifecycle
- **WARNING**: Warning conditions that don't prevent operation
- **ERROR**: Error conditions that may affect functionality

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

**Using Direct Polars Datatypes:**
```python
col_def = ColumnDefinition(
    name="customer_id",
    alias="Customer ID",
    datatype=pl.Int64,
    nullable=False
)
```

**Using String Datatype Names (Recommended):**
```python
# More readable and user-friendly
col_def = ColumnDefinition(
    name="customer_id",
    alias="Customer ID",
    datatype="Int64",  # String name instead of pl.Int64
    nullable=False
)

# Supports all Polars datatypes
complex_col = ColumnDefinition(
    name="metadata",
    alias="Metadata",
    datatype="Struct",  # Complex datatype as string
    nullable=True
)

timestamp_col = ColumnDefinition(
    name="created_at",
    alias="Created At",
    datatype="Datetime",  # Automatically configured with defaults
    nullable=False
)
```

**Mixed Usage in Schemas:**
```python
schema = ComparisonSchema(
    columns={
        # Mix string names and direct datatypes
        "id": ColumnDefinition("id", "ID", "Int64", False),  # String
        "name": ColumnDefinition("name", "Name", pl.Utf8, True),  # Direct
        "created": ColumnDefinition("created", "Created", "Datetime", False),  # String
        "tags": ColumnDefinition("tags", "Tags", pl.List(pl.Utf8), True),  # Direct
    },
    primary_key_columns=["id"]
)
```

**⚠️ Important Notes for Complex Types:**

- **List types** require an inner type: `pl.List(pl.Utf8)` ✅, not `pl.List` ❌
- **Struct types** require field definitions: `pl.Struct([])` ✅ or `pl.Struct({"field": pl.Utf8})` ✅, not `pl.Struct` ❌
- The framework will provide clear error messages if you accidentally use unparameterized complex types

#### `ColumnMapping`
Maps columns between left and right datasets.

```python
mapping = ColumnMapping(
    left="customer_id",
    right="cust_id",
    name="customer_id"
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

#### `ComparisonResult`
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

#### `ReportingService`
Generate human-readable reports with multiple formats.

```python
from splurge_lazyframe_compare import ReportingService

reporter = ReportingService()

# Summary report
summary_report = reporter.generate_summary_report(results=results)
print(summary_report)

# Detailed report with samples
detailed_report = reporter.generate_detailed_report(
    results=results,
    max_samples=10
)
print(detailed_report)

# Summary table in different formats
summary_grid = reporter.generate_summary_table(
    results=results,
    table_format="grid"  # Options: grid, simple, pipe, orgtbl
)
print(summary_grid)

# Export results to files
exported_files = reporter.export_results(
    results=results,
    format="csv",  # Options: csv, parquet, json
    output_dir="./comparison_results"
)
```

## Data Quality Validation

The framework includes comprehensive data quality validation through the ValidationService:

```python
from splurge_lazyframe_compare import ValidationService

validator = ValidationService()

# Validate schema against expected structure
schema_result = validator.validate_dataframe_schema(
    df=df,
    expected_schema=comparison_schema
)

# Check primary key uniqueness
pk_result = validator.validate_primary_key_uniqueness(
    df=df,
    primary_key_columns=["customer_id", "order_date"]
)

# Validate data completeness
completeness_result = validator.validate_completeness(
    df=df,
    required_columns=["customer_id", "amount"]
)

# Validate data types
dtype_result = validator.validate_data_types(
    df=df,
    expected_types={"customer_id": pl.Int64, "amount": pl.Float64}
)

# Validate numeric ranges
range_result = validator.validate_numeric_ranges(
    df=df,
    column_ranges={"amount": {"min": 0, "max": 10000}}
)

# Validate string patterns (regex)
pattern_result = validator.validate_string_patterns(
    df=df,
    column_patterns={"email": r"^[^@]+@[^@]+\.[^@]+$"}
)

# Validate uniqueness constraints
uniqueness_result = validator.validate_uniqueness(
    df=df,
    unique_columns=["customer_id"]
)

# Access validation results
if schema_result.is_valid:
    print("Schema validation passed!")
else:
    print("Schema validation failed:")
    for error in schema_result.errors:
        print(f"  - {error}")
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
    print(f"Schema validation failed: {e}")
    if hasattr(e, 'validation_errors'):
        for error in e.validation_errors:
            print(f"  - {error}")
except PrimaryKeyViolationError as e:
    print(f"Primary key violation: {e}")
    if hasattr(e, 'duplicate_keys'):
        print(f"Duplicate keys found: {e.duplicate_keys}")
except ColumnMappingError as e:
    print(f"Column mapping error: {e}")
    if hasattr(e, 'mapping_errors'):
        for error in e.mapping_errors:
            print(f"  - {error}")
```

## Examples

See the `examples/` directory for comprehensive working examples demonstrating all framework capabilities:

### Core Usage Examples
- **`basic_comparison_example.py`** - Basic usage demonstration with schema definition, column mapping, and report generation
- **`service_example.py`** - Service-oriented architecture patterns and dependency injection

### Performance Examples
- **`performance_comparison_example.py`** - Performance benchmarking with large datasets (100K+ records)
- **`detailed_performance_benchmark.py`** - Comprehensive performance analysis with statistical reporting

### Reporting Examples
- **`tabulated_report_example.py`** - Multiple table formats (grid, simple, pipe, orgtbl) and export functionality

### Running Examples
```bash
# Basic comparison
python examples/basic_comparison_example.py

# Performance testing
python examples/performance_comparison_example.py

# Detailed benchmarking
python examples/detailed_performance_benchmark.py

# Service architecture
python examples/service_example.py

# Tabulated reporting
python examples/tabulated_report_example.py
```

Each example includes comprehensive documentation and demonstrates real-world usage patterns.

## Performance Characteristics

Based on the included performance benchmarks, the framework demonstrates excellent performance:

### Key Metrics
- **Processing Speed**: 2.2M+ records/second for large datasets
- **Memory Efficiency**: Lazy evaluation prevents memory issues
- **Scalability**: Performance improves with larger datasets due to Polars optimizations
- **Time per Record**: 0.0004 ms/record for optimal performance

### Benchmark Results Summary
- **1K-5K records**: 118K-475K records/second
- **10K-25K records**: 841K-1.4M records/second
- **50K+ records**: 2.2M+ records/second

The framework leverages Polars' lazy evaluation and vectorized operations for optimal performance across all dataset sizes.

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

### Logging in Development

The framework uses structured logging throughout. During development, you can enable debug logging to see detailed information:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Now all framework operations will be logged with detailed information
```

### Code Quality

```bash
# Run linting
ruff check .

# Run formatting
ruff format .

# Type checking (if mypy is configured)
mypy splurge_lazyframe_compare
```

### Recent Improvements

- **Production-ready logging**: Replaced all `print()` statements with proper Python logging module
- **Structured logging**: Consistent log format with timestamps, service names, and operation context
- **Performance monitoring**: Built-in performance tracking and slow operation detection
- **Error handling**: Robust exception handling with custom exceptions and graceful recovery

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

- **Python**: 3.10 or higher
- **Polars**: >= 1.32.0 (core data processing)
- **tabulate**: >= 0.9.0 (table formatting for reports)
- **logging**: Built-in (structured logging and monitoring)
- **typing**: Built-in (for type annotations)

### Optional Dependencies
- Additional packages may be required for specific export formats or advanced features

### Installation Options
```bash
# Install with all dependencies
pip install splurge-lazyframe-compare

# Install in development mode
pip install -e .
```
