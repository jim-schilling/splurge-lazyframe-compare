# Splurge LazyFrame Comparison Framework — Detailed Documentation

This document contains full usage instructions, configuration options, API reference, examples, logging, development notes, and other long-form material for `splurge_lazyframe_compare`.

## Features

- Service-oriented architecture — modular services for comparison, validation, reporting, and data preparation
- Multi-column primary key support (composite keys)
- Flexible schema definitions supporting direct Polars datatypes and string type names
- Column-to-column mapping between datasets with different column names
- Value differences, left-only, and right-only comparisons
- Comprehensive reporting with multiple table formats and export options (CSV, Parquet, JSON)
- Data validation utilities and helpful error messages
- Type-safe implementation with annotations
- Performance optimized via Polars lazy evaluation
- Auto-configuration utilities to generate ComparisonConfig from LazyFrames
- Configuration management (load/save/merge/environment overrides)


## Installation

```bash
pip install splurge-lazyframe-compare
```

For development:

```bash
pip install -e .
pip install -r dev-requirements.txt  # optional if you maintain a dev requirements file
```

## Quick Start (expanded)

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
    pk_columns=["customer_id", "order_date"]
)

right_schema = ComparisonSchema(
    columns={
        "cust_id": ColumnDefinition("cust_id", "Customer ID", "Int64", False),          # String name
        "order_dt": ColumnDefinition("order_dt", "Order Date", "Date", False),          # String name
        "total_amount": ColumnDefinition("total_amount", "Order Amount", pl.Float64, False), # Direct datatype
        "order_status": ColumnDefinition("order_status", "Order Status", "String", True), # String name
    },
    pk_columns=["cust_id", "order_dt"]
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
    pk_columns=["customer_id", "order_date"]
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

## Auto-Configuration from LazyFrames

For cases where your LazyFrames have identical column names and you want to quickly set up a comparison, you can use the automatic configuration generator:

```python
from splurge_lazyframe_compare.utils import create_comparison_config_from_lazyframes

# Your LazyFrames with identical column names
left_data = {
    "customer_id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "email": ["alice@example.com", "bob@example.com", "charlie@example.com", "david@example.com", "eve@example.com"],
    "balance": [100.50, 250.00, 75.25, 300.00, 150.75],
    "active": [True, True, False, True, True]
}

right_data = {
    "customer_id": [1, 2, 3, 4, 6],  # ID 5 missing, ID 6 added
    "name": ["Alice", "Bob", "Charlie", "Dave", "Frank"],  # David -> Dave, Eve -> Frank
    "email": ["alice@example.com", "bob@example.com", "charlie@example.com", "dave@example.com", "frank@example.com"],
    "balance": [100.50, 250.00, 75.25, 320.00, 200.00],  # David's balance changed
    "active": [True, True, False, True, False]  # Frank is inactive
}

left_df = pl.LazyFrame(left_data)
right_df = pl.LazyFrame(right_data)

# Specify primary key columns
primary_keys = ["customer_id"]

# Generate ComparisonConfig automatically (keyword-only parameters)
config = create_comparison_config_from_lazyframes(
    left=left_df,
    right=right_df,
    pk_columns=primary_keys
)

print(f"Auto-generated config with {len(config.column_mappings)} column mappings")
print(f"Primary key columns: {config.pk_columns}")

# Use immediately for comparison
from splurge_lazyframe_compare.services.comparison_service import ComparisonService

comparison_service = ComparisonService()
results = comparison_service.execute_comparison(
    left=left_df,
    right=right_df,
    config=config
)

print(f"Comparison completed - found {results.summary.value_differences_count} differences")

## Advanced Usage

### Numeric Tolerance

```python
# Allow small differences in numeric columns
config = ComparisonConfig(
    left_schema=left_schema,
    right_schema=right_schema,
    column_mappings=mappings,
    pk_columns=["customer_id", "order_date"],
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
    pk_columns=["customer_id", "order_date"],
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
    pk_columns=["customer_id", "order_date"],
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



## Configuration Management

The framework provides comprehensive configuration management utilities for loading, saving, and manipulating comparison configurations:

### Configuration Files

```python
from splurge_lazyframe_compare.utils.config_helpers import (
    load_config_from_file,
    save_config_to_file,
    create_default_config,
    validate_config
)

# Create a default configuration template
default_config = create_default_config()
print("Default config keys:", list(default_config.keys()))

# Save configuration to file
save_config_to_file(default_config, "my_comparison_config.json")

# Load configuration from file
loaded_config = load_config_from_file("my_comparison_config.json")

# Validate configuration
validation_errors = validate_config(loaded_config)
if validation_errors:
    print("Configuration errors:", validation_errors)
else:
    print("Configuration is valid")
```

### Environment Variable Configuration

```python
from splurge_lazyframe_compare.utils.config_helpers import (
    get_env_config,
    apply_environment_overrides,
    merge_configs
)

# Load configuration from environment variables (prefixed with SPLURGE_)
env_config = get_env_config()
print("Environment config:", env_config)

# Apply environment overrides to existing config
final_config = apply_environment_overrides(default_config)

# Merge multiple configuration sources
custom_config = {"comparison": {"max_samples": 100}}
merged_config = merge_configs(default_config, custom_config)
```

### Configuration Utilities

```python
from splurge_lazyframe_compare.utils.config_helpers import (
    get_config_value,
    create_config_from_dataframes
)

# Get nested configuration values
max_samples = get_config_value(merged_config, "reporting.max_samples", default_value=10)

# Create configuration from existing DataFrames
basic_config = create_config_from_dataframes(
    left_df=left_df,
    right_df=right_df,
    primary_keys=["customer_id"],
    auto_map_columns=True
)
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
from splurge_lazyframe_compare.utils.logging_helpers import (
    get_logger,
    configure_logging,
)

# Configure logging at application startup (no side-effects on import)
configure_logging(level="INFO", fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')

# Get a logger for your module
logger = get_logger(__name__)
```

### Performance Monitoring

```python
from splurge_lazyframe_compare.utils.logging_helpers import performance_monitor

# Automatically log performance metrics
with performance_monitor("ComparisonService", "find_differences") as ctx:
    result = perform_comparison()  # Your actual operation here
    # Add custom metrics (result may not always have len() method)
    if hasattr(result, '__len__'):
        ctx["records_processed"] = len(result)
    ctx["operation_status"] = "completed"
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
```

### Interpreting Validation Errors

- SchemaValidationError: indicates schema mismatches (missing columns, wrong dtypes, nullability, or unmapped PKs). Inspect the message substrings for actionable details such as missing columns or dtype mismatches. Primary key violations surface as duplicates or missing mappings.
- PrimaryKeyViolationError: raised when primary key constraints are violated. Ensure all PKs exist and are unique in input LazyFrames.

Exception types and original tracebacks are preserved by services; messages include service name and operation context for clarity.

## CLI Usage

The package provides a `slc` CLI.

```bash
slc --help
slc compare --dry-run
slc report --dry-run
slc export --dry-run
```

The dry-run subcommands validate inputs and demonstrate execution without running a full comparison.

### CLI Errors and Exit Codes

- The CLI surfaces domain errors using custom exceptions and stable exit codes:
  - Configuration issues (e.g., invalid JSON, failed validation) raise `ConfigError` and exit with code `2`.
  - Data source issues (e.g., missing files, unsupported extensions) raise `DataSourceError` and exit with code `2`.
  - Unexpected errors exit with code `1` and include a brief message; enable debug logging for full tracebacks.

Example messages:
```
Configuration error: Invalid configuration: <details>
Compare failed: Unsupported file extension: .txt
Export failed: Data file not found: <path>
```

## Large-data Export Tips

- Prefer parquet format for performance and compression. CSV is human-friendly but slower for large datasets.
- Ensure sufficient temporary disk space when exporting large LazyFrames; parquet writes may buffer data.
- For JSON summaries we export a compact, versioned envelope:

```json
{
  "schema_version": "1.0",
  "summary": {
    "total_left_records": 123,
    "total_right_records": 123,
    "matching_records": 120,
    "value_differences_count": 3,
    "left_only_count": 0,
    "right_only_count": 0,
    "comparison_timestamp": "2025-01-01T00:00:00"
  }
}
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
    pk_columns=["id"]
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
    pk_columns=["id"]
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
    pk_columns=["customer_id", "order_date"],
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

... (remaining detailed sections preserved)
Contents


- Features and overview
- Installation
- Quick start (expanded)
- Auto-configuration from LazyFrames
- Advanced usage (numeric tolerance, null handling, case-insensitive comparison)
- Export and reporting options
- Configuration management and environment overrides
- Service architecture and examples
- Logging and monitoring
- Interpreting validation errors and exceptions
- CLI usage and exit codes
- Examples (examples/ directory)
- Development (testing, linting, type checking)
- Contributing


## Features


- Service-oriented architecture — modular services for comparison, validation, reporting, and data preparation
- Multi-column primary key support (composite keys)
- Flexible schema definitions supporting direct Polars datatypes and string type names
- Column-to-column mapping between datasets with different column names
- Value differences, left-only, and right-only comparisons
- Comprehensive reporting with multiple table formats and export options (CSV, Parquet, JSON)
- Data validation utilities and helpful error messages
- Type-safe implementation with annotations
- Performance optimized via Polars lazy evaluation
- Auto-configuration utilities to generate ComparisonConfig from LazyFrames
- Configuration management (load/save/merge/environment overrides)


## Installation


```bash
pip install splurge-lazyframe-compare
```


For development:


```bash
pip install -e .
pip install -r dev-requirements.txt  # optional if you maintain a dev requirements file
```


## Quick Start (expanded)


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

# See this document (`docs/README-DETAILS.md`) for full examples and explanations

```


### Auto-Configuration from LazyFrames


Use `create_comparison_config_from_lazyframes()` to auto-generate a `ComparisonConfig` when left and right LazyFrames share column names.


```python
from splurge_lazyframe_compare.utils import create_comparison_config_from_lazyframes

config = create_comparison_config_from_lazyframes(left=left_lf, right=right_lf, pk_columns=["id"])

```


## Advanced Usage


Numeric tolerance, null value handling, and case-insensitive string comparison are controlled via `ComparisonConfig` options. See the API reference below for full parameter lists.


## Export Results and Reporting


The `ReportingService` supports generating summary and detailed reports and exporting results to CSV, Parquet, and JSON. See the examples in `examples/` for detailed usage patterns.


## Configuration Management


Utilities for loading, saving, merging, and applying environment overrides are available in `splurge_lazyframe_compare.utils.config_helpers`.




## Service Architecture


Key services include:


- `ComparisonOrchestrator`: coordinates comparison activities and manages service dependencies
- `ComparisonService`: core comparison logic and result generation
- `ValidationService`: schema and data quality checks
- `ReportingService`: report generation and exports
- `DataPreparationService`: preprocessing and schema transformations


Refer to the in-code docstrings and the examples in `examples/` for service-level usage patterns.


## Logging and Monitoring


Use `configure_logging()` and `get_logger()` from `splurge_lazyframe_compare.utils.logging_helpers` to configure structured logging. Use `performance_monitor()` to capture timing and performance metrics.


## Exceptions and Error Handling


The package exposes domain exceptions in `splurge_lazyframe_compare.exceptions`, including:


- `SchemaValidationError`
- `PrimaryKeyViolationError`
- `ColumnMappingError`
- `ConfigError`
- `DataSourceError`


Catch these in CLI code or integrations to map to appropriate exit codes or user-facing messages.


## CLI


The `slc` CLI ships with the package and exposes `compare`, `report`, and `export` subcommands. `--dry-run` helps validate wiring without executing heavy comparisons.


## Examples


See the `examples/` folder for working examples:


- `basic_comparison_example.py`
- `service_example.py`
- `auto_config_example.py`
- `detailed_performance_benchmark.py`



## Development


Run tests and checks locally:


```bash
# Run unit tests
pytest

# Lint and format
ruff check .
ruff format .

# Type checking
mypy splurge_lazyframe_compare
```


Note: tests in this repository were updated using the `splurge-test-namer` utility which relies on module-level `DOMAINS` sentinels. If you change a module's `DOMAINS` sentinel, re-run the tool to keep test naming consistent:


```bash
splurge-test-namer --import-root splurge_lazyframe_compare --test-root tests --repo-root .
```


## Contributing


Please follow the standard fork/branch/PR workflow. Add tests for new features and ensure linters and type checks pass.


## API Reference (high level)


See in-code docstrings and the usage examples for detailed parameter lists. Core concepts: `ComparisonSchema`, `ColumnDefinition`, `ColumnMapping`, `ComparisonConfig`, `LazyFrameComparator`, `ComparisonResult`, and `ReportingService`.


## Contact and Support


File issues on GitHub and reference the `chore/update-tests` branch if reporting test or CI issues related to recent changes.


---

For full, copy-ready code examples and more details, search the `examples/` directory or explore the package modules.