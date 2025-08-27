# Polars LazyFrame Comparison Framework - Detailed Plan

## 1. Project Overview

### 1.1 Purpose
Create a comprehensive Python framework for comparing two Polars LazyFrames with configurable schemas, primary keys, and column mappings. The framework will identify value differences, missing records, and provide detailed comparison reports.

### 1.2 Key Features
- Multi-column primary key support
- Flexible schema definition with friendly names and data types
- Column-to-column mapping between datasets
- Three core comparison patterns: value differences, left-only records, right-only records
- Comprehensive reporting and validation
- Type-safe implementation with full annotations

## 2. Architecture Design

### 2.1 Core Components

```
polars_comparison/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── schema.py          # Schema definition and validation
│   ├── comparator.py      # Main comparison engine
│   └── results.py         # Result containers and reporting
├── validators/
│   ├── __init__.py
│   └── data_validators.py # Data quality validators
├── exceptions/
│   ├── __init__.py
│   └── comparison_exceptions.py
└── utils/
    ├── __init__.py
    └── type_helpers.py
```

### 2.2 Class Hierarchy

```
ComparisonSchema
├── ColumnDefinition
└── PrimaryKeyDefinition

LazyFrameComparator
├── ComparisonResults
│   ├── ValueDifferences
│   ├── LeftOnlyRecords
│   └── RightOnlyRecords
└── ComparisonReport
```

## 3. Data Models

### 3.1 Schema Definition

```python
@dataclass
class ColumnDefinition:
    column_name: str
    friendly_name: str
    polars_dtype: pl.DataType
    nullable: bool
    
@dataclass
class ComparisonSchema:
    columns: Dict[str, ColumnDefinition]
    primary_key_columns: List[str]
    
    def validate(self) -> None:
        """Validate schema consistency"""
        
@dataclass
class ColumnMapping:
    left_column: str
    right_column: str
    comparison_name: str
```

### 3.2 Configuration Model

```python
@dataclass
class ComparisonConfig:
    left_schema: ComparisonSchema
    right_schema: ComparisonSchema
    column_mappings: List[ColumnMapping]
    primary_key_columns: List[str]
    ignore_case: bool = False
    null_equals_null: bool = True
    tolerance: Optional[Dict[str, float]] = None  # For numeric comparisons
```

## 4. Core Implementation

### 4.1 Schema Module (`core/schema.py`)

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import polars as pl

@dataclass
class ColumnDefinition:
    """Defines a column with metadata for comparison."""
    column_name: str
    friendly_name: str
    polars_dtype: pl.DataType
    nullable: bool
    
    def validate_column_exists(self, df: pl.LazyFrame) -> bool:
        """Check if column exists in DataFrame."""
        
    def validate_data_type(self, df: pl.LazyFrame) -> bool:
        """Validate column data type matches definition."""

@dataclass
class ComparisonSchema:
    """Schema definition for a LazyFrame in comparison."""
    columns: Dict[str, ColumnDefinition]
    primary_key_columns: List[str]
    
    def validate_schema(self, df: pl.LazyFrame) -> List[str]:
        """Validate DataFrame against schema, return validation errors."""
        
    def get_primary_key_definition(self) -> List[ColumnDefinition]:
        """Get column definitions for primary key columns."""
        
    def get_compare_columns(self) -> List[str]:
        """Get non-primary-key columns for comparison."""
```

### 4.2 Comparator Module (`core/comparator.py`)

```python
from typing import Dict, List, Optional, Tuple
import polars as pl
from .schema import ComparisonSchema, ColumnMapping
from .results import ComparisonResults

class LazyFrameComparator:
    """Main comparison engine for Polars LazyFrames."""
    
    def __init__(self, config: ComparisonConfig):
        self.config = config
        self._validate_config()
    
    def compare(
        self, 
        left: pl.LazyFrame, 
        right: pl.LazyFrame
    ) -> ComparisonResults:
        """Execute complete comparison between two LazyFrames."""
        
    def _find_value_differences(
        self, 
        left: pl.LazyFrame, 
        right: pl.LazyFrame
    ) -> pl.LazyFrame:
        """Find records with same keys but different values."""
        
    def _find_left_only_records(
        self, 
        left: pl.LazyFrame, 
        right: pl.LazyFrame
    ) -> pl.LazyFrame:
        """Find records that exist only in left DataFrame."""
        
    def _find_right_only_records(
        self, 
        left: pl.LazyFrame, 
        right: pl.LazyFrame
    ) -> pl.LazyFrame:
        """Find records that exist only in right DataFrame."""
        
    def _prepare_dataframes(
        self, 
        left: pl.LazyFrame, 
        right: pl.LazyFrame
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """Prepare and validate DataFrames for comparison."""
        
    def _apply_column_mappings(
        self, 
        left: pl.LazyFrame, 
        right: pl.LazyFrame
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """Apply column mappings and create standardized column names."""
        
    def _validate_config(self) -> None:
        """Validate comparison configuration."""
```

### 4.3 Results Module (`core/results.py`)

```python
from dataclasses import dataclass
from typing import Dict, List, Optional
import polars as pl

@dataclass
class ComparisonSummary:
    """Summary statistics for comparison results."""
    total_left_records: int
    total_right_records: int
    matching_records: int
    value_differences_count: int
    left_only_count: int
    right_only_count: int
    comparison_timestamp: str

@dataclass
class ValueDifference:
    """Details of a specific value difference."""
    primary_key_values: Dict[str, any]
    column_name: str
    left_value: any
    right_value: any
    friendly_column_name: str

@dataclass
class ComparisonResults:
    """Container for all comparison results."""
    summary: ComparisonSummary
    value_differences: pl.LazyFrame
    left_only_records: pl.LazyFrame
    right_only_records: pl.LazyFrame
    config: ComparisonConfig
    
    def to_report(self) -> 'ComparisonReport':
        """Generate human-readable comparison report."""
        
    def export_results(
        self, 
        format: str = "parquet", 
        output_dir: str = "."
    ) -> Dict[str, str]:
        """Export results to files."""

class ComparisonReport:
    """Human-readable comparison report generator."""
    
    def __init__(self, results: ComparisonResults):
        self.results = results
    
    def generate_summary_report(self) -> str:
        """Generate summary report as string."""
        
    def generate_detailed_report(self) -> str:
        """Generate detailed report with sample differences."""
        
    def export_to_html(self, filename: str) -> None:
        """Export report as HTML file."""
```

## 5. Implementation Details

### 5.1 Value Difference Detection

```python
def _find_value_differences(self, left: pl.LazyFrame, right: pl.LazyFrame) -> pl.LazyFrame:
    """
    Algorithm:
    1. Apply column mappings to standardize column names
    2. Inner join on primary key columns
    3. For each mapped column pair, create difference condition
    4. Filter records where any column differs
    5. Add metadata columns indicating which columns differ
    6. Include both left and right values for differing columns
    """
    
    # Join on primary key
    joined = left.join(
        right, 
        on=self.config.primary_key_columns, 
        how="inner", 
        suffix="_right"
    )
    
    # Create difference conditions for each mapped column
    diff_conditions = []
    for mapping in self.config.column_mappings:
        left_col = mapping.left_column
        right_col = f"{mapping.right_column}_right"
        
        # Handle null comparisons based on config
        if self.config.null_equals_null:
            condition = ~pl.col(left_col).eq_missing(pl.col(right_col))
        else:
            condition = pl.col(left_col) != pl.col(right_col)
            
        diff_conditions.append(condition)
    
    # Filter rows with any differences
    return joined.filter(pl.any_horizontal(diff_conditions))
```

### 5.2 Primary Key Handling

```python
def _validate_primary_key_uniqueness(self, df: pl.LazyFrame, schema_name: str) -> None:
    """Validate that primary key columns are unique."""
    
    duplicates = (df
                 .group_by(self.config.primary_key_columns)
                 .len()
                 .filter(pl.col("len") > 1))
    
    duplicate_count = duplicates.select(pl.len()).collect().item()
    
    if duplicate_count > 0:
        raise PrimaryKeyViolationError(
            f"Duplicate primary keys found in {schema_name}: {duplicate_count} duplicates"
        )
```

### 5.3 Schema Validation

```python
def validate_schema(self, df: pl.LazyFrame) -> List[str]:
    """Comprehensive schema validation."""
    errors = []
    
    # Check column existence
    df_columns = set(df.columns)
    schema_columns = set(self.columns.keys())
    
    missing_columns = schema_columns - df_columns
    extra_columns = df_columns - schema_columns
    
    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")
    
    # Check data types for existing columns
    for col_name, col_def in self.columns.items():
        if col_name in df_columns:
            actual_dtype = df.select(pl.col(col_name)).dtypes[0]
            if actual_dtype != col_def.polars_dtype:
                errors.append(
                    f"Column {col_name}: expected {col_def.polars_dtype}, "
                    f"got {actual_dtype}"
                )
    
    # Validate nullable constraints
    for col_name, col_def in self.columns.items():
        if col_name in df_columns and not col_def.nullable:
            null_count = df.select(pl.col(col_name).is_null().sum()).collect().item()
            if null_count > 0:
                errors.append(
                    f"Column {col_name}: {null_count} null values found "
                    f"but column defined as non-nullable"
                )
    
    return errors
```

## 6. Usage Examples

### 6.1 Basic Usage

```python
import polars as pl
from polars_comparison import (
    LazyFrameComparator, 
    ComparisonConfig, 
    ComparisonSchema, 
    ColumnDefinition,
    ColumnMapping
)

# Define schemas
left_schema = ComparisonSchema(
    columns={
        "customer_id": ColumnDefinition("customer_id", "Customer ID", pl.Int64, False),
        "order_date": ColumnDefinition("order_date", "Order Date", pl.Date, False),
        "amount": ColumnDefinition("amount", "Order Amount", pl.Float64, False),
        "status": ColumnDefinition("status", "Order Status", pl.Utf8, True)
    },
    primary_key_columns=["customer_id", "order_date"]
)

right_schema = ComparisonSchema(
    columns={
        "cust_id": ColumnDefinition("cust_id", "Customer ID", pl.Int64, False),
        "order_dt": ColumnDefinition("order_dt", "Order Date", pl.Date, False),
        "total_amount": ColumnDefinition("total_amount", "Order Amount", pl.Float64, False),
        "order_status": ColumnDefinition("order_status", "Order Status", pl.Utf8, True)
    },
    primary_key_columns=["cust_id", "order_dt"]
)

# Define column mappings
mappings = [
    ColumnMapping("customer_id", "cust_id", "Customer ID"),
    ColumnMapping("order_date", "order_dt", "Order Date"),
    ColumnMapping("amount", "total_amount", "Order Amount"),
    ColumnMapping("status", "order_status", "Order Status")
]

# Create configuration
config = ComparisonConfig(
    left_schema=left_schema,
    right_schema=right_schema,
    column_mappings=mappings,
    primary_key_columns=["customer_id", "order_date"]
)

# Execute comparison
comparator = LazyFrameComparator(config)
results = comparator.compare(left_df, right_df)

# Generate report
report = results.to_report()
print(report.generate_summary_report())
```

### 6.2 Advanced Usage with Tolerances

```python
# Configuration with numeric tolerances
config = ComparisonConfig(
    left_schema=left_schema,
    right_schema=right_schema,
    column_mappings=mappings,
    primary_key_columns=["customer_id", "order_date"],
    tolerance={"amount": 0.01}  # Allow 1 cent difference in amounts
)
```

## 7. Testing Strategy

### 7.1 Unit Tests
- Schema validation logic
- Column mapping transformations
- Individual comparison pattern methods
- Configuration validation

### 7.2 Integration Tests
- End-to-end comparison workflows
- Large dataset performance tests
- Edge cases (empty DataFrames, all nulls, etc.)
- Memory usage validation

### 7.3 Test Data Generation
```python
def generate_test_data() -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """Generate controlled test datasets with known differences."""
    
def create_performance_test_data(num_rows: int) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """Generate large datasets for performance testing."""
```

## 8. Performance Considerations

### 8.1 Memory Management
- Use LazyFrame operations throughout
- Minimize intermediate materializations
- Implement streaming for very large datasets

### 8.2 Optimization Strategies
- Leverage Polars' lazy evaluation
- Use efficient join strategies
- Implement column pruning for unused columns
- Consider partitioned processing for extremely large datasets

### 8.3 Scalability Features
- Support for chunked processing
- Progress reporting for long-running comparisons
- Memory usage monitoring and warnings

## 9. Error Handling

### 9.1 Custom Exceptions
```python
class ComparisonError(Exception):
    """Base exception for comparison framework."""

class SchemaValidationError(ComparisonError):
    """Schema validation failures."""

class PrimaryKeyViolationError(ComparisonError):
    """Primary key constraint violations."""

class ColumnMappingError(ComparisonError):
    """Column mapping configuration errors."""
```

### 9.2 Validation Layers
- Configuration validation at initialization
- Schema validation before comparison
- Data quality checks during processing
- Result validation after comparison

## 10. Extensibility

### 10.1 Plugin Architecture
- Custom comparison functions for specific data types
- Pluggable difference detection algorithms
- Custom report formatters
- Additional output formats

### 10.2 Future Enhancements
- Integration with data profiling tools
- Support for nested/struct data types
- Temporal comparisons (time-series data)
- Integration with data catalogs
- REST API wrapper
- CLI interface

## 11. Documentation and Examples

### 11.1 User Guide
- Installation instructions
- Quick start tutorial
- Advanced configuration examples
- Best practices guide
- Performance tuning guide

### 11.2 API Documentation
- Comprehensive docstrings for all public methods
- Type annotations for all parameters
- Usage examples in docstrings
- Error handling documentation

This framework provides a robust foundation for Polars-based data comparison with the flexibility to handle complex real-world scenarios while maintaining high performance and type safety.