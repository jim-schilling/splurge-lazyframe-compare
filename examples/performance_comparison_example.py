#!/usr/bin/env python3
"""Performance example demonstrating the Polars LazyFrame comparison framework.

This example benchmarks comparison performance with:
- 25 columns of mixed data types (string, date, datetime, integer, float)
- 25,000 records in left dataset
- 25,250 records in right dataset
- Controlled mix of value differences, left-only, and right-only records
- Comprehensive timing benchmarks
"""

import time
from datetime import date, datetime, timedelta
from typing import Any

import polars as pl

from splurge_lazyframe_compare import (
    ColumnDefinition,
    ColumnMapping,
    ComparisonConfig,
    ComparisonSchema,
    LazyFrameComparator,
)


def generate_mixed_data(
    *,
    num_records: int,
    start_id: int = 1,
    include_differences: bool = True
) -> dict[str, list[Any]]:
    """Generate mixed data with 75 columns of various types.

    Args:
        num_records: Number of records to generate.
        start_id: Starting ID for the dataset.
        include_differences: Whether to include some variation for differences.

    Returns:
        Dictionary with column names as keys and lists of values as values.
    """
    data = {}
    
    # Primary key columns (3 columns)
    data["id"] = list(range(start_id, start_id + num_records))
    data["batch_date"] = [
        date(2024, 1, 1) + timedelta(days=i % 365) 
        for i in range(num_records)
    ]
    data["sequence_id"] = [i % 1000 for i in range(num_records)]
    
    # String columns (35 columns)
    string_columns = [
        "name", "email", "category", "status", "region", "department", "title",
        "first_name", "last_name", "middle_name", "nickname", "username",
        "phone", "mobile", "address_line1", "address_line2", "city", "state",
        "country", "postal_code", "timezone", "language", "currency",
        "company_name", "division", "team", "role", "level", "grade",
        "certification", "license", "membership", "affiliation", "preference",
        "notes"
    ]
    
    for i, col in enumerate(string_columns):
        if col == "name":
            data[col] = [f"User_{i:06d}" for i in range(num_records)]
        elif col == "email":
            data[col] = [f"user_{i:06d}@example.com" for i in range(num_records)]
        elif col == "category":
            data[col] = [f"Category_{i % 10}" for i in range(num_records)]
        elif col == "status":
            data[col] = ["active" if i % 3 == 0 else "inactive" if i % 3 == 1 else "pending" for i in range(num_records)]
        elif col == "region":
            data[col] = [f"Region_{chr(65 + (i % 26))}" for i in range(num_records)]
        elif col == "department":
            data[col] = [f"Dept_{i % 5}" for i in range(num_records)]
        elif col == "title":
            data[col] = [f"Title_{i % 8}" for i in range(num_records)]
        elif col == "first_name":
            data[col] = [f"First_{i:06d}" for i in range(num_records)]
        elif col == "last_name":
            data[col] = [f"Last_{i:06d}" for i in range(num_records)]
        elif col == "middle_name":
            data[col] = [f"Middle_{i:06d}" for i in range(num_records)]
        elif col == "nickname":
            data[col] = [f"Nick_{i:06d}" for i in range(num_records)]
        elif col == "username":
            data[col] = [f"user_{i:06d}" for i in range(num_records)]
        elif col == "phone":
            data[col] = [f"+1-555-{i:03d}-{i:04d}" for i in range(num_records)]
        elif col == "mobile":
            data[col] = [f"+1-555-{i:03d}-{i:04d}" for i in range(num_records)]
        elif col == "address_line1":
            data[col] = [f"{i} Main Street" for i in range(num_records)]
        elif col == "address_line2":
            data[col] = [f"Apt {i % 100}" for i in range(num_records)]
        elif col == "city":
            data[col] = [f"City_{i % 50}" for i in range(num_records)]
        elif col == "state":
            data[col] = [f"State_{i % 20}" for i in range(num_records)]
        elif col == "country":
            data[col] = [f"Country_{i % 10}" for i in range(num_records)]
        elif col == "postal_code":
            data[col] = [f"{i:05d}" for i in range(num_records)]
        elif col == "timezone":
            data[col] = [f"UTC+{i % 12}" for i in range(num_records)]
        elif col == "language":
            data[col] = [f"Lang_{i % 5}" for i in range(num_records)]
        elif col == "currency":
            data[col] = [f"Curr_{i % 3}" for i in range(num_records)]
        elif col == "company_name":
            data[col] = [f"Company_{i % 20}" for i in range(num_records)]
        elif col == "division":
            data[col] = [f"Division_{i % 8}" for i in range(num_records)]
        elif col == "team":
            data[col] = [f"Team_{i % 15}" for i in range(num_records)]
        elif col == "role":
            data[col] = [f"Role_{i % 12}" for i in range(num_records)]
        elif col == "level":
            data[col] = [f"Level_{i % 6}" for i in range(num_records)]
        elif col == "grade":
            data[col] = [f"Grade_{i % 10}" for i in range(num_records)]
        elif col == "certification":
            data[col] = [f"Cert_{i % 8}" for i in range(num_records)]
        elif col == "license":
            data[col] = [f"License_{i % 5}" for i in range(num_records)]
        elif col == "membership":
            data[col] = [f"Member_{i % 4}" for i in range(num_records)]
        elif col == "affiliation":
            data[col] = [f"Affil_{i % 7}" for i in range(num_records)]
        elif col == "preference":
            data[col] = [f"Pref_{i % 6}" for i in range(num_records)]
        elif col == "notes":
            data[col] = [f"Note_{i:06d}" for i in range(num_records)]
    
    # Integer columns (25 columns)
    int_columns = [
        "age", "experience_years", "projects_count", "team_size", "salary_grade",
        "employee_id", "manager_id", "department_id", "location_id", "skill_level",
        "years_employed", "vacation_days", "sick_days", "overtime_hours",
        "meetings_attended", "trainings_completed", "certifications_count",
        "languages_known", "timezone_offset", "security_level", "access_level",
        "priority_level", "performance_rating", "satisfaction_score", "tenure_months"
    ]
    
    for i, col in enumerate(int_columns):
        if col == "age":
            data[col] = [20 + (i % 60) for i in range(num_records)]
        elif col == "experience_years":
            data[col] = [i % 20 for i in range(num_records)]
        elif col == "projects_count":
            data[col] = [i % 50 for i in range(num_records)]
        elif col == "team_size":
            data[col] = [1 + (i % 15) for i in range(num_records)]
        elif col == "salary_grade":
            data[col] = [1 + (i % 10) for i in range(num_records)]
        elif col == "employee_id":
            data[col] = [1000 + i for i in range(num_records)]
        elif col == "manager_id":
            data[col] = [500 + (i % 100) for i in range(num_records)]
        elif col == "department_id":
            data[col] = [100 + (i % 20) for i in range(num_records)]
        elif col == "location_id":
            data[col] = [200 + (i % 30) for i in range(num_records)]
        elif col == "skill_level":
            data[col] = [1 + (i % 5) for i in range(num_records)]
        elif col == "years_employed":
            data[col] = [i % 15 for i in range(num_records)]
        elif col == "vacation_days":
            data[col] = [i % 25 for i in range(num_records)]
        elif col == "sick_days":
            data[col] = [i % 10 for i in range(num_records)]
        elif col == "overtime_hours":
            data[col] = [i % 40 for i in range(num_records)]
        elif col == "meetings_attended":
            data[col] = [i % 100 for i in range(num_records)]
        elif col == "trainings_completed":
            data[col] = [i % 20 for i in range(num_records)]
        elif col == "certifications_count":
            data[col] = [i % 8 for i in range(num_records)]
        elif col == "languages_known":
            data[col] = [1 + (i % 4) for i in range(num_records)]
        elif col == "timezone_offset":
            data[col] = [i % 24 for i in range(num_records)]
        elif col == "security_level":
            data[col] = [1 + (i % 5) for i in range(num_records)]
        elif col == "access_level":
            data[col] = [1 + (i % 3) for i in range(num_records)]
        elif col == "priority_level":
            data[col] = [1 + (i % 4) for i in range(num_records)]
        elif col == "performance_rating":
            data[col] = [1 + (i % 5) for i in range(num_records)]
        elif col == "satisfaction_score":
            data[col] = [1 + (i % 10) for i in range(num_records)]
        elif col == "tenure_months":
            data[col] = [i % 120 for i in range(num_records)]
    
    # Float columns (12 columns)
    float_columns = [
        "salary", "bonus", "performance_score", "efficiency_ratio", "cost_per_hour",
        "hourly_rate", "commission_rate", "tax_rate", "discount_rate", "interest_rate",
        "exchange_rate", "conversion_rate"
    ]
    
    for i, col in enumerate(float_columns):
        if col == "salary":
            data[col] = [50000.0 + (i * 1000.0) + (i % 1000) * 0.5 for i in range(num_records)]
        elif col == "bonus":
            data[col] = [i * 500.0 + (i % 100) * 0.25 for i in range(num_records)]
        elif col == "performance_score":
            data[col] = [3.0 + (i % 20) * 0.1 for i in range(num_records)]
        elif col == "efficiency_ratio":
            data[col] = [0.5 + (i % 50) * 0.01 for i in range(num_records)]
        elif col == "cost_per_hour":
            data[col] = [25.0 + (i % 30) * 0.5 for i in range(num_records)]
        elif col == "hourly_rate":
            data[col] = [20.0 + (i % 40) * 0.25 for i in range(num_records)]
        elif col == "commission_rate":
            data[col] = [0.05 + (i % 10) * 0.01 for i in range(num_records)]
        elif col == "tax_rate":
            data[col] = [0.15 + (i % 20) * 0.01 for i in range(num_records)]
        elif col == "discount_rate":
            data[col] = [0.10 + (i % 15) * 0.005 for i in range(num_records)]
        elif col == "interest_rate":
            data[col] = [0.03 + (i % 8) * 0.002 for i in range(num_records)]
        elif col == "exchange_rate":
            data[col] = [1.0 + (i % 10) * 0.1 for i in range(num_records)]
        elif col == "conversion_rate":
            data[col] = [0.02 + (i % 5) * 0.005 for i in range(num_records)]
    
    # Date columns (2 columns)
    data["hire_date"] = [
        date(2020, 1, 1) + timedelta(days=i % 365) 
        for i in range(num_records)
    ]
    data["last_review_date"] = [
        date(2023, 1, 1) + timedelta(days=i % 365) 
        for i in range(num_records)
    ]
    
    # Datetime columns (2 columns)
    data["created_at"] = [
        datetime(2024, 1, 1) + timedelta(hours=i) 
        for i in range(num_records)
    ]
    data["updated_at"] = [
        datetime(2024, 6, 1) + timedelta(hours=i * 2) 
        for i in range(num_records)
    ]
    
    # Introduce some differences if requested
    if include_differences:
        # Modify some values to create differences
        for i in range(num_records):
            if i % 100 == 0:  # Every 100th record has some differences
                data["salary"][i] += 1000.0
                data["status"][i] = "terminated" if data["status"][i] == "active" else "active"
                data["performance_score"][i] += 0.5
                data["age"][i] += 1
                data["bonus"][i] += 500.0
    
    return data


def create_performance_data() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Create performance test data with specified characteristics.

    Returns:
        Tuple of (left_df, right_df) with performance test data.
    """
    print("Generating left dataset (100,000 records)...")
    left_data = generate_mixed_data(num_records=100_000, start_id=1, include_differences=False)
    left_df = pl.LazyFrame(left_data)
    
    print("Generating right dataset (101,000 records)...")
    right_data = generate_mixed_data(num_records=101_000, start_id=1, include_differences=True)
    right_df = pl.LazyFrame(right_data)
    
    return left_df, right_df


def define_performance_schemas() -> tuple[ComparisonSchema, ComparisonSchema]:
    """Define schemas for the performance test datasets.

    Returns:
        Tuple of (left_schema, right_schema).
    """
    # Define column definitions for all 75 columns
    column_definitions = {
        # Primary key columns (3 columns)
        "id": ColumnDefinition("id", "Employee ID", pl.Int64, False),
        "batch_date": ColumnDefinition("batch_date", "Batch Date", pl.Date, False),
        "sequence_id": ColumnDefinition("sequence_id", "Sequence ID", pl.Int64, False),
        
        # String columns (35 columns)
        "name": ColumnDefinition("name", "Employee Name", pl.Utf8, False),
        "email": ColumnDefinition("email", "Email Address", pl.Utf8, False),
        "category": ColumnDefinition("category", "Category", pl.Utf8, False),
        "status": ColumnDefinition("status", "Status", pl.Utf8, False),
        "region": ColumnDefinition("region", "Region", pl.Utf8, False),
        "department": ColumnDefinition("department", "Department", pl.Utf8, False),
        "title": ColumnDefinition("title", "Job Title", pl.Utf8, False),
        "first_name": ColumnDefinition("first_name", "First Name", pl.Utf8, False),
        "last_name": ColumnDefinition("last_name", "Last Name", pl.Utf8, False),
        "middle_name": ColumnDefinition("middle_name", "Middle Name", pl.Utf8, False),
        "nickname": ColumnDefinition("nickname", "Nickname", pl.Utf8, False),
        "username": ColumnDefinition("username", "Username", pl.Utf8, False),
        "phone": ColumnDefinition("phone", "Phone", pl.Utf8, False),
        "mobile": ColumnDefinition("mobile", "Mobile", pl.Utf8, False),
        "address_line1": ColumnDefinition("address_line1", "Address Line 1", pl.Utf8, False),
        "address_line2": ColumnDefinition("address_line2", "Address Line 2", pl.Utf8, False),
        "city": ColumnDefinition("city", "City", pl.Utf8, False),
        "state": ColumnDefinition("state", "State", pl.Utf8, False),
        "country": ColumnDefinition("country", "Country", pl.Utf8, False),
        "postal_code": ColumnDefinition("postal_code", "Postal Code", pl.Utf8, False),
        "timezone": ColumnDefinition("timezone", "Timezone", pl.Utf8, False),
        "language": ColumnDefinition("language", "Language", pl.Utf8, False),
        "currency": ColumnDefinition("currency", "Currency", pl.Utf8, False),
        "company_name": ColumnDefinition("company_name", "Company Name", pl.Utf8, False),
        "division": ColumnDefinition("division", "Division", pl.Utf8, False),
        "team": ColumnDefinition("team", "Team", pl.Utf8, False),
        "role": ColumnDefinition("role", "Role", pl.Utf8, False),
        "level": ColumnDefinition("level", "Level", pl.Utf8, False),
        "grade": ColumnDefinition("grade", "Grade", pl.Utf8, False),
        "certification": ColumnDefinition("certification", "Certification", pl.Utf8, False),
        "license": ColumnDefinition("license", "License", pl.Utf8, False),
        "membership": ColumnDefinition("membership", "Membership", pl.Utf8, False),
        "affiliation": ColumnDefinition("affiliation", "Affiliation", pl.Utf8, False),
        "preference": ColumnDefinition("preference", "Preference", pl.Utf8, False),
        "notes": ColumnDefinition("notes", "Notes", pl.Utf8, False),
        
        # Integer columns (25 columns)
        "age": ColumnDefinition("age", "Age", pl.Int64, False),
        "experience_years": ColumnDefinition("experience_years", "Experience Years", pl.Int64, False),
        "projects_count": ColumnDefinition("projects_count", "Projects Count", pl.Int64, False),
        "team_size": ColumnDefinition("team_size", "Team Size", pl.Int64, False),
        "salary_grade": ColumnDefinition("salary_grade", "Salary Grade", pl.Int64, False),
        "employee_id": ColumnDefinition("employee_id", "Employee ID", pl.Int64, False),
        "manager_id": ColumnDefinition("manager_id", "Manager ID", pl.Int64, False),
        "department_id": ColumnDefinition("department_id", "Department ID", pl.Int64, False),
        "location_id": ColumnDefinition("location_id", "Location ID", pl.Int64, False),
        "skill_level": ColumnDefinition("skill_level", "Skill Level", pl.Int64, False),
        "years_employed": ColumnDefinition("years_employed", "Years Employed", pl.Int64, False),
        "vacation_days": ColumnDefinition("vacation_days", "Vacation Days", pl.Int64, False),
        "sick_days": ColumnDefinition("sick_days", "Sick Days", pl.Int64, False),
        "overtime_hours": ColumnDefinition("overtime_hours", "Overtime Hours", pl.Int64, False),
        "meetings_attended": ColumnDefinition("meetings_attended", "Meetings Attended", pl.Int64, False),
        "trainings_completed": ColumnDefinition("trainings_completed", "Trainings Completed", pl.Int64, False),
        "certifications_count": ColumnDefinition("certifications_count", "Certifications Count", pl.Int64, False),
        "languages_known": ColumnDefinition("languages_known", "Languages Known", pl.Int64, False),
        "timezone_offset": ColumnDefinition("timezone_offset", "Timezone Offset", pl.Int64, False),
        "security_level": ColumnDefinition("security_level", "Security Level", pl.Int64, False),
        "access_level": ColumnDefinition("access_level", "Access Level", pl.Int64, False),
        "priority_level": ColumnDefinition("priority_level", "Priority Level", pl.Int64, False),
        "performance_rating": ColumnDefinition("performance_rating", "Performance Rating", pl.Int64, False),
        "satisfaction_score": ColumnDefinition("satisfaction_score", "Satisfaction Score", pl.Int64, False),
        "tenure_months": ColumnDefinition("tenure_months", "Tenure Months", pl.Int64, False),
        
        # Float columns (12 columns)
        "salary": ColumnDefinition("salary", "Salary", pl.Float64, False),
        "bonus": ColumnDefinition("bonus", "Bonus", pl.Float64, False),
        "performance_score": ColumnDefinition("performance_score", "Performance Score", pl.Float64, False),
        "efficiency_ratio": ColumnDefinition("efficiency_ratio", "Efficiency Ratio", pl.Float64, False),
        "cost_per_hour": ColumnDefinition("cost_per_hour", "Cost Per Hour", pl.Float64, False),
        "hourly_rate": ColumnDefinition("hourly_rate", "Hourly Rate", pl.Float64, False),
        "commission_rate": ColumnDefinition("commission_rate", "Commission Rate", pl.Float64, False),
        "tax_rate": ColumnDefinition("tax_rate", "Tax Rate", pl.Float64, False),
        "discount_rate": ColumnDefinition("discount_rate", "Discount Rate", pl.Float64, False),
        "interest_rate": ColumnDefinition("interest_rate", "Interest Rate", pl.Float64, False),
        "exchange_rate": ColumnDefinition("exchange_rate", "Exchange Rate", pl.Float64, False),
        "conversion_rate": ColumnDefinition("conversion_rate", "Conversion Rate", pl.Float64, False),
        
        # Date columns (2 columns)
        "hire_date": ColumnDefinition("hire_date", "Hire Date", pl.Date, False),
        "last_review_date": ColumnDefinition("last_review_date", "Last Review Date", pl.Date, False),
        
        # Datetime columns (2 columns)
        "created_at": ColumnDefinition("created_at", "Created At", pl.Datetime, False),
        "updated_at": ColumnDefinition("updated_at", "Updated At", pl.Datetime, False),
    }
    
    # Both schemas are identical for this performance test
    left_schema = ComparisonSchema(
        columns=column_definitions,
        primary_key_columns=["id", "batch_date", "sequence_id"],
    )
    
    right_schema = ComparisonSchema(
        columns=column_definitions,
        primary_key_columns=["id", "batch_date", "sequence_id"],
    )
    
    return left_schema, right_schema


def create_performance_column_mappings() -> list[ColumnMapping]:
    """Create column mappings for the performance test.

    Returns:
        List of ColumnMapping objects.
    """
    # All columns map to themselves since both datasets have identical schemas
    column_names = [
        # Primary key columns (3 columns)
        "id", "batch_date", "sequence_id",
        
        # String columns (35 columns)
        "name", "email", "category", "status", "region", "department", "title",
        "first_name", "last_name", "middle_name", "nickname", "username",
        "phone", "mobile", "address_line1", "address_line2", "city", "state",
        "country", "postal_code", "timezone", "language", "currency",
        "company_name", "division", "team", "role", "level", "grade",
        "certification", "license", "membership", "affiliation", "preference",
        "notes",
        
        # Integer columns (25 columns)
        "age", "experience_years", "projects_count", "team_size", "salary_grade",
        "employee_id", "manager_id", "department_id", "location_id", "skill_level",
        "years_employed", "vacation_days", "sick_days", "overtime_hours",
        "meetings_attended", "trainings_completed", "certifications_count",
        "languages_known", "timezone_offset", "security_level", "access_level",
        "priority_level", "performance_rating", "satisfaction_score", "tenure_months",
        
        # Float columns (12 columns)
        "salary", "bonus", "performance_score", "efficiency_ratio", "cost_per_hour",
        "hourly_rate", "commission_rate", "tax_rate", "discount_rate", "interest_rate",
        "exchange_rate", "conversion_rate",
        
        # Date columns (2 columns)
        "hire_date", "last_review_date",
        
        # Datetime columns (2 columns)
        "created_at", "updated_at"
    ]
    
    return [
        ColumnMapping(col, col, col) for col in column_names
    ]


def benchmark_comparison(
    *,
    left_df: pl.LazyFrame,
    right_df: pl.LazyFrame,
    config: ComparisonConfig
) -> dict[str, float]:
    """Benchmark the comparison performance.

    Args:
        left_df: Left LazyFrame to compare.
        right_df: Right LazyFrame to compare.
        config: Comparison configuration.

    Returns:
        Dictionary with timing results.
    """
    print("Initializing comparator...")
    start_time = time.time()
    comparator = LazyFrameComparator(config)
    init_time = time.time() - start_time
    
    print("Running comparison...")
    start_time = time.time()
    results = comparator.compare(left=left_df, right=right_df)
    comparison_time = time.time() - start_time
    
    print("Collecting results...")
    start_time = time.time()
    summary = results.summary
    value_differences_count = results.value_differences.select(pl.len()).collect().item()
    left_only_count = results.left_only_records.select(pl.len()).collect().item()
    right_only_count = results.right_only_records.select(pl.len()).collect().item()
    collect_time = time.time() - start_time
    
    return {
        "initialization_time": init_time,
        "comparison_time": comparison_time,
        "collection_time": collect_time,
        "total_time": init_time + comparison_time + collect_time,
        "value_differences_count": value_differences_count,
        "left_only_count": left_only_count,
        "right_only_count": right_only_count,
    }


def print_performance_results(results: dict[str, float]) -> None:
    """Print formatted performance results.

    Args:
        results: Dictionary with performance results.
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK RESULTS")
    print("=" * 80)
    
    print(f"\nTiming Results:")
    print(f"  Initialization time: {results['initialization_time']:.4f} seconds")
    print(f"  Comparison time:     {results['comparison_time']:.4f} seconds")
    print(f"  Collection time:     {results['collection_time']:.4f} seconds")
    print(f"  Total time:          {results['total_time']:.4f} seconds")
    
    print(f"\nComparison Results:")
    print(f"  Value differences:   {results['value_differences_count']:,} records")
    print(f"  Left-only records:   {results['left_only_count']:,} records")
    print(f"  Right-only records:  {results['right_only_count']:,} records")
    
    # Calculate performance metrics
    total_records = 100_000 + 101_000
    records_per_second = total_records / results['comparison_time']
    
    print(f"\nPerformance Metrics:")
    print(f"  Records processed:   {total_records:,}")
    print(f"  Processing rate:     {records_per_second:,.0f} records/second")
    print(f"  Time per record:     {results['comparison_time'] / total_records * 1000:.4f} ms/record")


def main() -> None:
    """Run the performance comparison example."""
    print("=" * 80)
    print("POLARS LAZYFRAME COMPARISON FRAMEWORK - PERFORMANCE EXAMPLE")
    print("=" * 80)
    
    print(f"\nTest Configuration:")
    print(f"  Left dataset:  100,000 records")
    print(f"  Right dataset: 101,000 records")
    print(f"  Total columns: 75 columns (mixed data types)")
    print(f"  Primary keys:  id, batch_date, sequence_id")
    
    # Create performance test data
    print(f"\n1. Creating performance test data...")
    left_df, right_df = create_performance_data()
    
    # Verify data characteristics
    left_count = left_df.select(pl.len()).collect().item()
    right_count = right_df.select(pl.len()).collect().item()
    print(f"   Left dataset: {left_count:,} records")
    print(f"   Right dataset: {right_count:,} records")
    print(f"   Left columns: {len(left_df.collect_schema().names())}")
    print(f"   Right columns: {len(right_df.collect_schema().names())}")
    
    # Define schemas
    print(f"\n2. Defining schemas...")
    left_schema, right_schema = define_performance_schemas()
    print(f"   Schema columns: {len(left_schema.columns)}")
    print(f"   Primary keys: {left_schema.primary_key_columns}")
    
    # Create column mappings
    print(f"\n3. Creating column mappings...")
    mappings = create_performance_column_mappings()
    print(f"   Column mappings: {len(mappings)}")
    
    # Create comparison configuration
    print(f"\n4. Creating comparison configuration...")
    config = ComparisonConfig(
        left_schema=left_schema,
        right_schema=right_schema,
        column_mappings=mappings,
        primary_key_columns=["id", "batch_date", "sequence_id"],
        ignore_case=False,
        null_equals_null=True,
    )
    
    # Run performance benchmark
    print(f"\n5. Running performance benchmark...")
    benchmark_results = benchmark_comparison(
        left_df=left_df,
        right_df=right_df,
        config=config
    )
    
    # Display results
    print_performance_results(benchmark_results)
    
    # Optional: Export results
    print(f"\n6. Exporting results...")
    try:
        comparator = LazyFrameComparator(config)
        results = comparator.compare(left=left_df, right=right_df)
        exported_files = results.export_results(format="csv", output_dir="./comparison_results")
        print(f"   Exported files: {list(exported_files.keys())}")
    except Exception as e:
        print(f"   Export skipped: {e}")
    
    print(f"\n" + "=" * 80)
    print("PERFORMANCE EXAMPLE COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
