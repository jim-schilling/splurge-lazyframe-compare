#!/usr/bin/env python3
"""Detailed performance benchmark for the Polars LazyFrame comparison framework.

This script provides comprehensive performance analysis including:
- Multiple iterations for statistical accuracy
- Different dataset sizes
- Memory usage tracking
- Detailed timing breakdowns
- Performance comparison across different configurations
"""

import gc
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
    include_differences: bool = True,
    difference_rate: float = 0.01
) -> dict[str, list[Any]]:
    """Generate mixed data with 75 columns of various types.

    Args:
        num_records: Number of records to generate.
        start_id: Starting ID for the dataset.
        include_differences: Whether to include some variation for differences.
        difference_rate: Rate of records with differences (0.0 to 1.0).

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
        # Modify some values to create differences based on difference_rate
        difference_interval = int(1 / difference_rate) if difference_rate > 0 else 0
        for i in range(num_records):
            if difference_interval > 0 and i % difference_interval == 0:
                data["salary"][i] += 1000.0
                data["status"][i] = "terminated" if data["status"][i] == "active" else "active"
                data["performance_score"][i] += 0.5
                data["age"][i] += 1
                data["bonus"][i] += 500.0
    
    return data


def create_test_datasets(
    *,
    left_size: int,
    right_size: int,
    difference_rate: float = 0.01
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Create test datasets with specified characteristics.

    Args:
        left_size: Number of records in left dataset.
        right_size: Number of records in right dataset.
        difference_rate: Rate of records with differences.

    Returns:
        Tuple of (left_df, right_df) with test data.
    """
    print(f"  Generating left dataset ({left_size:,} records)...")
    left_data = generate_mixed_data(
        num_records=left_size, 
        start_id=1, 
        include_differences=False
    )
    left_df = pl.LazyFrame(left_data)
    
    print(f"  Generating right dataset ({right_size:,} records)...")
    right_data = generate_mixed_data(
        num_records=right_size, 
        start_id=1, 
        include_differences=True,
        difference_rate=difference_rate
    )
    right_df = pl.LazyFrame(right_data)
    
    return left_df, right_df


def define_test_schemas() -> tuple[ComparisonSchema, ComparisonSchema]:
    """Define schemas for the test datasets.

    Returns:
        Tuple of (left_schema, right_schema).
    """
    # Define column definitions for all 75 columns
    column_definitions = {
        # Primary key columns (3 columns)
        "id": ColumnDefinition(name="id", alias="Employee ID", datatype=pl.Int64, nullable=False),
        "batch_date": ColumnDefinition(name="batch_date", alias="Batch Date", datatype=pl.Date, nullable=False),
        "sequence_id": ColumnDefinition(name="sequence_id", alias="Sequence ID", datatype=pl.Int64, nullable=False),
        
        # String columns (35 columns)
        "name": ColumnDefinition(name="name", alias="Employee Name", datatype=pl.Utf8, nullable=False),
        "email": ColumnDefinition(name="email", alias="Email Address", datatype=pl.Utf8, nullable=False),
        "category": ColumnDefinition(name="category", alias="Category", datatype=pl.Utf8, nullable=False),
        "status": ColumnDefinition(name="status", alias="Status", datatype=pl.Utf8, nullable=False),
        "region": ColumnDefinition(name="region", alias="Region", datatype=pl.Utf8, nullable=False),
        "department": ColumnDefinition(name="department", alias="Department", datatype=pl.Utf8, nullable=False),
        "title": ColumnDefinition(name="title", alias="Job Title", datatype=pl.Utf8, nullable=False),
        "first_name": ColumnDefinition(name="first_name", alias="First Name", datatype=pl.Utf8, nullable=False),
        "last_name": ColumnDefinition(name="last_name", alias="Last Name", datatype=pl.Utf8, nullable=False),
        "middle_name": ColumnDefinition(name="middle_name", alias="Middle Name", datatype=pl.Utf8, nullable=False),
        "nickname": ColumnDefinition(name="nickname", alias="Nickname", datatype=pl.Utf8, nullable=False),
        "username": ColumnDefinition(name="username", alias="Username", datatype=pl.Utf8, nullable=False),
        "phone": ColumnDefinition(name="phone", alias="Phone", datatype=pl.Utf8, nullable=False),
        "mobile": ColumnDefinition(name="mobile", alias="Mobile", datatype=pl.Utf8, nullable=False),
        "address_line1": ColumnDefinition(name="address_line1", alias="Address Line 1", datatype=pl.Utf8, nullable=False),
        "address_line2": ColumnDefinition(name="address_line2", alias="Address Line 2", datatype=pl.Utf8, nullable=False),
        "city": ColumnDefinition(name="city", alias="City", datatype=pl.Utf8, nullable=False),
        "state": ColumnDefinition(name="state", alias="State", datatype=pl.Utf8, nullable=False),
        "country": ColumnDefinition(name="country", alias="Country", datatype=pl.Utf8, nullable=False),
        "postal_code": ColumnDefinition(name="postal_code", alias="Postal Code", datatype=pl.Utf8, nullable=False),
        "timezone": ColumnDefinition(name="timezone", alias="Timezone", datatype=pl.Utf8, nullable=False),
        "language": ColumnDefinition(name="language", alias="Language", datatype=pl.Utf8, nullable=False),
        "currency": ColumnDefinition(name="currency", alias="Currency", datatype=pl.Utf8, nullable=False),
        "company_name": ColumnDefinition(name="company_name", alias="Company Name", datatype=pl.Utf8, nullable=False),
        "division": ColumnDefinition(name="division", alias="Division", datatype=pl.Utf8, nullable=False),
        "team": ColumnDefinition(name="team", alias="Team", datatype=pl.Utf8, nullable=False),
        "role": ColumnDefinition(name="role", alias="Role", datatype=pl.Utf8, nullable=False),
        "level": ColumnDefinition(name="level", alias="Level", datatype=pl.Utf8, nullable=False),
        "grade": ColumnDefinition(name="grade", alias="Grade", datatype=pl.Utf8, nullable=False),
        "certification": ColumnDefinition(name="certification", alias="Certification", datatype=pl.Utf8, nullable=False),
        "license": ColumnDefinition(name="license", alias="License", datatype=pl.Utf8, nullable=False),
        "membership": ColumnDefinition(name="membership", alias="Membership", datatype=pl.Utf8, nullable=False),
        "affiliation": ColumnDefinition(name="affiliation", alias="Affiliation", datatype=pl.Utf8, nullable=False),
        "preference": ColumnDefinition(name="preference", alias="Preference", datatype=pl.Utf8, nullable=False),
        "notes": ColumnDefinition(name="notes", alias="Notes", datatype=pl.Utf8, nullable=False),
        
        # Integer columns (25 columns)
        "age": ColumnDefinition(name="age", alias="Age", datatype=pl.Int64, nullable=False),
        "experience_years": ColumnDefinition(name="experience_years", alias="Experience Years", datatype=pl.Int64, nullable=False),
        "projects_count": ColumnDefinition(name="projects_count", alias="Projects Count", datatype=pl.Int64, nullable=False),
        "team_size": ColumnDefinition(name="team_size", alias="Team Size", datatype=pl.Int64, nullable=False),
        "salary_grade": ColumnDefinition(name="salary_grade", alias="Salary Grade", datatype=pl.Int64, nullable=False),
        "employee_id": ColumnDefinition(name="employee_id", alias="Employee ID", datatype=pl.Int64, nullable=False),
        "manager_id": ColumnDefinition(name="manager_id", alias="Manager ID", datatype=pl.Int64, nullable=False),
        "department_id": ColumnDefinition(name="department_id", alias="Department ID", datatype=pl.Int64, nullable=False),
        "location_id": ColumnDefinition(name="location_id", alias="Location ID", datatype=pl.Int64, nullable=False),
        "skill_level": ColumnDefinition(name="skill_level", alias="Skill Level", datatype=pl.Int64, nullable=False),
        "years_employed": ColumnDefinition(name="years_employed", alias="Years Employed", datatype=pl.Int64, nullable=False),
        "vacation_days": ColumnDefinition(name="vacation_days", alias="Vacation Days", datatype=pl.Int64, nullable=False),
        "sick_days": ColumnDefinition(name="sick_days", alias="Sick Days", datatype=pl.Int64, nullable=False),
        "overtime_hours": ColumnDefinition(name="overtime_hours", alias="Overtime Hours", datatype=pl.Int64, nullable=False),
        "meetings_attended": ColumnDefinition(name="meetings_attended", alias="Meetings Attended", datatype=pl.Int64, nullable=False),
        "trainings_completed": ColumnDefinition(name="trainings_completed", alias="Trainings Completed", datatype=pl.Int64, nullable=False),
        "certifications_count": ColumnDefinition(name="certifications_count", alias="Certifications Count", datatype=pl.Int64, nullable=False),
        "languages_known": ColumnDefinition(name="languages_known", alias="Languages Known", datatype=pl.Int64, nullable=False),
        "timezone_offset": ColumnDefinition(name="timezone_offset", alias="Timezone Offset", datatype=pl.Int64, nullable=False),
        "security_level": ColumnDefinition(name="security_level", alias="Security Level", datatype=pl.Int64, nullable=False),
        "access_level": ColumnDefinition(name="access_level", alias="Access Level", datatype=pl.Int64, nullable=False),
        "priority_level": ColumnDefinition(name="priority_level", alias="Priority Level", datatype=pl.Int64, nullable=False),
        "performance_rating": ColumnDefinition(name="performance_rating", alias="Performance Rating", datatype=pl.Int64, nullable=False),
        "satisfaction_score": ColumnDefinition(name="satisfaction_score", alias="Satisfaction Score", datatype=pl.Int64, nullable=False),
        "tenure_months": ColumnDefinition(name="tenure_months", alias="Tenure Months", datatype=pl.Int64, nullable=False),
        
        # Float columns (12 columns)
        "salary": ColumnDefinition(name="salary", alias="Salary", datatype=pl.Float64, nullable=False),
        "bonus": ColumnDefinition(name="bonus", alias="Bonus", datatype=pl.Float64, nullable=False),
        "performance_score": ColumnDefinition(name="performance_score", alias="Performance Score", datatype=pl.Float64, nullable=False),
        "efficiency_ratio": ColumnDefinition(name="efficiency_ratio", alias="Efficiency Ratio", datatype=pl.Float64, nullable=False),
        "cost_per_hour": ColumnDefinition(name="cost_per_hour", alias="Cost Per Hour", datatype=pl.Float64, nullable=False),
        "hourly_rate": ColumnDefinition(name="hourly_rate", alias="Hourly Rate", datatype=pl.Float64, nullable=False),
        "commission_rate": ColumnDefinition(name="commission_rate", alias="Commission Rate", datatype=pl.Float64, nullable=False),
        "tax_rate": ColumnDefinition(name="tax_rate", alias="Tax Rate", datatype=pl.Float64, nullable=False),
        "discount_rate": ColumnDefinition(name="discount_rate", alias="Discount Rate", datatype=pl.Float64, nullable=False),
        "interest_rate": ColumnDefinition(name="interest_rate", alias="Interest Rate", datatype=pl.Float64, nullable=False),
        "exchange_rate": ColumnDefinition(name="exchange_rate", alias="Exchange Rate", datatype=pl.Float64, nullable=False),
        "conversion_rate": ColumnDefinition(name="conversion_rate", alias="Conversion Rate", datatype=pl.Float64, nullable=False),
        
        # Date columns (2 columns)
        "hire_date": ColumnDefinition(name="hire_date", alias="Hire Date", datatype=pl.Date, nullable=False),
        "last_review_date": ColumnDefinition(name="last_review_date", alias="Last Review Date", datatype=pl.Date, nullable=False),
        
        # Datetime columns (2 columns)
        "created_at": ColumnDefinition(name="created_at", alias="Created At", datatype=pl.Datetime, nullable=False),
        "updated_at": ColumnDefinition(name="updated_at", alias="Updated At", datatype=pl.Datetime, nullable=False),
    }
    
    # Both schemas are identical for this test
    left_schema = ComparisonSchema(
        columns=column_definitions,
        pk_columns=["id", "batch_date", "sequence_id"],
    )
    
    right_schema = ComparisonSchema(
        columns=column_definitions,
        pk_columns=["id", "batch_date", "sequence_id"],
    )
    
    return left_schema, right_schema


def create_column_mappings() -> list[ColumnMapping]:
    """Create column mappings for the test.

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
        ColumnMapping(left=col, right=col, name=col) for col in column_names
    ]


def run_single_benchmark(
    *,
    left_df: pl.LazyFrame,
    right_df: pl.LazyFrame,
    config: ComparisonConfig,
    iteration: int
) -> dict[str, float]:
    """Run a single benchmark iteration.

    Args:
        left_df: Left LazyFrame to compare.
        right_df: Right LazyFrame to compare.
        config: Comparison configuration.
        iteration: Iteration number for logging.

    Returns:
        Dictionary with timing results.
    """
    print(f"    Iteration {iteration}: Initializing comparator...")
    start_time = time.time()
    comparator = LazyFrameComparator(config)
    init_time = time.time() - start_time
    
    print(f"    Iteration {iteration}: Running comparison...")
    start_time = time.time()
    results = comparator.compare(left=left_df, right=right_df)
    comparison_time = time.time() - start_time
    
    print(f"    Iteration {iteration}: Collecting results...")
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


def run_benchmark_suite(
    *,
    left_size: int,
    right_size: int,
    iterations: int = 5,
    difference_rate: float = 0.01
) -> dict[str, Any]:
    """Run a complete benchmark suite with multiple iterations.

    Args:
        left_size: Number of records in left dataset.
        right_size: Number of records in right dataset.
        iterations: Number of iterations to run.
        difference_rate: Rate of records with differences.

    Returns:
        Dictionary with comprehensive benchmark results.
    """
    print(f"\nRunning benchmark suite:")
    print(f"  Left dataset: {left_size:,} records")
    print(f"  Right dataset: {right_size:,} records")
    print(f"  Iterations: {iterations}")
    print(f"  Difference rate: {difference_rate:.2%}")
    
    # Create test data
    left_df, right_df = create_test_datasets(
        left_size=left_size,
        right_size=right_size,
        difference_rate=difference_rate
    )
    
    # Define schemas and mappings
    left_schema, right_schema = define_test_schemas()
    mappings = create_column_mappings()
    
    # Create configuration
    config = ComparisonConfig(
        left_schema=left_schema,
        right_schema=right_schema,
        column_mappings=mappings,
        primary_key_columns=["id", "batch_date", "sequence_id"],
        ignore_case=False,
        null_equals_null=True,
    )
    
    # Run iterations
    results = []
    for i in range(iterations):
        # Force garbage collection before each iteration
        gc.collect()
        
        iteration_result = run_single_benchmark(
            left_df=left_df,
            right_df=right_df,
            config=config,
            iteration=i + 1
        )
        results.append(iteration_result)
    
    # Calculate statistics
    total_records = left_size + right_size
    
    # Calculate averages and standard deviations
    avg_init_time = sum(r["initialization_time"] for r in results) / len(results)
    avg_comparison_time = sum(r["comparison_time"] for r in results) / len(results)
    avg_collection_time = sum(r["collection_time"] for r in results) / len(results)
    avg_total_time = sum(r["total_time"] for r in results) / len(results)
    
    # Calculate processing rates
    avg_records_per_second = total_records / avg_comparison_time
    avg_ms_per_record = (avg_comparison_time / total_records) * 1000
    
    # Get consistent result counts (should be the same across iterations)
    value_differences_count = results[0]["value_differences_count"]
    left_only_count = results[0]["left_only_count"]
    right_only_count = results[0]["right_only_count"]
    
    return {
        "dataset_info": {
            "left_size": left_size,
            "right_size": right_size,
            "total_records": total_records,
            "difference_rate": difference_rate,
        },
        "timing_stats": {
            "avg_initialization_time": avg_init_time,
            "avg_comparison_time": avg_comparison_time,
            "avg_collection_time": avg_collection_time,
            "avg_total_time": avg_total_time,
            "min_comparison_time": min(r["comparison_time"] for r in results),
            "max_comparison_time": max(r["comparison_time"] for r in results),
        },
        "performance_metrics": {
            "avg_records_per_second": avg_records_per_second,
            "avg_ms_per_record": avg_ms_per_record,
        },
        "comparison_results": {
            "value_differences_count": value_differences_count,
            "left_only_count": left_only_count,
            "right_only_count": right_only_count,
        },
        "raw_results": results,
    }


def print_benchmark_results(results: dict[str, Any]) -> None:
    """Print formatted benchmark results.

    Args:
        results: Dictionary with benchmark results.
    """
    dataset_info = results["dataset_info"]
    timing_stats = results["timing_stats"]
    performance_metrics = results["performance_metrics"]
    comparison_results = results["comparison_results"]
    
    print("\n" + "=" * 80)
    print("DETAILED BENCHMARK RESULTS")
    print("=" * 80)
    
    print(f"\nDataset Information:")
    print(f"  Left dataset:      {dataset_info['left_size']:,} records")
    print(f"  Right dataset:     {dataset_info['right_size']:,} records")
    print(f"  Total records:     {dataset_info['total_records']:,}")
    print(f"  Difference rate:   {dataset_info['difference_rate']:.2%}")
    
    print(f"\nTiming Statistics:")
    print(f"  Avg initialization: {timing_stats['avg_initialization_time']:.4f} seconds")
    print(f"  Avg comparison:     {timing_stats['avg_comparison_time']:.4f} seconds")
    print(f"  Avg collection:     {timing_stats['avg_collection_time']:.4f} seconds")
    print(f"  Avg total time:     {timing_stats['avg_total_time']:.4f} seconds")
    print(f"  Min comparison:     {timing_stats['min_comparison_time']:.4f} seconds")
    print(f"  Max comparison:     {timing_stats['max_comparison_time']:.4f} seconds")
    
    print(f"\nPerformance Metrics:")
    print(f"  Avg processing rate: {performance_metrics['avg_records_per_second']:,.0f} records/second")
    print(f"  Avg time per record: {performance_metrics['avg_ms_per_record']:.4f} ms/record")
    
    print(f"\nComparison Results:")
    print(f"  Value differences:   {comparison_results['value_differences_count']:,} records")
    print(f"  Left-only records:   {comparison_results['left_only_count']:,} records")
    print(f"  Right-only records:  {comparison_results['right_only_count']:,} records")


def main() -> None:
    """Run the detailed performance benchmark."""
    print("=" * 80)
    print("POLARS LAZYFRAME COMPARISON FRAMEWORK - DETAILED PERFORMANCE BENCHMARK")
    print("=" * 80)
    
    # Define test scenarios
    test_scenarios = [
        {"left_size": 1_000, "right_size": 1_050, "iterations": 3},
        {"left_size": 5_000, "right_size": 5_250, "iterations": 3},
        {"left_size": 10_000, "right_size": 10_250, "iterations": 3},
        {"left_size": 25_000, "right_size": 25_250, "iterations": 3},
        {"left_size": 50_000, "right_size": 50_250, "iterations": 2},
        {"left_size": 100_000, "right_size": 101_000, "iterations": 1},
    ]
    
    all_results = []
    
    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario['left_size']:,} vs {scenario['right_size']:,} records")
        print(f"{'='*60}")
        
        result = run_benchmark_suite(
            left_size=scenario["left_size"],
            right_size=scenario["right_size"],
            iterations=scenario["iterations"],
            difference_rate=0.01
        )
        
        print_benchmark_results(result)
        all_results.append(result)
    
    # Print summary comparison
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY COMPARISON")
    print(f"{'='*80}")
    
    print(f"\n{'Size':<15} {'Records/sec':<15} {'ms/record':<12} {'Total Time':<12}")
    print("-" * 60)
    
    for result in all_results:
        dataset_info = result["dataset_info"]
        performance_metrics = result["performance_metrics"]
        timing_stats = result["timing_stats"]
        
        size_str = f"{dataset_info['total_records']:,}"
        records_per_sec = f"{performance_metrics['avg_records_per_second']:,.0f}"
        ms_per_record = f"{performance_metrics['avg_ms_per_record']:.4f}"
        total_time = f"{timing_stats['avg_total_time']:.4f}"
        
        print(f"{size_str:<15} {records_per_sec:<15} {ms_per_record:<12} {total_time:<12}")
    
    print(f"\n{'='*80}")
    print("DETAILED BENCHMARK COMPLETED!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
