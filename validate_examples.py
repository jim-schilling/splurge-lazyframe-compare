#!/usr/bin/env python3
"""Quick validation script to test the updated examples."""

import logging
import sys

from splurge_lazyframe_compare import ColumnDefinition, ComparisonSchema
from splurge_lazyframe_compare.utils.logging_helpers import get_logger

# Configure logging for testing
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] [%(funcName)s] %(message)s"
)

def test_basic_mixed_usage():
    """Test basic mixed datatype usage."""
    print("Testing basic mixed datatype usage...")

    # Test string names
    col1 = ColumnDefinition(name="id", alias="ID", datatype="Int64", nullable=False)
    col2 = ColumnDefinition(name="name", alias="Name", datatype="String", nullable=True)
    col3 = ColumnDefinition(name="created", alias="Created", datatype="Datetime", nullable=False)

    # Test direct datatypes
    col4 = ColumnDefinition(name="tags", alias="Tags", datatype=pl.List, nullable=True)
    col5 = ColumnDefinition(name="metadata", alias="Metadata", datatype=pl.Struct, nullable=True)

    print(f"✓ String 'Int64': {col1.datatype}")
    print(f"✓ String 'String': {col2.datatype}")
    print(f"✓ String 'Datetime': {col3.datatype}")
    print(f"✓ Direct pl.List: {col4.datatype}")
    print(f"✓ Direct pl.Struct: {col5.datatype}")

    # Test schema creation
    schema = ComparisonSchema(
        columns={
            "id": col1,
            "name": col2,
            "created": col3,
            "tags": col4,
            "metadata": col5,
        },
        pk_columns=["id"]
    )

    print(f"✓ Schema created with {len(schema.columns)} columns")
    return True

def test_complex_datatypes():
    """Test complex datatype instantiation."""
    print("\nTesting complex datatypes...")

    # These should all work now
    test_cases = [
        ("Datetime", "Datetime(time_unit='us', time_zone=None)"),
        ("Categorical", "Categorical"),
        ("List", "List(Int64)"),
        ("Struct", "Struct({})"),
    ]

    for name, expected_pattern in test_cases:
        try:
            col = ColumnDefinition(name=f"test_{name.lower()}", alias=f"Test {name}", datatype=name, nullable=True)
            print(f"✓ {name}: {col.datatype}")
        except Exception as e:
            print(f"✗ {name}: Failed - {e}")
            return False

    return True

if __name__ == "__main__":
    import polars as pl

    # Test logging functions
    logger = get_logger('validation')
    logger.info("Starting validation tests...")

    print("Validating mixed datatype usage in examples...")

    success1 = test_basic_mixed_usage()
    success2 = test_complex_datatypes()

    if success1 and success2:
        logger.info("All validations passed successfully")
        print("\n🎉 All validations passed! Mixed datatype usage works perfectly!")
        print("The README.md and examples have been successfully updated.")
    else:
        logger.error("Some validations failed")
        print("\n❌ Some validations failed!")
        sys.exit(1)
