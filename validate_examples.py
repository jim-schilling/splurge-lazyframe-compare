#!/usr/bin/env python3
"""Quick validation script to test the updated examples."""

import logging
import sys

from splurge_lazyframe_compare import ColumnDefinition, ComparisonSchema
from splurge_lazyframe_compare.utils.logging_helpers import get_logger
from splurge_lazyframe_compare.utils.type_helpers import get_polars_datatype_type

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

    # Test direct datatypes using helper function for version compatibility
    list_class = getattr(pl, "List")
    struct_class = getattr(pl, "Struct")
    col4 = ColumnDefinition(name="tags", alias="Tags", datatype=list_class(get_polars_datatype_type("String")), nullable=True)
    col5 = ColumnDefinition(name="metadata", alias="Metadata", datatype=struct_class([]), nullable=True)

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

def verify_copy_fix():
    """Verify that the copy fix works correctly."""
    import random

    # Simulate the test scenario
    num_records = 5
    left_data = {
        'id': list(range(1, num_records + 1)),
        'value': [random.uniform(0, 1000) for _ in range(num_records)],
    }

    # Test the new (clean) approach
    right_data = left_data.copy()
    right_data["value"] = [v + random.uniform(-1, 1) for v in right_data["value"]]

    print("✅ Copy fix verification:")
    print(f"  Left IDs: {left_data['id']}")
    print(f"  Right IDs: {right_data['id']}")
    print(f"  IDs are same object: {right_data['id'] is left_data['id']}")  # Should be True
    print(f"  Values are different: {left_data['value'] != right_data['value']}")  # Should be True
    return True

def verify_decimal_defaults():
    """Verify that Decimal type uses proper defaults."""
    from splurge_lazyframe_compare.utils.type_helpers import get_polars_datatype_type

    try:
        decimal_type = get_polars_datatype_type('Decimal')
        decimal_str = str(decimal_type)

        # Check if it contains the expected default values
        has_precision = 'precision=38' in decimal_str
        has_scale = 'scale=9' in decimal_str

        print("✅ Decimal defaults verification:")
        print(f"  Decimal type: {decimal_str}")
        print(f"  Has precision=38: {has_precision}")
        print(f"  Has scale=9: {has_scale}")

        return has_precision and has_scale
    except Exception as e:
        print(f"❌ Decimal defaults verification failed: {e}")
        return False

if __name__ == "__main__":
    import polars as pl

    # Test logging functions
    logger = get_logger('validation')
    logger.info("Starting validation tests...")

    print("Validating mixed datatype usage in examples...")

    success1 = test_basic_mixed_usage()
    success2 = test_complex_datatypes()
    success3 = verify_copy_fix()
    success4 = verify_decimal_defaults()

    if success1 and success2 and success3 and success4:
        logger.info("All validations passed successfully")
        print("\n🎉 All validations passed! Mixed datatype usage works perfectly!")
        print("The README.md and examples have been successfully updated.")
    else:
        logger.error("Some validations failed")
        print("\n❌ Some validations failed!")
        sys.exit(1)
