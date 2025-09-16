import polars as pl

from splurge_lazyframe_compare.models.schema import ColumnDefinition, ColumnMapping, ComparisonConfig, ComparisonSchema
from splurge_lazyframe_compare.services.preparation_service import DataPreparationService


def _make_config(ignore_case: bool = False) -> tuple[ComparisonConfig, list[str]]:
    left_columns = {
        "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
        "name": ColumnDefinition(name="name", alias="Name", datatype=pl.Utf8, nullable=False),
    }
    right_columns = {
        "customer_id": ColumnDefinition(name="customer_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
        "full_name": ColumnDefinition(name="full_name", alias="Full Name", datatype=pl.Utf8, nullable=False),
    }
    left_schema = ComparisonSchema(columns=left_columns, pk_columns=["id"])
    right_schema = ComparisonSchema(columns=right_columns, pk_columns=["customer_id"])
    mappings = [
        ColumnMapping(left="id", right="customer_id", name="id"),
        ColumnMapping(left="name", right="full_name", name="name"),
    ]
    config = ComparisonConfig(
        left_schema=left_schema,
        right_schema=right_schema,
        column_mappings=mappings,
        pk_columns=["id"],
        ignore_case=ignore_case,
    )
    expected_order = ["PK_id", "L_name", "R_name"]
    return config, expected_order


def test_apply_column_mappings_and_orders_columns():
    service = DataPreparationService()
    config, expected_order = _make_config()

    left = pl.LazyFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
    right = pl.LazyFrame({"customer_id": [1, 3], "full_name": ["Alice", "Charlie"]})

    mapped_left, mapped_right = service.apply_column_mappings(left=left, right=right, config=config)

    assert mapped_left.collect().columns == ["PK_id", "L_name"]
    assert mapped_right.collect().columns == ["PK_id", "R_name"]

    assert service.get_alternating_column_order(config=config) == expected_order
    assert service.get_left_only_column_order(config=config) == ["PK_id", "L_name"]
    assert service.get_right_only_column_order(config=config) == ["PK_id", "R_name"]


def test_prepare_dataframes_with_ignore_case():
    service = DataPreparationService()
    config, _ = _make_config(ignore_case=True)

    left = pl.LazyFrame({"id": [1], "name": ["ALICE"]})
    right = pl.LazyFrame({"customer_id": [1], "full_name": ["alice"]})

    prepared_left, prepared_right = service.prepare_dataframes(left=left, right=right, config=config)

    df_left = prepared_left.collect()
    df_right = prepared_right.collect()

    assert df_left["L_name"][0] == "alice"
    assert df_right["R_name"][0] == "alice"
