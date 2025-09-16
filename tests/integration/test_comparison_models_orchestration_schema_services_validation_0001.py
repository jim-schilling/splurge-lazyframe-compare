import polars as pl
import pytest

from splurge_lazyframe_compare.services.orchestrator import ComparisonOrchestrator
from splurge_lazyframe_compare.models.schema import (
    ComparisonConfig,
    ComparisonSchema,
    ColumnDefinition,
    ColumnMapping,
)


@pytest.mark.perf
def test_medium_dataset_smoke() -> None:
    num = 50_000
    left = pl.LazyFrame({
        "id": list(range(num)),
        "value": [i % 5 for i in range(num)],
    })
    right = pl.LazyFrame({
        "cust_id": list(range(num)),
        "val": [i % 5 for i in range(num)],
    })

    left_schema = ComparisonSchema(
        columns={
            "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
            "value": ColumnDefinition(name="value", alias="Value", datatype=pl.Int64, nullable=False),
        },
        pk_columns=["id"],
    )
    right_schema = ComparisonSchema(
        columns={
            "cust_id": ColumnDefinition(name="cust_id", alias="ID", datatype=pl.Int64, nullable=False),
            "val": ColumnDefinition(name="val", alias="Value", datatype=pl.Int64, nullable=False),
        },
        pk_columns=["cust_id"],
    )

    config = ComparisonConfig(
        left_schema=left_schema,
        right_schema=right_schema,
        column_mappings=[
            ColumnMapping(name="id", left="id", right="cust_id"),
            ColumnMapping(name="value", left="value", right="val"),
        ],
        pk_columns=["id"],
    )

    orchestrator = ComparisonOrchestrator()
    result = orchestrator.compare_dataframes(config=config, left=left, right=right)
    assert result.summary.matching_records == num

