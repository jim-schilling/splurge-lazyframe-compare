import polars as pl
import pytest

from splurge_lazyframe_compare import ComparisonOrchestrator
from splurge_lazyframe_compare.models.schema import ColumnDefinition, ColumnMapping, ComparisonConfig, ComparisonSchema


def _make_basic_config() -> ComparisonConfig:
    left_columns = {
        "id": ColumnDefinition(name="id", alias="ID", datatype=pl.Int64, nullable=False),
        "val": ColumnDefinition(name="val", alias="Value", datatype=pl.Int64, nullable=False),
    }
    right_columns = {
        "cust_id": ColumnDefinition(name="cust_id", alias="Customer ID", datatype=pl.Int64, nullable=False),
        "amount": ColumnDefinition(name="amount", alias="Amount", datatype=pl.Int64, nullable=False),
    }
    left_schema = ComparisonSchema(columns=left_columns, pk_columns=["id"])
    right_schema = ComparisonSchema(columns=right_columns, pk_columns=["cust_id"])
    mappings = [
        ColumnMapping(left="id", right="cust_id", name="id"),
        ColumnMapping(left="val", right="amount", name="val"),
    ]
    return ComparisonConfig(
        left_schema=left_schema,
        right_schema=right_schema,
        column_mappings=mappings,
        pk_columns=["id"],
    )


def _make_frames() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    left = pl.LazyFrame({"id": [1, 2], "val": [10, 20]})
    right = pl.LazyFrame({"cust_id": [1, 3], "amount": [10, 30]})
    return left, right


def test_generate_report_from_result_variants():
    orch = ComparisonOrchestrator()
    cfg = _make_basic_config()
    left, right = _make_frames()
    res = orch.compare_dataframes(config=cfg, left=left, right=right)

    # summary
    s = orch.generate_report_from_result(result=res, report_type="summary")
    assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in s

    # detailed
    d = orch.generate_report_from_result(result=res, report_type="detailed", max_samples=1, table_format="simple")
    assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in d

    # table
    t = orch.generate_report_from_result(result=res, report_type="table")
    assert "Metric" in t or "Left DataFrame" in t


def test_generate_report_from_result_invalid_type():
    orch = ComparisonOrchestrator()
    cfg = _make_basic_config()
    left, right = _make_frames()
    res = orch.compare_dataframes(config=cfg, left=left, right=right)

    with pytest.raises(ValueError):
        orch.generate_report_from_result(result=res, report_type="unknown")


def test_getters_use_reporting_service():
    orch = ComparisonOrchestrator()
    cfg = _make_basic_config()
    left, right = _make_frames()
    res = orch.compare_dataframes(config=cfg, left=left, right=right)

    summary = orch.get_comparison_summary(result=res)
    assert "SPLURGE LAZYFRAME COMPARISON SUMMARY" in summary

    table = orch.get_comparison_table(result=res, table_format="simple")
    assert isinstance(table, str)
    assert len(table) > 0


def test_input_validation_errors():
    orch = ComparisonOrchestrator()
    cfg = _make_basic_config()
    left, right = _make_frames()

    with pytest.raises(ValueError, match="left must be a polars LazyFrame"):
        orch.compare_dataframes(config=cfg, left="x", right=right)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="right must be a polars LazyFrame"):
        orch.compare_dataframes(config=cfg, left=left, right="y")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="config must be a ComparisonConfig"):
        orch.compare_dataframes(config="z", left=left, right=right)  # type: ignore[arg-type]
