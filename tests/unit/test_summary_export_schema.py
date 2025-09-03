import json
import tempfile
from pathlib import Path

import polars as pl

from splurge_lazyframe_compare.services.reporting_service import ReportingService
from splurge_lazyframe_compare.models.comparison import ComparisonResult, ComparisonSummary
from splurge_lazyframe_compare.utils.constants import SUMMARY_SCHEMA_VERSION


def _make_empty_result() -> ComparisonResult:
    summary = ComparisonSummary(
        total_left_records=0,
        total_right_records=0,
        matching_records=0,
        value_differences_count=0,
        left_only_count=0,
        right_only_count=0,
        comparison_timestamp="2025-01-01T00:00:00",
    )
    empty = pl.LazyFrame({"_": []})
    return ComparisonResult(
        summary=summary,
        value_differences=empty,
        left_only_records=empty,
        right_only_records=empty,
        config=None,  # type: ignore[arg-type]
    )


def test_export_summary_envelope_has_version_and_summary() -> None:
    reporter = ReportingService()
    results = _make_empty_result()

    with tempfile.TemporaryDirectory() as tmp:
        exported = reporter.export_results(results=results, format="parquet", output_dir=tmp)
        summary_path = Path(exported["summary"])  # always json
        assert summary_path.exists()

        data = json.loads(summary_path.read_text())
        assert data["schema_version"] == SUMMARY_SCHEMA_VERSION
        assert "summary" in data
        assert set(data["summary"].keys()) >= {
            "total_left_records",
            "total_right_records",
            "matching_records",
            "value_differences_count",
            "left_only_count",
            "right_only_count",
            "comparison_timestamp",
        }

