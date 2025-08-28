"""Result containers and reporting for the comparison framework."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from tabulate import tabulate

from splurge_lazyframe_compare.core.schema import ComparisonConfig

# Private constants
_DEFAULT_OUTPUT_DIR = "."
_DEFAULT_FORMAT = "parquet"
_FORMAT_PARQUET = "parquet"
_FORMAT_CSV = "csv"
_FORMAT_JSON = "json"
_VALUE_DIFFERENCES_FILENAME = "value_differences_{}.{}"
_LEFT_ONLY_FILENAME = "left_only_records_{}.{}"
_RIGHT_ONLY_FILENAME = "right_only_records_{}.{}"
_SUMMARY_FILENAME = "comparison_summary_{}.json"
_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
_REPORT_HEADER_LENGTH = 60
_SECTION_SEPARATOR_LENGTH = 40
_REPORT_HEADER = "=" * _REPORT_HEADER_LENGTH
_REPORT_TITLE = "SPLURGE LAZYFRAME COMPARISON SUMMARY"
_RECORD_COUNTS_SECTION = "RECORD COUNTS:"
_COMPARISON_RESULTS_SECTION = "COMPARISON RESULTS:"
_PERCENTAGES_SECTION = "PERCENTAGES (of Left DataFrame):"
_VALUE_DIFFERENCES_SECTION = "VALUE DIFFERENCES SAMPLES:"
_LEFT_ONLY_SECTION = "LEFT-ONLY RECORDS SAMPLES:"
_RIGHT_ONLY_SECTION = "RIGHT-ONLY RECORDS SAMPLES:"
_SECTION_SEPARATOR = "-" * _SECTION_SEPARATOR_LENGTH
_ZERO_THRESHOLD = 0
_DEFAULT_MAX_SAMPLES = 10
_JSON_INDENT = 2
_PERCENTAGE_MULTIPLIER = 100
_PERCENTAGE_FORMAT = ".1f"


@dataclass
class ComparisonSummary:
    """Summary statistics for comparison results.

    Attributes:
        total_left_records: Total number of records in left DataFrame.
        total_right_records: Total number of records in right DataFrame.
        matching_records: Number of records that match exactly.
        value_differences_count: Number of records with value differences.
        left_only_count: Number of records only in left DataFrame.
        right_only_count: Number of records only in right DataFrame.
        comparison_timestamp: Timestamp when comparison was performed.
    """

    total_left_records: int
    total_right_records: int
    matching_records: int
    value_differences_count: int
    left_only_count: int
    right_only_count: int
    comparison_timestamp: str

    @classmethod
    def create(
        cls,
        *,
        total_left_records: int,
        total_right_records: int,
        value_differences: pl.LazyFrame,
        left_only_records: pl.LazyFrame,
        right_only_records: pl.LazyFrame,
    ) -> "ComparisonSummary":
        """Create a comparison summary from result DataFrames.

        Args:
            total_left_records: Total records in left DataFrame.
            total_right_records: Total records in right DataFrame.
            value_differences: LazyFrame containing value differences.
            left_only_records: LazyFrame containing left-only records.
            right_only_records: LazyFrame containing right-only records.

        Returns:
            ComparisonSummary with calculated statistics.
        """
        value_differences_count = value_differences.select(pl.len()).collect().item()
        left_only_count = left_only_records.select(pl.len()).collect().item()
        right_only_count = right_only_records.select(pl.len()).collect().item()

        # Calculate matching records
        matching_records = total_left_records - left_only_count - value_differences_count

        return cls(
            total_left_records=total_left_records,
            total_right_records=total_right_records,
            matching_records=matching_records,
            value_differences_count=value_differences_count,
            left_only_count=left_only_count,
            right_only_count=right_only_count,
            comparison_timestamp=datetime.now().isoformat(),
        )


@dataclass
class ValueDifference:
    """Details of a specific value difference.

    Attributes:
        primary_key_values: Dictionary of primary key column values.
        column_name: Name of the column with the difference.
        left_value: Value from left DataFrame.
        right_value: Value from right DataFrame.
        friendly_column_name: Human-readable column name.
    """

    primary_key_values: dict[str, Any]
    column_name: str
    left_value: Any
    right_value: Any
    friendly_column_name: str


@dataclass
class ComparisonResults:
    """Container for all comparison results.

    Attributes:
        summary: Summary statistics for the comparison.
        value_differences: LazyFrame containing records with value differences.
        left_only_records: LazyFrame containing records only in left DataFrame.
        right_only_records: LazyFrame containing records only in right DataFrame.
        config: Configuration used for the comparison.
    """

    summary: ComparisonSummary
    value_differences: pl.LazyFrame
    left_only_records: pl.LazyFrame
    right_only_records: pl.LazyFrame
    config: ComparisonConfig

    def to_report(self) -> "ComparisonReport":
        """Generate human-readable comparison report.

        Returns:
            ComparisonReport object for generating reports.
        """
        return ComparisonReport(self)

    def export_results(
        self,
        *,
        format: str = _DEFAULT_FORMAT,
        output_dir: str = _DEFAULT_OUTPUT_DIR
    ) -> dict[str, str]:
        """Export results to files.

        Args:
            format: Output format for files (parquet, csv, json).
            output_dir: Directory to save output files.

        Returns:
            Dictionary mapping result type to file path.
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
        results = {}

        # Export value differences
        if self.value_differences.select(pl.len()).collect().item() > _ZERO_THRESHOLD:
            value_diff_path = output_path / _VALUE_DIFFERENCES_FILENAME.format(timestamp, format)
            self._export_lazyframe(lazyframe=self.value_differences, file_path=value_diff_path, format=format)
            results["value_differences"] = str(value_diff_path)

        # Export left-only records
        if self.left_only_records.select(pl.len()).collect().item() > _ZERO_THRESHOLD:
            left_only_path = output_path / _LEFT_ONLY_FILENAME.format(timestamp, format)
            self._export_lazyframe(lazyframe=self.left_only_records, file_path=left_only_path, format=format)
            results["left_only_records"] = str(left_only_path)

        # Export right-only records
        if self.right_only_records.select(pl.len()).collect().item() > _ZERO_THRESHOLD:
            right_only_path = output_path / _RIGHT_ONLY_FILENAME.format(timestamp, format)
            self._export_lazyframe(lazyframe=self.right_only_records, file_path=right_only_path, format=format)
            results["right_only_records"] = str(right_only_path)

        # Export summary
        summary_path = output_path / _SUMMARY_FILENAME.format(timestamp)
        self._export_summary(file_path=summary_path)
        results["summary"] = str(summary_path)

        return results

    def _export_lazyframe(
        self,
        *,
        lazyframe: pl.LazyFrame,
        file_path: Path,
        format: str = _DEFAULT_FORMAT
    ) -> None:
        """Export a LazyFrame to a file.

        Args:
            lazyframe: LazyFrame to export.
            file_path: Path to save the file.
            format: Output format.
        """
        if format.lower() == _FORMAT_PARQUET:
            lazyframe.sink_parquet(file_path)
        elif format.lower() == _FORMAT_CSV:
            lazyframe.sink_csv(file_path)
        elif format.lower() == _FORMAT_JSON:
            lazyframe.sink_json(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_summary(self, *, file_path: Path) -> None:
        """Export summary to JSON file.

        Args:
            file_path: Path to save the summary file.
        """
        import json

        summary_dict = {
            "total_left_records": self.summary.total_left_records,
            "total_right_records": self.summary.total_right_records,
            "matching_records": self.summary.matching_records,
            "value_differences_count": self.summary.value_differences_count,
            "left_only_count": self.summary.left_only_count,
            "right_only_count": self.summary.right_only_count,
            "comparison_timestamp": self.summary.comparison_timestamp,
        }

        with open(file_path, "w") as f:
            json.dump(summary_dict, f, indent=_JSON_INDENT)


class ComparisonReport:
    """Human-readable comparison report generator.

    Attributes:
        results: ComparisonResults object containing the comparison data.
    """

    def __init__(self, results: ComparisonResults) -> None:
        """Initialize the report generator.

        Args:
            results: ComparisonResults object to generate reports from.
        """
        self.results = results

    def generate_summary_report(self) -> str:
        """Generate summary report as string.

        Returns:
            Formatted summary report string.
        """
        summary = self.results.summary

        report_lines = [
            _REPORT_HEADER,
            _REPORT_TITLE,
            _REPORT_HEADER,
            f"Comparison Timestamp: {summary.comparison_timestamp}",
            "",
            _RECORD_COUNTS_SECTION,
            f"  Left DataFrame:  {summary.total_left_records:,} records",
            f"  Right DataFrame: {summary.total_right_records:,} records",
            "",
            _COMPARISON_RESULTS_SECTION,
            f"  Matching Records:      {summary.matching_records:,}",
            f"  Value Differences:     {summary.value_differences_count:,}",
            f"  Left-Only Records:     {summary.left_only_count:,}",
            f"  Right-Only Records:    {summary.right_only_count:,}",
            "",
        ]

        # Calculate percentages
        if summary.total_left_records > _ZERO_THRESHOLD:
            match_pct = (summary.matching_records / summary.total_left_records) * _PERCENTAGE_MULTIPLIER
            diff_pct = (summary.value_differences_count / summary.total_left_records) * _PERCENTAGE_MULTIPLIER
            left_only_pct = (summary.left_only_count / summary.total_left_records) * _PERCENTAGE_MULTIPLIER

            report_lines.extend([
                _PERCENTAGES_SECTION,
                f"  Matching:      {match_pct:{_PERCENTAGE_FORMAT}}%",
                f"  Differences:   {diff_pct:{_PERCENTAGE_FORMAT}}%",
                f"  Left-Only:     {left_only_pct:{_PERCENTAGE_FORMAT}}%",
                "",
            ])

        if summary.total_right_records > _ZERO_THRESHOLD:
            right_only_pct = (summary.right_only_count / summary.total_right_records) * _PERCENTAGE_MULTIPLIER
            report_lines.append(f"  Right-Only:    {right_only_pct:{_PERCENTAGE_FORMAT}}% (of Right DataFrame)")

        report_lines.append("=" * _REPORT_HEADER_LENGTH)

        return "\n".join(report_lines)

    def generate_detailed_report(
        self,
        *,
        max_samples: int = _DEFAULT_MAX_SAMPLES,
        table_format: str = "grid"
    ) -> str:
        """Generate detailed report with sample differences.

        Args:
            max_samples: Maximum number of sample records to include.
            table_format: Table format for tabulate (grid, simple, pipe, orgtbl, etc.).

        Returns:
            Formatted detailed report string.
        """
        report_lines = [self.generate_summary_report(), ""]

        # Add value differences samples
        if self.results.summary.value_differences_count > _ZERO_THRESHOLD:
            report_lines.extend([
                _VALUE_DIFFERENCES_SECTION,
                _SECTION_SEPARATOR,
            ])

            # Get sample of value differences
            sample_diff = self.results.value_differences.limit(max_samples).collect()
            if not sample_diff.is_empty():
                try:
                    # Convert Polars DataFrame to list of lists for tabulate
                    headers = sample_diff.columns
                    data = sample_diff.to_dicts()
                    table_data = [[row[col] for col in headers] for row in data]

                    table = tabulate(
                        table_data,
                        headers=headers,
                        tablefmt=table_format,
                        showindex=False,
                        numalign='right',
                        stralign='left'
                    )
                    report_lines.append(table)
                except Exception:
                    # Fallback to original format if tabulate fails
                    report_lines.append(str(sample_diff))
            else:
                report_lines.append("No value differences found.")

            report_lines.append("")

        # Add left-only samples
        if self.results.summary.left_only_count > _ZERO_THRESHOLD:
            report_lines.extend([
                _LEFT_ONLY_SECTION,
                _SECTION_SEPARATOR,
            ])

            sample_left = self.results.left_only_records.limit(max_samples).collect()
            if not sample_left.is_empty():
                try:
                    # Convert Polars DataFrame to list of lists for tabulate
                    headers = sample_left.columns
                    data = sample_left.to_dicts()
                    table_data = [[row[col] for col in headers] for row in data]

                    table = tabulate(
                        table_data,
                        headers=headers,
                        tablefmt=table_format,
                        showindex=False,
                        numalign='right',
                        stralign='left'
                    )
                    report_lines.append(table)
                except Exception:
                    # Fallback to original format if tabulate fails
                    report_lines.append(str(sample_left))
            else:
                report_lines.append("No left-only records found.")

            report_lines.append("")

        # Add right-only samples
        if self.results.summary.right_only_count > _ZERO_THRESHOLD:
            report_lines.extend([
                _RIGHT_ONLY_SECTION,
                _SECTION_SEPARATOR,
            ])

            sample_right = self.results.right_only_records.limit(max_samples).collect()
            if not sample_right.is_empty():
                try:
                    # Convert Polars DataFrame to list of lists for tabulate
                    headers = sample_right.columns
                    data = sample_right.to_dicts()
                    table_data = [[row[col] for col in headers] for row in data]

                    table = tabulate(
                        table_data,
                        headers=headers,
                        tablefmt=table_format,
                        showindex=False,
                        numalign='right',
                        stralign='left'
                    )
                    report_lines.append(table)
                except Exception:
                    # Fallback to original format if tabulate fails
                    report_lines.append(str(sample_right))
            else:
                report_lines.append("No right-only records found.")

        return "\n".join(report_lines)

    def generate_summary_table(self, *, table_format: str = "grid") -> str:
        """Generate summary statistics as a formatted table.

        Args:
            table_format: Table format for tabulate (grid, simple, pipe, orgtbl, etc.).

        Returns:
            Formatted summary table string.
        """
        summary = self.results.summary

        # Prepare table data
        table_data = [
            ["Left DataFrame", summary.total_left_records, "records"],
            ["Right DataFrame", summary.total_right_records, "records"],
            ["Matching Records", summary.matching_records, "records"],
            ["Value Differences", summary.value_differences_count, "records"],
            ["Left-Only Records", summary.left_only_count, "records"],
            ["Right-Only Records", summary.right_only_count, "records"],
        ]

        # Calculate percentages if applicable
        if summary.total_left_records > _ZERO_THRESHOLD:
            match_pct = (summary.matching_records / summary.total_left_records) * _PERCENTAGE_MULTIPLIER
            diff_pct = (summary.value_differences_count / summary.total_left_records) * _PERCENTAGE_MULTIPLIER
            left_only_pct = (summary.left_only_count / summary.total_left_records) * _PERCENTAGE_MULTIPLIER

            table_data.extend([
                ["Matching %", f"{match_pct:{_PERCENTAGE_FORMAT}}%", "of left records"],
                ["Differences %", f"{diff_pct:{_PERCENTAGE_FORMAT}}%", "of left records"],
                ["Left-Only %", f"{left_only_pct:{_PERCENTAGE_FORMAT}}%", "of left records"],
            ])

        if summary.total_right_records > _ZERO_THRESHOLD:
            right_only_pct = (summary.right_only_count / summary.total_right_records) * _PERCENTAGE_MULTIPLIER
            table_data.append(["Right-Only %", f"{right_only_pct:{_PERCENTAGE_FORMAT}}%", "of right records"])

        # Format with tabulate
        try:
            table = tabulate(
                table_data,
                headers=["Metric", "Value", "Unit"],
                tablefmt=table_format,
                numalign='right',
                stralign='left'
            )
            return table
        except Exception:
            # Fallback to simple format
            return tabulate(table_data, headers=["Metric", "Value", "Unit"], tablefmt="simple")

    def generate_tabulated_report(
        self,
        *,
        max_samples: int = _DEFAULT_MAX_SAMPLES,
        table_format: str = "grid"
    ) -> str:
        """Generate a complete report with tabulated formatting.

        Args:
            max_samples: Maximum number of sample records to include.
            table_format: Table format for tabulate (grid, simple, pipe, orgtbl, etc.).

        Returns:
            Formatted tabulated report string.
        """
        report_lines = [
            _REPORT_HEADER,
            _REPORT_TITLE,
            _REPORT_HEADER,
            f"Comparison Timestamp: {self.results.summary.comparison_timestamp}",
            "",
            "SUMMARY STATISTICS:",
            _SECTION_SEPARATOR,
            "",
            self.generate_summary_table(table_format=table_format),
            "",
        ]

        # Add value differences samples
        if self.results.summary.value_differences_count > _ZERO_THRESHOLD:
            report_lines.extend([
                _VALUE_DIFFERENCES_SECTION,
                _SECTION_SEPARATOR,
                f"Showing up to {max_samples} sample records:",
                "",
            ])

            # Get sample of value differences
            sample_diff = self.results.value_differences.limit(max_samples).collect()
            if not sample_diff.is_empty():
                try:
                    # Convert Polars DataFrame to list of lists for tabulate
                    headers = sample_diff.columns
                    data = sample_diff.to_dicts()
                    table_data = [[row[col] for col in headers] for row in data]

                    table = tabulate(
                        table_data,
                        headers=headers,
                        tablefmt=table_format,
                        showindex=False,
                        numalign='right',
                        stralign='left'
                    )
                    report_lines.append(table)
                except Exception:
                    # Fallback to original format if tabulate fails
                    report_lines.append(str(sample_diff))
            else:
                report_lines.append("No value differences found.")

            report_lines.append("")

        # Add left-only samples
        if self.results.summary.left_only_count > _ZERO_THRESHOLD:
            report_lines.extend([
                _LEFT_ONLY_SECTION,
                _SECTION_SEPARATOR,
                f"Showing up to {max_samples} sample records:",
                "",
            ])

            sample_left = self.results.left_only_records.limit(max_samples).collect()
            if not sample_left.is_empty():
                try:
                    # Convert Polars DataFrame to list of lists for tabulate
                    headers = sample_left.columns
                    data = sample_left.to_dicts()
                    table_data = [[row[col] for col in headers] for row in data]

                    table = tabulate(
                        table_data,
                        headers=headers,
                        tablefmt=table_format,
                        showindex=False,
                        numalign='right',
                        stralign='left'
                    )
                    report_lines.append(table)
                except Exception:
                    # Fallback to original format if tabulate fails
                    report_lines.append(str(sample_left))
            else:
                report_lines.append("No left-only records found.")

            report_lines.append("")

        # Add right-only samples
        if self.results.summary.right_only_count > _ZERO_THRESHOLD:
            report_lines.extend([
                _RIGHT_ONLY_SECTION,
                _SECTION_SEPARATOR,
                f"Showing up to {max_samples} sample records:",
                "",
            ])

            sample_right = self.results.right_only_records.limit(max_samples).collect()
            if not sample_right.is_empty():
                try:
                    # Convert Polars DataFrame to list of lists for tabulate
                    headers = sample_right.columns
                    data = sample_right.to_dicts()
                    table_data = [[row[col] for col in headers] for row in data]

                    table = tabulate(
                        table_data,
                        headers=headers,
                        tablefmt=table_format,
                        showindex=False,
                        numalign='right',
                        stralign='left'
                    )
                    report_lines.append(table)
                except Exception:
                    # Fallback to original format if tabulate fails
                    report_lines.append(str(sample_right))
            else:
                report_lines.append("No right-only records found.")

        report_lines.append("=" * _REPORT_HEADER_LENGTH)
        return "\n".join(report_lines)

    def export_to_html(self, filename: str) -> None:
        """Export report as HTML file.

        Args:
            filename: Path to save the HTML file.
        """
        summary = self.results.summary

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Polars LazyFrame Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .section {{ margin: 15px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4f8; border-radius: 5px; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c5aa0; }}
        .metric-label {{ font-size: 12px; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Polars LazyFrame Comparison Report</h1>
        <p>Generated on: {summary.comparison_timestamp}</p>
    </div>

    <div class="summary">
        <h2>Summary</h2>
        <div class="metric">
            <div class="metric-value">{summary.total_left_records:,}</div>
            <div class="metric-label">Left Records</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.total_right_records:,}</div>
            <div class="metric-label">Right Records</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.matching_records:,}</div>
            <div class="metric-label">Matching</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.value_differences_count:,}</div>
            <div class="metric-label">Differences</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.left_only_count:,}</div>
            <div class="metric-label">Left Only</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.right_only_count:,}</div>
            <div class="metric-label">Right Only</div>
        </div>
    </div>

    <div class="section">
        <h2>Configuration</h2>
        <p><strong>Primary Key Columns:</strong> {', '.join(self.results.config.primary_key_columns)}</p>
        <p><strong>Column Mappings:</strong> {len(self.results.config.column_mappings)} mappings</p>
        <p><strong>Null Equals Null:</strong> {self.results.config.null_equals_null}</p>
        <p><strong>Ignore Case:</strong> {self.results.config.ignore_case}</p>
    </div>
</body>
</html>
        """

        with open(filename, "w") as f:
            f.write(html_content)
