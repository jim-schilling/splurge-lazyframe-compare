"""Reporting service for the comparison framework."""

import json
import os
from datetime import datetime
from pathlib import Path

import polars as pl
from tabulate import tabulate

from splurge_lazyframe_compare.models.comparison import ComparisonResult
from splurge_lazyframe_compare.services.base_service import BaseService
from splurge_lazyframe_compare.utils.constants import (
    COMPARISON_RESULTS_SECTION,
    DEFAULT_FORMAT,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_OUTPUT_DIR,
    JSON_INDENT,
    LEFT_ONLY_FILENAME,
    LEFT_ONLY_SECTION,
    PERCENTAGES_SECTION,
    RECORD_COUNTS_SECTION,
    REPORT_HEADER,
    REPORT_HEADER_LENGTH,
    REPORT_TITLE,
    RIGHT_ONLY_FILENAME,
    RIGHT_ONLY_SECTION,
    SECTION_SEPARATOR,
    SUMMARY_FILENAME,
    TIMESTAMP_FORMAT,
    VALUE_DIFFERENCES_FILENAME,
    VALUE_DIFFERENCES_SECTION,
    ZERO_THRESHOLD,
)
from splurge_lazyframe_compare.utils.file_operations import FileOperationConstants, export_lazyframe
from splurge_lazyframe_compare.utils.formatting import (
    format_large_number,
    format_percentage,
)


class ReportingService(BaseService):
    """Reporting service for comparison results.

    This service handles the generation of human-readable reports,
    export of results to various formats, and presentation of
    comparison findings.
    """

    def __init__(self) -> None:
        """Initialize the reporting service."""
        super().__init__("ReportingService")

    def _validate_inputs(self, **kwargs) -> None:
        """Validate service inputs."""
        if "results" in kwargs and not isinstance(kwargs["results"], ComparisonResult):
            raise ValueError("results must be a ComparisonResult")

    def generate_summary_report(self, *, results: ComparisonResult) -> str:
        """Generate summary report as string.

        Args:
            results: ComparisonResult object containing the comparison data.

        Returns:
            Formatted summary report string.
        """
        self._validate_inputs(results=results)
        try:
            summary = results.summary

            report_lines = [
                REPORT_HEADER,
                REPORT_TITLE,
                REPORT_HEADER,
                f"Comparison Timestamp: {summary.comparison_timestamp}",
                "",
                RECORD_COUNTS_SECTION,
                f"  Left DataFrame:  {format_large_number(summary.total_left_records)} records",
                f"  Right DataFrame: {format_large_number(summary.total_right_records)} records",
                "",
                COMPARISON_RESULTS_SECTION,
                f"  Matching Records:      {format_large_number(summary.matching_records)}",
                f"  Value Differences:     {format_large_number(summary.value_differences_count)}",
                f"  Left-Only Records:     {format_large_number(summary.left_only_count)}",
                f"  Right-Only Records:    {format_large_number(summary.right_only_count)}",
                "",
            ]

            # Calculate percentages
            if summary.total_left_records > ZERO_THRESHOLD:
                match_pct = summary.matching_records / summary.total_left_records
                diff_pct = summary.value_differences_count / summary.total_left_records
                left_only_pct = summary.left_only_count / summary.total_left_records

                report_lines.extend([
                    PERCENTAGES_SECTION,
                    f"  Matching:      {format_percentage(match_pct)}",
                    f"  Differences:   {format_percentage(diff_pct)}",
                    f"  Left-Only:     {format_percentage(left_only_pct)}",
                    "",
                ])

            if summary.total_right_records > ZERO_THRESHOLD:
                right_only_pct = summary.right_only_count / summary.total_right_records
                report_lines.append(f"  Right-Only:    {format_percentage(right_only_pct)} (of Right DataFrame)")

            report_lines.append("=" * REPORT_HEADER_LENGTH)

            return "\n".join(report_lines)

        except Exception as e:
            self._handle_error(e, {"operation": "summary_report_generation"})

    def generate_detailed_report(
        self,
        *,
        results: ComparisonResult,
        max_samples: int = DEFAULT_MAX_SAMPLES,
        table_format: str = "grid",
    ) -> str:
        """Generate detailed report with sample differences.

        Args:
            results: ComparisonResult object containing the comparison data.
            max_samples: Maximum number of sample records to include.
            table_format: Table format for tabulate (grid, simple, pipe, orgtbl, etc.).

        Returns:
            Formatted detailed report string.
        """
        self._validate_inputs(results=results)
        try:
            report_lines = [self.generate_summary_report(results=results), ""]

            # Add value differences samples
            if results.summary.value_differences_count > ZERO_THRESHOLD:
                report_lines.extend([
                    VALUE_DIFFERENCES_SECTION,
                    SECTION_SEPARATOR,
                ])

                # Get sample of value differences
                sample_diff = results.value_differences.limit(max_samples).collect()
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
            if results.summary.left_only_count > ZERO_THRESHOLD:
                report_lines.extend([
                    LEFT_ONLY_SECTION,
                    SECTION_SEPARATOR,
                ])

                sample_left = results.left_only_records.limit(max_samples).collect()
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
            if results.summary.right_only_count > ZERO_THRESHOLD:
                report_lines.extend([
                    RIGHT_ONLY_SECTION,
                    SECTION_SEPARATOR,
                ])

                sample_right = results.right_only_records.limit(max_samples).collect()
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

        except Exception as e:
            self._handle_error(e, {"operation": "detailed_report_generation"})

    def generate_summary_table(self, *, results: ComparisonResult, table_format: str = "grid") -> str:
        """Generate summary statistics as a formatted table.

        Args:
            results: ComparisonResult object containing the comparison data.
            table_format: Table format for tabulate (grid, simple, pipe, orgtbl, etc.).

        Returns:
            Formatted summary table string.
        """
        self._validate_inputs(results=results)
        try:
            summary = results.summary

            # Prepare table data
            table_data = [
                ["Left DataFrame", format_large_number(summary.total_left_records), "records"],
                ["Right DataFrame", format_large_number(summary.total_right_records), "records"],
                ["Matching Records", format_large_number(summary.matching_records), "records"],
                ["Value Differences", format_large_number(summary.value_differences_count), "records"],
                ["Left-Only Records", format_large_number(summary.left_only_count), "records"],
                ["Right-Only Records", format_large_number(summary.right_only_count), "records"],
            ]

            # Calculate percentages if applicable
            if summary.total_left_records > ZERO_THRESHOLD:
                match_pct = summary.matching_records / summary.total_left_records
                diff_pct = summary.value_differences_count / summary.total_left_records
                left_only_pct = summary.left_only_count / summary.total_left_records

                table_data.extend([
                    ["Matching %", format_percentage(match_pct), "of left records"],
                    ["Differences %", format_percentage(diff_pct), "of left records"],
                    ["Left-Only %", format_percentage(left_only_pct), "of left records"],
                ])

            if summary.total_right_records > ZERO_THRESHOLD:
                right_only_pct = summary.right_only_count / summary.total_right_records
                table_data.append(["Right-Only %", format_percentage(right_only_pct), "of right records"])

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

        except Exception as e:
            self._handle_error(e, {"operation": "summary_table_generation"})

    def export_results(
        self,
        *,
        results: ComparisonResult,
        format: str = DEFAULT_FORMAT,
        output_dir: str = DEFAULT_OUTPUT_DIR,
    ) -> dict[str, str]:
        """Export results to files.

        Args:
            results: ComparisonResult object containing the comparison data.
            format: Output format for files (parquet, csv, json).
            output_dir: Directory to save output files.

        Returns:
            Dictionary mapping result type to file path.
        """
        self._validate_inputs(results=results)
        try:
            # Validate format
            if format not in FileOperationConstants.SUPPORTED_FORMATS:
                raise ValueError(f"Unsupported format: {format}. Supported formats are: {', '.join(FileOperationConstants.SUPPORTED_FORMATS)}")

            output_path = Path(output_dir)

            # Validate that the output directory path is valid and accessible
            parent_dir = output_path.parent

            # For absolute paths, check if parent directory exists and is writable
            if output_path.is_absolute():
                if not parent_dir.exists():
                    raise ValueError(f"Parent directory does not exist: {parent_dir}")
                if not os.access(parent_dir, os.W_OK):
                    raise ValueError(f"Parent directory is not writable: {parent_dir}")
            else:
                # For relative paths, check if parent directory exists and is writable
                if str(parent_dir) != "." and not parent_dir.exists():
                    raise ValueError(f"Parent directory does not exist: {parent_dir}")
                if str(parent_dir) != "." and not os.access(parent_dir, os.W_OK):
                    raise ValueError(f"Parent directory is not writable: {parent_dir}")

            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
            exported_files = {}

            # Export value differences
            if results.value_differences.select(pl.len()).collect().item() > ZERO_THRESHOLD:
                value_diff_path = output_path / VALUE_DIFFERENCES_FILENAME.format(timestamp, format)
                self._export_lazyframe(
                    lazyframe=results.value_differences,
                    file_path=value_diff_path,
                    format=format
                )
                exported_files["value_differences"] = str(value_diff_path)

            # Export left-only records
            if results.left_only_records.select(pl.len()).collect().item() > ZERO_THRESHOLD:
                left_only_path = output_path / LEFT_ONLY_FILENAME.format(timestamp, format)
                self._export_lazyframe(
                    lazyframe=results.left_only_records,
                    file_path=left_only_path,
                    format=format
                )
                exported_files["left_only_records"] = str(left_only_path)

            # Export right-only records
            if results.right_only_records.select(pl.len()).collect().item() > ZERO_THRESHOLD:
                right_only_path = output_path / RIGHT_ONLY_FILENAME.format(timestamp, format)
                self._export_lazyframe(
                    lazyframe=results.right_only_records,
                    file_path=right_only_path,
                    format=format
                )
                exported_files["right_only_records"] = str(right_only_path)

            # Export summary
            summary_path = output_path / SUMMARY_FILENAME.format(timestamp)
            self._export_summary(results=results, file_path=summary_path)
            exported_files["summary"] = str(summary_path)

            return exported_files

        except Exception as e:
            self._handle_error(e, {"operation": "results_export"})

    def _export_lazyframe(
        self,
        *,
        lazyframe: pl.LazyFrame,
        file_path: Path,
        format: str = DEFAULT_FORMAT
    ) -> None:
        """Export a LazyFrame to a file.

        Args:
            lazyframe: LazyFrame to export.
            file_path: Path to save the file.
            format: Output format.
        """
        try:
            export_lazyframe(lazyframe, file_path, format)
        except Exception as e:
            self._handle_error(e, {"operation": "lazyframe_export", "format": format})

    def _export_summary(self, *, results: ComparisonResult, file_path: Path) -> None:
        """Export summary to JSON file.

        Args:
            results: ComparisonResult object containing the comparison data.
            file_path: Path to save the summary file.
        """
        try:
            summary = results.summary
            summary_dict = {
                "total_left_records": summary.total_left_records,
                "total_right_records": summary.total_right_records,
                "matching_records": summary.matching_records,
                "value_differences_count": summary.value_differences_count,
                "left_only_count": summary.left_only_count,
                "right_only_count": summary.right_only_count,
                "comparison_timestamp": summary.comparison_timestamp,
            }

            with open(file_path, "w") as f:
                json.dump(summary_dict, f, indent=JSON_INDENT)

        except Exception as e:
            self._handle_error(e, {"operation": "summary_export"})
