"""Main orchestrator service for the comparison framework."""


import polars as pl

from splurge_lazyframe_compare.models.comparison import ComparisonResult
from splurge_lazyframe_compare.models.schema import ComparisonConfig
from splurge_lazyframe_compare.services.base_service import BaseService
from splurge_lazyframe_compare.services.comparison_service import ComparisonService
from splurge_lazyframe_compare.services.reporting_service import ReportingService


class ComparisonOrchestrator(BaseService):
    """Main orchestrator for LazyFrame comparisons with HTML security."""

    @staticmethod
    def _escape_html(text: str) -> str:
        """Escape HTML special characters to prevent XSS attacks.

        Args:
            text: Text to escape

        Returns:
            HTML-escaped text
        """
        if text is None:
            return ""
        # Use a comprehensive set of HTML entity replacements
        html_entities = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '/': '&#x2F;',
            '=': '&#x3D;',  # Escape equals sign to prevent attribute injection
        }

        result = str(text)
        for char, entity in html_entities.items():
            result = result.replace(char, entity)
        return result

    def __init__(
        self,
        *,
        comparison_service: ComparisonService | None = None,
        reporting_service: ReportingService | None = None
    ) -> None:
        """Initialize the comparison orchestrator.

        Args:
            comparison_service: Optional comparison service instance.
            reporting_service: Optional reporting service instance.
        """
        super().__init__("ComparisonOrchestrator")

        # Use provided services or create defaults
        self.comparison_service = comparison_service or ComparisonService()
        self.reporting_service = reporting_service or ReportingService()

    def _validate_inputs(self, **kwargs) -> None:
        """Validate orchestrator inputs."""
        if "config" in kwargs and not isinstance(kwargs["config"], ComparisonConfig):
            raise ValueError("config must be a ComparisonConfig")
        if "left" in kwargs and not isinstance(kwargs["left"], pl.LazyFrame):
            raise ValueError("left must be a polars LazyFrame")
        if "right" in kwargs and not isinstance(kwargs["right"], pl.LazyFrame):
            raise ValueError("right must be a polars LazyFrame")

    def compare_dataframes(
        self,
        *,
        config: ComparisonConfig,
        left: pl.LazyFrame,
        right: pl.LazyFrame
    ) -> ComparisonResult:
        """Execute a complete comparison between two LazyFrames.

        This is the main entry point for the comparison framework. It orchestrates
        the comparison process using the service architecture.

        Args:
            config: Comparison configuration defining schemas and mappings.
            left: Left LazyFrame to compare.
            right: Right LazyFrame to compare.

        Returns:
            ComparisonResult containing all comparison results and data.

        Raises:
            SchemaValidationError: If DataFrames don't match their schemas.
            PrimaryKeyViolationError: If primary key constraints are violated.
        """
        try:
            self._validate_inputs(config=config, left=left, right=right)

            # Execute comparison using the comparison service
            result = self.comparison_service.execute_comparison(
                left=left,
                right=right,
                config=config
            )

            return result

        except Exception as e:
            self._handle_error(e, {"operation": "dataframe_comparison"})

    def compare_and_report(
        self,
        *,
        config: ComparisonConfig,
        left: pl.LazyFrame,
        right: pl.LazyFrame,
        include_samples: bool = True,
        max_samples: int = 10,
        table_format: str = "grid"
    ) -> str:
        """Compare DataFrames and generate a complete report.

        Args:
            config: Comparison configuration.
            left: Left LazyFrame to compare.
            right: Right LazyFrame to compare.
            include_samples: Whether to include sample records in the report.
            max_samples: Maximum number of sample records to include.
            table_format: Table format for sample data.

        Returns:
            Complete formatted comparison report.
        """
        try:
            self._validate_inputs(config=config, left=left, right=right)

            # Validate parameters
            if max_samples < 0:
                raise ValueError("max_samples must be non-negative")

            # Execute comparison
            result = self.compare_dataframes(
                config=config,
                left=left,
                right=right
            )

            # Generate detailed report
            if include_samples:
                report = self.reporting_service.generate_detailed_report(
                    results=result,
                    max_samples=max_samples,
                    table_format=table_format
                )
            else:
                report = self.reporting_service.generate_summary_report(results=result)

            return report

        except Exception as e:
            self._handle_error(e, {"operation": "compare_and_report"})

    def compare_and_export(
        self,
        *,
        config: ComparisonConfig,
        left: pl.LazyFrame,
        right: pl.LazyFrame,
        output_dir: str = ".",
        format: str = "parquet"
    ) -> dict[str, str]:
        """Compare DataFrames and export results to files.

        Args:
            config: Comparison configuration.
            left: Left LazyFrame to compare.
            right: Right LazyFrame to compare.
            output_dir: Directory to save output files.
            format: Output format for files (parquet, csv, json).

        Returns:
            Dictionary mapping result type to file path.
        """
        try:
            self._validate_inputs(config=config, left=left, right=right)

            # Execute comparison
            result = self.compare_dataframes(
                config=config,
                left=left,
                right=right
            )

            # Export results
            exported_files = self.reporting_service.export_results(
                results=result,
                output_dir=output_dir,
                format=format
            )

            return exported_files

        except Exception as e:
            self._handle_error(e, {"operation": "compare_and_export"})

    def generate_report_from_result(
        self,
        *,
        result: ComparisonResult,
        report_type: str = "detailed",
        max_samples: int = 10,
        table_format: str = "grid"
    ) -> str:
        """Generate a report from existing comparison results.

        Args:
            result: ComparisonResult object containing comparison data.
            report_type: Type of report ("summary", "detailed", "table", "html").
            max_samples: Maximum number of sample records to include.
            table_format: Table format for sample data.

        Returns:
            Formatted report string.
        """
        try:
            if report_type == "summary":
                return self.reporting_service.generate_summary_report(results=result)
            elif report_type == "detailed":
                return self.reporting_service.generate_detailed_report(
                    results=result,
                    max_samples=max_samples,
                    table_format=table_format
                )
            elif report_type == "table":
                return self.reporting_service.generate_summary_table(
                    results=result,
                    table_format=table_format
                )
            elif report_type == "html":
                # HTML reporting not yet implemented, return summary report
                return self.reporting_service.generate_summary_report(results=result)
            else:
                raise ValueError(f"Unknown report type: {report_type}")

        except Exception as e:
            self._handle_error(e, {"operation": "generate_report_from_result", "report_type": report_type})

    def export_result_to_html(
        self,
        *,
        result: ComparisonResult,
        filename: str
    ) -> None:
        """Export comparison result to HTML file.

        Args:
            result: ComparisonResult object containing comparison data.
            filename: Path to save the HTML file.
        """
        try:
            self._export_result_to_html_file(result, filename)

        except Exception as e:
            self._handle_error(e, {"operation": "export_result_to_html", "filename": filename})

    def _export_result_to_html_file(self, result: ComparisonResult, filename: str) -> None:
        """Export comparison result to HTML file.

        Args:
            result: ComparisonResult object containing comparison data.
            filename: Path to save the HTML file.
        """
        try:
            # Generate HTML content
            html_content = self._generate_html_report(result)

            # Write to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)

        except Exception as e:
            self._handle_error(e, {"operation": "_export_result_to_html_file", "filename": filename})

    def _generate_html_report(self, result: ComparisonResult) -> str:
        """Generate HTML report content.

        Args:
            result: ComparisonResult object containing comparison data.

        Returns:
            HTML content as string.
        """
        try:
            summary = result.summary

            # Escape data values to prevent XSS
            escaped_timestamp = self._escape_html(summary.comparison_timestamp)

            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SPLURGE LazyFrame Comparison Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .section {{
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .status-good {{ color: #28a745; }}
        .status-warning {{ color: #ffc107; }}
        .status-error {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SPLURGE LazyFrame Comparison Report</h1>
        <p>Generated on {escaped_timestamp}</p>
    </div>

    <div class="section">
        <h2>📊 Record Counts</h2>
        <div class="metric-grid">
            <div class="metric-item">
                <div class="metric-value">{summary.total_left_records:,}</div>
                <div class="metric-label">Left DataFrame Records</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">{summary.total_right_records:,}</div>
                <div class="metric-label">Right DataFrame Records</div>
            </div>
            <div class="metric-item">
                <div class="metric-value status-good">{summary.matching_records:,}</div>
                <div class="metric-label">Matching Records</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>🔍 Comparison Results</h2>
        <div class="metric-grid">
            <div class="metric-item">
                <div class="metric-value {'status-warning' if summary.value_differences_count > 0 else 'status-good'}">{summary.value_differences_count:,}</div>
                <div class="metric-label">Value Differences</div>
            </div>
            <div class="metric-item">
                <div class="metric-value {'status-warning' if summary.left_only_count > 0 else 'status-good'}">{summary.left_only_count:,}</div>
                <div class="metric-label">Left-Only Records</div>
            </div>
            <div class="metric-item">
                <div class="metric-value {'status-warning' if summary.right_only_count > 0 else 'status-good'}">{summary.right_only_count:,}</div>
                <div class="metric-label">Right-Only Records</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>📈 Percentages (of Left DataFrame)</h2>
        <div class="metric-grid">
            <div class="metric-item">
                <div class="metric-value status-good">{summary.matching_percentage:.1f}%</div>
                <div class="metric-label">Matching</div>
            </div>
            <div class="metric-item">
                <div class="metric-value {'status-warning' if summary.differences_percentage > 0 else 'status-good'}">{summary.differences_percentage:.1f}%</div>
                <div class="metric-label">Differences</div>
            </div>
            <div class="metric-item">
                <div class="metric-value {'status-warning' if summary.left_only_percentage > 0 else 'status-good'}">{summary.left_only_percentage:.1f}%</div>
                <div class="metric-label">Left-Only</div>
            </div>
        </div>
    </div>

    <div class="section">
        <h2>📋 Summary</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Status</th>
            </tr>
            <tr>
                <td>Total Records Compared</td>
                <td>{summary.total_left_records + summary.total_right_records:,}</td>
                <td class="status-good">✓</td>
            </tr>
            <tr>
                <td>Data Quality Score</td>
                <td>{(summary.matching_records / max(summary.total_left_records, 1) * 100):.1f}%</td>
                <td class="status-good">✓</td>
            </tr>
            <tr>
                <td>Issues Found</td>
                <td>{summary.value_differences_count + summary.left_only_count + summary.right_only_count:,}</td>
                <td class="{'status-error' if (summary.value_differences_count + summary.left_only_count + summary.right_only_count) > 0 else 'status-good'}">{'⚠️' if (summary.value_differences_count + summary.left_only_count + summary.right_only_count) > 0 else '✓'}</td>
            </tr>
        </table>
    </div>
</body>
</html>"""

            return html

        except Exception as e:
            self._handle_error(e, {"operation": "_generate_html_report"})

    def get_comparison_summary(self, *, result: ComparisonResult) -> str:
        """Get a quick summary of comparison results.

        Args:
            result: ComparisonResult object containing comparison data.

        Returns:
            Formatted summary string.
        """
        try:
            return self.reporting_service.generate_summary_report(results=result)

        except Exception as e:
            self._handle_error(e, {"operation": "get_comparison_summary"})

    def get_comparison_table(self, *, result: ComparisonResult, table_format: str = "grid") -> str:
        """Get comparison results as a formatted table.

        Args:
            result: ComparisonResult object containing comparison data.
            table_format: Table format for display.

        Returns:
            Formatted table string.
        """
        try:
            return self.reporting_service.generate_summary_table(
                results=result,
                table_format=table_format
            )

        except Exception as e:
            self._handle_error(e, {"operation": "get_comparison_table"})
