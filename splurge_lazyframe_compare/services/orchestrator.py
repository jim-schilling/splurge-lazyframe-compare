"""Main orchestrator service for the comparison framework."""


import polars as pl

from splurge_lazyframe_compare.models.comparison import ComparisonResult
from splurge_lazyframe_compare.models.schema import ComparisonConfig
from splurge_lazyframe_compare.services.base_service import BaseService
from splurge_lazyframe_compare.services.comparison_service import ComparisonService
from splurge_lazyframe_compare.services.reporting_service import ReportingService


class ComparisonOrchestrator(BaseService):
    """Main orchestrator for LazyFrame comparisons."""

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
        table_format: str = "grid",
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
        format: str = "parquet",
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
        table_format: str = "grid",
    ) -> str:
        """Generate a report from existing comparison results.

        Args:
            result: ComparisonResult object containing comparison data.
            report_type: Type of report ("summary", "detailed", "table").
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
            else:
                raise ValueError(f"Unknown report type: {report_type}")

        except Exception as e:
            self._handle_error(e, {"operation": "generate_report_from_result", "report_type": report_type})


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

    def get_comparison_table(
        self,
        *,
        result: ComparisonResult,
        table_format: str = "grid",
    ) -> str:
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
