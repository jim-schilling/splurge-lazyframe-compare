"""Service-based comparator interface for the comparison framework."""


import polars as pl

from splurge_lazyframe_compare.models.comparison import ComparisonResult
from splurge_lazyframe_compare.models.schema import ComparisonConfig
from splurge_lazyframe_compare.services.orchestrator import ComparisonOrchestrator


class LazyFrameComparator:
    """Service-based LazyFrame comparator.

    Primary interface for the comparison framework, built on the
    service architecture for modular and maintainable comparisons.
    """

    def __init__(self, config: ComparisonConfig) -> None:
        """Initialize the comparator with configuration.

        Args:
            config: Comparison configuration defining schemas and mappings.
        """
        self.config = config
        self.orchestrator = ComparisonOrchestrator()

    def compare(
        self,
        *,
        left: pl.LazyFrame,
        right: pl.LazyFrame
    ) -> ComparisonResult:
        """Execute complete comparison between two LazyFrames.

        Args:
            left: Left LazyFrame to compare.
            right: Right LazyFrame to compare.

        Returns:
            ComparisonResults containing all comparison results.

        Raises:
            SchemaValidationError: If DataFrames don't match their schemas.
            PrimaryKeyViolationError: If primary key constraints are violated.
        """
        return self.orchestrator.compare_dataframes(
            config=self.config,
            left=left,
            right=right
        )

    def compare_and_report(
        self,
        *,
        left: pl.LazyFrame,
        right: pl.LazyFrame,
        include_samples: bool = True,
        max_samples: int = 10,
        table_format: str = "grid"
    ) -> str:
        """Compare DataFrames and generate a complete report.

        Args:
            left: Left LazyFrame to compare.
            right: Right LazyFrame to compare.
            include_samples: Whether to include sample records in the report.
            max_samples: Maximum number of sample records to include.
            table_format: Table format for sample data.

        Returns:
            Complete formatted comparison report.
        """
        return self.orchestrator.compare_and_report(
            config=self.config,
            left=left,
            right=right,
            include_samples=include_samples,
            max_samples=max_samples,
            table_format=table_format
        )

    def compare_and_export(
        self,
        *,
        left: pl.LazyFrame,
        right: pl.LazyFrame,
        output_dir: str = ".",
        format: str = "parquet"
    ) -> dict[str, str]:
        """Compare DataFrames and export results to files.

        Args:
            left: Left LazyFrame to compare.
            right: Right LazyFrame to compare.
            output_dir: Directory to save output files.
            format: Output format for files (parquet, csv, json).

        Returns:
            Dictionary mapping result type to file path.
        """
        return self.orchestrator.compare_and_export(
            config=self.config,
            left=left,
            right=right,
            output_dir=output_dir,
            format=format
        )

    def to_report(self) -> "ComparisonReport":
        """Create a report generator for the current configuration.

        Returns:
            ComparisonReport object for generating reports.
        """
        return ComparisonReport(self.orchestrator, self.config)


class ComparisonReport:
    """Report generator using the service architecture."""

    def __init__(
        self,
        orchestrator: ComparisonOrchestrator,
        config: ComparisonConfig
    ) -> None:
        """Initialize the report generator.

        Args:
            orchestrator: ComparisonOrchestrator instance.
            config: Comparison configuration.
        """
        self.orchestrator = orchestrator
        self.config = config
        self._last_result: ComparisonResult | None = None

    def generate_from_result(self, result: ComparisonResult) -> str:
        """Generate report from a comparison result.

        Args:
            result: ComparisonResult to generate report from.

        Returns:
            Formatted report string.
        """
        self._last_result = result
        return self.orchestrator.generate_report_from_result(
            result=result,
            report_type="detailed"
        )

    def generate_summary_report(self, result: ComparisonResult | None = None) -> str:
        """Generate summary report.

        Args:
            result: Optional comparison result. If not provided, uses last result.

        Returns:
            Formatted summary report string.
        """
        if result is None and self._last_result is None:
            raise ValueError("No comparison result available. Call generate_from_result first.")

        target_result = result or self._last_result
        return self.orchestrator.generate_report_from_result(
            result=target_result,
            report_type="summary"
        )

    def generate_detailed_report(
        self,
        *,
        result: ComparisonResult | None = None,
        max_samples: int = 10,
        table_format: str = "grid"
    ) -> str:
        """Generate detailed report with samples.

        Args:
            result: Optional comparison result. If not provided, uses last result.
            max_samples: Maximum number of sample records to include.
            table_format: Table format for sample data.

        Returns:
            Formatted detailed report string.
        """
        if result is None and self._last_result is None:
            raise ValueError("No comparison result available. Call generate_from_result first.")

        target_result = result or self._last_result
        return self.orchestrator.generate_report_from_result(
            result=target_result,
            report_type="detailed",
            max_samples=max_samples,
            table_format=table_format
        )

    def generate_summary_table(
        self,
        *,
        result: ComparisonResult | None = None,
        table_format: str = "grid"
    ) -> str:
        """Generate summary statistics as a table.

        Args:
            result: Optional comparison result. If not provided, uses last result.
            table_format: Table format for display.

        Returns:
            Formatted summary table string.
        """
        if result is None and self._last_result is None:
            raise ValueError("No comparison result available. Call generate_from_result first.")

        target_result = result or self._last_result
        return self.orchestrator.generate_report_from_result(
            result=target_result,
            report_type="table"
        )


