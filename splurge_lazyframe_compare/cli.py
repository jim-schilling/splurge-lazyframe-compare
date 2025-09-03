import argparse
import json
from pathlib import Path

import polars as pl

from splurge_lazyframe_compare.services.orchestrator import ComparisonOrchestrator
from splurge_lazyframe_compare.utils.config_helpers import (
    apply_environment_overrides,
    load_config_from_file,
    validate_config,
)
from splurge_lazyframe_compare.utils.logging_helpers import configure_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slc",
        description="Splurge LazyFrame Compare CLI",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--config", help="Path to JSON configuration file")

    sub = parser.add_subparsers(dest="command", required=True)

    # compare subcommand
    compare = sub.add_parser("compare", help="Compare two LazyFrames")
    compare.add_argument("--left", required=False, help="Path to left dataset (parquet/csv/ndjson)")
    compare.add_argument("--right", required=False, help="Path to right dataset (parquet/csv/ndjson)")
    compare.add_argument("--format", default="parquet", choices=["parquet", "csv", "json"], help="Export format")
    compare.add_argument("--output-dir", default="comparison_results", help="Output directory for exports")
    compare.add_argument("--dry-run", action="store_true", help="Validate inputs and exit")

    # report subcommand
    report = sub.add_parser("report", help="Generate report from comparison results")
    report.add_argument("--dry-run", action="store_true", help="Validate inputs and exit")

    # export subcommand
    export = sub.add_parser("export", help="Export comparison results to files")
    export.add_argument("--left", required=False, help="Path to left dataset (parquet/csv/ndjson)")
    export.add_argument("--right", required=False, help="Path to right dataset (parquet/csv/ndjson)")
    export.add_argument("--format", default="parquet", choices=["parquet", "csv", "json"], help="Export format")
    export.add_argument("--output-dir", default="comparison_results", help="Output directory for exports")
    export.add_argument("--dry-run", action="store_true", help="Validate inputs and exit")

    return parser


def _load_and_validate_config(config_path: str | None) -> dict:
    base_cfg: dict = {}
    if config_path:
        base_cfg = load_config_from_file(config_path)
        errors = validate_config(base_cfg)
        if errors:
            raise ValueError("Invalid configuration: " + "; ".join(errors))
    # Apply environment overrides (supports SPLURGE_ prefix)
    return apply_environment_overrides(base_cfg or {})


def _scan_lazyframe(path_str: str | None) -> pl.LazyFrame | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(p)
    if suffix == ".csv":
        return pl.scan_csv(p)
    if suffix in (".json", ".ndjson"):
        return pl.scan_ndjson(p)
    raise ValueError(f"Unsupported file extension: {suffix}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Configure logging for CLI runs
    configure_logging(level=args.log_level)

    # Load config + environment overrides
    try:
        cfg = _load_and_validate_config(args.config)
    except Exception as e:  # noqa: BLE001
        print(f"Configuration error: {e}")
        return 2

    if args.command == "compare":
        if args.dry_run:
            print("Dry run: compare would execute with provided parameters")
            return 0
        try:
            left_lf = _scan_lazyframe(args.left)
            right_lf = _scan_lazyframe(args.right)
            if left_lf is None or right_lf is None:
                print("Error: --left and --right paths are required unless using a config that supplies sources")
                return 2

            # Build ComparisonConfig from config mappings
            from splurge_lazyframe_compare.models.schema import (
                ColumnMapping,
                ComparisonConfig,
                ComparisonSchema,
            )

            # Derive schemas from data and config
            left_schema = left_lf.collect_schema()
            right_schema = right_lf.collect_schema()

            def to_schema(schema: pl.Schema) -> dict:
                from splurge_lazyframe_compare.models.schema import ColumnDefinition
                return {
                    name: ColumnDefinition(name=name, alias=name, datatype=dtype, nullable=True)
                    for name, dtype in zip(schema.names(), schema.dtypes(), strict=False)
                }

            pk_cols = cfg.get("primary_key_columns", [])
            mappings_cfg = cfg.get("column_mappings", [])
            mappings = [
                ColumnMapping(name=m["name"], left=m["left"], right=m["right"]) for m in mappings_cfg
            ]

            config = ComparisonConfig(
                left_schema=ComparisonSchema(columns=to_schema(left_schema), pk_columns=pk_cols),
                right_schema=ComparisonSchema(columns=to_schema(right_schema), pk_columns=[
                    next((m["right"] for m in mappings_cfg if m["name"] == pk), pk) for pk in pk_cols
                ]),
                column_mappings=mappings,
                pk_columns=pk_cols,
                ignore_case=cfg.get("ignore_case", False),
                null_equals_null=cfg.get("null_equals_null", True),
                tolerance=cfg.get("tolerance", {}),
            )

            orch = ComparisonOrchestrator()
            result = orch.compare_dataframes(config=config, left=left_lf, right=right_lf)
            report = orch.generate_report_from_result(result=result, report_type="summary")
            print(report)
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"Compare failed: {e}")
            return 2
    if args.command == "report":
        if args.dry_run:
            print("Dry run: report would generate summary/detailed report")
            return 0
        print("Report command requires input parameters in future iteration")
        return 2
    if args.command == "export":
        if args.dry_run:
            print("Dry run: export would write results to output directory")
            return 0
        try:
            left_lf = _scan_lazyframe(args.left)
            right_lf = _scan_lazyframe(args.right)
            if left_lf is None or right_lf is None:
                print("Error: --left and --right paths are required unless using a config that supplies sources")
                return 2

            # Reuse config build from compare path
            from splurge_lazyframe_compare.models.schema import (
                ColumnMapping,
                ComparisonConfig,
                ComparisonSchema,
            )

            left_schema = left_lf.collect_schema()
            right_schema = right_lf.collect_schema()

            def to_schema(schema: pl.Schema) -> dict:
                from splurge_lazyframe_compare.models.schema import ColumnDefinition
                return {
                    name: ColumnDefinition(name=name, alias=name, datatype=dtype, nullable=True)
                    for name, dtype in zip(schema.names(), schema.dtypes(), strict=False)
                }

            pk_cols = cfg.get("primary_key_columns", [])
            mappings_cfg = cfg.get("column_mappings", [])
            mappings = [
                ColumnMapping(name=m["name"], left=m["left"], right=m["right"]) for m in mappings_cfg
            ]

            config = ComparisonConfig(
                left_schema=ComparisonSchema(columns=to_schema(left_schema), pk_columns=pk_cols),
                right_schema=ComparisonSchema(columns=to_schema(right_schema), pk_columns=[
                    next((m["right"] for m in mappings_cfg if m["name"] == pk), pk) for pk in pk_cols
                ]),
                column_mappings=mappings,
                pk_columns=pk_cols,
                ignore_case=cfg.get("ignore_case", False),
                null_equals_null=cfg.get("null_equals_null", True),
                tolerance=cfg.get("tolerance", {}),
            )

            orch = ComparisonOrchestrator()
            exported = orch.compare_and_export(
                config=config, left=left_lf, right=right_lf, output_dir=args.output_dir, format=args.format
            )
            print(json.dumps(exported, indent=2))
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"Export failed: {e}")
            return 2

    return 0


